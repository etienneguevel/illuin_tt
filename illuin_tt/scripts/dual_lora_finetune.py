"""
LoRA fine-tuning with one encoder per field (problem description + source code).

Each field listed under `encoders` in the config is embedded by its own encoder (LoRA or frozen),
an attention pooling layer mixes the field embeddings per sample, and the MLP head is trained on
the pooled embedding. Training / evaluation / outputs otherwise mirror lora_finetune.py.

Usage:
  uv run python -m illuin_tt.scripts.dual_lora_finetune --config-path configs/Qwen3-Embedding-4B_dual_lora.yaml
"""
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from illuin_tt.datasets.text_dataset import TextDataset
from illuin_tt.eval import eval_model
from illuin_tt.loaders import TARGET_TAGS, make_train_test_val
from illuin_tt.models.lr import LrMulticlass
from illuin_tt.models.mlp import MLP, FixedThreshold
from illuin_tt.models.pooling import AttentionPooling
from illuin_tt.report import _md_table, summary_table, threshold_section, write_report
from illuin_tt.scripts.lora_finetune import embed, get_args, load_embedder, trainable_state
from illuin_tt.utils import get_device

MODEL_NAME = "Dual LoRA"


def collate(batch):
    texts, labels = zip(*batch)
    return {f: [t[f] for t in texts] for f in texts[0]}, torch.as_tensor(np.stack(labels))


def is_trainable(model) -> bool:
    return any(p.requires_grad for p in model.parameters())


def set_train(models, pooling, head, mode: bool):
    # Frozen encoders always stay in eval mode
    for m in models.values():
        m.train(mode and is_trainable(m))
    pooling.train(mode)
    head.train(mode)


def fused(models, pooling, texts: dict[str, list[str]]) -> torch.Tensor:
    """Differentiable forward pass: {field: texts} -> pooled fp32 embeddings."""
    embs = {}
    for f, model in models.items():
        with torch.set_grad_enabled(torch.is_grad_enabled() and is_trainable(model)):
            embs[f] = embed(model, texts[f])
    return pooling(embs)


@torch.no_grad()
def eval_loss(models, pooling, head, dataset, criterion, batch_size, device):
    set_train(models, pooling, head, False)
    total, n = 0.0, 0
    for texts, labels in DataLoader(dataset, batch_size, collate_fn=collate):
        logits = head(fused(models, pooling, texts))
        total += criterion(logits, labels.to(device)).item() * len(labels)
        n += len(labels)
    return total / n


def finetune(cfg, models, pooling, head, train_dataset, val_dataset, device):
    # Trainable params per encoder, each group with its own lr
    param_groups, trainable = [], []
    for f, model in models.items():
        params = [p for p in model.parameters() if p.requires_grad]
        n_total = sum(p.numel() for p in model.parameters())
        n_lora = sum(p.numel() for p in params)
        print(f"[{f}] trainable LoRA params: {n_lora:,} / {n_total:,} ({100 * n_lora / n_total:.2f}%)")
        if params:
            lr = cfg.encoders[f].get("lr_lora", cfg.lr_lora)
            param_groups.append({"params": params, "lr": lr})
            trainable += params

    print(f"Trainable pooling params: {sum(p.numel() for p in pooling.parameters()):,}")
    print(f"Trainable head params: {sum(p.numel() for p in head.parameters()):,}")
    param_groups += [
        {"params": list(pooling.parameters()), "lr": cfg.lr_pooling},
        {"params": list(head.parameters()), "lr": cfg.lr_head},
    ]
    trainable += list(pooling.parameters()) + list(head.parameters())

    dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, collate_fn=collate)

    # Weight positives by neg/pos ratio to counter class imbalance (rare tags)
    pos = train_dataset.labels.sum(0)
    pos_weight = torch.tensor(
        (len(train_dataset) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.01))
    total_steps = math.ceil(len(dataloader) / cfg.grad_accum) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.get("warmup_ratio", 0.1) * total_steps), total_steps
    )

    modules = [*models.values(), pooling, head]
    log = []
    best_loss, best_state = float("inf"), None
    for epoch in range(cfg.num_epochs):
        set_train(models, pooling, head, True)
        running, n = 0.0, 0
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")
        for step, (texts, labels) in enumerate(pbar):
            loss = criterion(head(fused(models, pooling, texts)), labels.to(device))

            (loss / cfg.grad_accum).backward()
            running, n = running + loss.item() * len(labels), n + len(labels)

            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable, cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{running / n:.4f}")

        val_loss = eval_loss(models, pooling, head, val_dataset, criterion, cfg.batch_size, device)
        log.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss})
        tqdm.write(f"epoch {epoch}: train loss {running / n:.4f}, val loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_state = val_loss, trainable_state(*modules)

    # Keep the epoch with the best validation loss (only trainable weights changed)
    for m, state in zip(modules, best_state):
        m.load_state_dict(state, strict=False)
    set_train(models, pooling, head, False)
    return log


@torch.no_grad()
def encode_dataset(models, pooling, dataset, batch_size, device) -> tuple[TensorDataset, torch.Tensor]:
    """Pooled embeddings of a split (+ the per-sample attention weights)."""
    embs = {
        f: model.encode(dataset.texts[f], batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True).float()
        for f, model in models.items()
    }
    pooled, weights = [], []
    for i in range(0, len(dataset), 256):
        p, w = pooling({f: e[i : i + 256].to(device) for f, e in embs.items()}, return_weights=True)
        pooled.append(p.cpu())
        weights.append(w.cpu())
    return TensorDataset(torch.cat(pooled), torch.as_tensor(dataset.labels)), torch.cat(weights)


def attention_section(fields: list[str], weights: torch.Tensor, labels: np.ndarray) -> str:
    """Mean attention weight per field on the test split, overall and per (positive) tag."""
    weights, labels = weights.numpy(), np.asarray(labels).astype(bool)
    rows = [["all", *(f"{x:.3f}" for x in weights.mean(0)), str(len(weights))]]
    for i, tag in TARGET_TAGS.items():
        mask = labels[:, i]
        if mask.any():
            rows.append([tag, *(f"{x:.3f}" for x in weights[mask].mean(0)), str(int(mask.sum()))])
    return "\n".join([
        "## Attention pooling weights",
        "",
        "Mean attention weight given to each field on the test split (per tag: samples with that tag).",
        "",
        _md_table(["Tag", *fields, "Support"], rows),
    ])


def main():
    args = get_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_path), OmegaConf.from_dotlist(args.overrides))
    fields = list(cfg.encoders)

    # Seed everything (the split relies on numpy's global RNG)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    files_train, y_train, files_val, y_val, files_test, y_test = make_train_test_val(
        cfg.data_path, True
    )
    num_classes = y_train.shape[-1]

    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    datasets = {
        "train": TextDataset(files_train, y_train, fields),
        "val": TextDataset(files_val, y_val, fields),
        "test": TextDataset(files_test, y_test, fields),
    }

    models = {f: load_embedder(cfg.encoders[f], device) for f in fields}
    pooling = AttentionPooling(
        {f: m.get_embedding_dimension() for f, m in models.items()},
        cfg.pooling.dim,
        cfg.pooling.get("dropout", 0.0),
    ).to(device)
    head = MLP(cfg.n_layers, cfg.pooling.dim, cfg.hidden_size, num_classes).to(device)

    train_log = finetune(cfg, models, pooling, head, datasets["train"], datasets["val"], device)

    # Re-embed every split with the fine-tuned encoders + pooling, then evaluate
    print("Encoding the splits with the fine-tuned models")
    encoded, att_weights = {}, {}
    for k, d in datasets.items():
        encoded[k], att_weights[k] = encode_dataset(
            models, pooling, d, cfg.get("encode_batch_size", 16), device
        )
    train_emb, train_labels = encoded["train"].tensors
    lr_est = LrMulticlass(num_classes).fit(train_emb.numpy(), train_labels.numpy())

    extra_sections = [attention_section(fields, att_weights["test"], datasets["test"].labels)]
    if cfg.get("tune_threshold", True):
        val_emb, val_labels = encoded["val"].tensors
        ths = head.tune_thresholds(val_emb.numpy(), val_labels.numpy())
        print("Tuned head thresholds: " + ", ".join(f"{t}={x:.2f}" for t, x in zip(TARGET_TAGS.values(), ths)))
        extra_sections.append(
            threshold_section(head, val_emb.numpy(), val_labels.numpy(), f"{MODEL_NAME} + MLP head")
        )

    results = {
        f"{MODEL_NAME} + MLP head": eval_model(head, encoded["test"]),
        f"{MODEL_NAME} + MLP head (ths=0.5)": eval_model(FixedThreshold(head, 0.5), encoded["test"]),
        f"{MODEL_NAME} + Logistic regression": eval_model(lr_est, encoded["test"]),
    }

    # Overrides (except output_dir) go in the run name so runs are distinguishable
    tags = [o.replace("=", "").replace("/", "-") for o in args.overrides if not o.startswith("output_dir=")]
    # Nested overrides (encoders.<field>.<key>) get long: cap the name below the 255 chars limit
    run_name = "_".join([Path(args.config_path).stem, *tags])[:200] + f"_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = Path(cfg.get("output_dir", "outputs")) / run_name
    write_report(out_dir, cfg, results, {k: len(d) for k, d in datasets.items()}, extra_sections)

    # Fine-tuned weights: LoRA adapters only (base models untouched) + pooling + head + LR
    for f, model in models.items():
        if is_trainable(model):
            model[0].auto_model.save_pretrained(out_dir / "lora_adapters" / f)
    torch.save(pooling.state_dict(), out_dir / "pooling.pt")
    torch.save(head.state_dict(), out_dir / "head.pt")
    (out_dir / "models").mkdir(exist_ok=True)
    joblib.dump(lr_est, out_dir / "models" / "logistic_regression.joblib")
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))

    print(summary_table(results))
    print(f"Report saved to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
