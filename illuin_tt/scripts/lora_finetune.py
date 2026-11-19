import json
import math
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
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
from illuin_tt.report import summary_table, threshold_section, write_report
from illuin_tt.utils import get_device


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. lora.r=8 num_epochs=1"
    )

    return parser.parse_args()


def collate(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.as_tensor(np.stack(labels))


def load_embedder(cfg, device) -> SentenceTransformer:
    model = SentenceTransformer(
        cfg.model_name, device=str(device), model_kwargs={"dtype": getattr(torch, cfg.dtype)}
    )
    model.max_seq_length = cfg.max_seq_length

    backbone = model[0].auto_model

    # No LoRA config: frozen encoder
    if cfg.get("lora") is None:
        model.requires_grad_(False)
        return model

    # Use activation checkpointing to save memory on forward pass
    if cfg.get("gradient_checkpointing", False):
        backbone.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})

    # Adds the adapters and freezes every non-LoRA weight
    backbone.add_adapter(
        LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            task_type="FEATURE_EXTRACTION",
        )
    )
    return model


def embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    """Differentiable forward pass: texts -> fp32 sentence embeddings."""
    features = model.preprocess(texts)
    features = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in features.items()}
    return model(features)["sentence_embedding"].float()


def trainable_state(*modules: nn.Module) -> list[dict]:
    return [
        {k: v.detach().clone() for k, v in m.named_parameters() if v.requires_grad} for m in modules
    ]


@torch.no_grad()
def eval_loss(model, head, dataset, criterion, batch_size):
    model.eval()
    head.eval()
    total, n = 0.0, 0
    for texts, labels in DataLoader(dataset, batch_size, collate_fn=collate):
        labels = labels.to(model.device)
        total += criterion(head(embed(model, texts)), labels).item() * len(texts)
        n += len(texts)
    return total / n


def finetune(cfg, model, head, train_dataset, val_dataset):
    device = model.device

    # Print number of trainahle parameters
    lora_params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable LoRA params: {n_lora:,} / {n_total:,} ({100 * n_lora / n_total:.2f}%)")
    print(f"Trainable head params: {sum(p.numel() for p in head.parameters()):,}")

    dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, collate_fn=collate)

    # Weight positives by neg/pos ratio to counter class imbalance (rare tags)
    pos = train_dataset.labels.sum(0)
    pos_weight = torch.tensor(
        (len(train_dataset) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Setup the optimizer and scheduler
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": cfg.lr_lora},
            {"params": head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = math.ceil(len(dataloader) / cfg.grad_accum) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.get("warmup_ratio", 0.1) * total_steps), total_steps
    )

    # Launch the training loop
    log = []
    best_loss, best_state = float("inf"), None
    for epoch in range(cfg.num_epochs):
        model.train()
        head.train()
        running, n = 0.0, 0
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")
        for step, (texts, labels) in enumerate(pbar):
            emb = embed(model, texts)
            loss = criterion(head(emb), labels.to(device))

            (loss / cfg.grad_accum).backward()
            running, n = running + loss.item() * len(texts), n + len(texts)

            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    lora_params + list(head.parameters()), cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{running / n:.4f}")

        val_loss = eval_loss(model, head, val_dataset, criterion, cfg.batch_size)
        log.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss})
        tqdm.write(f"epoch {epoch}: train loss {running / n:.4f}, val loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_state = val_loss, trainable_state(model, head)

    # Keep the epoch with the best validation loss (only trainable weights changed)
    model.load_state_dict(best_state[0], strict=False)
    head.load_state_dict(best_state[1], strict=False)
    model.eval()
    head.eval()
    return log


@torch.no_grad()
def encode_dataset(model, dataset, batch_size) -> TensorDataset:
    emb = model.encode(dataset.texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    return TensorDataset(emb.float().cpu(), torch.as_tensor(dataset.labels))


def main():
    args = get_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_path), OmegaConf.from_dotlist(args.overrides))

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
        "train": TextDataset(files_train, y_train, cfg.field),
        "val": TextDataset(files_val, y_val, cfg.field),
        "test": TextDataset(files_test, y_test, cfg.field),
    }

    model = load_embedder(cfg, device)
    head = MLP(cfg.n_layers, model.get_embedding_dimension(), cfg.hidden_size, num_classes).to(device)

    train_log = finetune(cfg, model, head, datasets["train"], datasets["val"])

    # Re-embed every split with the fine-tuned model, then evaluate like embedding_method.py
    print("Encoding the splits with the fine-tuned model")
    encoded = {k: encode_dataset(model, d, cfg.get("encode_batch_size", 16)) for k, d in datasets.items()}
    train_emb, train_labels = encoded["train"].tensors
    lr_est = LrMulticlass(num_classes).fit(train_emb.numpy(), train_labels.numpy())

    # Per-tag decision thresholds of the head, chosen on the fine-tuned val embeddings
    extra_sections = []
    if cfg.get("tune_threshold", True):
        val_emb, val_labels = encoded["val"].tensors
        ths = head.tune_thresholds(val_emb.numpy(), val_labels.numpy())
        print("Tuned head thresholds: " + ", ".join(f"{t}={x:.2f}" for t, x in zip(TARGET_TAGS.values(), ths)))
        extra_sections.append(
            threshold_section(head, val_emb.numpy(), val_labels.numpy(), "LoRA + MLP head")
        )

    results = {
        "LoRA + MLP head": eval_model(head, encoded["test"]),
        "LoRA + MLP head (ths=0.5)": eval_model(FixedThreshold(head, 0.5), encoded["test"]),
        "LoRA + Logistic regression": eval_model(lr_est, encoded["test"]),
    }

    # Overrides (except output_dir) go in the run name so runs are distinguishable
    tags = [o.replace("=", "").replace("/", "-") for o in args.overrides if not o.startswith("output_dir=")]
    run_name = "_".join([Path(args.config_path).stem, *tags, f"{datetime.now():%Y%m%d-%H%M%S}"])
    out_dir = Path(cfg.get("output_dir", "outputs")) / run_name
    write_report(out_dir, cfg, results, {k: len(d) for k, d in datasets.items()}, extra_sections)

    # Fine-tuned weights: LoRA adapter only (base model is untouched) + head
    model[0].auto_model.save_pretrained(out_dir / "lora_adapter")
    torch.save(head.state_dict(), out_dir / "head.pt")
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))

    print(summary_table(results))
    print(f"Report saved to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
