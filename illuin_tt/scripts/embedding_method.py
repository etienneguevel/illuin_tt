import copy
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from illuin_tt.datasets.embedding_dataset import EmbeddingDataset
from illuin_tt.eval import eval_model
from illuin_tt.loaders import TARGET_TAGS, make_train_test_val
from illuin_tt.models.knn import KnnMulticlass
from illuin_tt.models.lr import LrMulticlass
from illuin_tt.models.mlp import MLP, FixedThreshold
from illuin_tt.report import summary_table, threshold_section, write_report
from illuin_tt.utils import get_device


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. n_layers=3 hidden_size=512"
    )

    return parser.parse_args()


@torch.no_grad()
def eval_loss(model, dataset, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for emb, labels in DataLoader(dataset, 512):
        emb, labels = emb.to(device), labels.to(device)
        total += criterion(model(emb), labels).item() * len(emb)
        n += len(emb)
    return total / n


def mlp_fit(cfg, train_dataset, val_dataset, device):
    mlp = MLP(
        cfg.n_layers,
        train_dataset.emb_dim,
        cfg.hidden_size,
        train_dataset.num_classes
    ).to(device)

    dataloader = DataLoader(train_dataset, cfg.get("batch_size", 128), shuffle=True)

    # Weight positives by neg/pos ratio to counter class imbalance (rare tags)
    pos = train_dataset.labels.sum(0)
    pos_weight = torch.tensor((len(train_dataset) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(mlp.parameters(), lr=cfg.get("lr", 1e-3))

    # Start the training loop
    best_loss, best_state = float("inf"), None

    for epoch in tqdm(range(cfg.num_epochs)):
        mlp.train()
        for emb, labels in dataloader:
            emb, labels = emb.to(device), labels.to(device)
            loss = criterion(mlp(emb), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = eval_loss(mlp, val_dataset, criterion, device)
        tqdm.write(f"epoch {epoch}: val loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(mlp.state_dict())

    # Keep the epoch with the best validation loss
    mlp.load_state_dict(best_state)
    mlp.eval()
    return mlp


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

    print("Making the data dataset")
    cache_dir = Path(cfg.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    datasets = {
        split: EmbeddingDataset(
            cfg.model_name, files, labels, cfg.field, str(cache_dir / f"{split}.npy"), num_classes, cfg.max_tokens, device
            )
        for split, files, labels in [("train", files_train, y_train), ("val", files_val, y_val), ("test", files_test, y_test)]
    }

    # Make the different classifiers
    lr_est = LrMulticlass(num_classes).fit(datasets["train"].cache, datasets["train"].labels)
    knn_est = KnnMulticlass(num_classes, **cfg.get("knn", {})).fit(
        datasets["train"].cache, datasets["train"].labels
    )
    mlp = mlp_fit(cfg, datasets["train"], datasets["val"], device)

    # Per-tag decision thresholds of the MLP, chosen on the val split
    extra_sections = []
    if cfg.get("tune_threshold", True):
        ths = mlp.tune_thresholds(datasets["val"].cache, datasets["val"].labels)
        print("Tuned MLP thresholds: " + ", ".join(f"{t}={x:.2f}" for t, x in zip(TARGET_TAGS.values(), ths)))
        extra_sections.append(threshold_section(mlp, datasets["val"].cache, datasets["val"].labels))

    # Eval the models
    results = {
        "Logistic regression": eval_model(lr_est, datasets["test"]),
        "MLP": eval_model(mlp, datasets["test"]),
        "MLP (ths=0.5)": eval_model(FixedThreshold(mlp, 0.5), datasets["test"]),
        "kNN": eval_model(knn_est, datasets["test"]),
    }

    # Save a human-readable report (+ JSON copy) in a timestamped run directory
    # Overrides (except output_dir) go in the run name so sweep runs are distinguishable
    tags = [o.replace("=", "").replace("/", "-") for o in args.overrides if not o.startswith("output_dir=")]
    run_name = "_".join([Path(args.config_path).stem, *tags, f"{datetime.now():%Y%m%d-%H%M%S}"])
    out_dir = Path(cfg.get("output_dir", "outputs")) / run_name
    write_report(out_dir, cfg, results, {k: len(d) for k, d in datasets.items()}, extra_sections)

    # Save the trained models (MLP state dict includes the tuned thresholds, on CPU so it
    # loads anywhere; rebuild it with n_layers / hidden_size from config.yaml)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in mlp.state_dict().items()}, models_dir / "mlp.pt")
    joblib.dump(lr_est, models_dir / "logistic_regression.joblib")
    print(f"Models saved to: {models_dir}")

    print(summary_table(results))
    print(f"Report saved to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
