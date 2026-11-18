"""
Predict tags for new texts with a model trained by embedding_method.py or lora_finetune.py.

Loads the embedder of the training run (+ LoRA adapter if `lora_adapter` is set), embeds the
inputs, applies the chosen classifier and writes the predicted tags and per-tag scores to JSON.

Usage:
  uv run python -m illuin_tt.scripts.generate --config-path configs/generate.yaml input=path/to/texts
"""
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

from illuin_tt.loaders import TARGET_TAGS
from illuin_tt.models.mlp import MLP
from illuin_tt.utils import get_device


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. input=data/new.json model=mlp"
    )

    return parser.parse_args()


def _texts_from_json(path: Path, field: str) -> list[tuple[str, str]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [(str(path), data[field])]
    if isinstance(data, list):
        return [(f"{path}[{i}]", d[field]) for i, d in enumerate(data)]
    raise ValueError(f"{path}: expected a JSON object or a list of objects")


def load_inputs(input_path: str, field: str) -> list[tuple[str, str]]:
    """Return (source, text) pairs from a .txt, a .json, or a folder of .txt / .json files."""
    path = Path(input_path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix in (".txt", ".json"))
        if not files:
            raise ValueError(f"No .txt or .json files in {path}")
        return [doc for p in files for doc in load_inputs(str(p), field)]
    if path.suffix == ".txt":
        return [(str(path), path.read_text())]

    if path.suffix == ".json":
        return _texts_from_json(path, field)

    raise ValueError(f"Unsupported input {path}: expected a .txt, a .json or a folder")


def load_embedder(cfg, run_cfg, device) -> SentenceTransformer:
    """Embedder of the training run, with its dtype / max_seq_length and optional LoRA adapter."""
    model_kwargs = {"dtype": getattr(torch, run_cfg.dtype)} if run_cfg.get("dtype") else {}
    model = SentenceTransformer(run_cfg.model_name, device=str(device), model_kwargs=model_kwargs)
    if run_cfg.get("max_seq_length"):
        model.max_seq_length = run_cfg.max_seq_length

    if cfg.get("lora_adapter"):
        print(f"Loading LoRA adapter from {cfg.lora_adapter}")
        model[0].auto_model.load_adapter(str(cfg.lora_adapter))
    model.eval()
    return model


def load_classifier(cfg, run_dir: Path, run_cfg):
    """MLP (models/mlp.pt, or head.pt for LoRA runs) or logistic regression (models/*.joblib)."""
    if cfg.model == "logistic_regression":
        return joblib.load(run_dir / "models" / "logistic_regression.joblib")
    if cfg.model == "mlp":
        candidates = [run_dir / "models" / "mlp.pt", run_dir / "head.pt"]
        weights_path = next((p for p in candidates if p.exists()), None)
        if weights_path is None:
            raise FileNotFoundError(f"No models/mlp.pt or head.pt in {run_dir}")
        state = torch.load(weights_path, map_location="cpu")
        # Input / output sizes come from the first / last Linear layers
        weights = [v for k, v in state.items() if k.endswith("weight")]
        mlp = MLP(run_cfg.n_layers, weights[0].shape[1], run_cfg.hidden_size, weights[-1].shape[0])
        mlp.load_state_dict(state)
        return mlp.eval()
    raise ValueError(f"Unknown model {cfg.model!r}, choose 'mlp' or 'logistic_regression'")


def main():
    args = get_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_path), OmegaConf.from_dotlist(args.overrides))

    run_dir = Path(cfg.run_dir)
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    field = cfg.get("field") or run_cfg.field

    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    # Recover the input texts
    docs = load_inputs(cfg.input, field)
    print(f"Loaded {len(docs)} documents from {cfg.input}")

    # Load the embedder and classifier
    embedder = load_embedder(cfg, run_cfg, device)
    classifier = load_classifier(cfg, run_dir, run_cfg)
    if isinstance(classifier, MLP):
        classifier.to(device)

    # Encode the texts
    start = time.perf_counter()
    emb = embedder.encode(
        [text for _, text in docs],
        batch_size=cfg.get("batch_size", 8),
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    emb = emb.float().cpu().numpy()
    embed_time = time.perf_counter() - start

    # Make the tag predictions
    start = time.perf_counter()
    scores = np.asarray(classifier.predict_proba(emb))
    preds = np.asarray(classifier.predict(emb)).astype(int)
    classify_time = time.perf_counter() - start

    tags = list(TARGET_TAGS.values())
    predictions = [
        {
            "source": source,
            "tags": [t for t, p in zip(tags, pred) if p],
            "scores": {t: round(float(s), 4) for t, s in zip(tags, score)},
        }
        for (source, _), pred, score in zip(docs, preds, scores)
    ]

    output = Path(cfg.get("output", "predictions.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, indent=2))

    for p in predictions[:10]:
        print(f"{p['source']}: {', '.join(p['tags']) or '-'}")
    if len(predictions) > 10:
        print(f"... ({len(predictions) - 10} more)")

    total_time = embed_time + classify_time
    print(
        f"Inference time: {total_time:.2f}s for {len(docs)} documents "
        f"({1000 * total_time / len(docs):.1f} ms/doc, "
        f"embedding {embed_time:.2f}s, classification {classify_time:.3f}s)"
    )
    print(f"Predictions saved to: {output}")


if __name__ == "__main__":
    main()
