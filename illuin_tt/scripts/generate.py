"""
Predict tags for new texts with a model trained by embedding_method.py, lora_finetune.py or
dual_lora_finetune.py.

Loads the embedder of the training run (+ LoRA adapter if `lora_adapter` is set), embeds the
inputs, applies the chosen classifier and writes the predicted tags and per-tag scores to JSON.
For dual-encoder runs (an `encoders` block in the run config), every field is embedded by its own
encoder (+ adapter from <run_dir>/lora_adapters/<field>) and mixed by the saved attention pooling;
inputs must then be JSON objects holding every field.

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
from illuin_tt.models.pooling import AttentionPooling
from illuin_tt.utils import get_device


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. input=data/new.json model=mlp"
    )

    return parser.parse_args()


def _get_text(d: dict, field: str | list[str]):
    return d[field] if isinstance(field, str) else {f: d[f] for f in field}


def _texts_from_json(path: Path, field: str | list[str]) -> list[tuple[str, str | dict]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [(str(path), _get_text(data, field))]
    if isinstance(data, list):
        return [(f"{path}[{i}]", _get_text(d, field)) for i, d in enumerate(data)]
    raise ValueError(f"{path}: expected a JSON object or a list of objects")


def load_inputs(input_path: str, field: str | list[str]) -> list[tuple[str, str | dict]]:
    """
    Return (source, text) pairs from a .txt, a .json, or a folder of .txt / .json files.
    With a list of fields, text is a {field: text} dict and only .json inputs are supported.
    """
    path = Path(input_path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix in (".txt", ".json"))
        if not files:
            raise ValueError(f"No .txt or .json files in {path}")
        return [doc for p in files for doc in load_inputs(str(p), field)]
    if path.suffix == ".txt":
        if not isinstance(field, str):
            raise ValueError(f"{path}: dual-encoder runs need .json inputs holding the fields {field}")
        return [(str(path), path.read_text())]

    if path.suffix == ".json":
        return _texts_from_json(path, field)

    raise ValueError(f"Unsupported input {path}: expected a .txt, a .json or a folder")


def load_embedder(enc_cfg, lora_adapter, device) -> SentenceTransformer:
    """Embedder of the training run, with its dtype / max_seq_length and optional LoRA adapter."""
    model_kwargs = {"dtype": getattr(torch, enc_cfg.dtype)} if enc_cfg.get("dtype") else {}
    model = SentenceTransformer(enc_cfg.model_name, device=str(device), model_kwargs=model_kwargs)
    if enc_cfg.get("max_seq_length"):
        model.max_seq_length = enc_cfg.max_seq_length

    if lora_adapter:
        print(f"Loading LoRA adapter from {lora_adapter}")
        model[0].auto_model.load_adapter(str(lora_adapter))
    model.eval()
    return model


def load_dual(run_dir: Path, run_cfg, device) -> tuple[dict[str, SentenceTransformer], AttentionPooling]:
    """One embedder per field (+ its adapter if it was fine-tuned) and the attention pooling."""
    models = {}
    for f, enc_cfg in run_cfg.encoders.items():
        adapter = run_dir / "lora_adapters" / f
        models[f] = load_embedder(enc_cfg, adapter if adapter.exists() else None, device)

    pooling = AttentionPooling(
        {f: m.get_embedding_dimension() for f, m in models.items()}, run_cfg.pooling.dim
    )
    pooling.load_state_dict(torch.load(run_dir / "pooling.pt", map_location="cpu"))
    return models, pooling.to(device).eval()


def encode(embedders, pooling, texts: list, batch_size: int) -> np.ndarray:
    """Embed the texts (or {field: text} dicts through every encoder + pooling)."""
    kwargs = {"batch_size": batch_size, "convert_to_tensor": True, "show_progress_bar": True}
    if pooling is None:
        return embedders.encode(texts, **kwargs).float().cpu().numpy()

    with torch.no_grad():
        embs = {
            f: m.encode([t[f] for t in texts], **kwargs).float().to(next(pooling.parameters()).device)
            for f, m in embedders.items()
        }
        return pooling(embs).cpu().numpy()


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
    dual = "encoders" in run_cfg
    field = list(run_cfg.encoders) if dual else (cfg.get("field") or run_cfg.field)

    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    # Recover the input texts
    docs = load_inputs(cfg.input, field)
    print(f"Loaded {len(docs)} documents from {cfg.input}")

    # Load the embedder and classifier
    if dual:
        embedder, pooling = load_dual(run_dir, run_cfg, device)
    else:
        embedder, pooling = load_embedder(run_cfg, cfg.get("lora_adapter"), device), None
    classifier = load_classifier(cfg, run_dir, run_cfg)
    if isinstance(classifier, MLP):
        classifier.to(device)

    # Encode the texts
    start = time.perf_counter()
    emb = encode(embedder, pooling, [text for _, text in docs], cfg.get("batch_size", 8))
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
