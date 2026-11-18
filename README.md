# illuin-tt

This repo contains the code for the technical test with Illuin on a subset of the xCodeEval dataset.

The task consists of a multi-label classification problem of coding problems, focusing onto 8 tags:
`math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`.

Three approaches are implemented and report their results in the same format:

| Approach | Script | Features | Classifiers |
|---|---|---|---|
| TF-IDF baseline | `illuin_tt/scripts/tf_idf.py` | TF-IDF on words and word pairs | Logistic regression, gradient boosting |
| Embedding probing | `illuin_tt/scripts/embedding_method.py` | Frozen `Qwen/Qwen3-Embedding-*` embeddings | Logistic regression, MLP, kNN |
| LoRA fine-tuning | `illuin_tt/scripts/lora_finetune.py` | Qwen embeddings fine-tuned with LoRA | MLP head, logistic regression |

## Setup

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The package can be installed with other venv managers.

## Usage

Each experiment has a script that take a YAML config, followed by optional `key=value`
overrides to custom easily the run.

### TF-IDF baseline

```bash
uv run python -m illuin_tt.scripts.tf_idf --config-path configs/tf_idf.yaml
```

### Embedding probing

```bash
uv run python -m illuin_tt.scripts.embedding_method --config-path configs/Qwen3-Embedding-4B.yaml

# Probe the source code instead of the problem statement
uv run python -m illuin_tt.scripts.embedding_method --config-path configs/Qwen3-Embedding-4B_code.yaml

# Override any config value
uv run python -m illuin_tt.scripts.embedding_method --config-path configs/Qwen3-Embedding-4B.yaml \
    n_layers=3 hidden_size=512 knn.n_neighbors=25
```

Embeddings are computed once, then saved and reused from `cache_dir`.

### LoRA fine-tuning

```bash
uv run python -m illuin_tt.scripts.lora_finetune --config-path configs/Qwen3-Embedding-4B_lora.yaml
```

This experiment might take some time and the use of a GPU.

### Inference

Predict tags for new texts with a trained run. Set `run_dir` to a run folder from the embedding or
LoRA script, and `lora_adapter` to `<run_dir>/lora_adapter` for LoRA runs.

```bash
uv run python -m illuin_tt.scripts.generate --config-path configs/generate.yaml \
    run_dir=outputs/<run_name> input=path/to/texts output=predictions.json
```

`input` can be a `.txt` file, a `.json` file (an object or a list of objects, with the text in
`field`), or a folder of `.txt` / `.json` files.

## Outputs

Each run writes a timestamped folder in `outputs/`:

| File | Content |
|---|---|
| `report.md` | Test metrics per model and per tag, ROC curves, config |
| `metrics.json` | The same metrics in JSON, with the ROC curve points |
| `config.yaml` | The config used, overrides included |
| `roc_*.png` | ROC curves per model |
| `models/mlp.pt`, `models/logistic_regression.joblib` | Trained models (embedding probing) |
| `head.pt`, `lora_adapter/`, `train_log.json` | Head, adapter and losses (LoRA fine-tuning) |
