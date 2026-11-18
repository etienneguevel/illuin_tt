#!/usr/bin/env bash
# Sweep MLP architectures (n_layers x hidden_size) on top of an existing config.
# Embeddings are cached by EmbeddingDataset, so only the first run (if any) computes them.
#
# Usage:
#   scripts/mlp_sweep.sh [config.yaml]
# Env vars:
#   N_LAYERS="2 3 4"                 number of Linear layers (>= 2)
#   HIDDEN_SIZES="256 512 1024 2048" hidden widths
#   EXTRA="num_epochs=20 lr=5e-4"    extra OmegaConf overrides passed to every run
#
# Output: outputs/mlp_sweep_<config>_<timestamp>/{<run dirs>, logs/, summary.md}
set -uo pipefail

CONFIG="${1:-configs/Qwen3-Embedding-4B.yaml}"
N_LAYERS="${N_LAYERS:-2 3 4}"
HIDDEN_SIZES="${HIDDEN_SIZES:-256 512 1024 2048}"
EXTRA="${EXTRA:-}"

cd "$(dirname "$0")/.."

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

SWEEP_DIR="outputs/mlp_sweep_$(basename "$CONFIG" .yaml)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SWEEP_DIR/logs"
echo "Sweep dir: $SWEEP_DIR"

failed=()
for l in $N_LAYERS; do
    for h in $HIDDEN_SIZES; do
        name="l${l}_h${h}"
        echo "=== Run $name ==="
        # shellcheck disable=SC2086  # EXTRA is intentionally word-split into overrides
        if ! uv run python -m illuin_tt.scripts.embedding_method --config-path "$CONFIG" \
            n_layers="$l" hidden_size="$h" output_dir="$SWEEP_DIR" $EXTRA \
            2>&1 | tee "$SWEEP_DIR/logs/$name.log"; then
            echo "!!! Run $name failed (see $SWEEP_DIR/logs/$name.log)" >&2
            failed+=("$name")
        fi
    done
done

# Aggregate every run of the sweep into one table
uv run python - "$SWEEP_DIR" <<'EOF'
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from illuin_tt.report import _md_table

sweep = Path(sys.argv[1])
fmt = lambda x: "-" if x is None else f"{x:.3f}"
rows, lr = [], None
for metrics_path in sorted(sweep.glob("*/metrics.json")):
    res = json.loads(metrics_path.read_text())
    cfg = OmegaConf.load(metrics_path.parent / "config.yaml")
    m = res["MLP"]
    rows.append((m["macro_f1"], [
        str(cfg.n_layers), str(cfg.hidden_size),
        fmt(m["micro_f1"]), fmt(m["macro_f1"]), fmt(m["micro_roc_auc"]), fmt(m["macro_roc_auc"]),
        metrics_path.parent.name,
    ]))
    lr = lr or res.get("Logistic regression")

if not rows:
    print("No completed runs found.")
    sys.exit(0)

header = ["n_layers", "hidden_size", "Micro-F1", "Macro-F1", "Micro-AUC", "Macro-AUC", "Run"]
lines = [f"# MLP sweep: {sweep.name}", "", "## MLP runs (sorted by Macro-F1)", "",
         _md_table(header, [r for _, r in sorted(rows, key=lambda x: -x[0])])]
if lr:
    lines += ["", "## Logistic regression baseline", "",
              _md_table(["Micro-F1", "Macro-F1", "Micro-AUC", "Macro-AUC"],
                        [[fmt(lr["micro_f1"]), fmt(lr["macro_f1"]),
                          fmt(lr["micro_roc_auc"]), fmt(lr["macro_roc_auc"])]])]
text = "\n".join(lines) + "\n"
(sweep / "summary.md").write_text(text)
print(text)
EOF

if (( ${#failed[@]} )); then
    echo "Failed runs: ${failed[*]}" >&2
    exit 1
fi
echo "Summary: $SWEEP_DIR/summary.md"
