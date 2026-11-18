import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from illuin_tt.loaders import TARGET_TAGS
from illuin_tt.models.mlp import MLP


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table with padded columns, readable both raw and rendered."""
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    fmt = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([fmt(header), sep, *map(fmt, rows)])


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def summary_table(results: dict[str, dict]) -> str:
    return _md_table(
        ["Model", "Micro-F1", "Macro-F1", "Micro-AUC", "Macro-AUC"],
        [
            [m, _fmt(r["micro_f1"]), _fmt(r["macro_f1"]), _fmt(r.get("micro_roc_auc")), _fmt(r.get("macro_roc_auc"))]
            for m, r in results.items()
        ],
    )


def plot_roc(out_dir: Path, model: str, res: dict) -> str:
    """Save one figure with the ROC curve of every tag; returns the PNG file name."""
    name = "roc_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") + ".png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for tag, curve in res["roc_curves"].items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{tag} (AUC={res[tag]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {model}",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    return name


def threshold_section(mlp: MLP, x_val, y_val, model_name: str = "MLP") -> str:
    """Markdown table of the tuned per-tag thresholds with the val F1 they give vs 0.5."""
    probs, labels = mlp.predict_proba(x_val), np.asarray(y_val).astype(int)
    rows = []
    for i, tag in TARGET_TAGS.items():
        t = float(mlp.ths[i])
        f1_default = f1_score(labels[:, i], probs[:, i] > 0.5, zero_division=0)
        f1_tuned = f1_score(labels[:, i], probs[:, i] > t, zero_division=0)
        rows.append([tag, f"{t:.2f}", f"{f1_default:.3f}", f"{f1_tuned:.3f}"])
    return "\n".join([
        f"## {model_name} decision thresholds",
        "",
        f"Per-tag thresholds maximizing F1 on the val split (used for the `{model_name}` results above).",
        "",
        _md_table(["Tag", "Threshold", "Val F1 @0.5", "Val F1 @tuned"], rows),
    ])


def write_report(
    out_dir: Path,
    cfg,
    results: dict[str, dict],
    split_sizes: dict[str, int],
    extra_sections: list[str] | None = None,
):
    """`extra_sections`: markdown blocks appended after the per-model sections."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation report",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Split sizes: " + ", ".join(f"{k}={v}" for k, v in split_sizes.items()),
        "",
        "## Summary (test set)",
        "",
        summary_table(results),
    ]
    for model, res in results.items():
        rows = [
            [
                tag,
                f"{res[tag]['precision']:.3f}",
                f"{res[tag]['recall']:.3f}",
                f"{res[tag]['f1']:.3f}",
                f"{res[tag]['acc']:.3f}",
                _fmt(res[tag].get("roc_auc")),
                str(res[tag]["support"]),
            ]
            for tag in TARGET_TAGS.values()
        ]
        lines += [
            "",
            f"## {model}",
            "",
            _md_table(["Tag", "Precision", "Recall", "F1", "Balanced acc", "ROC AUC", "Support"], rows),
        ]
        # Models without scores (hard predictions only) have no ROC curves to plot
        if res.get("roc_curves"):
            lines += ["", f"![ROC - {model}]({plot_roc(out_dir, model, res)})"]
    for section in extra_sections or []:
        lines += ["", section.rstrip()]
    lines += ["", "## Config", "", "```yaml", OmegaConf.to_yaml(cfg).rstrip(), "```", ""]

    (out_dir / "report.md").write_text("\n".join(lines))
    OmegaConf.save(cfg, out_dir / "config.yaml")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)