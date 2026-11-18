import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer

from illuin_tt.eval import eval_arrays
from illuin_tt.loaders import make_train_test_val
from illuin_tt.models.gb import GbMulticlass
from illuin_tt.models.lr import LrMulticlass
from illuin_tt.report import summary_table, write_report


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. max_features=5000"
    )

    return parser.parse_args()


def load_documents(paths, field: str) -> list[str]:
    corpus = []
    for p in paths:
        with open(p) as f:
            corpus.append(json.load(f)[field])

    return corpus


def main():
    args = get_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_path), OmegaConf.from_dotlist(args.overrides))

    # Seed everything (the split relies on numpy's global RNG)
    np.random.seed(cfg.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = make_train_test_val(cfg.data_path, True)
    num_classes = y_train.shape[-1]

    # Fit the vocabulary / idf on train only, then transform the other splits
    vectorizer = TfidfVectorizer(
        max_features=cfg.get("max_features"),
        ngram_range=tuple(cfg.get("ngram_range", [1, 1])),
        min_df=cfg.get("min_df", 1),
        sublinear_tf=cfg.get("sublinear_tf", False),
    )
    X_train_vect = vectorizer.fit_transform(load_documents(X_train.ravel(), cfg.field))
    X_test_vect = vectorizer.transform(load_documents(X_test.ravel(), cfg.field))
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # Create and train the classifiers
    print("Fitting logistic regression")
    lr_est = LrMulticlass(num_classes).fit(X_train_vect, y_train)
    print("Fitting gradient boosting")
    gb_est = GbMulticlass(num_classes, random_state=cfg.seed).fit(X_train_vect, y_train)

    # Eval on the test set, like the other scripts
    results = {
        "TF-IDF + Logistic regression": eval_arrays(lr_est, X_test_vect, y_test),
        "TF-IDF + Gradient boosting": eval_arrays(gb_est, X_test_vect, y_test),
    }

    # Overrides (except output_dir) go in the run name so runs are distinguishable
    tags = [o.replace("=", "").replace("/", "-") for o in args.overrides if not o.startswith("output_dir=")]
    run_name = "_".join([Path(args.config_path).stem, *tags, f"{datetime.now():%Y%m%d-%H%M%S}"])
    out_dir = Path(cfg.get("output_dir", "outputs")) / run_name
    split_sizes = {"train": len(y_train), "val": len(y_val), "test": len(y_test)}
    write_report(out_dir, cfg, results, split_sizes)

    print(summary_table(results))
    print(f"Report saved to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
