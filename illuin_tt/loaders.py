import json
import os
from glob import glob

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

TARGET_TAGS = {
    0: "math",
    1: "graphs",
    2: "strings",
    3: "number theory",
    4: "trees",
    5: "geometry",
    6: "games",
    7: "probabilities"
}


def tag_encode(tags: set[str]) -> list[int]:

    return [1 if t in tags else 0 for t in TARGET_TAGS.values()]


def make_train_test_val(data_path: str, verbose=False):

    if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")

    print("Importing files from:", data_path)
    # Sorted so the split is reproducible regardless of filesystem order
    files = sorted(glob(os.path.join(data_path, "*.json")))
    tags = []

    # Open the files and load the tags
    for file in files:
        with open(file) as f:
            d = json.load(f)

        tags.append(d["tags"])

    # One hot (like) encode the tags and split them in train/val/test
    tags_enc = np.array([tag_encode(t) for t in tags])
    X_train, y_train, X_, y_ = iterative_train_test_split(
        np.array(files).reshape(-1, 1), tags_enc, 0.2
    )

    X_val, y_val, X_test, y_test = iterative_train_test_split(
        X_, y_, 0.5
    )

    if verbose:

        print(f"Size of training dataset: {len(y_train)}")
        print(f"Size of val dataset: {len(y_val)}")
        print(f"Size of test dataset: {len(y_test)}")

        print(f"Distribution of training labels: {y_train.mean(0)}")
        print(f"Distribution of val labels: {y_val.mean(0)}")
        print(f"Distribution of test labels: {y_test.mean(0)}")

    return X_train, y_train, X_val, y_val, X_test, y_test
