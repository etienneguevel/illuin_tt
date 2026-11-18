import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from illuin_tt.loaders import TARGET_TAGS


def eval_model(model, dataset):
    """Evaluate a model on a torch dataset of (embedding, label) pairs."""
    preds, scores, labels = [], [], []

    dataloader = DataLoader(dataset, 128)

    for emb, lab in tqdm(dataloader):
        preds.append(np.asarray(model.predict(emb)))
        scores.append(np.asarray(model.predict_proba(emb)))
        labels.append(lab.numpy())

    preds = np.concatenate(preds).astype(int)
    scores = np.concatenate(scores).astype(float)
    labels = np.concatenate(labels).astype(int)

    return compute_metrics(labels, preds, scores)


def eval_arrays(model, X, y):
    """Evaluate a model on in-memory inputs (numpy arrays or scipy sparse matrices)."""
    return compute_metrics(np.asarray(y), model.predict(X), model.predict_proba(X))


def compute_metrics(labels, preds, scores):
    """Threshold metrics from hard predictions and ROC metrics from scores, all (n, num_classes)."""
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    scores = np.asarray(scores).astype(float)
    metrics = {}

    curves = {}
    aucs = []
    for i, t in TARGET_TAGS.items():
        y_true, y_pred, y_score = labels[:, i], preds[:, i], scores[:, i]
        # ROC is undefined when the split contains a single class for this tag
        has_both = 0 < y_true.sum() < len(y_true)
        auc = float(roc_auc_score(y_true, y_score)) if has_both else None
        if has_both:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            curves[t] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            aucs.append(auc)

        metrics[t] = {
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "acc": balanced_accuracy_score(y_true, y_pred),
            "roc_auc": auc,
            "support": int(y_true.sum()),
        }

    metrics["micro_f1"] = f1_score(labels, preds, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    metrics["micro_roc_auc"] = float(roc_auc_score(labels.ravel(), scores.ravel()))
    metrics["macro_roc_auc"] = float(np.mean(aucs)) if aucs else None
    metrics["roc_curves"] = curves

    return metrics
