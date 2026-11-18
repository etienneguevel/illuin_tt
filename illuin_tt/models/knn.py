import numpy as np
from scipy.sparse import issparse
from sklearn.neighbors import KNeighborsClassifier


class KnnMulticlass:
    def __init__(
        self,
        num_classes: int,
        n_neighbors: int = 10,
        metric: str = "cosine",
        weights: str = "distance",
    ):
        self.num_classes = num_classes
        self.estimators = [
            KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
            for _ in range(num_classes)
        ]

    def fit(self, X_train, y_train):
        for i in range(self.num_classes):
            self.estimators[i].fit(X_train, y_train[:, i])
        return self

    def predict(self, x):
        # Deal with sparse tf-idf and dense embedding matrices
        x = x if issparse(x) else np.asarray(x)
        return np.stack([est.predict(x) for est in self.estimators], axis=1)

    def predict_proba(self, x):
        # Use to compute ROC
        # Deal with sparse tf-idf and dense embedding matrices
        x = x if issparse(x) else np.asarray(x)
        return np.stack([est.predict_proba(x)[:, 1] for est in self.estimators], axis=1)
