import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_layers: int, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 (use LrMulticlass for a linear model).")

        layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, out_size))

        self.layers = nn.Sequential(*layers)
        # Per-class decision thresholds (saved with the state dict), see tune_thresholds
        self.register_buffer("ths", torch.full((out_size,), 0.5))

    def forward(self, x: torch.Tensor):
        # (bs, in_size) -> (bs, out_size)
        return self.layers(x)

    @torch.no_grad()
    def predict_proba(self, x):
        device = next(self.parameters()).device
        logits = self.forward(torch.as_tensor(x, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()

    @torch.no_grad()
    def predict(self, x, ths=None):
        ths = (
            self.ths.cpu().numpy() if ths is None else np.asarray(ths)
        )
        return (self.predict_proba(x) > ths).astype(int)

    @torch.no_grad()
    def tune_thresholds(self, x_val, y_val, grid=None) -> np.ndarray:
        """
        Per class, pick the threshold of `grid` maximizing F1 on (x_val, y_val); ties go to the
        value closest to 0.5, classes without positives keep 0.5. Stored in self.ths and returned.
        """
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else np.asarray(grid)
        probs = self.predict_proba(x_val)
        y_val = np.asarray(y_val).astype(int)

        ths = np.full(probs.shape[1], 0.5)
        for i in range(probs.shape[1]):
            if y_val[:, i].sum() == 0:
                continue
            f1s = np.array([f1_score(y_val[:, i], probs[:, i] > t, zero_division=0) for t in grid])
            best = np.flatnonzero(np.isclose(f1s, f1s.max()))
            ths[i] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]

        self.ths.copy_(torch.as_tensor(ths, dtype=self.ths.dtype))
        return ths


class FixedThreshold:
    """Evaluate a model with a fixed decision threshold (same scores, different cut-off)."""

    def __init__(self, model, ths: float):
        self.model, self.ths = model, ths

    def predict(self, x):
        return self.model.predict(x, self.ths)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
