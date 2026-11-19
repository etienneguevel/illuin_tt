import torch
from torch import nn


class AttentionPooling(nn.Module):
    """
    Mix one embedding per field into a single vector with per-sample attention weights.

    Each field is projected to `dim` (so the encoders may have different sizes), scored with
    s_f = v^T tanh(W h_f), and the projections are summed with softmax(s) weights.
    Field order follows `in_dims` (the config's `encoders` order).
    """

    def __init__(self, in_dims: dict[str, int], dim: int, dropout: float = 0.0):
        super().__init__()
        self.fields = list(in_dims)
        self.proj = nn.ModuleDict({f: nn.Linear(d, dim) for f, d in in_dims.items()})
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, embs: dict[str, torch.Tensor], return_weights: bool = False):
        # (bs, n_fields, dim)
        h = torch.stack([self.proj[f](embs[f]) for f in self.fields], dim=1)
        # (bs, n_fields)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=1)
        pooled = self.dropout((weights.unsqueeze(-1) * h).sum(1))
        return (pooled, weights) if return_weights else pooled
