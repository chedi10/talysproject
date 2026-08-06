from __future__ import annotations

"""Deep Tabular Network — MLP + embeddings for categorical features."""

import torch
import torch.nn as nn


class DeepTabularNet(nn.Module):
    """
    Modern tabular deep learning:
    - Embeddings for categorical columns
    - BatchNorm + GELU MLP for numeric + embedded features
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        embed_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.25,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        self.n_numeric = n_numeric
        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(c, 2), embed_dim) for c in cat_cardinalities]
        )
        in_dim = n_numeric + embed_dim * len(cat_cardinalities)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = [e(x_cat[:, i].long()) for i, e in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb, dim=1)
        return self.mlp(x).squeeze(-1)
