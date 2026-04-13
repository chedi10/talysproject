from __future__ import annotations

"""
Minimal GraphSAGE implementation (mean aggregator) in pure PyTorch.

No torch-geometric dependency to keep Windows setup simple.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_aggregate(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute mean neighbor embedding for each node.

    Parameters
    ----------
    h : (N, D) node embeddings
    edge_index : (2, E) where edge_index[0]=src, edge_index[1]=dst
        Message passing: src -> dst

    Returns
    -------
    neigh_mean : (N, D)
    """
    src = edge_index[0]
    dst = edge_index[1]

    n, d = h.shape
    out = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    deg = torch.zeros((n,), device=h.device, dtype=h.dtype)

    out.index_add_(0, dst, h[src])
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))

    deg = deg.clamp_min(1.0).unsqueeze(1)
    return out / deg


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neigh = mean_aggregate(h, edge_index)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim, dropout=dropout)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns logits per node: shape (N,)
        """
        h = self.sage1(x, edge_index)
        h = self.sage2(h, edge_index)
        logits = self.head(h).squeeze(-1)
        return logits

