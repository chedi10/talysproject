from __future__ import annotations



"""Graph Attention Network (GAT) with edge-weighted attention."""



import torch

import torch.nn as nn

import torch.nn.functional as F





class GATLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.2):

        super().__init__()

        assert out_dim % n_heads == 0

        self.n_heads = n_heads

        self.head_dim = out_dim // n_heads

        self.lin = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_src = nn.Parameter(torch.empty(n_heads, self.head_dim))

        self.attn_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))

        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))

        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

        self.dropout = dropout

        self.leaky = 0.2



    def forward(

        self,

        h: torch.Tensor,

        edge_index: torch.Tensor,

        edge_weight: torch.Tensor | None = None,

    ) -> torch.Tensor:

        n = h.size(0)

        src, dst = edge_index[0], edge_index[1]

        h_proj = self.lin(h).view(n, self.n_heads, self.head_dim)



        e_src = (h_proj[src] * self.attn_src).sum(-1)

        e_dst = (h_proj[dst] * self.attn_dst).sum(-1)

        scores = F.leaky_relu(e_src + e_dst, negative_slope=self.leaky)



        if edge_weight is not None:

            scores = scores * edge_weight.unsqueeze(-1)



        alpha = torch.zeros_like(scores)

        for head in range(self.n_heads):

            sh = scores[:, head]

            max_v = torch.zeros(n, device=h.device, dtype=h.dtype)

            max_v.scatter_reduce_(0, dst, sh, reduce="amax", include_self=False)

            exp = torch.exp(sh - max_v[dst])

            denom = torch.zeros(n, device=h.device, dtype=h.dtype)

            denom.index_add_(0, dst, exp)

            alpha[:, head] = exp / denom[dst].clamp_min(1e-9)



        msg = h_proj[src] * alpha.unsqueeze(-1)

        out = torch.zeros(n, self.n_heads, self.head_dim, device=h.device, dtype=h.dtype)

        idx = dst.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim)

        out.scatter_add_(0, idx, msg)

        out = out.reshape(n, -1)

        return F.elu(F.dropout(out, p=self.dropout, training=self.training))





class GATClassifier(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_heads: int = 4, dropout: float = 0.2):

        super().__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, n_heads=n_heads, dropout=dropout)

        self.gat2 = GATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)

        self.head = nn.Linear(hidden_dim, 1)



    def forward(

        self,

        x: torch.Tensor,

        edge_index: torch.Tensor,

        edge_weight: torch.Tensor | None = None,

    ) -> torch.Tensor:

        h = self.gat1(x, edge_index, edge_weight)

        h = self.gat2(h, edge_index, edge_weight)

        return self.head(h).squeeze(-1)

