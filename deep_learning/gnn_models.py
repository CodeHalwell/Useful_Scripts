"""Simple graph neural network layers using PyTorch."""

import torch
from torch import nn


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Return symmetrically normalized adjacency matrix."""
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = normalize_adjacency(adj)
        x = adj_norm @ x
        return self.linear(x)


class GATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.linear = nn.Linear(in_features, out_features * heads, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, out_features * 2))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        N = h.size(0)
        h = h.view(N, self.heads, -1)
        a_input = torch.cat([h.repeat(1, 1, N).view(N, N, self.heads, -1),
                             h.repeat(N, 1, 1)], dim=-1)
        e = (a_input * self.attn).sum(-1)
        e = torch.where(adj > 0, e, torch.full_like(e, -9e15))
        attention = torch.softmax(e, dim=1)
        h_prime = torch.einsum("ijh,jhd->ihd", attention, h)
        return h_prime.reshape(N, -1)


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer with mean, sum or max aggregation."""

    def __init__(self, in_features: int, out_features: int, agg: str = "mean"):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        self.agg = agg

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.agg == "mean":
            neigh = torch.matmul(adj, x) / (adj.sum(1, keepdim=True) + 1e-10)
        elif self.agg == "sum":
            neigh = torch.matmul(adj, x)
        elif self.agg == "max":
            neigh, _ = torch.max(adj.unsqueeze(-1) * x.unsqueeze(0), dim=1)
        else:
            raise ValueError("Unknown aggregation: {}".format(self.agg))
        h = torch.cat([x, neigh], dim=1)
        return self.linear(h)
