from typing import Union

import torch, torch.nn as nn
from torch import Tensor


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = torch.sigmoid(weights)
        return self._top_k_graph(scores, h)

    def _top_k_graph(self, scores: Tensor, h: Tensor) -> Tensor:
        bs, n_nodes, n_feat = h.size()
        n_nodes_to_keep = max(1, int(n_nodes * self.k))
        _, top_indices = torch.topk(scores, n_nodes_to_keep, dim=1)
        h_weighted = h * scores
        top_indices = top_indices.expand(-1, -1, n_feat)
        h_pooled = torch.gather(h_weighted, 1, top_indices)
        return h_pooled