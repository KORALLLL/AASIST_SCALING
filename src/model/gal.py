import torch, torch.nn as nn
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, temperature: float = 1.0, dropout: float = 0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_drop(x)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            aggregated_nodes, _ = self.mha(query=x, key=x, value=x)
        x_proj = self.proj_with_att(aggregated_nodes)
        x_res = self.proj_without_att(x)
        x = x_proj + x_res
        x = self._apply_bn(x)
        x = self.act(x)
        return x

    def _apply_bn(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x