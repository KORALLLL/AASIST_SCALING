from typing import Tuple

import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor


class HeterogeneousAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q1 = nn.Linear(embed_dim, embed_dim)
        self.to_q2 = nn.Linear(embed_dim, embed_dim)
        self.to_k1 = nn.Linear(embed_dim, embed_dim)
        self.to_k2 = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, num_type1: int) -> Tensor:
        batch_size, nb_nodes, _ = x.shape
        x1 = x[:, :num_type1, :]
        x2 = x[:, num_type1:, :]

        def reshape_for_heads(tensor: Tensor):
            b, n, d = tensor.shape
            return tensor.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q1, q2 = self.to_q1(x1), self.to_q2(x2)
        k1, k2 = self.to_k1(x1), self.to_k2(x2)
        q = torch.cat([q1, q2], dim=1)
        k = torch.cat([k1, k2], dim=1)
        v = self.to_v(x)
        q = reshape_for_heads(q)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, nb_nodes, self.embed_dim)
        return self.dropout(self.to_out(out))


# Heterogeneous Graph Attention Layer combining different node types and a master node.
class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, temperature: float = 1.0, dropout: float = 0.2):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.het_attention = HeterogeneousAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.mha_master = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout,
                                                batch_first=True)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x1: Tensor, x2: Tensor, master: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        num_type1 = x1.size(1)
        x1_proj = self.proj_type1(x1)
        x2_proj = self.proj_type2(x2)
        x = torch.cat([x1_proj, x2_proj], dim=1)
        x = self.input_drop(x)
        master = self._update_master(x, master)
        aggregated_nodes = self.het_attention(x, num_type1)
        x_proj = self.proj_with_att(aggregated_nodes)
        x_res = self.proj_without_att(x)
        x = x_proj + x_res
        x = self._apply_bn(x)
        x = self.act(x)
        x1_out = x.narrow(1, 0, x1.size(1))
        x2_out = x.narrow(1, x1.size(1), x2.size(1))
        return x1_out, x2_out, master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            master_aggregated, _ = self.mha_master(query=master, key=x, value=x)
        master_proj = self.proj_with_attM(master_aggregated)
        master_res = self.proj_without_attM(master)
        return master_proj + master_res

    def _apply_bn(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x
