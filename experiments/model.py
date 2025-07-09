import random
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Wav2Vec2Model


# Model for feature extraction.
class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.out_dim = self.model.config.hidden_size

    def forward(self, input_data: Tensor) -> Tensor:
        self.model.eval()
        if input_data.ndim == 3:
            input_tmp = input_data.squeeze(-1)
        else:
            input_tmp = input_data
        emb = self.model(input_tmp).last_hidden_state
        return emb


# Homogeneous Graph Attention Layer.
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


# Core attention mechanism for nodes of different types.
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


# Differentiable graph pooling layer based on top-k selection.
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


# Residual block for the CNN encoder.
class Residual_block(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False):
        super().__init__()
        self.first = first
        in_channels, out_channels = nb_filts
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), padding=(1, 1))
        self.act = nn.GELU()
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 3), padding=(0, 1))
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv_downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = x if self.first else self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        filts = [1, 32], [32, 32], [32, 64], [64, 64]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5]
        temperatures = [2, 2, 2]
        num_heads = 4

        self.ssl_model = SSLModel()

        print("Freezing Wav2Vec2Model parameters...")
        for param in self.ssl_model.model.parameters():
            param.requires_grad = False
        print("Wav2Vec2Model parameters frozen.")

        self.feature_proj = nn.Linear(self.ssl_model.out_dim, 128)

        self.pre_encoder_bn = nn.BatchNorm2d(num_features=1)
        encoder_blocks = [Residual_block(filts[0], first=True)]
        encoder_blocks.extend([Residual_block(filts[1]), Residual_block(filts[2]),
                               Residual_block(filts[3]), Residual_block(filts[3]), Residual_block(filts[3])])
        self.encoder = nn.Sequential(*encoder_blocks)
        self.post_encoder_bn = nn.BatchNorm2d(num_features=filts[-1][-1])

        self.attention = nn.Sequential(
            nn.Conv2d(filts[-1][-1], 128, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, filts[-1][-1], kernel_size=1),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.gat_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], num_heads=num_heads, temperature=temperatures[0])
        self.gat_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], num_heads=num_heads, temperature=temperatures[1])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)

        self.masters = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, gat_dims[0])),
            nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        ])
        self.hgat_streams = nn.ModuleList()
        for _ in range(2):
            stream_layers = nn.ModuleDict({
                "hgat1": HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], num_heads=num_heads,
                                                 temperature=temperatures[2]),
                "hgat2": HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], num_heads=num_heads,
                                                 temperature=temperatures[2]),
                "pool_hS": GraphPool(pool_ratios[2], gat_dims[1], 0.3),
                "pool_hT": GraphPool(pool_ratios[2], gat_dims[1], 0.3),
            })
            self.hgat_streams.append(stream_layers)

        # Modules for combining outputs from the two parallel streams
        combiner_dropout = 0.2
        self.stream_combiner_T = nn.MultiheadAttention(gat_dims[1], num_heads, dropout=combiner_dropout,
                                                       batch_first=True)
        self.stream_combiner_S = nn.MultiheadAttention(gat_dims[1], num_heads, dropout=combiner_dropout,
                                                       batch_first=True)
        self.stream_combiner_master = nn.MultiheadAttention(gat_dims[1], num_heads, dropout=combiner_dropout,
                                                            batch_first=True)

        self.norm_T = nn.LayerNorm(gat_dims[1])
        self.norm_S = nn.LayerNorm(gat_dims[1])
        self.norm_master = nn.LayerNorm(gat_dims[1])

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        self.dropout = nn.Dropout(0.5)
        self.dropout_way = nn.Dropout(0.2)
        self.act = nn.GELU()

    # Helper function for a single heterogeneous graph stream
    def _forward_graph_stream(self, out_T_in, out_S_in, master_in, stream_layers):
        out_T, out_S, master = stream_layers["hgat1"](out_T_in, out_S_in, master=master_in)
        out_S = stream_layers["pool_hS"](out_S)
        out_T = stream_layers["pool_hT"](out_T)
        out_T_aug, out_S_aug, master_aug = stream_layers["hgat2"](out_T, out_S, master=master)
        # Residual connections for augmented features
        out_T = out_T + out_T_aug
        out_S = out_S + out_S_aug
        master = master + master_aug
        return out_T, out_S, master

    def forward(self, x: Tensor, return_activations=False) -> Tensor:
        # 1. Extract features with Wav2Vec2.
        x_ssl_feat = self.ssl_model(x)
        x_proj = self.feature_proj(x_ssl_feat)
        x = x_proj

        # 2. Process features with a CNN encoder.
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.act(self.pre_encoder_bn(F.max_pool2d(x, (3, 3))))
        x = self.act(self.post_encoder_bn(self.encoder(x)))

        # 3. Create temporal (e_T) and spectral (e_S) graph nodes via attention.
        w = self.attention(x)
        w_s = F.softmax(w, dim=-1)
        m_s = torch.sum(x * w_s, dim=-1)
        e_S = m_s.transpose(1, 2) + self.pos_S
        w_t = F.softmax(w, dim=-2)
        m_t = torch.sum(x * w_t, dim=-2)
        e_T = m_t.transpose(1, 2)

        # 4. Initial homogeneous graph processing and pooling.
        out_S_init = self.pool_S(self.gat_S(e_S))
        out_T_init = self.pool_T(self.gat_T(e_T))

        # 5. Process through two parallel heterogeneous graph streams.
        stream_outputs = []
        for i in range(2):
            master_init = self.masters[i].expand(x.size(0), -1, -1)
            out_T, out_S, master = self._forward_graph_stream(
                out_T_init, out_S_init, master_init, self.hgat_streams[i]
            )
            stream_outputs.append([out_T, out_S, master])

        s1_out_T, s1_out_S, s1_master = [self.dropout_way(o) for o in stream_outputs[0]]
        s2_out_T, s2_out_S, s2_master = [self.dropout_way(o) for o in stream_outputs[1]]

        # 6. Combine stream outputs using self-attention.

        # Combine temporal (T) nodes
        in_T = torch.cat([s1_out_T, s2_out_T], dim=1)
        attn_out_T, _ = self.stream_combiner_T(query=in_T, key=in_T, value=in_T)
        out_T = self.norm_T(in_T + attn_out_T)  # Residual connection and normalization

        # Combine spectral (S) nodes
        in_S = torch.cat([s1_out_S, s2_out_S], dim=1)
        attn_out_S, _ = self.stream_combiner_S(query=in_S, key=in_S, value=in_S)
        out_S = self.norm_S(in_S + attn_out_S)

        # Combine master nodes
        in_master = torch.cat([s1_master, s2_master], dim=1)
        attn_out_master, _ = self.stream_combiner_master(query=in_master, key=in_master, value=in_master)
        master_combined = self.norm_master(in_master + attn_out_master)
        master = master_combined.mean(dim=1, keepdim=True)  # Average master nodes to get a single vector

        # 7. Aggregate graph features for final classification.
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        final_embedding = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )

        # 8. Final classification layer.
        output = self.out_layer(self.dropout(final_embedding))

        if return_activations:
            return output, x_ssl_feat, x_proj
        else:
            return output