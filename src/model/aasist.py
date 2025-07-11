from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .frontend import SSLModel
from .gal import GraphAttentionLayer
from .hs_gal import HtrgGraphAttentionLayer
from .pool import GraphPool
from .res import Residual_block


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
        self.feature_proj = nn.Linear(1024, 128)

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