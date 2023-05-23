import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


'''
- two different positional encoding methods
- PositionEncodingSuperGule usually cost more GPU memory.
'''
from src.model.module import PositionEncodingSuperGule, PositionEncodingSine


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)


class FeatureMatchTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=8, num_layers=4):
        super(FeatureMatchTransformer, self).__init__()

        self.d_model = d_model  # default 32
        self.nhead = nhead  # default 8
        encoder_layer = EncoderLayer(d_model, nhead)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self._reset_parameters()

        # self.pos_encoding = PositionEncodingSuperGule(d_model)
        self.pos_encoding = PositionEncodingSine(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature_list=None):
        """
        :param feature_list: NV x (b,C,H,W)  b=B//nv
        :return:
        """
        view_nums = len(feature_list)
        _, _, H, _ = feature_list[0].shape
        # pos_encoding
        for i in range(view_nums):
            feature_list[i] = einops.rearrange(self.pos_encoding(feature_list[i]), 'n c h w -> n (h w) c')
        # attention layer
        for layer in self.layers:
            # self attention
            for idx in range(view_nums):
                feature_list[idx] = layer(feature_list[idx], feature_list[idx])
            # cross attention
            tmp_list = [_.clone() for _ in feature_list]
            for ref_idx in range(view_nums):
                for src_idx in range(view_nums):
                    if ref_idx == src_idx: continue
                    tmp_list[ref_idx] = layer(tmp_list[ref_idx], feature_list[src_idx])
            feature_list = tmp_list

        for i in range(view_nums):
            feature_list[i] = einops.rearrange(feature_list[i], 'n (h w) c -> n c h w', h=H)

        return feature_list





