"""
    Transformer class definition

    The implementation mainly follows the implementation found in the PyTorch
        with added support of pre-residual connection normalization.

    Resources used to develop this script:
        - https://github.com/jwang0306/transformer-pytorch
"""
import torch
import torch.nn as nn

from pytorch3d.ops import GraphConv


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.pff = nn.Sequential(
            nn.Linear(hidden_size, filter_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(filter_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, src):
        src = self.pff(src)

        return src


class GCNMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = GraphConv(hid_dim, hid_dim)
        self.fc_k = GraphConv(hid_dim, hid_dim)
        self.fc_v = GraphConv(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

        self.edge = torch.load("./template/multi_edge").cuda().to(torch.int64)
        self.mask = torch.load("./template/multi_mask").cuda()

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        Q = []
        K = []
        V = []
        for i in range(batch_size):
            Q.append(self.fc_q(query[i], self.edge))
            K.append(self.fc_k(key[i], self.edge))
            V.append(self.fc_v(value[i], self.edge))
        Q = torch.stack(Q, dim=0)
        K = torch.stack(K, dim=0)
        V = torch.stack(V, dim=0)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        energy = energy.masked_fill(self.mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, pre_lnorm, dropout):
        super(EncoderLayer, self).__init__()
        # self-attention part
        self.self_attn = GCNMultiHeadAttentionLayer(
            hidden_size, n_head, dropout=dropout
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        # feed forward network part
        self.pff = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm = nn.LayerNorm(hidden_size)

        self.self_attn2 = GCNMultiHeadAttentionLayer(
            hidden_size, n_head, dropout=dropout
        )
        self.self_attn_norm2 = nn.LayerNorm(hidden_size)
        # feed forward network part
        self.pff2 = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm2 = nn.LayerNorm(hidden_size)

        self.self_attn3 = GCNMultiHeadAttentionLayer(
            hidden_size, n_head, dropout=dropout
        )
        self.self_attn_norm3 = nn.LayerNorm(hidden_size)
        # feed forward network part
        self.pff3 = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm3 = nn.LayerNorm(hidden_size)

        self.self_attn4 = MultiHeadAttentionLayer(hidden_size, n_head, dropout=dropout)
        self.self_attn_norm4 = nn.LayerNorm(hidden_size)
        # feed forward network part
        self.pff4 = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm4 = nn.LayerNorm(hidden_size)

    def forward(self, src):
        pre = self.self_attn_norm(src)
        temp = self.self_attn(pre, pre, pre)
        src = src + temp  # residual connection
        pre = self.pff_norm(src)
        src = src + self.pff(pre)  # residual connection

        pre = self.self_attn_norm2(src)
        temp = self.self_attn2(pre, pre, pre)
        src = src + temp  # residual connection
        pre = self.pff_norm2(src)
        src = src + self.pff2(pre)  # residual connection

        pre = self.self_attn_norm3(src)
        temp = self.self_attn3(pre, pre, pre)
        src = src + temp  # residual connection
        pre = self.pff_norm3(src)
        src = src + self.pff3(pre)  # residual connection

        pre = self.self_attn_norm4(src)
        temp = self.self_attn4(pre, pre, pre)
        src = src + temp  # residual connection
        pre = self.pff_norm4(src)
        src = src + self.pff4(pre)  # residual connection

        # attention = torch.stack((attention1, attention2, attention3, attention4), dim=0)

        return src


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, dropout, n_layers, pre_lnorm):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_scale = hidden_size**0.5
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, filter_size, n_head, pre_lnorm, dropout)
                for _ in range(n_layers)
            ]
        )
        self.pre_lnorm = pre_lnorm
        self.last_norm = nn.LayerNorm(hidden_size)

    def forward(self, src):
        attention = []
        for layer in self.layers:
            src = layer(src)
            # attention.append(att)

        src = self.last_norm(src)

        # attention = torch.stack(attention, dim=0)
        return src



class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        nhead,
        dropout,
        num_encoder_layers,
        num_decoder_layers,
        pre_lnorm=True,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            d_model, dim_feedforward, nhead, dropout, num_encoder_layers, pre_lnorm
        )

    def forward(self, src, trg=None, trg_mask=None):
        enc_out = self.encoder(src)
        return enc_out

