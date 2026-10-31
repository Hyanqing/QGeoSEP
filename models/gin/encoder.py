import numpy as np
import torch
from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.nn.norm import GraphNorm

from models.conv import GraphAttentionLayer


class EGeoGNNBlock(nn.Module):
    def __init__(self, latent_size, dropout_rate, last_act, **kwargs):
        super(EGeoGNNBlock, self).__init__()

        self.latent_size = latent_size
        self.last_act = last_act

        self.gnn = GIN(latent_size, latent_size, num_layers=2)
        self.norm = nn.LayerNorm(latent_size)
        self.graph_norm = GraphNorm(latent_size)

        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, node_hidden, edge_hidden, edge_index, node_batches=None):
        """tbd"""
        out = self.gnn(x=node_hidden, edge_attr=edge_hidden, edge_index=edge_index)
        out = self.norm(out)
        out = self.graph_norm(out, node_batches)
        if self.last_act:
            out = self.act(out)
        if self.training:
            out = self.dropout(out)
        out = out + node_hidden
        return out


class AttentionEGeoGNNBlock(nn.Module):
    def __init__(self, latent_size, dropout_rate, last_act, heads=8):
        super(AttentionEGeoGNNBlock, self).__init__()

        self.latent_size = latent_size
        self.last_act = last_act

        # 使用自定义注意力层
        self.attention_layer = GraphAttentionLayer(
            latent_size, latent_size, latent_size,
            heads=heads, dropout=dropout_rate
        )

        self.norm = nn.LayerNorm(latent_size)
        self.graph_norm = GraphNorm(latent_size)

        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, node_hidden, edge_hidden, edge_index, node_batches=None, return_attention_weights=False):
        if return_attention_weights:
            out, attention_weights = self.attention_layer(
                node_hidden, edge_index, edge_hidden, return_attention=True
            )
        else:
            out = self.attention_layer(node_hidden, edge_index, edge_hidden)

        out = self.norm(out)
        out = self.graph_norm(out, node_batches)
        if self.last_act:
            out = self.act(out)
        if self.training:
            out = self.dropout(out)
        out = out + node_hidden

        if return_attention_weights:
            return out, attention_weights
        else:
            return out


class AdapterEGeoGNNBlock(nn.Module):
    def __init__(self, latent_size, dropout_rate, last_act, heads=8, use_adapter_attention=True):
        super().__init__()
        self.latent_size = latent_size
        self.last_act = last_act
        self.use_adapter_attention = use_adapter_attention

        # 原始预训练路径，名字尽量保持不变，便于加载旧 checkpoint
        self.gnn = GIN(latent_size, latent_size, num_layers=2)
        self.norm = nn.LayerNorm(latent_size)
        self.graph_norm = GraphNorm(latent_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        if last_act:
            self.act = nn.ReLU()

        self.attention_layer = GraphAttentionLayer(
            latent_size, latent_size, latent_size,
            heads=heads, dropout=dropout_rate
        )
        self.attn_norm = nn.LayerNorm(latent_size)
        self.attn_graph_norm = GraphNorm(latent_size)
        self.attn_dropout = nn.Dropout(p=dropout_rate)

        # 关键：初值设成 0，让初始行为等价于原始 EGeoGNNBlock
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_hidden, edge_hidden, edge_index, node_batches=None, return_attention_weights=False):
        base_delta = self.gnn(x=node_hidden, edge_attr=edge_hidden, edge_index=edge_index)
        base_delta = self.norm(base_delta)
        base_delta = self.graph_norm(base_delta, node_batches)
        if self.last_act:
            base_delta = self.act(base_delta)
        if self.training:
            base_delta = self.dropout(base_delta)

        out = node_hidden + base_delta

        attention_weights = None

        if return_attention_weights:
            attn_delta, attention_weights = self.attention_layer(
                node_hidden, edge_index, edge_hidden, return_attention=True
            )
        else:
            attn_delta = self.attention_layer(node_hidden, edge_index, edge_hidden)

        attn_delta = self.attn_norm(attn_delta)
        attn_delta = self.attn_graph_norm(attn_delta, node_batches)
        if self.last_act:
            attn_delta = self.act(attn_delta)
        if self.training:
            attn_delta = self.attn_dropout(attn_delta)

        out = out + torch.tanh(self.alpha) * attn_delta

        if return_attention_weights:
            return out, attention_weights
        return out


class AtomBondEmbedding(nn.Module):
    def __init__(self, vocab_sizes, latent_size):
        super(AtomBondEmbedding, self).__init__()

        self.embed_layers = nn.ModuleList()
        for vocab_size in vocab_sizes:
            embed = nn.Embedding(
                vocab_size,
                latent_size,
            )
            self.embed_layers.append(embed)

    def forward(self, node_features):
        out_embed = 0
        for i, layer in enumerate(self.embed_layers):
            out_embed += layer(node_features[:, i])
        return out_embed


class RBF(nn.Module):
    def __init__(self, centers, gamma, device):
        super(RBF, self).__init__()
        self.centers = torch.tensor(centers, dtype=torch.float32).reshape(1, -1).to(device)
        self.gamma = gamma

    def forward(self, x):
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondFloatRBF(nn.Module):
    def __init__(self, latent_size, device):
        super(BondFloatRBF, self).__init__()

        centers = np.arange(0, 2, 0.1)
        gamma = 10.0

        self.rbf = RBF(centers, gamma, device)
        self.linear = nn.Linear(len(centers), latent_size)

    def forward(self, bond_lengths):
        rbf_x = self.rbf(bond_lengths)
        out_embed = self.linear(rbf_x)
        return out_embed


class BondAngleFloatRBF(nn.Module):
    def __init__(self, latent_size, device):
        super(BondAngleFloatRBF, self).__init__()

        centers = np.arange(0, np.pi, 0.01)
        gamma = 10.0

        self.rbf = RBF(centers, gamma, device)
        self.linear = nn.Linear(len(centers), latent_size)

    def forward(self, bond_angles):
        rbf_x = self.rbf(bond_angles)
        out_embed = self.linear(rbf_x)
        return out_embed


class DihedralAngleFloatRBF(nn.Module):
    def __init__(self, latent_size, device):
        super(DihedralAngleFloatRBF, self).__init__()

        centers = np.arange(-np.pi, np.pi, 0.01)
        gamma = 10

        self.rbf = RBF(centers, gamma, device)
        self.linear = nn.Linear(len(centers), latent_size)

    def forward(self, dihedral_angles):
        rbf_x = self.rbf(dihedral_angles)
        out_embed = self.linear(rbf_x)
        return out_embed
