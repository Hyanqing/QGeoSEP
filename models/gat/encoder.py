import torch
from torch import nn

from models.conv import DropoutIfTraining, MLP, NodeAttn


class NodeEncoder(nn.Module):
    def __init__(
            self, latent_size: int, hidden_size: int, n_layers: int,
            use_layer_norm: bool, layernorm_before: bool, use_bn: bool,
            dropnode_rate: float, encoder_dropout: float
    ):
        super().__init__()

        self.encoder = DropoutIfTraining(
            p=dropnode_rate,
            submodule=MLP(
                latent_size * 4,
                [hidden_size] * n_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                layernorm_before=layernorm_before,
                dropout=encoder_dropout,
                use_bn=use_bn,
            )
        )

        self.node_attn = NodeAttn(latent_size, num_heads=None)

    def forward(self, edges, node_feat, edge_attr, u, num_nodes):
        max_node_id = node_feat.size(0)

        row = edges[0]  # attention
        col = edges[1]  # attention

        sent_attributes = self.node_attn(node_feat[row], node_feat[col], edge_attr, row, max_node_id)
        received_attributes = self.node_attn(node_feat[col], node_feat[row], edge_attr, col, max_node_id)
        global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
        feat_list = [node_feat, sent_attributes, received_attributes, global_nodes]
        node_feat = self.encoder(torch.cat(feat_list, dim=1))

        return node_feat


class GlobalEncoder(nn.Module):
    def __init__(
            self, latent_size: int, hidden_size: int, n_layers: int,
            use_layer_norm: bool, layernorm_before: bool, use_bn: bool, encoder_dropout: float, aggregate_fn,
    ):
        super().__init__()
        self.encoder = MLP(
            latent_size * 4,
            [hidden_size] * n_layers + [latent_size],
            use_layer_norm=use_layer_norm,
            layernorm_before=layernorm_before,
            dropout=encoder_dropout,
            use_bn=use_bn,
        )

        self.aggregate_fn = aggregate_fn

    def forward(self, u, atom_attr, bond_attr, angle_attr, atom_batch, bond_batch, angle_batch):
        n_graph = u.size(0)
        atom_attributes = self.aggregate_fn(atom_attr, atom_batch, size=n_graph)
        edge_attributes = self.aggregate_fn(bond_attr, bond_batch, size=n_graph)
        angle_attributes = self.aggregate_fn(angle_attr, angle_batch, size=n_graph)
        feat_list = [u, atom_attributes, edge_attributes, angle_attributes]
        u = self.encoder(torch.cat(feat_list, dim=-1))
        return u


class EGeoGNNBlock(nn.Module):
    def __init__(
            self, latent_size: int, hidden_size: int, n_layers: int,
            use_layer_norm: bool, layernorm_before: bool, use_bn: bool,
            dropnode_rate: float, encoder_dropout: float,
            aggregate_fn,
    ):
        super().__init__()

        self.atom_encoder = NodeEncoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            layernorm_before=layernorm_before,
            use_bn=use_bn,
            dropnode_rate=dropnode_rate,
            encoder_dropout=encoder_dropout
        )

        self.bond_encoder = NodeEncoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            layernorm_before=layernorm_before,
            use_bn=use_bn,
            dropnode_rate=dropnode_rate,
            encoder_dropout=encoder_dropout
        )

        self.angle_encoder = NodeEncoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            layernorm_before=layernorm_before,
            use_bn=use_bn,
            dropnode_rate=dropnode_rate,
            encoder_dropout=encoder_dropout
        )

        self.global_encoder = GlobalEncoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            layernorm_before=layernorm_before,
            use_bn=use_bn,
            encoder_dropout=encoder_dropout,
            aggregate_fn=aggregate_fn,
        )

    def forward(self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
                atom_attr, bond_attr, angle_attr, dihedral_attr, u,
                num_atoms, num_bonds, num_angles,
                atom_batch, bond_batch, angle_batch):
        atom_attr = self.atom_encoder(
            edges=AtomBondGraph_edges,
            node_feat=atom_attr,
            edge_attr=bond_attr,
            u=u,
            num_nodes=num_atoms
        )
        bond_attr = self.bond_encoder(
            edges=BondAngleGraph_edges,
            node_feat=bond_attr,
            edge_attr=angle_attr,
            u=u,
            num_nodes=num_bonds
        )
        angle_attr = self.angle_encoder(
            edges=AngleDihedralGraph_edges,
            node_feat=angle_attr,
            edge_attr=dihedral_attr,
            u=u,
            num_nodes=num_angles
        )

        u = self.global_encoder(
            u=u,
            atom_attr=atom_attr,
            bond_attr=bond_attr,
            angle_attr=angle_attr,
            atom_batch=atom_batch,
            bond_batch=bond_batch,
            angle_batch=angle_batch
        )

        return atom_attr, bond_attr, angle_attr, u
