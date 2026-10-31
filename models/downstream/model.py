import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.conv import MLP
from models.gin import EGeoGNNModel


class DownstreamModel(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel,
            hidden_size, dropout_rate, n_layers,
            endpoints, frozen_encoder,
            task_type='regression',
            inference=False
    ):
        super(DownstreamModel, self).__init__()
        self.task_type = task_type
        self.endpoints = endpoints
        self.frozen_encoder = frozen_encoder

        num_tasks = len(endpoints)

        self.layer_mapping = dict(zip(endpoints, range(num_tasks)))

        self.downstream_layer = nn.ModuleList([
            MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=False,
                activation=nn.ReLU,
                layernorm_before=False,
                use_bn=False,
                dropout=dropout_rate
            ) for _ in range(num_tasks)
        ])

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.latent_size)

        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

        self.inference = inference

        self.loss_func = nn.MSELoss()

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            tgt_endpoints
    ):
        if self.frozen_encoder:
            self.compound_encoder.eval()
            with torch.no_grad():
                _, _, _, _, graph_repr = self.compound_encoder(
                    AtomBondGraph_edges=AtomBondGraph_edges,
                    BondAngleGraph_edges=BondAngleGraph_edges,
                    AngleDihedralGraph_edges=AngleDihedralGraph_edges,
                    pos=pos, x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
                    bond_angles=bond_angles, dihedral_angles=dihedral_angles,
                    num_graphs=num_graphs, atom_batch=atom_batch,
                    num_atoms=num_atoms, num_bonds=num_bonds, num_angles=num_angles,
                    masked_atom_indices=None,
                    masked_bond_indices=None,
                    masked_angle_indices=None,
                    masked_dihedral_indices=None
                )
        else:
            _, _, _, _, graph_repr = self.compound_encoder(
                AtomBondGraph_edges=AtomBondGraph_edges,
                BondAngleGraph_edges=BondAngleGraph_edges,
                AngleDihedralGraph_edges=AngleDihedralGraph_edges,
                pos=pos, x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
                bond_angles=bond_angles, dihedral_angles=dihedral_angles,
                num_graphs=num_graphs, atom_batch=atom_batch,
                num_atoms=num_atoms, num_bonds=num_bonds, num_angles=num_angles,
                masked_atom_indices=None,
                masked_bond_indices=None,
                masked_angle_indices=None,
                masked_dihedral_indices=None
            )

        graph_repr = self.norm(graph_repr)
        if self.inference:
            graph_repr = graph_repr.repeat_interleave(len(tgt_endpoints), axis=0)
            tgt_endpoints = np.tile(tgt_endpoints, num_graphs)

        pred = torch.cat([
            self.downstream_layer[self.layer_mapping[endpoint]](graph_repr[i])
            for i, endpoint in enumerate(tgt_endpoints)
        ])

        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred

    def compute_loss(self, pred, target):
        loss_dict = {}

        loss = self.loss_func(pred, target)
        loss_dict['loss'] = loss.detach().item()

        return loss, loss_dict
