import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.conv import MLP
from models.gin import EGeoGNNModel


class DownstreamMTModel(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel,
            hidden_size, dropout_rate, n_layers,
            endpoints, frozen_encoder,
            task_type='regression',
            inference=False
    ):
        super(DownstreamMTModel, self).__init__()
        self.task_type = task_type
        self.num_tasks = len(endpoints)
        self.frozen_encoder = frozen_encoder

        self.downstream_layer = MLP(
            input_size=compound_encoder.latent_size,
            output_sizes=[hidden_size] * n_layers + [self.num_tasks],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.latent_size)

        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()
            self.loss_func = nn.BCELoss()
        else:
            self.loss_func = nn.MSELoss()

        self.inference = inference

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch
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
        pred = self.downstream_layer(graph_repr)

        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred

    def compute_loss(self, pred, target):
        loss_dict = {}

        loss = self.loss_func(pred, target)
        loss_dict['loss'] = loss.detach().item()

        return loss, loss_dict
