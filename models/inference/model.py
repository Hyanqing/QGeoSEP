import numpy as np
from torch import nn

from models.downstream import DownstreamModel, DownstreamMTModel


class InferenceModel(nn.Module):
    def __init__(
            self,
            toxicity_model: DownstreamModel,
            physchem_model: DownstreamModel,
            toxicity_endpoints, physchem_endpoints,
            endpoint_statuses
    ):
        super(InferenceModel, self).__init__()

        self.toxicity_model = toxicity_model
        self.physchem_model = physchem_model

        self.toxicity_endpoints = toxicity_endpoints
        self.physchem_endpoints = physchem_endpoints
        self.endpoint_statuses = endpoint_statuses

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch
    ):
        toxicity_pred_scaled = self.toxicity_model(
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            tgt_endpoints=self.toxicity_endpoints
        )

        physchem_pred_scaled = self.physchem_model(
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            tgt_endpoints=self.physchem_endpoints
        )

        toxicity_pred_scaled = toxicity_pred_scaled.detach().cpu().numpy().reshape(num_graphs, -1).T
        physchem_pred_scaled = physchem_pred_scaled.detach().cpu().numpy().reshape(num_graphs, -1).T
        pred_scaled = np.vstack([toxicity_pred_scaled, physchem_pred_scaled])

        pred_mean = np.array([
            self.endpoint_statuses[endpoint]['mean'] for endpoint in self.toxicity_endpoints + self.physchem_endpoints
        ]).reshape(-1, 1)

        pred_std = np.array([
            self.endpoint_statuses[endpoint]['std'] for endpoint in self.toxicity_endpoints + self.physchem_endpoints
        ]).reshape(-1, 1)

        return pred_scaled * pred_std + pred_mean


class InferenceMTModel(nn.Module):
    def __init__(self, pred_model: DownstreamMTModel, endpoints, endpoint_statuses):
        super(InferenceMTModel, self).__init__()

        self.pred_model = pred_model
        self.endpoints = endpoints
        self.endpoint_statuses = endpoint_statuses

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch
    ):

        pred_scaled = self.pred_model(
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
        )

        pred_mean = np.array([
            self.endpoint_statuses[endpoint]['mean'] for endpoint in self.endpoints
        ])

        pred_std = np.array([
            self.endpoint_statuses[endpoint]['std'] for endpoint in self.endpoints
        ])

        return pred_scaled * pred_std + pred_mean