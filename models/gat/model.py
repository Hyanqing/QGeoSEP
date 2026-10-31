from typing import List
import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from datasets import EgeognnPretrainedDataset
from datasets.featurizer import get_feature_dims
from models.conv import MLP
from models.gat.encoder import EGeoGNNBlock
from models.gin.encoder import BondAngleFloatRBF, DihedralAngleFloatRBF

_REDUCER_NAMES = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class EGeoGNNModel(nn.Module):
    def __init__(
            self, latent_size: int, hidden_size: int, n_layers: int,
            use_layer_norm: bool, layernorm_before: bool, use_bn: bool,
            dropnode_rate: float, encoder_dropout: float, num_message_passing_steps: int,
            atom_names: List[str], bond_names: List[str], global_reducer: str,
            device
    ):
        super().__init__()

        self.bond_init = MLP(
            sum(get_feature_dims(bond_names)),
            [hidden_size] * n_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.atom_init = MLP(
            sum(get_feature_dims(atom_names)),
            [hidden_size] * n_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        # self.global_init = nn.Parameter(torch.zeros((1, latent_size)))
        self.global_init = nn.Parameter(torch.randn((1, latent_size)))

        self.layers = nn.ModuleList([
            EGeoGNNBlock(
                latent_size=latent_size, hidden_size=hidden_size, n_layers=n_layers,
                use_layer_norm=use_layer_norm, layernorm_before=layernorm_before, use_bn=use_bn,
                dropnode_rate=dropnode_rate, encoder_dropout=encoder_dropout,
                aggregate_fn=_REDUCER_NAMES[global_reducer]
            ) for _ in range(num_message_passing_steps)
        ])

        self.latent_size = latent_size

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.pos_embedding = MLP(3, [latent_size, latent_size])
        self.dis_embedding = MLP(1, [latent_size, latent_size])
        self.angle_embedding = BondAngleFloatRBF(latent_size, device=device)
        self.dihedral_embedding = DihedralAngleFloatRBF(latent_size, device=device)

        self.encoder_dropout = encoder_dropout

    def one_hot_atoms(self, atoms, masked_atom_indices=None):
        _atoms = copy.deepcopy(atoms)
        vocab_sizes = get_feature_dims(self.atom_names)
        one_hots = []
        for i in range(_atoms.shape[1]):
            if masked_atom_indices is not None:
                _atoms[:, i][masked_atom_indices] = vocab_sizes[i] - 1
            one_hots.append(
                F.one_hot(_atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device).to(torch.float32)
            )
        return torch.cat(one_hots, dim=1)

    def one_hot_bonds(self, bonds, masked_bond_indices=None):
        _bonds = copy.deepcopy(bonds)
        vocab_sizes = get_feature_dims(self.bond_names)
        one_hots = []
        for i in range(bonds.shape[1]):
            if masked_bond_indices is not None:
                _bonds[:, i][masked_bond_indices] = vocab_sizes[i] - 1
            one_hots.append(
                F.one_hot(_bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device).to(torch.float32)
            )
        return torch.cat(one_hots, dim=1)

    def extend_x_edge(self, pos, x, edge_attr, bond_lengths):
        extended_x = x + self.pos_embedding(pos)
        extended_edge_attr = edge_attr + self.dis_embedding(bond_lengths.unsqueeze(-1))
        return extended_x, extended_edge_attr

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            masked_atom_indices=None, masked_bond_indices=None,
            masked_angle_indices=None, masked_dihedral_indices=None,
    ):
        onehot_x = self.one_hot_atoms(x, masked_atom_indices=masked_atom_indices)
        onehot_bond_attr = self.one_hot_bonds(bond_attr, masked_bond_indices=masked_bond_indices)

        _bond_angles = copy.deepcopy(bond_angles).to(bond_angles.device)
        _dihedral_angles = copy.deepcopy(dihedral_angles).to(dihedral_angles.device)

        if masked_angle_indices is not None:
            _bond_angles[masked_angle_indices] = 1e-5
        if masked_dihedral_indices is not None:
            _dihedral_angles[masked_dihedral_indices] = 1e-5

        graph_idx = torch.arange(num_graphs).to(x.device)
        bond_batch = torch.repeat_interleave(graph_idx, num_bonds, dim=0)
        angle_batch = torch.repeat_interleave(graph_idx, num_angles, dim=0)

        atom_attr = self.atom_init(onehot_x)
        bond_attr = self.bond_init(onehot_bond_attr)
        angle_attr = self.angle_embedding(_bond_angles.unsqueeze(-1))
        dihedral_attr = self.dihedral_embedding(_dihedral_angles.unsqueeze(-1))

        u = self.global_init.expand(num_graphs, -1)

        atom_attr, bond_attr = self.extend_x_edge(
            pos=pos,
            x=atom_attr,
            edge_attr=bond_attr,
            bond_lengths=bond_lengths,
        )

        for i, layer in enumerate(self.layers):
            atom_attr_1, bond_attr_1, angle_attr_1, u_1 = layer(
                AtomBondGraph_edges=AtomBondGraph_edges,
                BondAngleGraph_edges=BondAngleGraph_edges,
                AngleDihedralGraph_edges=AngleDihedralGraph_edges,
                atom_attr=atom_attr,
                bond_attr=bond_attr,
                angle_attr=angle_attr,
                dihedral_attr=dihedral_attr,
                u=u,
                num_atoms=num_atoms,
                num_bonds=num_bonds,
                num_angles=num_angles,
                atom_batch=atom_batch,
                bond_batch=bond_batch,
                angle_batch=angle_batch
            )
            atom_attr = F.dropout(atom_attr_1, p=self.encoder_dropout, training=self.training) + atom_attr
            bond_attr = F.dropout(bond_attr_1, p=self.encoder_dropout, training=self.training) + bond_attr
            angle_attr = F.dropout(angle_attr_1, p=self.encoder_dropout, training=self.training) + angle_attr
            # dihedral_attr = F.dropout(dihedral_attr_1, p=self.dropout, training=self.training) + dihedral_attr
            u = F.dropout(u_1, p=self.encoder_dropout, training=self.training) + u

        return atom_attr, bond_attr, angle_attr, dihedral_attr, u


class EGEM(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel, pretrain_tasks: List[str],
            n_layers: int, hidden_size: int, dropout_rate: float,
            use_layer_norm: bool, use_bn: bool, adc_vocab: int
    ):
        super().__init__()

        self.compound_encoder = compound_encoder
        self.pretrain_tasks = pretrain_tasks

        # bond length with regression
        if 'Blr' in pretrain_tasks:
            self.Blr_mlp = MLP(
                input_size=compound_encoder.latent_size * 2,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Blr_loss = nn.SmoothL1Loss()

        # bond angle with regression
        if 'Bar' in pretrain_tasks:
            self.Bar_mlp = MLP(
                input_size=compound_encoder.latent_size * 3,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Bar_loss = nn.SmoothL1Loss()

        # dihedral angle with regression
        if 'Dar' in pretrain_tasks:
            self.Dar_mlp = MLP(
                input_size=compound_encoder.latent_size * 4,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Dar_loss = nn.SmoothL1Loss()

        # atom distance with classification
        if 'Adc' in pretrain_tasks:
            self.Adc_mlp = MLP(
                input_size=compound_encoder.latent_size * 2,
                output_sizes=[hidden_size] * n_layers + [adc_vocab + 3],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Adc_loss = nn.CrossEntropyLoss()

        # cm5 charges with regression
        if 'CM5' in pretrain_tasks:
            self.cm5_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.cm5_loss = nn.SmoothL1Loss()

        # espc charges with regression
        if 'ESPC' in pretrain_tasks:
            self.espc_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.espc_loss = nn.SmoothL1Loss()

        # hirshfeld charges with regression
        if 'HIRSHFELD' in pretrain_tasks:
            self.hirshfeld_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.hirshfeld_loss = nn.SmoothL1Loss()

        # npa charges with regression
        if 'NPA' in pretrain_tasks:
            self.npa_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.npa_loss = nn.SmoothL1Loss()

        # bond wiberg order with regression
        if 'WIBERG' in pretrain_tasks:
            self.wiberg_mlp = MLP(
                input_size=compound_encoder.latent_size * 2,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.wiberg_loss = nn.SmoothL1Loss()

    def _get_Blr_loss(self, atom_attr, bond_lengths, AtomBondGraph_edges, masked_bond_indices=None):
        masked_atom_i, masked_atom_j = AtomBondGraph_edges.index_select(1, masked_bond_indices)
        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        atom_attr_j = atom_attr.index_select(0, masked_atom_j)
        atom_attr_ij = torch.cat((atom_attr_i, atom_attr_j), dim=1)

        pred = self.Blr_mlp(atom_attr_ij)
        return self.Blr_loss(
            pred,
            bond_lengths[masked_bond_indices].unsqueeze(-1)
        )

    def _get_Bar_loss(
            self, atom_attr, bond_angles,
            AtomBondGraph_edges, BondAngleGraph_edges,
            masked_angle_indices=None
    ):
        masked_bond_i, masked_bond_j = BondAngleGraph_edges.index_select(1, masked_angle_indices)

        masked_atom_i, masked_atom_j = AtomBondGraph_edges.index_select(1, masked_bond_i)
        _, masked_atom_k = AtomBondGraph_edges.index_select(1, masked_bond_j)

        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        atom_attr_j = atom_attr.index_select(0, masked_atom_j)
        atom_attr_k = atom_attr.index_select(0, masked_atom_k)
        atom_attr_ijk = torch.cat((atom_attr_i, atom_attr_j, atom_attr_k), dim=1)

        pred = self.Bar_mlp(atom_attr_ijk)
        return self.Bar_loss(
            pred,
            bond_angles[masked_angle_indices].unsqueeze(-1)
        )

    def _get_Dar_loss(
            self, atom_attr, dihedral_angles,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            masked_dihedral_indices=None
    ):
        masked_angle_i, masked_angle_j = AngleDihedralGraph_edges.index_select(1, masked_dihedral_indices)

        masked_bond_i, _ = BondAngleGraph_edges.index_select(1, masked_angle_i)
        _, masked_bond_k = BondAngleGraph_edges.index_select(1, masked_angle_j)

        masked_atom_i, masked_atom_j = AtomBondGraph_edges.index_select(1, masked_bond_i)
        masked_atom_k, masked_atom_l = AtomBondGraph_edges.index_select(1, masked_bond_k)

        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        atom_attr_j = atom_attr.index_select(0, masked_atom_j)
        atom_attr_k = atom_attr.index_select(0, masked_atom_k)
        atom_attr_l = atom_attr.index_select(0, masked_atom_l)
        atom_attr_ijkl = torch.cat((atom_attr_i, atom_attr_j, atom_attr_k, atom_attr_l), dim=1)

        pred = self.Dar_mlp(atom_attr_ijkl)
        return self.Dar_loss(
            pred,
            dihedral_angles[masked_dihedral_indices].unsqueeze(-1)
        )

    def compute_loss(
            self, bond_lengths, bond_angles, dihedral_angles,
            AtomBondGraph_edges, atom_attr, BondAngleGraph_edges, AngleDihedralGraph_edges,
            masked_bond_indices=None, masked_angle_indices=None, masked_dihedral_indices=None
    ):
        loss = 0
        loss_dict = {}

        if "Blr" in self.pretrain_tasks:
            bond_length_loss = self._get_Blr_loss(
                atom_attr=atom_attr,
                bond_lengths=bond_lengths,
                AtomBondGraph_edges=AtomBondGraph_edges,
                masked_bond_indices=masked_bond_indices
            )
            loss += bond_length_loss
            loss_dict["bond_length_loss"] = bond_length_loss.detach().item()

        if "Bar" in self.pretrain_tasks:
            bond_angle_loss = self._get_Bar_loss(
                atom_attr=atom_attr,
                bond_angles=bond_angles,
                AtomBondGraph_edges=AtomBondGraph_edges,
                BondAngleGraph_edges=BondAngleGraph_edges,
                masked_angle_indices=masked_angle_indices
            )
            loss += bond_angle_loss
            loss_dict["bond_angle_loss"] = bond_angle_loss.detach().item()

        if "Dar" in self.pretrain_tasks:
            dihedral_angle_loss = self._get_Dar_loss(
                atom_attr=atom_attr,
                dihedral_angles=dihedral_angles,
                AtomBondGraph_edges=AtomBondGraph_edges,
                BondAngleGraph_edges=BondAngleGraph_edges,
                AngleDihedralGraph_edges=AngleDihedralGraph_edges,
                masked_dihedral_indices = masked_dihedral_indices
            )
            loss += dihedral_angle_loss
            loss_dict["dihedral_angle_loss"] = dihedral_angle_loss.detach().item()

        loss_dict["loss"] = loss.detach().item()
        return loss, loss_dict

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            masked_atom_indices=None, masked_bond_indices=None,
            masked_angle_indices=None, masked_dihedral_indices=None
    ):

        atom_attr, bond_attr, angle_attr, dihedral_attr, u = self.compound_encoder(
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            masked_atom_indices=masked_atom_indices,
            masked_bond_indices=masked_bond_indices,
            masked_angle_indices=masked_angle_indices,
            masked_dihedral_indices=masked_dihedral_indices
        )

        return self.compute_loss(
            atom_attr=atom_attr,
            bond_lengths=bond_lengths,
            bond_angles=bond_angles,
            dihedral_angles=dihedral_angles,
            AtomBondGraph_edges=AtomBondGraph_edges,
            BondAngleGraph_edges=BondAngleGraph_edges,
            AngleDihedralGraph_edges=AngleDihedralGraph_edges,
            masked_bond_indices=masked_bond_indices,
            masked_angle_indices=masked_angle_indices,
            masked_dihedral_indices=masked_dihedral_indices,
        )
        # self.compute_loss(bond_attr, masked_bond_indices=masked_bond_indices)


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    import json
    from tqdm import tqdm

    with open("../../input_feats.json") as f:
        configs = json.load(f)

    dataset = EgeognnPretrainedDataset(
        root='../../data/demo', dataset_name='demo',
        remove_hs=True, base_path="../../data/demo",
        atom_names=configs["atom_names"], bond_names=configs["bond_names"],
        with_provided_3d=False, mask_ratio=0.12
    )

    demo_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=10
    )

    encoder = EGeoGNNModel(
        latent_size=256, hidden_size=512, n_layers=4,
        use_layer_norm=True, layernorm_before=True, use_bn=True,
        dropnode_rate=0.01, encoder_dropout=0.1, num_message_passing_steps=4,
        atom_names=configs["atom_names"], bond_names=configs["bond_names"],
        global_reducer='sum'
    )

    model = EGEM(
        compound_encoder=encoder,
        pretrain_tasks=['Blr', 'Bar', 'Dar'],
        n_layers=4,
        hidden_size=128,
        dropout_rate=0.3,
        use_bn=False,
        use_layer_norm=False,
        adc_vocab=4
    )

    pbar = tqdm(demo_loader, desc="Iteration")

    for step, batch in enumerate(pbar):
        model(
            AtomBondGraph_edges=batch.AtomBondGraph_edges,
            BondAngleGraph_edges=batch.BondAngleGraph_edges,
            AngleDihedralGraph_edges=batch.AngleDihedralGraph_edges,
            pos=batch.atom_poses,
            x=batch.node_feat,
            bond_attr=batch.edge_attr,
            bond_lengths=batch.bond_lengths,
            bond_angles=batch.bond_angles,
            dihedral_angles=batch.dihedral_angles,
            num_graphs=batch.num_graphs,
            num_atoms=batch.n_atoms,
            num_bonds=batch.n_bonds,
            num_angles=batch.n_angles,
            atom_batch=batch.batch,
            masked_atom_indices=batch.masked_atom_indices,
            masked_bond_indices=batch.masked_bond_indices,
            masked_angle_indices=batch.masked_angle_indices,
            masked_dihedral_indices=batch.masked_dihedral_indices,
        )
