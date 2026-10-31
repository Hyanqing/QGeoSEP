import copy
from typing import List

import torch
from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from datasets.featurizer import get_feature_dims
from models.conv import MLP
from models.gin.encoder import AtomBondEmbedding, BondAngleFloatRBF, DihedralAngleFloatRBF, \
    EGeoGNNBlock, BondFloatRBF, AttentionEGeoGNNBlock, AdapterEGeoGNNBlock

_REDUCER_NAMES = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class EGeoGNNModel(nn.Module):
    def __init__(
            self, latent_size, encoder_dropout, n_layers, atom_names, bond_names, device,
            without_dihedral=False, use_attention=False, attention_heads=8, **kwargs
    ):
        super(EGeoGNNModel, self).__init__()
        self.without_dihedral = without_dihedral

        self.latent_size = latent_size
        self.dropout_rate = encoder_dropout
        self.n_layers = n_layers

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.init_atom_embedding = AtomBondEmbedding(get_feature_dims(self.atom_names), self.latent_size)
        self.init_bond_embedding = AtomBondEmbedding(get_feature_dims(self.bond_names), self.latent_size)
        self.init_bond_float_rbf = BondFloatRBF(self.latent_size, device=device)
        self.init_bond_angle_float_rbf = BondAngleFloatRBF(self.latent_size, device=device)

        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        if not self.without_dihedral:
            self.dihedral_angle_float_rbf_list = nn.ModuleList()

        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()
        if not self.without_dihedral:
            self.angle_dihedral_block_list = nn.ModuleList()

        self.use_attention = use_attention
        if use_attention:
            # block_class = AttentionEGeoGNNBlock
            block_class = AdapterEGeoGNNBlock
        else:
            block_class = EGeoGNNBlock

        for layer_id in range(self.n_layers):
            self.bond_embedding_list.append(
                AtomBondEmbedding(get_feature_dims(self.bond_names), self.latent_size)
            )
            self.bond_float_rbf_list.append(
                BondFloatRBF(self.latent_size, device=device)
            )
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.latent_size, device=device)
            )
            if not self.without_dihedral:
                self.dihedral_angle_float_rbf_list.append(
                    DihedralAngleFloatRBF(self.latent_size, device=device)
                )
            self.atom_bond_block_list.append(
                block_class(
                    self.latent_size, self.dropout_rate,
                    last_act=(layer_id != self.n_layers - 1),
                    heads=attention_heads
                )
            )
            # self.bond_angle_block_list.append(
            #     block_class(
            #         self.latent_size, self.dropout_rate,
            #         last_act=(layer_id != self.n_layers - 1),
            #         heads=attention_heads
            #     )
            # )

            self.bond_angle_block_list.append(
                block_class(
                    self.latent_size, self.dropout_rate,
                    last_act=(layer_id != self.n_layers - 1),
                    heads=attention_heads
                )
            )

            # if self.use_attention and layer_id == 0:
            #     bond_angle_block = AttentionEGeoGNNBlock(
            #         self.latent_size,
            #         self.dropout_rate,
            #         last_act=(layer_id != self.n_layers - 1),
            #         heads=attention_heads
            #     )
            # else:
            #     bond_angle_block = EGeoGNNBlock(
            #         self.latent_size,
            #         self.dropout_rate,
            #         last_act=(layer_id != self.n_layers - 1)
            #     )
            #
            # self.bond_angle_block_list.append(bond_angle_block)

            if not self.without_dihedral:
                self.angle_dihedral_block_list.append(
                    block_class(
                        self.latent_size, self.dropout_rate,
                        last_act=(layer_id != self.n_layers - 1),
                        heads=attention_heads
                    )
                )

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def mask_attr(
            self, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices
    ):
        _x = copy.deepcopy(x)
        _bond_attr = copy.deepcopy(bond_attr)
        _bond_lengths = copy.deepcopy(bond_lengths)
        _bond_angles = copy.deepcopy(bond_angles)
        _dihedral_angles = copy.deepcopy(dihedral_angles)

        atom_vocab_sizes = get_feature_dims(self.atom_names)
        for i in range(_x.shape[1]):
            if masked_atom_indices is not None:
                _x[:, i][masked_atom_indices] = atom_vocab_sizes[i] - 1

        bond_vocab_sizes = get_feature_dims(self.bond_names)
        for i in range(_bond_attr.shape[1]):
            if masked_bond_indices is not None:
                _bond_attr[:, i][masked_bond_indices] = bond_vocab_sizes[i] - 1

        if masked_bond_indices is not None:
            _bond_lengths[masked_bond_indices] = 0
        if masked_angle_indices is not None:
            _bond_angles[masked_angle_indices] = 0
        if masked_dihedral_indices is not None:
            _dihedral_angles[masked_dihedral_indices] = 0

        return _x, _bond_attr, _bond_lengths, _bond_angles, _dihedral_angles

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            atom_batch, num_bonds, num_angles, num_graphs,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices,
            return_attention_weights=False, **kwargs
    ):
        x, bond_attr, bond_lengths, bond_angles, dihedral_angles = self.mask_attr(
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices
        )

        node_hidden = self.init_atom_embedding(x)
        bond_embed = self.init_bond_embedding(bond_attr)
        bond_hidden = bond_embed + self.init_bond_float_rbf(bond_lengths)

        if self.without_dihedral:
            node_hidden_list = [node_hidden]
            edge_hidden_list = [bond_hidden]

            graph_idx = torch.arange(num_graphs).to(x.device)
            bond_batch = torch.repeat_interleave(graph_idx, num_bonds, dim=0)

            for layer_id in range(self.n_layers):
                node_hidden = self.atom_bond_block_list[layer_id](
                    node_hidden=node_hidden_list[layer_id],
                    edge_hidden=edge_hidden_list[layer_id],
                    edge_index=AtomBondGraph_edges,
                    node_batches=atom_batch
                )

                cur_edge_hidden = self.bond_embedding_list[layer_id](bond_attr)
                cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](bond_lengths)
                cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angles)
                edge_hidden = self.bond_angle_block_list[layer_id](
                    node_hidden=cur_edge_hidden,
                    edge_hidden=cur_angle_hidden,
                    edge_index=BondAngleGraph_edges,
                    node_batches=bond_batch
                )

                node_hidden_list.append(node_hidden)
                edge_hidden_list.append(edge_hidden)

            node_repr = node_hidden_list[-1]
            edge_repr = edge_hidden_list[-1]
            graph_repr = global_mean_pool(node_repr, atom_batch, size=num_graphs)

            return node_repr, edge_repr, None, None, graph_repr

        angle_hidden = self.init_bond_angle_float_rbf(bond_angles)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [bond_hidden]
        angle_hidden_list = [angle_hidden]

        graph_idx = torch.arange(num_graphs).to(x.device)
        bond_batch = torch.repeat_interleave(graph_idx, num_bonds, dim=0)
        angle_batch = torch.repeat_interleave(graph_idx, num_angles, dim=0)

        attention_weights = {}

        for layer_id in range(self.n_layers):
            if return_attention_weights and self.use_attention:
                node_hidden, atom_bond_attention = self.atom_bond_block_list[layer_id](
                    node_hidden=node_hidden_list[layer_id],
                    edge_hidden=edge_hidden_list[layer_id],
                    edge_index=AtomBondGraph_edges,
                    node_batches=atom_batch,
                    return_attention_weights=True
                )
                attention_weights[f'layer_{layer_id}_atom_bond'] = {
                    "attn": atom_bond_attention,
                    "edges": AtomBondGraph_edges
                }
            else:
                node_hidden = self.atom_bond_block_list[layer_id](
                    node_hidden=node_hidden_list[layer_id],
                    edge_hidden=edge_hidden_list[layer_id],
                    edge_index=AtomBondGraph_edges,
                    node_batches=atom_batch
                )

            cur_edge_hidden = self.bond_embedding_list[layer_id](bond_attr)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](bond_lengths)

            if return_attention_weights and self.use_attention and layer_id == 0:
            # if return_attention_weights and self.use_attention:
                edge_hidden, bond_angle_attention = self.bond_angle_block_list[layer_id](
                    node_hidden=cur_edge_hidden,
                    edge_hidden=angle_hidden_list[layer_id],
                    edge_index=BondAngleGraph_edges,
                    node_batches=bond_batch,
                    return_attention_weights=True
                )
            else:
                edge_hidden = self.bond_angle_block_list[layer_id](
                    node_hidden=cur_edge_hidden,
                    edge_hidden=angle_hidden_list[layer_id],
                    edge_index=BondAngleGraph_edges,
                    node_batches=bond_batch
                )


            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angles)
            cur_dihedral_hidden = self.dihedral_angle_float_rbf_list[layer_id](dihedral_angles)

            if not self.without_dihedral:
                cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angles)
                cur_dihedral_hidden = self.dihedral_angle_float_rbf_list[layer_id](dihedral_angles)
                if return_attention_weights and self.use_attention:
                    angle_hidden, angle_dihedral_attention = self.angle_dihedral_block_list[layer_id](
                        node_hidden=cur_angle_hidden,
                        edge_hidden=cur_dihedral_hidden,
                        edge_index=AngleDihedralGraph_edges,
                        node_batches=angle_batch,
                        return_attention_weights=True
                    )
                else:
                    angle_hidden = self.angle_dihedral_block_list[layer_id](
                        node_hidden=cur_angle_hidden,
                        edge_hidden=cur_dihedral_hidden,
                        edge_index=AngleDihedralGraph_edges,
                        node_batches=angle_batch
                    )

            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
            angle_hidden_list.append(angle_hidden)

        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        angle_repr = angle_hidden_list[-1]
        dihedral_repr = cur_dihedral_hidden
        graph_repr = global_mean_pool(node_repr, atom_batch, size=num_graphs)

        return node_repr, edge_repr, angle_repr, dihedral_repr, graph_repr, attention_weights


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

    def _get_cm5_loss(self, atom_attr, cm5_charges, masked_atom_indices=None):
        masked_atom_i = masked_atom_indices
        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        pred = self.cm5_mlp(atom_attr_i)
        return self.cm5_loss(
            pred,
            cm5_charges[masked_atom_indices].unsqueeze(-1)
        )

    def _get_espc_loss(self, atom_attr, espc_charges, masked_atom_indices=None):
        masked_atom_i = masked_atom_indices
        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        pred = self.espc_mlp(atom_attr_i)
        return self.espc_loss(
            pred,
            espc_charges[masked_atom_indices].unsqueeze(-1)
        )

    def _get_hirshfeld_loss(self, atom_attr, hirshfeld_charges, masked_atom_indices=None):
        masked_atom_i = masked_atom_indices
        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        pred = self.hirshfeld_mlp(atom_attr_i)
        return self.hirshfeld_loss(
            pred,
            hirshfeld_charges[masked_atom_indices].unsqueeze(-1)
        )

    def _get_npa_loss(self, atom_attr, npa_charges, masked_atom_indices=None):
        masked_atom_i = masked_atom_indices
        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        pred = self.npa_mlp(atom_attr_i)
        return self.npa_loss(
            pred,
            npa_charges[masked_atom_indices].unsqueeze(-1)
        )

    def _get_wiberg_loss(self, atom_attr, bond_orders, AtomBondGraph_edges, masked_bond_indices=None):
        masked_atom_i, masked_atom_j = AtomBondGraph_edges.index_select(1, masked_bond_indices)
        atom_attr_i = atom_attr.index_select(0, masked_atom_i)
        atom_attr_j = atom_attr.index_select(0, masked_atom_j)
        atom_attr_ij = torch.cat((atom_attr_i, atom_attr_j), dim=1)

        pred = self.wiberg_mlp(atom_attr_ij)
        return self.wiberg_loss(
            pred,
            bond_orders[masked_bond_indices].unsqueeze(-1)
        )

    def compute_loss(
            self, bond_lengths, bond_angles, dihedral_angles,
            AtomBondGraph_edges, atom_attr, BondAngleGraph_edges, AngleDihedralGraph_edges,
            cm5_charges, espc_charges, hirshfeld_charges, npa_charges, bond_orders,
            masked_atom_indices=None, masked_bond_indices=None,
            masked_angle_indices=None, masked_dihedral_indices=None
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
                masked_dihedral_indices=masked_dihedral_indices
            )
            loss += dihedral_angle_loss
            loss_dict["dihedral_angle_loss"] = dihedral_angle_loss.detach().item()

        if "CM5" in self.pretrain_tasks:
            cm5_charge_loss = self._get_cm5_loss(
                atom_attr=atom_attr,
                cm5_charges=cm5_charges,
                masked_atom_indices=masked_atom_indices
            )
            loss += cm5_charge_loss
            loss_dict["cm5_charge_loss"] = cm5_charge_loss.detach().item()

        if "ESPC" in self.pretrain_tasks:
            espc_charge_loss = self._get_espc_loss(
                atom_attr=atom_attr,
                espc_charges=espc_charges,
                masked_atom_indices=masked_atom_indices
            )
            loss += espc_charge_loss
            loss_dict["espc_charge_loss"] = espc_charge_loss.detach().item()

        if "HIRSHFELD" in self.pretrain_tasks:
            hirshfeld_charge_loss = self._get_hirshfeld_loss(
                atom_attr=atom_attr,
                hirshfeld_charges=hirshfeld_charges,
                masked_atom_indices=masked_atom_indices
            )
            loss += hirshfeld_charge_loss
            loss_dict["hirshfeld_charge_loss"] = hirshfeld_charge_loss.detach().item()

        if "NPA" in self.pretrain_tasks:
            npa_charge_loss = self._get_npa_loss(
                atom_attr=atom_attr,
                npa_charges=npa_charges,
                masked_atom_indices=masked_atom_indices
            )
            loss += npa_charge_loss
            loss_dict["npa_charge_loss"] = npa_charge_loss.detach().item()

        if "WIBERG" in self.pretrain_tasks:
            wiberg_order_loss = self._get_wiberg_loss(
                atom_attr=atom_attr,
                bond_orders=bond_orders,
                AtomBondGraph_edges=AtomBondGraph_edges,
                masked_bond_indices=masked_bond_indices
            )
            loss += wiberg_order_loss
            loss_dict["wiberg_order_loss"] = wiberg_order_loss.detach().item()

        loss_dict["loss"] = loss.detach().item()
        return loss, loss_dict

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            cm5_charges, espc_charges, hirshfeld_charges, npa_charges, bond_orders,
            num_graphs, num_bonds, num_angles, atom_batch,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices,
            **kwargs
    ):

        atom_attr, _, _, _, _ = self.compound_encoder(
            AtomBondGraph_edges=AtomBondGraph_edges,
            BondAngleGraph_edges=BondAngleGraph_edges,
            AngleDihedralGraph_edges=AngleDihedralGraph_edges,
            x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
            bond_angles=bond_angles, dihedral_angles=dihedral_angles,
            num_graphs=num_graphs, num_bonds=num_bonds,
            num_angles=num_angles, atom_batch=atom_batch,
            masked_atom_indices=masked_atom_indices,
            masked_bond_indices=masked_bond_indices,
            masked_angle_indices=masked_angle_indices,
            masked_dihedral_indices=masked_dihedral_indices
        )

        return self.compute_loss(
            atom_attr=atom_attr,
            bond_lengths=bond_lengths,
            bond_angles=bond_angles,
            cm5_charges=cm5_charges,
            espc_charges=espc_charges,
            hirshfeld_charges=hirshfeld_charges,
            npa_charges=npa_charges,
            bond_orders=bond_orders,
            dihedral_angles=dihedral_angles,
            AtomBondGraph_edges=AtomBondGraph_edges,
            BondAngleGraph_edges=BondAngleGraph_edges,
            AngleDihedralGraph_edges=AngleDihedralGraph_edges,
            masked_atom_indices=masked_atom_indices,
            masked_bond_indices=masked_bond_indices,
            masked_angle_indices=masked_angle_indices,
            masked_dihedral_indices=masked_dihedral_indices,
        )
