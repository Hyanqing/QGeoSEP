import numpy as np
import torch
from torch import nn

from models.conv import MLP, MultiHeadAttentionLayer, BidirectionalGatedFusion
from models.gin import EGeoGNNModel
from models.gin.encoder import RBF


class AuxiliaryFloatRBF(nn.Module):
    def __init__(self, c, latent_size, device, hidden_size=32, encoder_mode="mlp"):
        super(AuxiliaryFloatRBF, self).__init__()
        assert encoder_mode in ["rbf", "mlp"]
        self.encoder_mode = encoder_mode

        if self.encoder_mode == "rbf":
            centers = np.arange(*c)
            gamma = 10
            self.rbf = RBF(centers, gamma, device)
            self.linear = nn.Linear(len(centers), latent_size)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size)
            )
            self.norm = nn.LayerNorm(latent_size)

    def forward(self, auxiliary):
        if self.encoder_mode == "rbf":
            rbf_x = self.rbf(auxiliary)
            out_embed = self.linear(rbf_x)
        else:
            x = auxiliary.reshape(-1, 1)
            out_embed = self.encoder(x)
            out_embed = self.norm(out_embed)
        return out_embed


class DownstreamTransformerModelV0(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel,
            hidden_size, dropout_rate, n_layers,
            endpoints, frozen_encoder, device,
            task_type='regression', n_labels=2,
            **kwargs
    ):
        super(DownstreamTransformerModelV0, self).__init__()

        assert task_type in ['regression', 'classification']
        if task_type == "classification":
            assert n_labels >= 2

        self.task_type = task_type
        self.n_labels = n_labels

        self.endpoints = endpoints
        self.frozen_encoder = frozen_encoder
        self.latent_size = compound_encoder.latent_size
        self.num_tasks = len(endpoints)

        self.endpoint_embedding = nn.Embedding(self.num_tasks, self.latent_size)

        self.attention = MultiHeadAttentionLayer(
            hid_dim=compound_encoder.latent_size,
            n_heads=8,
            dropout=dropout_rate,
            device=device
        )

        self.downstream_layer = MLP(
            input_size=compound_encoder.latent_size,
            output_sizes=[hidden_size] * n_layers + [1 if self.task_type == 'regression' else n_labels],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )

        self.endpoint_layer = MLP(
            input_size=compound_encoder.latent_size,
            output_sizes=[hidden_size] * n_layers + [len(self.endpoints)],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )

        self.compound_encoder = compound_encoder

        self.graph_feat_norm = nn.LayerNorm(self.latent_size)
        self.endpoint_attn_layer_norm = nn.LayerNorm(self.latent_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.device = device

        self.loss_func = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.endpoint_loss_func = nn.CrossEntropyLoss()

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            tgt_endpoints, inference=False, **kwargs
    ):

        # endpoint_index = torch.tensor([self.endpoints.index(x) for x in tgt_endpoints]).reshape(-1, 1).to(self.device)
        endpoint_embedding = self.endpoint_embedding(tgt_endpoints)

        if self.frozen_encoder:
            self.compound_encoder.eval()
            with torch.no_grad():
                _, _, _, _, graph_repr, attention_weights = self.compound_encoder(
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
            _, _, _, _, graph_repr, attention_weights = self.compound_encoder(
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

        if inference:
            n = graph_repr.shape[0]
            graph_repr = graph_repr.repeat_interleave(tgt_endpoints.shape[0], dim=0).to(self.device)
            endpoint_embedding = endpoint_embedding.tile(n, 1, 1).to(self.device)

        graph_repr = self.graph_feat_norm(graph_repr).unsqueeze(1)

        _graph_repr, _ = self.attention(graph_repr, endpoint_embedding, endpoint_embedding)
        graph_repr = self.endpoint_attn_layer_norm(graph_repr + self.dropout(_graph_repr)).squeeze(1)

        pred = self.downstream_layer(graph_repr)
        proba = self.endpoint_layer(graph_repr)

        if inference:
            if self.task_type == 'classification':
                pred = self.softmax(pred)
            proba = self.softmax(proba)

        return pred, proba, attention_weights

    def compute_loss(self, pred_value, target_value, endpoint_proba, endpoints, endpoint_weight=0.8, gamma=1):
        loss_dict = {}

        value_loss = self.loss_func(pred_value, target_value) if self.task_type == "regression" \
            else self.loss_func(pred_value, target_value.squeeze(1).to(torch.int64))

        endpoint_loss = self.endpoint_loss_func(endpoint_proba, endpoints.squeeze(1).to(torch.int64))
        loss = value_loss + endpoint_loss * endpoint_weight * gamma

        loss_dict['value_loss'] = value_loss.detach().item()
        loss_dict['endpoint_loss'] = endpoint_loss.detach().item()
        loss_dict['loss'] = loss.detach().item()

        return loss, loss_dict


class DownstreamTransformerModel(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel,
            hidden_size, dropout_rate, n_layers,
            endpoints, frozen_encoder, device,
            task_type='regression', n_labels=2,
            use_auxiliary=False, auxiliary_centers=None,
            auxiliary_encoder_mode="mlp",
            auxiliary_aggregate_mode="concat",
            auxiliary_fusion_mode="residual_gate"
    ):
        super(DownstreamTransformerModel, self).__init__()

        assert task_type in ['regression', 'classification']
        if task_type == "classification":
            assert n_labels >= 2

        assert not use_auxiliary or auxiliary_centers is not None
        self.use_auxiliary = use_auxiliary
        assert auxiliary_encoder_mode in ["rbf", "mlp"]
        assert auxiliary_aggregate_mode in ["sum", "concat"]
        assert auxiliary_fusion_mode in ["bidirectional_gate", "residual_gate"]
        self.auxiliary_encoder_mode = auxiliary_encoder_mode
        self.auxiliary_aggregate_mode = auxiliary_aggregate_mode
        self.auxiliary_fusion_mode = auxiliary_fusion_mode

        self.task_type = task_type
        self.n_labels = n_labels

        self.endpoints = endpoints
        self.frozen_encoder = frozen_encoder
        self.latent_size = compound_encoder.latent_size
        self.num_tasks = len(endpoints)

        self.endpoint_embedding = nn.Embedding(self.num_tasks, self.latent_size)

        self.attention = MultiHeadAttentionLayer(
            hid_dim=compound_encoder.latent_size,
            n_heads=8,
            dropout=dropout_rate,
            device=device
        )

        self.downstream_expert_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_size, self.latent_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_size * 2, self.latent_size * 2)
            ) for _ in range(self.num_tasks)
        ])

        self.downstream_shared_layer = MLP(
            input_size=self.latent_size * 2,
            output_sizes=[hidden_size] * n_layers + [1 if self.task_type == 'regression' else n_labels],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )

        self.endpoint_specific_layer = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_size // 2, self.latent_size // 2)
        )

        self.endpoint_layer = MLP(
            input_size=compound_encoder.latent_size // 2,
            output_sizes=[hidden_size] * 2 + [len(self.endpoints)],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )

        if self.use_auxiliary:
            self.num_auxiliary = len(auxiliary_centers)
            self.auxiliary_layers = nn.ModuleList([
                AuxiliaryFloatRBF(
                    c=auxiliary_center,
                    latent_size=self.latent_size,
                    device=device,
                    encoder_mode=self.auxiliary_encoder_mode
                ) for auxiliary_center in auxiliary_centers
            ])
            if self.auxiliary_aggregate_mode == "concat":
                self.aux_proj = nn.Sequential(
                    nn.Linear(self.latent_size * self.num_auxiliary, self.latent_size),
                    nn.ReLU(),
                    nn.LayerNorm(self.latent_size)
                )
            if self.auxiliary_fusion_mode == "bidirectional_gate":
                self.fusion_layer = BidirectionalGatedFusion(feature_dim=self.latent_size)
            else:
                self.aux_to_graph = nn.Sequential(
                    nn.Linear(self.latent_size, self.latent_size),
                    nn.ReLU(),
                    nn.LayerNorm(self.latent_size)
                )
                self.fusion_gate = nn.Sequential(
                    nn.Linear(self.latent_size * 2, self.latent_size),
                    nn.Sigmoid()
                )
                self.aux_residual_alpha = nn.Parameter(torch.tensor(0.0))
                self._init_residual_gate_for_safe_start()

        self.compound_encoder = compound_encoder

        self.graph_feat_norm = nn.LayerNorm(self.latent_size)
        self.endpoint_attn_layer_norm = nn.LayerNorm(self.latent_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.device = device

        self.loss_func = nn.MSELoss(reduction="none") if self.task_type == 'regression' \
            else nn.CrossEntropyLoss(reduction="none")
        self.endpoint_loss_func = nn.CrossEntropyLoss(reduction="none")

        self.value_log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        self.endpoint_log_vars = nn.Parameter(torch.zeros(self.num_tasks))

        self.softmax = nn.Softmax(dim=1)

    def _init_residual_gate_for_safe_start(self):
        aux_linear = self.aux_to_graph[0]
        nn.init.zeros_(aux_linear.weight)
        nn.init.zeros_(aux_linear.bias)

        gate_linear = self.fusion_gate[0]
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, -8.0)

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            tgt_endpoints, inference=False, return_attention_weights=False,
            auxiliaries=None
    ):

        assert not self.use_auxiliary or auxiliaries is not None

        endpoint_embedding = self.endpoint_embedding(tgt_endpoints)
        attn_flag = self.compound_encoder.use_attention and return_attention_weights

        if self.frozen_encoder:
            self.compound_encoder.eval()
            with torch.no_grad():
                _, _, _, _, graph_repr, attention_weights = self.compound_encoder(
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
                    masked_dihedral_indices=None,
                    return_attention_weights=attn_flag
                )
        else:
            _, _, _, _, graph_repr, attention_weights = self.compound_encoder(
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
                masked_dihedral_indices=None,
                return_attention_weights=attn_flag
            )

        if inference:
            n = graph_repr.shape[0]
            graph_repr = graph_repr.repeat_interleave(tgt_endpoints.shape[0], dim=0).to(self.device)
            auxiliaries = auxiliaries.repeat_interleave(tgt_endpoints.shape[0], dim=1).to(self.device)
            endpoint_embedding = endpoint_embedding.tile(n, 1, 1).to(self.device)
            tgt_endpoints = tgt_endpoints.tile(n, 1).to(self.device)

        graph_repr = self.graph_feat_norm(graph_repr).unsqueeze(1)

        _graph_repr, _ = self.attention(graph_repr, endpoint_embedding, endpoint_embedding)
        graph_repr = self.endpoint_attn_layer_norm(graph_repr + self.dropout(_graph_repr)).squeeze(1)

        endpoint_repr = self.endpoint_specific_layer(graph_repr)
        endpoint_proba = self.endpoint_layer(endpoint_repr)

        if self.use_auxiliary:
            if self.auxiliary_aggregate_mode == "sum":
                aux_repr = 0
                for i, aux in enumerate(auxiliaries):
                    aux_repr = aux_repr + self.auxiliary_layers[i](aux)
            else:
                aux_repr_list = []
                for i, aux in enumerate(auxiliaries):
                    aux_repr_list.append(self.auxiliary_layers[i](aux))
                aux_repr = torch.cat(aux_repr_list, dim=-1)
                aux_repr = self.aux_proj(aux_repr)

            if self.auxiliary_fusion_mode == "bidirectional_gate":
                graph_repr = self.fusion_layer(graph_repr, aux_repr)
            else:
                aux_delta = self.aux_to_graph(aux_repr)
                gate_input = torch.cat([graph_repr, aux_repr], dim=-1)
                gate = self.fusion_gate(gate_input)
                graph_repr = graph_repr + self.aux_residual_alpha * gate * aux_delta

        pred_value = []
        for i, _repr in enumerate(graph_repr):
            expert_layer = self.downstream_expert_layers[tgt_endpoints[i]]
            pred = self.downstream_shared_layer(expert_layer(_repr))
            pred_value.append(pred)
        pred_value = torch.concat(pred_value).unsqueeze(dim=1)

        if inference:
            if self.task_type == 'classification':
                pred_value = self.softmax(pred_value)

        return pred_value, endpoint_proba, attention_weights

    @staticmethod
    def uncertainty_weighted_loss(losses, endpoints, log_vars):
        total_loss = 0
        for i, loss in enumerate(losses):
            # 不确定性加权: 1/(2*σ^2) * L + log(σ)
            j = endpoints[i]
            precision = torch.exp(-log_vars[j])
            total_loss += precision * loss + 0.5 * log_vars[j]

        return total_loss

    def compute_loss(self, pred_value, target_value, endpoint_proba, endpoints, endpoint_weight=0.8, gamma=1):
        loss_dict = {}

        if self.task_type == "regression":
            _value_loss = self.loss_func(pred_value, target_value)
        else:
            _value_loss = self.loss_func(pred_value, target_value.squeeze(1).to(torch.int64))
        value_loss = self.uncertainty_weighted_loss(_value_loss, endpoints, self.value_log_vars)

        _endpoint_loss = self.endpoint_loss_func(endpoint_proba, endpoints.squeeze(1).to(torch.int64))
        endpoint_loss = self.uncertainty_weighted_loss(_endpoint_loss, endpoints, self.endpoint_log_vars)

        loss = value_loss + endpoint_loss * endpoint_weight * gamma
        loss_dict['value_loss'] = value_loss.detach().item()
        loss_dict['endpoint_loss'] = endpoint_loss.detach().item()
        loss_dict['loss'] = loss.detach().item()

        return loss, loss_dict
