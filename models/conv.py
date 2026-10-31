import torch
from torch import nn
import math
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_max, scatter_add


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(dropout))
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class AdvancedMLP(nn.Module):
    def __init__(
            self,
            input_size,
            output_sizes,
            use_layer_norm=False,
            activation=nn.ReLU,
            dropout=0.0,
            layernorm_before=False,
            use_bn=False,
            residual_connection=False,
    ):
        super().__init__()
        module_list = []
        self.residual_connection = residual_connection

        if layernorm_before and use_layer_norm:
            module_list.append(nn.LayerNorm(input_size))

        for i, size in enumerate(output_sizes):
            module_list.append(nn.Linear(input_size, size))

            if i < len(output_sizes) - 1:
                if use_bn:
                    module_list.append(nn.BatchNorm1d(size))
                elif use_layer_norm and not layernorm_before:
                    module_list.append(nn.LayerNorm(size))

                module_list.append(activation())

                if dropout > 0:
                    module_list.append(nn.Dropout(dropout))

            input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.module_list):
            if (isinstance(layer, nn.Linear) and
                    i + 1 < len(self.module_list) and
                    self.residual_connection and
                    x.shape[-1] == layer.out_features):
                residual = x
                x = layer(x)
                if i > 0 and residual.shape == x.shape:
                    x = x + residual
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()


class EndpointMLP(nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes=None,
            activation=nn.ReLU,
            dropout=0.3,
            use_bn=True,
    ):
        super().__init__()
        module_list = []

        if hidden_sizes is None:
            hidden_sizes = [input_size // 2]

        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            module_list.append(nn.Linear(sizes[i], sizes[i + 1]))

            if i < len(sizes) - 2:
                if use_bn:
                    module_list.append(nn.BatchNorm1d(sizes[i + 1]))
                module_list.append(activation())
                if dropout > 0:
                    module_list.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()


class DropoutIfTraining(nn.Module):
    """
    Borrow this implementation from deepmind
    """

    def __init__(self, p=0.0, submodule=None):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.submodule = submodule if submodule else nn.Identity()

    def forward(self, x):
        x = self.submodule(x)
        newones = x.new_ones((x.size(0), 1))
        newones = F.dropout(newones, p=self.p, training=self.training)
        out = x * newones
        return out


class NodeAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0

        self.w1 = nn.Linear(3 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3 = nn.Linear(2 * emb_dim, emb_dim)
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, q, k_v, k_e, index, nnode, return_attn_weights=False):
        """
        q: [N, C]
        k: [N, 2*c]
        v: [N, 2*c]
        """
        x = torch.cat([q, k_v, k_e], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)

        attn_scores = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weights = softmax(attn_scores, index)  # [E, num_heads, 1]

        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weights * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)

        if return_attn_weights:
            return x, attn_weights.squeeze(-1)  # [E, num_heads]
        else:
            return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
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

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_dim, heads=8, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_features // heads

        # 线性变换
        self.W_node = nn.Linear(in_features, out_features, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_features, bias=False)

        # 注意力机制
        self.attention = nn.Linear(3 * self.head_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        num_nodes = x.size(0)

        # 变换节点和边特征
        h = self.W_node(x)  # [num_nodes, out_features]
        e = self.W_edge(edge_attr)  # [num_edges, out_features]

        # 重塑为多头
        h = h.view(-1, self.heads, self.head_dim)  # [num_nodes, heads, head_dim]

        # 获取源节点和目标节点
        src, dst = edge_index

        # 准备注意力计算
        h_src = h[src]  # [num_edges, heads, head_dim]
        h_dst = h[dst]  # [num_edges, heads, head_dim]
        e_transformed = e.view(-1, self.heads, self.head_dim)  # [num_edges, heads, head_dim]

        # 计算注意力分数
        attention_input = torch.cat([h_src, h_dst, e_transformed], dim=-1)  # [num_edges, heads, 3*head_dim]
        attention_scores = self.attention(attention_input).squeeze(-1)  # [num_edges, heads]
        attention_scores = self.leaky_relu(attention_scores)

        # 应用softmax
        max_score = scatter_max(attention_scores, dst, dim=0, dim_size=num_nodes)[0][dst]
        exp_scores = torch.exp(attention_scores - max_score)
        attention_weights = exp_scores / (scatter_add(exp_scores, dst, dim=0, dim_size=num_nodes)[dst] + 1e-16)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        out = h_src * attention_weights.unsqueeze(-1)  # [num_edges, heads, head_dim]

        # 聚合
        out = scatter_add(out, dst, dim=0, dim_size=num_nodes)  # [num_nodes, heads, head_dim]
        out = out.view(num_nodes, -1)  # [num_nodes, out_features]

        if return_attention:
            return out, attention_weights
        else:
            return out


class BidirectionalGatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        concat_feat = torch.cat([feat1, feat2], dim=-1)

        gate1_weights = self.gate1(concat_feat)  # feat1的重要性
        gate2_weights = self.gate2(concat_feat)  # feat2的重要性

        # 归一化
        total_weights = gate1_weights + gate2_weights + 1e-8
        gate1_weights = gate1_weights / total_weights
        gate2_weights = gate2_weights / total_weights

        fused_feature = gate1_weights * feat1 + gate2_weights * feat2
        return fused_feature
