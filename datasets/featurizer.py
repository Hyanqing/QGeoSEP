import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from datasets.utils import MoleculePositionToolKit

ALLOWABLE_FEATURES = {
    "atomic_num": list(range(1, 119)) + ["misc", "masked"],
    "chiral_tag": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "masked"
    ],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc", "masked"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc", "masked"],
    "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc", "masked"],
    "num_radical_e": [0, 1, 2, 3, 4, "misc", "masked"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc", "masked"],
    "is_aromatic": [False, True, "masked"],
    "is_in_ring": [False, True, "masked"],
    "bond_type": [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
        "ANGLE",
        "DIHEDRAL",
        "misc",
        "masked"
    ],
    "bond_stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
        "masked"
    ],
    "is_conjugated": [False, True, "masked"],
}


class MoleculeFeatureToolKit:
    @staticmethod
    def safe_index(array, target):
        try:
            return array.index(target)
        except ValueError:
            return len(array) - 1

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'is_in_ring':
            return int(atom.IsInRing())
        # elif name == 'valence_out_shell':
        #     return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))

        bond_lengths = np.array(bond_lengths, 'float32')
        return np.array(edges[:, 0]), np.array(edges[:, 1]), bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""

        # def _get_vec(atom_poses, edge):
        #     return atom_poses[edge[0]] - atom_poses[edge[1]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        atoms = []

        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]

            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue

                src_edge = edges[src_edge_i]
                temp = np.hstack([src_edge, tar_edge[1:]])
                if temp[0] == temp[-1]:
                    continue

                atoms.append(temp)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(
                    atom_poses[src_edge[0]] - atom_poses[src_edge[1]],
                    atom_poses[tar_edge[1]] - atom_poses[tar_edge[0]]
                )

                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        _atoms = np.array(atoms)

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
            node_i_indices = np.zeros([0, ], 'int64')
            node_j_indices = np.zeros([0, ], 'int64')
            node_k_indices = np.zeros([0, ], 'int64')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
            node_i_indices, node_j_indices, node_k_indices = _atoms.T

        return node_i_indices, node_j_indices, node_k_indices, super_edges, bond_angles, bond_angle_dirs

    @staticmethod
    def get_supersuperedge_dihedral(mol, superedges, edges):
        """get supersuperedge dihedral"""

        def _get_dihedral(mol, a, b, c, d):
            return rdMolTransforms.GetDihedralRad(mol.GetConformer(), a, b, c, d)

        E = len(superedges)
        superedge_indices = np.arange(E)
        supersuper_edges = []
        dihedral_angles = []
        atoms = []

        for tar_superedge_i in range(E):
            tar_superedge = superedges[tar_superedge_i]
            src_superedge_indices = superedge_indices[superedges[:, 1] == tar_superedge[0]]

            for src_superedge_i in src_superedge_indices:
                if src_superedge_i == tar_superedge_i:
                    continue

                src_superedge = superedges[src_superedge_i]
                if src_superedge[0] == tar_superedge[1]:
                    continue

                src_edges = [edges[src_superedge[0]], edges[src_superedge[1]]]
                tar_edges = [edges[tar_superedge[0]], edges[tar_superedge[1]]]

                atom_a = int(src_edges[0][0])
                atom_b = int(src_edges[0][1])
                atom_c = int(tar_edges[1][0])
                atom_d = int(tar_edges[1][1])
                if len({atom_a, atom_b, atom_c, atom_d}) != 4:
                    continue

                atoms.append([atom_a, atom_b, atom_c, atom_d])
                supersuper_edges.append([src_superedge_i, tar_superedge_i])
                try:
                    dihedral = _get_dihedral(mol, atom_a, atom_b, atom_c, atom_d)
                except Exception:
                    dihedral = 0
                dihedral_angles.append(dihedral)
                # bond_angle_dirs.append(src_superedge[1] == tar_edge[0])  # H -> H or H -> T

        _atoms = np.array(atoms)

        if len(supersuper_edges) == 0:
            supersuper_edges = np.zeros([0, 2], 'int64')
            dihedral_angles = np.zeros([0, ], 'float32')
            node_i_indices = np.zeros([0, ], 'int64')
            node_j_indices = np.zeros([0, ], 'int64')
            node_k_indices = np.zeros([0, ], 'int64')
            node_l_indices = np.zeros([0, ], 'int64')
        else:
            supersuper_edges = np.array(supersuper_edges, 'int64')
            dihedral_angles = np.array(dihedral_angles, 'float32')
            node_i_indices, node_j_indices, node_k_indices, node_l_indices = _atoms.T

        dihedral_angles = np.nan_to_num(dihedral_angles)
        return node_i_indices, node_j_indices, node_k_indices, node_l_indices, supersuper_edges, dihedral_angles

    @staticmethod
    def get_bond_borders(edges, wiberg):
        bond_orders = []
        for src_node_i, tar_node_j in edges:
            bond_orders.append(wiberg[src_node_i, tar_node_j])
        bond_orders = np.array(bond_orders, 'float32')
        return bond_orders


def mol_to_graph_data(mol, atom_names, bond_names):
    if len(mol.GetAtoms()) == 0:
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(
            [
                MoleculeFeatureToolKit.safe_index(
                    ALLOWABLE_FEATURES[name], MoleculeFeatureToolKit.get_atom_value(atom, name)
                )
                for name in atom_names
            ]
        )
    x = np.array(atom_features_list, dtype=np.int64)

    # edges
    num_bond_features = len(bond_names)
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                MoleculeFeatureToolKit.safe_index(
                    ALLOWABLE_FEATURES[name], MoleculeFeatureToolKit.get_bond_value(bond, name)
                )
                for name in bond_names
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64)
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:
        edge_index = np.empty((0, 2), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    # super_edges
    # edge_num = edge_index.shape[0]
    # edge_index_idxs = np.arange(edge_num)
    # super_edges = []
    # super_edges_idxs = []
    #
    # for tgt_i in range(edge_num):
    #     tgt_edge = edge_index[:, tgt_i]
    #     src_edge_idxs = edge_index_idxs[edge_index[1] == tgt_edge[0]]
    #     for src_i in src_edge_idxs:
    #         if src_i == tgt_i:
    #             continue
    #         if edge_index[0, src_i] == edge_index[1, tgt_i]:
    #             continue
    #         src_edge = edge_index[:, src_i]
    #         super_edges.append([
    #             src_edge[0].item(), src_edge[1].item(),
    #             tgt_edge[0].item(), tgt_edge[1].item()
    #         ])
    #         super_edges_idxs.append([src_i, tgt_i])
    #
    # super_edges = np.array(super_edges, dtype=np.int64)
    # super_edges_idxs = np.array(super_edges_idxs, dtype=np.int64)

    # # supersuper_edges
    # super_edges_num = len(super_edges_idxs)
    # super_edges_index_idxs = np.arange(super_edges_num)
    # supersuper_edges = []
    # supersuper_edges_idxs = []
    # for tgt_i in range(super_edges_num):
    #     tgt_super_edge = super_edges_idxs[tgt_i]
    #     src_super_edge_idxs = super_edges_index_idxs[super_edges_idxs[:, 1] == tgt_super_edge[0]]
    #     for src_i in src_super_edge_idxs:
    #         if src_i == tgt_i:
    #             continue
    #         if super_edges_idxs[src_i, 0] == super_edges_idxs[tgt_i, 1]:
    #             continue
    #         src_super_edge = super_edges_idxs[src_i]
    #         supersuper_edges.append([
    #             edge_index[0, src_super_edge[0]].item(), edge_index[1, src_super_edge[0]].item(),
    #             edge_index[0, tgt_super_edge[1]].item(), edge_index[1, tgt_super_edge[1]].item(),
    #         ])
    #         supersuper_edges_idxs.append([src_i, tgt_i])
    #
    # supersuper_edges = np.array(supersuper_edges, dtype=np.int64)
    # supersuper_edges_idxs = np.array(supersuper_edges_idxs, dtype=np.int64)
    #
    graph = dict()
    graph["edges"] = edge_index
    graph["edge_attr"] = edge_attr
    graph["node_feat"] = x
    # graph["super_edge_index"] = super_edges.T
    # graph["supersuper_edge_index"] = supersuper_edges.T
    # graph["angle_index"] = super_edges_idxs.T
    # graph["dihedral_index"] = supersuper_edges_idxs.T
    #
    graph["num_nodes"] = len(x)
    graph["num_edges"] = len(edge_attr)
    # graph["n_nodes"] = len(node_feat)
    # graph["n_edges"] = len(edge_attr)
    # graph["n_angles"] = len(super_edges)
    # graph["n_dihedrals"] = len(supersuper_edges)
    return graph


def mol_to_egeognn_graph_data(mol, atom_names, bond_names, atom_poses):
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol, atom_names, bond_names)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['Bl_node_i'], data['Bl_node_j'], data['bond_length'] = \
        MoleculeFeatureToolKit.get_bond_lengths(data['edges'], data['atom_pos'])

    node_i_indices, node_j_indices, node_k_indices, BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
        MoleculeFeatureToolKit.get_superedge_angles(data['edges'], data['atom_pos'])

    # data['Ba_node_i'] = node_i_indices
    # data['Ba_node_j'] = node_j_indices
    # data['Ba_node_k'] = node_k_indices
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['BondAngleGraph_edge_attr'] = np.array(bond_angles, 'float32')
    data['bond_angle'] = np.array(bond_angles, 'float32')
    data["num_angles"] = len(bond_angles)

    node_i_indices, node_j_indices, node_k_indices, node_l_indices, AngleDihedralGraph_edges, dihedral_angles = \
        MoleculeFeatureToolKit.get_supersuperedge_dihedral(mol, BondAngleGraph_edges, data["edges"])
    # data['Da_node_i'] = node_i_indices
    # data['Da_node_j'] = node_j_indices
    # data['Da_node_k'] = node_k_indices
    # data['Da_node_l'] = node_l_indices
    data['AngleDihedralGraph_edges'] = AngleDihedralGraph_edges
    data['AngleDihedralGraph_edge_attr'] = np.array(dihedral_angles, 'float32')
    data['dihedral_angle'] = np.array(dihedral_angles, 'float32')
    # data['num_dihedral'] = len(dihedral_angles)

    props = mol.GetPropsAsDict()
    if 'cm5' in props:
        data['atom_cm5'] = np.array(json.loads(props['cm5'])).astype('float32')

    if 'espc' in props:
        data['atom_espc'] = np.array(json.loads(props['espc'])).astype('float32')

    if 'cm5' in props:
        data['atom_hirshfeld'] = np.array(json.loads(props['hirshfeld'])).astype('float32')

    if 'cm5' in props:
        data['atom_npa'] = np.array(json.loads(props['npa'])).astype('float32')

    if 'wiberg' in props:
        wiberg = np.array(json.loads(props['wiberg'])).astype('float32')
        data['bond_order'] = MoleculeFeatureToolKit.get_bond_borders(data['edges'], wiberg)

    return data


def mask_egeognn_graph(graph, mask_ratio):
    masked_atom_indices = []
    masked_bond_indices = []
    masked_angle_indices = []
    masked_dihedral_indices = []

    n_atoms = graph["num_nodes"]
    edges = graph["edges"]
    BondAngleGraph_edges = graph["BondAngleGraph_edges"]
    AngleDihedralGraph_edges = graph["AngleDihedralGraph_edges"]

    masked_size = max(1, int(n_atoms * mask_ratio))  # at least 1 atom will be selected.

    # mask atoms and edges
    full_atom_indices = np.arange(n_atoms)
    full_bond_indices = np.arange(len(edges))

    target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)

    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[edges[:, 1] == atom_index]

        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        nei_atom_indices = edges[left_nei_bond_indices, 1]

        masked_atom_indices.append([atom_index])
        masked_atom_indices.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    masked_atom_indices = np.concatenate(masked_atom_indices, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)

    # mask angles
    full_angle_indices = np.arange(graph['BondAngleGraph_edges'].shape[0])
    for bond_index in masked_bond_indices:
        left_indices = full_angle_indices[BondAngleGraph_edges[:, 0] == bond_index]
        right_indices = full_angle_indices[BondAngleGraph_edges[:, 1] == bond_index]
        masked_angle_indices.append(np.append(left_indices, right_indices))

    if len(masked_angle_indices) != 0:
        masked_angle_indices = np.concatenate(masked_angle_indices, 0)

    # mask dihedral angles
    full_dihedral_indices = np.arange(graph['AngleDihedralGraph_edges'].shape[0])
    for angle_index in masked_angle_indices:
        left_indices = full_dihedral_indices[AngleDihedralGraph_edges[:, 0] == angle_index]
        right_indices = full_dihedral_indices[AngleDihedralGraph_edges[:, 1] == angle_index]
        masked_dihedral_indices.append(np.append(left_indices, right_indices))

    if len(masked_dihedral_indices) != 0:
        masked_dihedral_indices = np.concatenate(masked_dihedral_indices, 0)

    graph['masked_atom_indices'] = masked_atom_indices
    graph['masked_bond_indices'] = masked_bond_indices
    graph['masked_angle_indices'] = masked_angle_indices
    graph['masked_dihedral_indices'] = masked_dihedral_indices

    return graph


def get_feature_dims(atom_names):
    return list(map(len, [ALLOWABLE_FEATURES[name] for name in atom_names]))


if __name__ == "__main__":
    import json

    with open("../configs/input_feats.json") as f:
        configs = json.load(f)

    m = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    pos = MoleculePositionToolKit.get_2d_atom_poses(m)

    mol_to_egeognn_graph_data(m, configs["atom_names"], configs["bond_names"], atom_poses=pos)
    print("DONE!")
