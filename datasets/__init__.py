import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from sklearn.metrics import pairwise_distances
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from datasets.featurizer import mol_to_egeognn_graph_data, mask_egeognn_graph
from datasets.utils import MoleculePositionToolKit
from utils import calculate_oxygen_balance_and_nitrogen_content


class EgeognnPretrainedDataset(TorchDataset):
    def __init__(
            self,
            base_path, dataset_name,
            atom_names, bond_names,
            with_provided_3d, mask_ratio,
            remove_hs=False, use_mpi=False,
            force_generate=False
    ):
        self.loaded_batches = []
        self.smiles_list = []
        self.sdf_files = []

        if remove_hs:
            self.folder = os.path.join(base_path, f"egeognn_{dataset_name}_rh")
        else:
            self.folder = os.path.join(base_path, f"egeognn_{dataset_name}")

        self.dataset_name = dataset_name
        self.remove_hs = remove_hs
        self.base_path = base_path
        self.with_provided_3d = with_provided_3d

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.mask_ratio = mask_ratio

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

        processed_files = os.listdir(self.processed_dir)
        if (not force_generate) and processed_files and processed_files[0].endswith(".pt"):
            print(f'Loading data from {self.processed_dir}')
            for file in tqdm(processed_files):
                batch_file_path = os.path.join(self.processed_dir, file)
                loaded_data = torch.load(batch_file_path)
                self.loaded_batches.extend(loaded_data)
            return

        if not with_provided_3d:
            input_file = os.path.join(self.base_path, f"{self.dataset_name}.smi")
            with open(input_file) as f:
                smiles_list = f.readlines()
            random.shuffle(smiles_list)
            self.smiles_list = smiles_list
        else:
            self.sdf_files = list(Path(base_path).glob("*.sdf"))

        self.use_mpi = use_mpi
        # for fucking bug of nscc-tj
        self.queue = None
        if self.use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            self.queue = np.zeros(size)
        self.loaded_batches = self.process()

        super().__init__()

    def __len__(self):
        return len(self.loaded_batches)

    def __getitem__(self, item):
        return self.loaded_batches[item]

    def load_smiles(self, smiles):
        if "." in smiles:
            return
        tmp_mol = Chem.MolFromSmiles(smiles.strip())
        try:
            mol = RemoveHs(tmp_mol) if self.remove_hs else tmp_mol
            if mol.GetNumBonds() < 1:
                return
            return mol
        except Exception:
            return

    def load_smiles_list(self, smiles_list):
        print("Convert Smiles to molecule")
        mol_list = []
        with ThreadPoolExecutor() as executor:
            for result in tqdm(executor.map(self.load_smiles, smiles_list), total=len(smiles_list)):
                mol_list.append(result)
        return mol_list

    def load_sdf_file(self, sdf_file):
        tmp_mol = Chem.SDMolSupplier(sdf_file)[0]
        try:
            mol = RemoveHs(tmp_mol) if self.remove_hs else tmp_mol
            if mol.GetNumBonds() < 1:
                return
            return mol
        except Exception:
            return

    def load_sdf_file_list(self, sdf_file_list):
        print("Convert SDF file to molecule")
        mol_list = []
        with ThreadPoolExecutor() as executor:
            for result in tqdm(executor.map(self.load_sdf_file, sdf_file_list), total=len(sdf_file_list)):
                mol_list.append(result)
        return mol_list

    def process(self):
        print("Converting pickle files into graphs...")
        return self.process_egeognn()

    def get_idx_split(self):
        path = os.path.join(self.root, "split")
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

    def prepare_pretrain_task(self, graph):
        """
        prepare graph for pretrain task
        """
        n = len(graph['atom_pos'])
        dist_matrix = pairwise_distances(graph['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        graph['Ad_node_i'] = indice.reshape([-1, 1])
        graph['Ad_node_j'] = indice.T.reshape([-1, 1])
        graph['atom_distance'] = dist_matrix.reshape([-1, 1])

        graph = mask_egeognn_graph(graph, mask_ratio=self.mask_ratio)

        graph['atom_cm5'] = np.array(graph.get('atom_cm5', []))
        graph['atom_espc'] = np.array(graph.get('atom_espc', []))
        graph['atom_hirshfeld'] = np.array(graph.get('atom_hirshfeld', []))
        graph['atom_npa'] = np.array(graph.get('atom_npa', []))
        graph['bond_order'] = np.array(graph.get('bond_order', []))

        return graph

    def mol_to_egeognn_graph_data_raw3d(self, mol):
        atom_poses = MoleculePositionToolKit.get_atom_poses(mol)
        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def mol_to_egeognn_graph_data_MMFF3d(self, mol):
        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=3, numThreads=1)
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def process_molecule(self, mol):

        try:
            if self.with_provided_3d:
                graph = self.mol_to_egeognn_graph_data_raw3d(mol)
            else:
                graph = self.mol_to_egeognn_graph_data_MMFF3d(mol)

            graph = self.prepare_pretrain_task(graph)

            data = CustomData()
            data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
            data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
            data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

            data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)

            data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)
            data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)
            data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

            data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
            data.n_atoms = graph["num_nodes"]
            data.n_bonds = graph["num_edges"]
            data.n_angles = graph["num_angles"]

            data.masked_atom_indices = torch.from_numpy(graph["masked_atom_indices"]).to(torch.int64)
            data.masked_bond_indices = torch.from_numpy(graph["masked_bond_indices"]).to(torch.int64)
            data.masked_angle_indices = torch.from_numpy(graph["masked_angle_indices"]).to(torch.int64)
            data.masked_dihedral_indices = torch.from_numpy(graph["masked_dihedral_indices"]).to(torch.int64)

            data.cm5_charges = torch.from_numpy(graph['atom_cm5']).to(torch.float32)
            data.espc_charges = torch.from_numpy(graph['atom_espc']).to(torch.float32)
            data.hirshfeld_charges = torch.from_numpy(graph['atom_hirshfeld']).to(torch.float32)
            data.npa_charges = torch.from_numpy(graph['atom_npa']).to(torch.float32)
            data.bond_orders = torch.from_numpy(graph['bond_order']).to(torch.float32)

            data.smiles = Chem.MolToSmiles(mol)

            return data

        except Exception:
            return None

    def process_molecules(self, molecules):
        try:
            results = []
            with ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(self.process_molecule, molecules), total=len(molecules)):
                    if result is None:
                        print(f"Failed to process a molecule.")
                    results.append(result)
            return results
        except:
            return []

    @property
    def processed_dir(self):
        return os.path.join(self.folder, 'processed')

    def save_results(self, final_results, batch_size, rank):
        if len(final_results) == 0:
            return

        batch_data = []
        print('saving results...')

        for idx, data in enumerate(final_results):
            batch_data.append(data)
            if len(batch_data) >= batch_size:
                batch_file_path = os.path.join(self.processed_dir, f'batch_{idx // batch_size}_{rank}.pt')
                torch.save(batch_data, batch_file_path)
                batch_data = []

        if batch_data:
            batch_file_path = os.path.join(self.processed_dir, f'batch_{len(final_results) // batch_size}_{rank}.pt')
            torch.save(batch_data, batch_file_path)

    def process_egeognn(self):
        if self.use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            chunk_size = (len(self.smiles_list) or len(self.sdf_files)) // size

            start_index = rank * chunk_size
            end_index = (rank + 1) * chunk_size if rank != size - 1 else (len(self.smiles_list) or len(self.sdf_files))

            if not self.with_provided_3d:
                local_molecules = self.load_smiles_list(self.smiles_list[start_index:end_index])
            else:
                local_molecules = self.load_sdf_file_list(self.sdf_files[start_index:end_index])

            local_molecules = [mol for mol in local_molecules if mol is not None]
            results = self.process_molecules(local_molecules)
            # all_results = comm.gather(results, root=0)

            final_results = [item for item in results if item is not None]

            self.save_results(final_results, batch_size=81920, rank=rank)
            print(f"NODE {rank} done!")

            self.queue[rank] = 1
            for node in range(size):
                if node != rank:
                    comm.isend(rank, dest=node, tag=11)

            print(f"NODE {rank}: waiting for other node...")
            while self.queue.sum() != size:
                for node in range(size):
                    if node != rank:
                        tgt = comm.recv(source=node, tag=11)
                        self.queue[tgt] = 1
                continue

            return []

        if not self.with_provided_3d:
            local_molecules = self.load_smiles_list(self.smiles_list)
        else:
            local_molecules = self.load_sdf_file_list(self.sdf_files)

        local_molecules = [mol for mol in local_molecules if mol is not None]
        results = self.process_molecules(local_molecules)

        final_results = [item for item in results if item is not None]
        self.save_results(final_results, batch_size=8192, rank=0)
        print("done")
        return final_results


class EgeognnFinetuneDataset(TorchDataset):
    def __init__(
            self, base_path, endpoints,
            atom_names, bond_names,
            remove_hs=False, dev=False,
            use_mpi=False, force_generate=False,
            preprocess_endpoints=None, task_type='regression',
            auxiliary_items=None, use_auxiliary=False
    ):
        assert task_type in ['regression', 'classification']
        if use_auxiliary:
            assert auxiliary_items is not None
        else:
            auxiliary_items = []

        self.task_type = task_type

        if preprocess_endpoints is None:
            preprocess_endpoints = []

        self.loaded_batches = []

        assert dev in [True, False]

        if remove_hs:
            self.folder = os.path.join(base_path, f"egeognn_downstream_finetune_rh")
        else:
            self.folder = os.path.join(base_path, f"egeognn_downstream_finetune")

        self.remove_hs = remove_hs
        self.base_path = base_path

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.dev = dev
        self.endpoints = endpoints
        self.use_mpi = use_mpi

        self.mol_list = []
        self.label_list = []
        self.endpoint_list = []
        self.auxiliary_list = []

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

        processed_files = os.listdir(self.processed_dir)
        if (not force_generate) and processed_files and processed_files[0].endswith(".pt"):
            print(f'Loading data from {self.processed_dir}')
            for file in tqdm(processed_files):
                batch_file_path = os.path.join(self.processed_dir, file)
                loaded_data = torch.load(batch_file_path)
                self.loaded_batches.extend(loaded_data)

            random.shuffle(self.loaded_batches)
            return

        curr_index = 0
        for endpoint in self.endpoints:
            input_file = os.path.join(self.base_path, f"{endpoint}.csv")
            df = pd.read_csv(input_file, nrows=[None, 10][self.dev])

            if endpoint in preprocess_endpoints:
                df = df[df[endpoint] != 0].copy()
                df[f"log10_{endpoint}"] = np.log10(df[endpoint].values)
                endpoint = f"log10_{endpoint}"

            _smiles_list = df['smiles'].values
            _label_list = df[endpoint].values
            # _auxiliary_list = df.loc[:, auxiliary_items].values
            _mol_list = [Chem.MolFromSmiles(smi) for smi in _smiles_list]
            _auxiliary_list = np.array([calculate_oxygen_balance_and_nitrogen_content(mol, auxiliary_items) for mol in _mol_list])
            # self.auxiliary_items = len(calculate_oxygen_balance_and_nitrogen_content(_mol_list[0]))
            for i, smiles in enumerate(_smiles_list):
                # if "." in smiles:
                #     continue
                tmp_mol = Chem.MolFromSmiles(smiles.strip())

                try:
                    mol = RemoveHs(tmp_mol) if self.remove_hs else tmp_mol
                    if mol.GetNumBonds() < 1:
                        continue

                    mol.SetProp('idx', f"{curr_index}")
                    self.endpoint_list.append(endpoint)
                    self.mol_list.append(mol)

                    label = _label_list[i]
                    label_mean = _label_list.mean() if self.task_type == 'regression' else 0
                    label_std = _label_list.std() + 1e-5 if self.task_type == 'regression' else 1

                    self.auxiliary_list.append(_auxiliary_list[i])
                    self.label_list.append(
                        [
                            (label - label_mean) / label_std,
                            label_mean, label_std
                        ]
                    )
                    curr_index += 1

                except Exception:
                    continue

        self.label_list = np.array(self.label_list)

        # for fucking bug of nscc-tj
        self.queue = None
        if self.use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            self.queue = np.zeros(size)
        self.loaded_batches = self.process()

        super().__init__()

    def __len__(self):
        return len(self.loaded_batches)

    def __getitem__(self, item):
        return self.loaded_batches[item]

    def process(self):
        print("Converting pickle files into graphs...")
        return self.process_egeognn_finetune()

    @property
    def processed_dir(self):
        return os.path.join(self.folder, 'processed')

    def save_results(self, final_results, batch_size, rank):
        if len(final_results) == 0:
            return

        batch_data = []
        print('saving results...')

        for idx, data in enumerate(final_results):
            batch_data.append(data)
            if len(batch_data) >= batch_size:
                batch_file_path = os.path.join(self.processed_dir, f'batch_{idx // batch_size}_{rank}.pt')
                torch.save(batch_data, batch_file_path)
                batch_data = []

        if batch_data:
            batch_file_path = os.path.join(self.processed_dir, f'batch_{len(final_results) // batch_size}_{rank}.pt')
            torch.save(batch_data, batch_file_path)

    def mol_to_egeognn_graph_data_MMFF3d(self, mol, numConfs=3):
        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=numConfs, numThreads=0)
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def process_molecule(self, mol):
        try:
            graph = self.mol_to_egeognn_graph_data_MMFF3d(mol)

            data = CustomData()
            data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
            data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
            data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

            data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)
            data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)
            data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

            data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
            data.n_atoms = graph["num_nodes"]
            data.n_bonds = graph["num_edges"]
            data.n_angles = graph["num_angles"]

            idx = int(mol.GetProp("idx"))
            data.label = torch.from_numpy(self.label_list[idx]).to(torch.float32)
            data.auxiliary = torch.from_numpy(self.auxiliary_list[idx]).to(torch.float32)
            data.endpoint = self.endpoint_list[idx]
            data.smiles = Chem.MolToSmiles(mol)

            return data

        except Exception:
            return None

    def process_molecules(self, molecules):
        try:
            results = []
            with ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(self.process_molecule, molecules), total=len(molecules)):
                    if result is None:
                        print(f"Failed to process a molecule.")
                    results.append(result)
            return results
        except:
            return []

    def process_egeognn_finetune(self):
        if self.use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            chunk_size = len(self.mol_list) // size

            start_index = rank * chunk_size
            end_index = (rank + 1) * chunk_size if rank != size - 1 else len(self.mol_list)

            local_molecules = self.mol_list[start_index:end_index]
            local_molecules = [mol for mol in local_molecules if mol is not None]
            results = self.process_molecules(local_molecules)
            # all_results = comm.gather(results, root=0)

            final_results = [item for item in results if item is not None]

            self.save_results(final_results, batch_size=81920, rank=rank)
            print(f"NODE {rank} done!")

            self.queue[rank] = 1
            for node in range(size):
                if node != rank:
                    comm.isend(rank, dest=node, tag=11)

            print(f"NODE {rank}: waiting for other node...")
            while self.queue.sum() != size:
                for node in range(size):
                    if node != rank:
                        tgt = comm.recv(source=node, tag=11)
                        self.queue[tgt] = 1
                continue

            return []

        local_molecules = [mol for mol in self.mol_list if mol is not None]
        results = self.process_molecules(local_molecules)
        final_results = [item for item in results if item is not None]

        self.save_results(final_results, batch_size=81920, rank=0)
        print("done")
        return final_results


class EgeognnFinetuneMTDataset(TorchDataset):
    def __init__(
            self, base_path, endpoints,
            atom_names, bond_names,
            remove_hs=False, dev=False,
            use_mpi=False, force_generate=False,
            preprocess_endpoints=False
    ):
        self.preprocess_endpoints = preprocess_endpoints
        self.loaded_batches = []

        assert dev in [True, False]

        if remove_hs:
            self.folder = os.path.join(base_path, f"egeognn_downstream_finetune_rh")
        else:
            self.folder = os.path.join(base_path, f"egeognn_downstream_finetune")

        self.remove_hs = remove_hs
        self.base_path = base_path

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.dev = dev
        self.endpoints = endpoints
        self.use_mpi = use_mpi

        self.mol_list = []
        self.label_list = []
        self.mean_list = []
        self.std_list = []

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

        processed_files = os.listdir(self.processed_dir)
        if (not force_generate) and processed_files and processed_files[0].endswith(".pt"):
            print(f'Loading data from {self.processed_dir}')
            for file in tqdm(processed_files):
                batch_file_path = os.path.join(self.processed_dir, file)
                loaded_data = torch.load(batch_file_path)
                self.loaded_batches.extend(loaded_data)
            return

        curr_index = 0
        for input_file in Path(base_path).glob('*.csv'):
            df = pd.read_csv(input_file, nrows=[None, 100][self.dev])

            if self.preprocess_endpoints:
                for endpoint in self.endpoints:
                    df[f"log10_{endpoint}"] = np.log10(df[endpoint].values)
                self.endpoints = [f"log10_{endpoint}" for endpoint in self.endpoints]

            _smiles_list = df['smiles'].values
            _label_list = df.loc[:, self.endpoints].values

            for i, smiles in enumerate(_smiles_list):
                if "." in smiles:
                    continue
                tmp_mol = Chem.MolFromSmiles(smiles.strip())

                try:
                    mol = RemoveHs(tmp_mol) if self.remove_hs else tmp_mol
                    if mol.GetNumBonds() < 1:
                        continue

                    mol.SetProp('idx', f"{curr_index}")
                    # self.endpoint_list.append(endpoint)
                    self.mol_list.append(mol)

                    label = _label_list[i]
                    label_mean = np.nanmean(_label_list, axis=0)
                    label_std = np.nanstd(_label_list, axis=0) + 1e-5

                    self.label_list.append((label - label_mean) / label_std)
                    self.mean_list.append(label_mean)
                    self.std_list.append(label_std)

                    curr_index += 1

                except Exception:
                    continue

        self.label_list = np.array(self.label_list)
        self.mean_list = np.array(self.mean_list)
        self.std_list = np.array(self.std_list)

        # for fucking bug of nscc-tj
        self.queue = None
        if self.use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            self.queue = np.zeros(size)

        self.loaded_batches = self.process()

        super().__init__()

    def __len__(self):
        return len(self.loaded_batches)

    def __getitem__(self, item):
        return self.loaded_batches[item]

    def process(self):
        print("Converting pickle files into graphs...")
        return self.process_egeognn_finetune()

    @property
    def processed_dir(self):
        return os.path.join(self.folder, 'processed')

    def save_results(self, final_results, batch_size, rank):
        if len(final_results) == 0:
            return

        batch_data = []
        print('saving results...')

        for idx, data in enumerate(final_results):
            batch_data.append(data)
            if len(batch_data) >= batch_size:
                batch_file_path = os.path.join(self.processed_dir, f'batch_{idx // batch_size}_{rank}.pt')
                torch.save(batch_data, batch_file_path)
                batch_data = []

        if batch_data:
            batch_file_path = os.path.join(self.processed_dir, f'batch_{len(final_results) // batch_size}_{rank}.pt')
            torch.save(batch_data, batch_file_path)

    def mol_to_egeognn_graph_data_MMFF3d(self, mol, numConfs=3):
        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=numConfs, numThreads=0)
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def process_molecule(self, mol):
        try:
            graph = self.mol_to_egeognn_graph_data_MMFF3d(mol)

            data = CustomData()
            data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
            data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
            data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

            data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)
            data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)
            data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

            data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
            data.n_atoms = graph["num_nodes"]
            data.n_bonds = graph["num_edges"]
            data.n_angles = graph["num_angles"]

            idx = int(mol.GetProp("idx"))
            data.label = torch.from_numpy(self.label_list[idx]).to(torch.float32)
            data.label_mean = torch.from_numpy(self.mean_list[idx]).to(torch.float32)
            data.label_std = torch.from_numpy(self.std_list[idx]).to(torch.float32)
            data.smiles = Chem.MolToSmiles(mol)

            return data

        except Exception:
            return None

    def process_molecules(self, molecules):
        try:
            results = []
            with ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(self.process_molecule, molecules), total=len(molecules)):
                    if result is None:
                        print(f"Failed to process a molecule.")
                    results.append(result)
            return results
        except:
            return []

    def process_egeognn_finetune(self):
        if self.use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            chunk_size = len(self.mol_list) // size

            start_index = rank * chunk_size
            end_index = (rank + 1) * chunk_size if rank != size - 1 else len(self.mol_list)

            local_molecules = self.mol_list[start_index:end_index]
            local_molecules = [mol for mol in local_molecules if mol is not None]
            results = self.process_molecules(local_molecules)
            # all_results = comm.gather(results, root=0)

            final_results = [item for item in results if item is not None]

            self.save_results(final_results, batch_size=81920, rank=rank)
            print(f"NODE {rank} done!")

            self.queue[rank] = 1
            for node in range(size):
                if node != rank:
                    comm.isend(rank, dest=node, tag=11)

            print(f"NODE {rank}: waiting for other node...")
            while self.queue.sum() != size:
                for node in range(size):
                    if node != rank:
                        tgt = comm.recv(source=node, tag=11)
                        self.queue[tgt] = 1
                continue

            return []

        local_molecules = [mol for mol in self.mol_list if mol is not None]
        results = self.process_molecules(local_molecules)
        final_results = [item for item in results if item is not None]

        self.save_results(final_results, batch_size=8192, rank=0)
        print("done")
        return final_results


class EgeognnInferenceDataset(TorchDataset):
    def __init__(
            self,
            smiles_list,
            atom_names,
            bond_names,
            remove_hs=False,
            seed=-1
    ):

        self.seed = seed
        self.remove_hs = remove_hs

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.mol_list = []
        self.endpoint_list = []

        self.smiles_list = smiles_list
        self.mol_list = [self.load_smiles(smiles) for smiles in self.smiles_list]
        self.mol_list = [x for x in self.mol_list if x]

        self.data_list = self.process_egeognn_finetune()

        super().__init__()

    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, item):
        return self.data_list[item]

    def load_smiles(self, smiles):
        tmp_mol = Chem.MolFromSmiles(smiles.strip())
        try:
            mol = RemoveHs(tmp_mol) if self.remove_hs else tmp_mol
            return mol
        except Exception:
            return

    def mol_to_egeognn_graph_data_MMFF3d(self, mol, numConfs=3):
        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(
                mol, numConfs=numConfs, numThreads=0, seed=self.seed
            )
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def process_molecule(self, mol):
        try:
            graph = self.mol_to_egeognn_graph_data_MMFF3d(mol, numConfs=10)

            data = CustomData()
            data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
            data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
            data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

            data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)
            data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)
            data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

            data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
            data.n_atoms = graph["num_nodes"]
            data.n_bonds = graph["num_edges"]
            data.n_angles = graph["num_angles"]

            data.smiles = Chem.MolToSmiles(mol)

            return data

        except Exception:
            return None

    def process_egeognn_finetune(self):
        data_list = [self.process_molecule(mol) for mol in self.mol_list]
        return [x for x in data_list if x is not None]


class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask|edges)", key)):
            return -1
        return 0

    def __inc__(self, key: str, value, *args, **kwargs):
        if 'batch' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        elif 'AtomBondGraph_edges' in key or key == 'face':
            return self.num_nodes
        elif 'BondAngleGraph_edges' in key:
            return self.AtomBondGraph_edges.size(1)
        elif 'AngleDihedralGraph_edges' in key:
            return self.BondAngleGraph_edges.size(1)
        elif 'masked_atom_indices' in key:
            return self.num_nodes
        elif 'masked_bond_indices' in key:
            return self.num_edges
        elif 'masked_angle_indices' in key:
            return len(self.bond_angles)
        elif 'masked_dihedral_indices' in key:
            return len(self.dihedral_angles)
        else:
            return 0


if __name__ == "__main__":
    with open("../configs/config.json") as f:
        configs = json.load(f)

    # dataset = EgeognnPretrainedDataset(
    #     root='../data/demo',
    #     dataset_name='demo',
    #     remove_hs=True,
    #     with_provided_3d=False,
    #     base_path="../data/demo",
    #     atom_names=configs["atom_names"],
    #     bond_names=configs["bond_names"],
    #     mask_ratio=0.1
    # )

    # ENDPOINTS = [
    #     'Cat_Intravenous_LD50',
    #     'Cat_Oral_LD50',
    #     'Chicken_Oral_LD50',
    #     'Dog_Oral_LD50',
    #     'Duck_Oral_LD50',
    #     'Guineapig_Oral_LD50',
    #     'Mouse_Intramuscular_LD50',
    #     'Mouse_Intraperitoneal_LD50',
    #     'Mouse_Intravenous_LD50',
    #     'Mouse_Oral_LD50',
    #     'Mouse_Subcutaneous_LD50',
    #     'Rabbit_Intravenous_LD50',
    #     'Rabbit_Oral_LD50',
    #     'Rat_Inhalation_LC50',
    #     'Rat_Intraperitoneal_LD50',
    #     'Rat_Intravenous_LD50',
    #     'Rat_Oral_LD50',
    #     'Rat_Skin_LD50',
    #     'Rat_Subcutaneous_LD50'
    # ]
    #
    # dataset = EgeognnFinetuneDataset(
    #     root='../data/downstream/toxicity',
    #     base_path="../data/downstream/toxicity",
    #     atom_names=configs["atom_names"],
    #     bond_names=configs["bond_names"],
    #     endpoints=ENDPOINTS,
    #     remove_hs=True,
    #     dev=True
    # )

    __smiles_list = [
        "C1=CC=C(C=C1)CSCC2=NS(=O)(=O)C3=CC(=C(C=C3N2)Cl)S(=O)(=O)N",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1CCC[C@H]1COc1cccc(Cl)c1",
        "C1=CC=C(C(=C1)C(=O)OC2=CC=CC=C2C(=O)O)O",
        "COc1ccc2[nH]cc(C[C@H]3CCCN3C)c2c1"
    ]
    dataset = EgeognnInferenceDataset(
        atom_names=configs["atom_names"],
        bond_names=configs["bond_names"],
        remove_hs=True,
        smiles_list=__smiles_list
    )

    demo_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=10
    )

    pbar = tqdm(demo_loader, desc="Iteration")

    for step, batch in enumerate(pbar):
        ...
