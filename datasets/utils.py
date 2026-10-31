import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Dataset


class MoleculePositionToolKit:
    @staticmethod
    def get_atom_poses(mol, conf_idx=0):
        """tbd"""
        return mol.GetConformer(int(conf_idx)).GetPositions()

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=10, numThreads=1, seed=-1):
        """the atoms of mol will be changed in some cases."""
        if os.environ.get('QGEM_FORCE_FIELD', 'mmff') == 'uff':
            # Explicit alternative: retain 3D optimization; never substitute 2D
            # coordinates or accept an unconverged UFF conformer in this mode.
            new_mol = Chem.AddHs(mol)
            if not AllChem.UFFHasAllMoleculeParams(new_mol):
                raise ValueError('UFF parameters are unavailable for this molecule')
            conformer_ids = list(AllChem.EmbedMultipleConfs(
                new_mol, numConfs=numConfs, numThreads=numThreads, randomSeed=seed))
            if not conformer_ids:
                raise ValueError('3D embedding failed before UFF optimization')
            results = AllChem.UFFOptimizeMoleculeConfs(new_mol, numThreads=1, maxIters=1000)
            converged = [(energy, cid) for cid, (status, energy) in zip(conformer_ids, results)
                         if status == 0 and np.isfinite(energy)]
            if not converged:
                raise ValueError('No UFF conformer converged within 1000 iterations')
            _, conformer_id = min(converged)
            new_mol = Chem.RemoveHs(new_mol)
            return new_mol, MoleculePositionToolKit.get_atom_poses(new_mol, conf_idx=conformer_id)
        new_mol = Chem.AddHs(mol)
        if not AllChem.MMFFHasAllMoleculeParams(new_mol):
            raise ValueError('MMFF parameters are unavailable for this molecule')
        conformer_ids = list(AllChem.EmbedMultipleConfs(
            new_mol, numConfs=numConfs, numThreads=numThreads, randomSeed=seed))
        if not conformer_ids:
            raise ValueError('3D embedding failed before MMFF optimization')
        results = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        converged = [(energy, cid) for cid, (status, energy) in zip(conformer_ids, results)
                     if status == 0 and np.isfinite(energy)]
        if not converged:
            raise ValueError('No MMFF conformer converged')
        _, index = min(converged)
        new_mol = Chem.RemoveHs(new_mol)
        return new_mol, MoleculePositionToolKit.get_atom_poses(new_mol, conf_idx=index)

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        return MoleculePositionToolKit.get_atom_poses(mol)


class GraphDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(GraphDataset, self).__init__(transform=transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    smiles = "NS(=O)(=O)c1cc2c(cc1Cl)NC(CSCc1ccccc1)=NS2(=O)=O"
    mol = Chem.MolFromSmiles(smiles)

    mmff_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=5)
    poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

    print(mmff_poses, poses)
