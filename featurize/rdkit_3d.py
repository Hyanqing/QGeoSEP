from rdkit.Chem import Descriptors3D
from datasets import MoleculePositionToolKit


def obtain_rdkit3d_desc(mol):
    mol, mmff_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=5)
    desc = Descriptors3D.CalcMolDescriptors3D(mol)
    return list(desc.values())


if __name__ == "__main__":
    from rdkit import  Chem
    smis = ["CCCC", "CCCCC", "CCCCCC", "CC(N)C(=O)O", "CC(N)C(=O)[O-].[Na+]"]
    for smi in smis:
        _mol = Chem.MolFromSmiles(smi)
        obtain_rdkit3d_desc(_mol)
