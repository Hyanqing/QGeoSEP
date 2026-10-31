from functools import partial
from multiprocessing import Pool

import numpy as np

from featurize.cats import CATS2D
from featurize.charge import GetCharge
from featurize.estate import EStateFP
from featurize.geary import GetGearyAuto
from featurize.ifg import IFG
from featurize.moe import GetMOE
from featurize.pubchem import PubChem
from featurize.rdkit_3d import obtain_rdkit3d_desc


def calculate_pubchem(mols, n_jobs=1):
    # 881 bits
    n_jobs = n_jobs if n_jobs >= 1 else None

    fps = PubChem()
    pool = Pool(n_jobs)
    fps = pool.map_async(fps.CalculatePubChem, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_cats(mols, PathLength=10, scale=3, n_jobs=1):
    # 150 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    kwargs = {
        "PathLength": PathLength,
        "scale": scale
    }
    func = partial(CATS2D, **kwargs)
    pool = Pool(n_jobs)
    fps = pool.map_async(func, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_charges(mols, n_jobs):
    # 25 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    pool = Pool(n_jobs)
    fps = pool.map_async(GetCharge, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_estate(mols, n_jobs):
    # 79 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    fp = EStateFP(val=True)
    pool = Pool(n_jobs)
    fps = pool.map_async(fp.CalculateEState, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_geary(mols, n_jobs):
    # 32 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    pool = Pool(n_jobs)
    fps = pool.map_async(GetGearyAuto, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_ifg(mols, n_jobs):
    # 19 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    fp = IFG(n_jobs=n_jobs)
    fps = fp.CalculateIFG(mols)
    return fps


def calculate_moe(mols, n_jobs):
    # 59 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    pool = Pool(n_jobs)
    fps = pool.map_async(GetMOE, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_rdkit3d(mols, n_jobs):
    # 11 bits
    n_jobs = n_jobs if n_jobs >= 1 else None
    pool = Pool(n_jobs)
    fps = pool.map_async(obtain_rdkit3d_desc, mols).get()
    pool.close()
    pool.join()
    fps = np.array(fps)
    return fps


def calculate_descriptor(mols):
    """

    :rtype: object
    """
    pubchem_fp = calculate_pubchem(mols, n_jobs=1)
    cats_fp = calculate_cats(mols, n_jobs=1)
    charges_fp = calculate_charges(mols, n_jobs=1)
    estate_fp = calculate_estate(mols, n_jobs=1)
    geary_fp = calculate_geary(mols, n_jobs=1)
    # ifg_fp = calculate_ifg(mols, n_jobs=1)
    moe_fp = calculate_moe(mols, n_jobs=1)
    rdkit3d_fp = calculate_rdkit3d(mols, n_jobs=1)

    return np.hstack([
        pubchem_fp,
        cats_fp,
        charges_fp,
        estate_fp,
        geary_fp,
        # ifg_fp,
        moe_fp,
        rdkit3d_fp
    ])


if __name__ == "__main__":
    from rdkit import Chem

    smis = [
        'C1=CC=CC(C(Br)C)=C1',
        'C1=CC2NC(=O)CC3C=2C(C(=O)C2C=CC=CC=23)=C1',
        'C1=CC=C2C(=O)C3C=CNC=3C(=O)C2=C1',
        'C1=NC(CCN)=CN1',
        'C1CCCC(CCO)C1',
        'C1=CC=C2N=C(O)C=CC2=C1',
        'C(OC)1=C(C)C=C2OC[C@]([H])3OC4C(C)=C(OC)C=CC=4C(=O)[C@@]3([H])C2=C1C',
        'C1=C2N=CC=NC2=C2N=CNC2=C1',
        'C1=C(O)C=CC(O)=C1',
        'CCC1(c2ccccc2)C(=O)NC(=O)NC1=O',
        'N1=CN=CN=C1',
        'C1=C2C=CC=CC2=CC2C=CC=CC1=2',  # NonGenotoxic_Carcinogenicity
        'C1=CC=C2C(=O)CC(=O)C2=C1',  # Pains
        'C1=CC=CC(COCO)=C1',  # Potential_Electrophilic
        'N1=NC=CN1C=O',  # Promiscuity
        'CC(=O)OC(=O)C1C=COC1',  # Skin_Sensitization
        'S',
        'CCCCC(=O)[H]',  # Biodegradable
        'C1=CN=C(C(=O)O)C=C1',  # Chelating
        'C(OC)1=CC=C2OCC3OC4C=C(OC)C=CC=4C(=O)C3C2=C1',
        'C1=C2N=CC=NC2=C2N=CNC2=C1',  # Genotoxic_Carcinogenicity_Mutagenicity
        'N(CC)(CCCCC)C(=S)N',  # Idiosyncratic
    ]
    _mols = [Chem.MolFromSmiles(smi) for smi in smis]

    # pubchem_fp = calculate_pubchem(_mols, n_jobs=1)
    # cats_fp = calculate_cats(_mols, n_jobs=1)
    # charges_fp = calculate_charges(_mols, n_jobs=1)
    # estate_fp = calculate_estate(_mols, n_jobs=1)
    # geary_fp = calculate_geary(_mols, n_jobs=1)
    # ifg_fp = calculate_ifg(_mols, n_jobs=1)
    # moe_fp = calculate_moe(_mols, n_jobs=1)
    # rdkit3d_fp = calculate_rdkit3d(_mols, n_jobs=1)
    #
    # mol_desc = np.hstack([
    #     pubchem_fp,
    #     cats_fp,
    #     charges_fp,
    #     estate_fp,
    #     geary_fp,
    #     ifg_fp,
    #     moe_fp,
    #     rdkit3d_fp
    # ])
    print(calculate_descriptor(_mols))
