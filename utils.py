import math
import os
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {} local rank {}): {}".format(
            args.rank, args.local_rank, "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                    self.eta_min
                    + (1 - self.eta_min)
                    * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                    / 2
            )

        else:
            return self.eta_min + self.lr_step * step


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


def calculate_oxygen_balance_and_nitrogen_content(mol: Chem.rdchem.Mol, auxiliary_items=None):

    METAL_OXYGEN_EQUIVALENTS = {
        "Ag": 0.5,  # Ag2O
        "Ba": 1.0,  # BaO
        "Ca": 1.0,  # CaO
        "Co": 4.0 / 3.0,  # Co3O4
        "Cs": 0.5,  # Cs2O
        "Cu": 1.0,  # CuO
        "Fe": 1.5,  # Fe2O3
        "K": 0.5,  # K2O
        "Li": 0.5,  # Li2O
        "Mg": 1.0,  # MgO
        "Mn": 4.0 / 3.0,  # Mn3O4
        "Na": 0.5,  # Na2O
        "Ni": 1.0,  # NiO
        "Pb": 1.0,  # PbO
        "Rb": 0.5,  # Rb2O
        "Ru": 2.0,  # RuO2
        "Sr": 1.0,  # SrO
        "Zn": 1.0,  # ZnO
    }

    molH = Chem.AddHs(mol)
    counts = Counter(atom.GetSymbol() for atom in molH.GetAtoms())

    nC = counts["C"]
    nH = counts["H"]
    nO = counts["O"]
    nN = counts["N"]
    nS = counts["S"]
    nX = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
    nB = counts["B"]

    metal_oxygen_demand = sum(
        counts[metal] * oxygen_eq
        for metal, oxygen_eq in METAL_OXYGEN_EQUIVALENTS.items()
    )

    oxygen_balance = (nO - 2*nC - nH/2 - 2*nS + nX/2 - 1.5*nB - metal_oxygen_demand) * 16.0 / Descriptors.ExactMolWt(molH) * 100
    nitrogen_content = (nN * 14.0 / Descriptors.ExactMolWt(mol)) * 100.0

    if nC == 0:
        CO_balance = 0
    else:
        CO_balance = (nO - nH/2 - 2*nS + nX/2 - 1.5*nB - metal_oxygen_demand) / nC

    features = {
        "oxygen_balance": oxygen_balance,
        "nitrogen_content": nitrogen_content,
        "CO_balance": CO_balance,
    }
    return [features[item] for item in auxiliary_items]


if __name__ == '__main__':
    smi = '[NH4+].[NH4+].[Na+].n1nc([N-]c2nnnn2-c2nn[n-]n2)n(-c2nn[n-]n2)n1'
    mol = Chem.MolFromSmiles(smi)
    a, b, c = calculate_oxygen_balance_and_nitrogen_content(mol)
    print(a, b, c)