import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from datasets import EgeognnInferenceDataset
from featurize import calculate_descriptor
from models.downstream import DownstreamTransformerModel, DownstreamTransformerModelV0
from models.gin import EGeoGNNModel
from utils import calculate_oxygen_balance_and_nitrogen_content


def sanitize_filename(text, max_len=80):
    text = re.sub(r"[^\w\-\.]+", "_", text)
    return text[:max_len].strip("_") or "mol"


def collect_attention_records(attention_weights, batch_smiles, atom_batch):
    atom_batch = atom_batch.detach().cpu()

    atom_offsets = [0]
    for smiles_idx in range(len(batch_smiles)):
        atom_count = int((atom_batch == smiles_idx).sum().item())
        atom_offsets.append(atom_offsets[-1] + atom_count)

    records = []
    for mol_idx, smiles in enumerate(batch_smiles):
        record = {
            "smiles": smiles,
            "layers": {},
        }

        atom_start = atom_offsets[mol_idx]
        atom_end = atom_offsets[mol_idx + 1]

        for layer_name, layer_payload in attention_weights.items():
            attn = layer_payload["attn"].detach().cpu()
            edges = layer_payload["edges"].detach().cpu()

            edge_mask = (
                (edges[0] >= atom_start) & (edges[0] < atom_end) &
                (edges[1] >= atom_start) & (edges[1] < atom_end)
            )
            selected_edges = edges[:, edge_mask] - atom_start

            if selected_edges.shape[1] == 0:
                continue

            if attn.shape[0] == edges.shape[1]:
                selected_attn = attn[edge_mask]
            else:
                unique_pairs = []
                seen_pairs = set()
                selected_global_indices = []
                for edge_idx in torch.nonzero(edge_mask, as_tuple=False).flatten().tolist():
                    pair = tuple(sorted((int(edges[0, edge_idx]), int(edges[1, edge_idx]))))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    unique_pairs.append((pair[0] - atom_start, pair[1] - atom_start))
                    selected_global_indices.append(edge_idx)

                total_unique_pairs = {
                    tuple(sorted((int(edges[0, edge_idx]), int(edges[1, edge_idx]))))
                    for edge_idx in range(edges.shape[1])
                }
                if attn.shape[0] != len(total_unique_pairs):
                    raise ValueError(
                        f"Unsupported attention shape for {layer_name}: "
                        f"attn={tuple(attn.shape)}, edges={tuple(edges.shape)}"
                    )

                selected_attn = attn[selected_global_indices]
                selected_edges = torch.tensor(unique_pairs, dtype=torch.long).T if unique_pairs else torch.empty((2, 0), dtype=torch.long)

            record["layers"][layer_name] = {
                "attn": selected_attn.clone(),
                "edges": selected_edges.clone(),
            }

        records.append(record)

    return records


def save_per_molecule_attention(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, record in enumerate(records):
        file_stem = f"{idx:03d}_{sanitize_filename(record['smiles'])}"
        payload = {
            "smiles": [record["smiles"]],
        }
        for layer_name, layer_payload in record["layers"].items():
            payload[layer_name] = {
                "attn": layer_payload["attn"],
                "edges": layer_payload["edges"],
            }
        torch.save(payload, os.path.join(output_dir, f"{file_stem}.pt"))


def inference(model: DownstreamTransformerModel, device, loader, endpoints, args, auxiliary_func=None, auxiliaries=None, auxiliary_items=None):
    if args.use_auxiliary:
        assert auxiliary_func is not None

    model.eval()
    model.compound_encoder.eval()

    results = []
    smiles = []
    attention_records = []

    for step, batch in enumerate(loader):

        tgt_endpoints = torch.tensor(range(len(endpoints))).int().reshape(-1, 1)

        if args.use_auxiliary:
            mols = [Chem.MolFromSmiles(smi) for smi in batch.smiles]
            auxiliaries = torch.tensor(
                np.array([auxiliary_func(mol, auxiliary_items=auxiliary_items) for mol in mols]).astype('float32')
            ).T.to(device)

        input_params = {
            "AtomBondGraph_edges": batch.AtomBondGraph_edges.to(device),
            "BondAngleGraph_edges": batch.BondAngleGraph_edges.to(device),
            "AngleDihedralGraph_edges": batch.AngleDihedralGraph_edges.to(device),
            "pos": batch.atom_poses.to(device),
            "x": batch.node_feat.to(device),
            "bond_attr": batch.edge_attr.to(device),
            "bond_lengths": batch.bond_lengths.to(device),
            "bond_angles": batch.bond_angles.to(device),
            "dihedral_angles": batch.dihedral_angles.to(device),
            "num_graphs": batch.num_graphs,
            "num_atoms": batch.n_atoms.to(device),
            "num_bonds": batch.n_bonds.to(device),
            "num_angles": batch.n_angles.to(device),
            "atom_batch": batch.batch.to(device),
            "tgt_endpoints": tgt_endpoints.to(device),
            "return_attention_weights": args.use_attn,
            "auxiliaries": auxiliaries,
            "inference": True
        }

        with torch.no_grad():
            pred_scaled, _, attention_weights = model(**input_params)
            results.append(pred_scaled.detach().cpu().numpy().flatten())
            smiles = [*smiles, *batch.smiles]
            if attention_weights is not None:
                attention_records.extend(
                    collect_attention_records(
                        attention_weights=attention_weights,
                        batch_smiles=batch.smiles,
                        atom_batch=batch.batch,
                    )
                )

    results = np.hstack(results).reshape(-1, len(endpoints))
    attention_payload = None
    if attention_records:
        attention_payload = {
            "records": attention_records,
            "smiles": smiles,
        }

    return results, attention_payload


def inference_main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    with open(args.model_config_path) as f:
        config = json.load(f)

    with open(args.endpoint_status_file) as f:
        endpoint_statuses = json.load(f)

    dataset = EgeognnInferenceDataset(
        remove_hs=args.remove_hs,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        smiles_list=args.smiles_list,
        seed=args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers
    )

    encoder_params = {
        "latent_size": args.latent_size,
        "hidden_size": args.encoder_hidden_size,
        "n_layers": args.num_encoder_layers,
        "use_layer_norm": True,
        "layernorm_before": True,
        "use_bn": True,
        "dropnode_rate": args.dropnode_rate,
        "encoder_dropout": args.encoder_dropout,
        "num_message_passing_steps": args.num_message_passing_steps,
        "atom_names": config["atom_names"],
        "bond_names": config["bond_names"],
        "global_reducer": 'sum',
        "device": device,
        "without_dihedral": args.without_dihedral,
        "use_attention": args.use_attn
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": args.task_type,
        "endpoints": args.endpoints,
        "frozen_encoder": True,
        "device": device,
        "use_auxiliary": args.use_auxiliary,
        "auxiliary_centers": np.array(args.auxiliary_centers).reshape(-1, 3) if args.use_auxiliary else None,
        "auxiliary_encoder_mode": args.aux_encoder_mode,
        "auxiliary_aggregate_mode": args.aux_aggregate_mode,
        "auxiliary_fusion_mode": args.aux_fusion_mode
    }
    if args.use_old:
        model = DownstreamTransformerModelV0(**model_params).to(device)
    else:
        model = DownstreamTransformerModel(**model_params).to(device)

    if args.model_eval_from is not None:
        assert os.path.exists(args.model_eval_from)
        checkpoint = torch.load(args.model_eval_from, map_location=device)["model_state_dict"]
        model.load_state_dict(checkpoint)
        print(f"load params from {args.model_eval_from}")

    model_without_ddp = model
    args.disable_tqdm = False

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    pred_scaled, attention_weights = inference(
        model=model,
        device=device,
        loader=loader,
        endpoints=args.endpoints,
        args=args,
        auxiliary_func=calculate_oxygen_balance_and_nitrogen_content,
        auxiliary_items=args.auxiliary_items
    )
    if args.task_type == "classification":
        pred_scaled = pred_scaled.argmax(dim=1, keepdim=True)

    pred_scaled_df = pd.DataFrame(pred_scaled, index=args.smiles_list, columns=args.endpoints)

    for endpoint in args.endpoints:
        mean = endpoint_statuses[endpoint]["mean"]
        std = endpoint_statuses[endpoint]["std"]
        pred_scaled_df[endpoint] = pred_scaled_df[endpoint] * std + mean

    if args.representation:
        mols = [Chem.MolFromSmiles(x) for x in attention_weights["smiles"]]
        desc = calculate_descriptor(mols)
        return np.hstack([desc, pred_scaled_df.values]).tolist()

    if args.return_json:
        return pred_scaled_df.T.to_dict()

    if args.out_file:
        pred_scaled_df.to_csv(args.out_file, index_label='SMILES')

    if attention_weights is not None and args.attention_weights_dir is not None:
        os.makedirs(args.attention_weights_dir, exist_ok=True)
        torch.save(attention_weights, os.path.join(args.attention_weights_dir, "attention_weights.pt"))
        if args.save_attention_per_molecule:
            save_per_molecule_attention(
                attention_weights["records"],
                os.path.join(args.attention_weights_dir, "per_molecule"),
            )

    return pred_scaled_df


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--task-type", type=str)

    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--endpoint-status-file", type=str, default=None)
    parser.add_argument("--out-file", type=str)

    parser.add_argument("--smiles-list", type=str, nargs="+", default=None)
    parser.add_argument("--smiles-file", type=str)
    parser.add_argument("--endpoints", type=str, nargs="+")

    parser.add_argument("--remove-hs", action='store_true', default=False)

    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--encoder-hidden-size", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--num-message-passing-steps", type=int, default=2)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)

    parser.add_argument("--encoder-eval-from", type=str, default=None)
    parser.add_argument("--model-eval-from", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-attn", action='store_true', default=False)
    parser.add_argument("--attention-weights-dir", type=str, default=None)
    parser.add_argument("--save-attention-per-molecule", action='store_true', default=False)

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--return-json", action='store_true', default=False)
    parser.add_argument("--without-dihedral", action='store_true', default=False)
    parser.add_argument("--representation", action='store_true', default=False)

    parser.add_argument("--use-auxiliary", action='store_true', default=False)
    parser.add_argument("--auxiliary-items", type=str, nargs="+", default=None)
    parser.add_argument("--auxiliary-centers", type=float, nargs="+", default=None)
    parser.add_argument("--aux-encoder-mode", type=str, default="rbf",
                        choices=["rbf", "mlp"])
    parser.add_argument("--aux-aggregate-mode", type=str, default="concat",
                        choices=["sum", "concat"])
    parser.add_argument("--aux-fusion-mode", type=str, default="residual_gate",
                        choices=["bidirectional_gate", "residual_gate"])

    parser.add_argument("--use-old", action='store_true', default=False)

    args = parser.parse_args()

    assert args.task_type in ['regression', 'classification']

    if args.representation:
        assert args.smiles_list is not None
        args.endpoints = [
            "alphaOpt", "density", "eGapOpt", "eLumoOpt", "espAvg", "espcMin", "espVarNeg", "hCorre", "mpi", "npaMin",
            "sa", "saPos", "uCorre", "zpe", "cm5Max", "e0K", "eHOMO", "eOpt", "espAvgNeg", "espMax", "espVarPos", "H",
            "mu", "nu", "saNeg", "S", "U", "cm5Min", "e", "eHomoOpt", "ese", "espAvgPos", "espMin", "gCorre",
            "hirshfeldMax", "muOpt", "pi", "saNonpolar", "theta", "volumeIMT", "cv", "eGap", "eLUMO", "eseOpt",
            "espcMax", "espVar", "G", "hirshfeldMin", "npaMax", "productOfSigmaSquareAndNu", "saPolar", "thetaOpt",
            "volumeMC"
        ]
        args.task_type = "regression"
        args.status_file = "./configs/tox_pc_mtl/qc_demo_status.json"

    if args.smiles_list is None:
        data = pd.read_csv(args.smiles_file, header=None)
        args.smiles_list = list(data[0].values)

    print(args)
    return inference_main(args)


if __name__ == '__main__':
    result = main_cli()
    print(result)
