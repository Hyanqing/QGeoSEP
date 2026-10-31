import argparse
import io
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error
from torch import optim
from torch.utils.data import DistributedSampler, random_split
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnFinetuneDataset
from datasets.utils import GraphDataset
from models.downstream import DownstreamTransformerModel
from models.downstream.model_transformer import DownstreamTransformerModelV0
from utils import exempt_parameters, init_distributed_mode


def train(
        model: DownstreamTransformerModel, device, loader,
        encoder_opt, head_opt, args, endpoint_status=None
):
    model.train()

    loss_accum_dict = defaultdict(float)
    counter = {}
    prediction_logs = {}

    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    for step, batch in enumerate(pbar):
        tgt_endpoints = torch.tensor(
            [args.endpoints.index(x) for x in batch.endpoint]
        ).reshape(-1, 1).to(device).int()

        auxiliaries = batch.auxiliary.reshape(-1, len(args.auxiliary_items)).T.to(device) \
            if args.use_auxiliary else None

        if endpoint_status is None:
            endpoint_status = {}

        if args.use_old:
            gamma = torch.tensor(
                [
                    -math.log10(endpoint_status[x]['sample'])
                    if x in endpoint_status else
                    1
                    for x in batch.endpoint
                ]
            ).mean()
        else:
            gamma = 1

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
            "tgt_endpoints": tgt_endpoints,
            "return_attention_weights": args.use_attn,
            "auxiliaries": auxiliaries
        }

        if len(batch.label) < args.batch_size * 0.5:
            continue
        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        if encoder_opt is not None:
            encoder_opt.zero_grad()
        head_opt.zero_grad()

        pred_scaled, endpoint_proba, _ = model(**input_params)

        label_scaled, label_mean, label_std = batch.label.reshape(-1, 3).T
        label_scaled = label_scaled.to(device).unsqueeze(1)
        label_mean = label_mean.to(device).unsqueeze(1)
        label_std = label_std.to(device).unsqueeze(1)

        if args.distributed:
            loss, loss_dict = model.module.compute_loss(
                pred_value=pred_scaled, target_value=label_scaled,
                endpoint_proba=endpoint_proba, endpoints=tgt_endpoints,
                endpoint_weight=0.8, gamma=gamma
            )
        else:
            loss, loss_dict = model.compute_loss(
                pred_value=pred_scaled, target_value=label_scaled,
                endpoint_proba=endpoint_proba, endpoints=tgt_endpoints,
                endpoint_weight=0.8, gamma=gamma
            )

        loss.backward()

        if encoder_opt is not None:
            encoder_opt.step()
        head_opt.step()

        label = label_scaled * label_std + label_mean
        pred = pred_scaled * label_std + label_mean
        if args.task_type == "classification":
            pred = pred.argmax(dim=1, keepdim=True)

        for i, endpoint in enumerate(batch.endpoint):
            if endpoint not in prediction_logs:
                prediction_logs[endpoint] = {'y_true': [], 'y_pred': []}
            prediction_logs[endpoint]['y_true'].append(label[i].detach().item())
            prediction_logs[endpoint]['y_pred'].append(pred[i].detach().item())

        for k, v in loss_dict.items():
            counter[k] = counter.get(k, 0) + 1
            loss_accum_dict[k] += v

        if step % args.log_interval == 0:
            description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
            pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= counter[k]

    for endpoint, results in prediction_logs.items():
        if args.task_type == "regression":
            loss_accum_dict[f"{endpoint}_r2"] = r2_score(results['y_true'], results['y_pred'])
            loss_accum_dict[f"{endpoint}_mse"] = mean_squared_error(results['y_true'], results['y_pred'])
            loss_accum_dict[f"{endpoint}_mae"] = mean_absolute_error(results['y_true'], results['y_pred'])
        else:
            loss_accum_dict[f"{endpoint}_acc"] = accuracy_score(results['y_true'], results['y_pred'])

    return loss_accum_dict, prediction_logs


def evaluate(
        model: DownstreamTransformerModel,
        device, loader, args, endpoint_status=None
):
    model.eval()
    model.compound_encoder.eval()

    loss_accum_dict = defaultdict(float)
    counter = {}
    prediction_logs = {}
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    for step, batch in enumerate(pbar):
        tgt_endpoints = torch.tensor(
            [args.endpoints.index(x) for x in batch.endpoint]
        ).reshape(-1, 1).to(device).int()

        auxiliaries = batch.auxiliary.reshape(-1, len(args.auxiliary_items)).T.to(device) \
            if args.use_auxiliary else None

        if endpoint_status is None:
            endpoint_status = {}

        if args.use_old:
            gamma = torch.tensor(
                [
                    -math.log10(endpoint_status[x]['sample'])
                    if x in endpoint_status else
                    1
                    for x in batch.endpoint
                ]
            ).mean()
        else:
            gamma = 1

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
            "tgt_endpoints": tgt_endpoints,
            "auxiliaries": auxiliaries
        }

        if len(batch.label) < args.batch_size * 0.5:
            continue
        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        with torch.no_grad():
            pred_scaled, endpoint_proba, _ = model(**input_params)

            label_scaled, label_mean, label_std = batch.label.reshape(-1, 3).T
            label_scaled = label_scaled.to(device).unsqueeze(1)
            label_mean = label_mean.to(device).unsqueeze(1)
            label_std = label_std.to(device).unsqueeze(1)

            if args.distributed:
                loss, loss_dict = model.module.compute_loss(
                    pred_value=pred_scaled, target_value=label_scaled,
                    endpoint_proba=endpoint_proba, endpoints=tgt_endpoints,
                    endpoint_weight=0.8, gamma=gamma
                )
            else:
                loss, loss_dict = model.compute_loss(
                    pred_value=pred_scaled, target_value=label_scaled,
                    endpoint_proba=endpoint_proba, endpoints=tgt_endpoints,
                    endpoint_weight=0.8, gamma=gamma
                )

        label = (label_scaled * label_std) + label_mean
        pred = (pred_scaled * label_std) + label_mean
        if args.task_type == "classification":
            pred = pred.argmax(dim=1, keepdim=True)

        for i, endpoint in enumerate(batch.endpoint):
            if endpoint not in prediction_logs:
                prediction_logs[endpoint] = {'y_true': [], 'y_pred': []}
            prediction_logs[endpoint]['y_true'].append(label[i].detach().item())
            prediction_logs[endpoint]['y_pred'].append(pred[i].detach().item())

        for k, v in loss_dict.items():
            counter[k] = counter.get(k, 0) + 1
            loss_accum_dict[k] += v

        if step % args.log_interval == 0:
            description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
            pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= counter[k]

    for endpoint, results in prediction_logs.items():
        if args.task_type == "regression":
            loss_accum_dict[f"{endpoint}_r2"] = r2_score(results['y_true'], results['y_pred'])
            loss_accum_dict[f"{endpoint}_mse"] = mean_squared_error(results['y_true'], results['y_pred'])
            loss_accum_dict[f"{endpoint}_mae"] = mean_absolute_error(results['y_true'], results['y_pred'])
        else:
            loss_accum_dict[f"{endpoint}_acc"] = accuracy_score(results['y_true'], results['y_pred'])

    return loss_accum_dict, prediction_logs


def get_label_stat(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])

    nan_mask = np.isnan(labels)

    return np.min(labels[~nan_mask]), np.max(labels[~nan_mask]), np.mean(labels[~nan_mask])


def random_split_by_endpoint(dataset, endpoints):
    train_dataset, test_dataset = [], []

    for endpoint in endpoints:
        idx = [i for (i, data) in enumerate(dataset) if data.endpoint == endpoint]
        train_set, test_set = random_split(
            GraphDataset([dataset[x] for x in idx]),
            [0.8, 0.2]
        )
        train_dataset.append(train_set)
        test_dataset.append(test_set)

    return ConcatDataset(train_dataset), ConcatDataset(test_dataset)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    with open(args.model_config_path) as f:
        config = json.load(f)

    endpoint_status = {}
    if args.endpoint_status_file is not None and args.endpoint_status_file != '':
        with open(args.endpoint_status_file) as f:
            endpoint_status = json.load(f)

    dataset = EgeognnFinetuneDataset(
        remove_hs=args.remove_hs,
        base_path=args.dataset_base_path,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        # rng_seed=args.rng_seed,
        dev=args.dev,
        endpoints=args.endpoints,
        use_mpi=args.use_mpi,
        force_generate=args.dataset,
        preprocess_endpoints=args.preprocess_endpoints,
        task_type=args.task_type,
        use_auxiliary=args.use_auxiliary,
        auxiliary_items=args.auxiliary_items
    )

    if args.dataset:
        return None

    args.endpoints = [
        endpoint
        if (endpoint not in args.preprocess_endpoints or args.task_type == "classification") else f"log10_{endpoint}"
        for endpoint in args.endpoints
    ]

    total_size = len(dataset)
    train_size = int(total_size * 0.8) if not args.train_all else total_size
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # train_dataset, test_dataset = random_split_by_endpoint(dataset, args.endpoints)
    print({"train": len(train_dataset), "test": len(test_dataset)})

    train_smiles = [x.smiles for x in train_dataset]
    test_smiles = [x.smiles for x in test_dataset]

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_sampler=batch_sampler_train,
    )

    if not args.train_all:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers
        )

    if args.model_ver == 'gat':
        from models.gat import EGeoGNNModel
    else:
        from models.gin import EGeoGNNModel

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

    if args.encoder_eval_from is not None:
        assert os.path.exists(args.encoder_eval_from)
        # checkpoint = torch.load(args.encoder_eval_from, map_location=device)["compound_encoder_state_dict"]
        # compound_encoder.load_state_dict(checkpoint)
        checkpoint = torch.load(args.encoder_eval_from, map_location=device)["compound_encoder_state_dict"]
        missing, unexpected = compound_encoder.load_state_dict(checkpoint, strict=False)

        print(f"load params from {args.encoder_eval_from}")

    model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": args.task_type,
        "endpoints": args.endpoints,
        "frozen_encoder": args.frozen_encoder,
        "n_labels": args.n_labels,
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

    model_without_ddp = model
    args.disable_tqdm = False

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with io.open(
            os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
    ) as tgt:
        print(args, file=tgt)

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    encoder_params = list(compound_encoder.parameters())
    head_params = exempt_parameters(model_without_ddp.parameters(), encoder_params)

    if args.frozen_encoder:
        for params in encoder_params:
            params.requires_grad = False
# -------
#     for p in compound_encoder.parameters():
#         p.requires_grad = False
#
#     for name, p in compound_encoder.named_parameters():
#         if any(key in name for key in [
#             "attention_layer",
#             "attn_norm",
#             "attn_graph_norm",
#             "alpha",
#         ]):
#             p.requires_grad = True
#
#     encoder_params = [p for p in compound_encoder.parameters() if p.requires_grad]
#     head_params = [
#         p for name, p in model_without_ddp.named_parameters()
#         if p.requires_grad and not name.startswith("compound_encoder.")
#     ]
    # -------
    if args.use_adamw:
        encoder_optimizer = optim.AdamW(
            encoder_params,
            lr=args.encoder_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        ) if not args.frozen_encoder else None
        # encoder_optimizer = optim.AdamW(
        #     encoder_params,
        #     lr=args.encoder_lr,
        #     betas=(0.9, args.beta),
        #     weight_decay=args.weight_decay,
        # ) if len(encoder_params) > 0 else None
        head_optimizer = optim.AdamW(
            head_params,
            lr=args.head_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )
    else:
        encoder_optimizer = optim.Adam(
            encoder_params,
            lr=args.encoder_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        ) if not args.frozen_encoder else None
        # encoder_optimizer = optim.Adam(
        #     encoder_params,
        #     lr=args.encoder_lr,
        #     betas=(0.9, args.beta),
        #     weight_decay=args.weight_decay,
        # ) if len(encoder_params) > 0 else None
        head_optimizer = optim.Adam(
            head_params,
            lr=args.head_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )

    detail = {
        "train_smiles": np.array(train_smiles),
        "test_smiles": np.array(test_smiles),
        "args": args,
    }
    torch.save(detail, os.path.join(args.checkpoint_dir, f"checkpoint_training_detail.pt"))

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))

        if args.distributed:
            sampler_train.set_epoch(epoch)

        print("Training...")
        train_dict, train_prediction_logs = train(
            model=model,
            device=device,
            loader=train_loader,
            encoder_opt=encoder_optimizer,
            head_opt=head_optimizer,
            args=args,
            endpoint_status=endpoint_status
        )

        test_dict = {}
        test_prediction_logs = {}
        if not args.train_all:
            print("Evaluating...")
            test_dict, test_prediction_logs = evaluate(
                model, device, test_loader, args,
                endpoint_status=endpoint_status
            )

        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")

        if args.checkpoint_dir:
            logs = {"epoch": epoch, "Train": train_dict, "Test": test_dict}
            with io.open(
                    os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

            checkpoint = {
                "epoch": epoch,
                "compound_encoder": compound_encoder.state_dict(),
                "model_state_dict": model_without_ddp.state_dict(),
                "train_history": train_prediction_logs if not args.train_all else np.array([]),
                "test_history": test_prediction_logs if not args.train_all else np.array([]),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))

            if args.enable_tb and not args.train_all:
                tb_writer = SummaryWriter(args.checkpoint_dir)

                tb_writer.add_scalar("loss/train", train_dict['loss'], epoch)
                tb_writer.add_scalar("loss/test", test_dict.get('loss', None), epoch)

                tb_writer.add_scalar("value_loss/train", train_dict['value_loss'], epoch)
                tb_writer.add_scalar("value_loss/test", test_dict.get('value_loss', None), epoch)

                tb_writer.add_scalar("endpoint_loss/train", train_dict['endpoint_loss'], epoch)
                tb_writer.add_scalar("endpoint_loss/test", test_dict.get('endpoint_loss', None), epoch)

                for k, v in test_dict.items():
                    if "_r2" in k:
                        tb_writer.add_scalar(f"r2_score/{k}", v, epoch)
                        continue
                    if "_mse" in k:
                        tb_writer.add_scalar(f"mse/{k}", v, epoch)
                    if "_acc" in k:
                        tb_writer.add_scalar(f"acc/{k}", v, epoch)
                    if "_mae" in k:
                        tb_writer.add_scalar(f"mae/{k}", v, epoch)
    return None


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action='store_true', default=False)
    parser.add_argument("--dataset", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--task-type", type=str)

    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--endpoint-status-file", type=str, default=None)
    parser.add_argument("--dataset-base-path", type=str)
    parser.add_argument("--endpoint-weight", type=float, default=0.8)

    parser.add_argument("--remove-hs", action='store_true', default=False)

    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--encoder-hidden-size", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--num-message-passing-steps", type=int, default=2)
    parser.add_argument("--n-labels", type=int, default=2)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)

    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--encoder-eval-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-adamw", action='store_true', default=False)

    parser.add_argument("--encoder_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-3)

    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-warmup", action='store_true', default=False)

    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--use_mpi", action='store_true', default=False)

    parser.add_argument("--endpoints", type=str, nargs="+")
    parser.add_argument("--preprocess-endpoints", type=str, nargs="+")
    parser.add_argument("--exclude-smiles", type=str, nargs="+")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--grad-norm", type=float, default=None)

    parser.add_argument("--model-ver", type=str, default='gat')
    parser.add_argument("--frozen-encoder", action='store_true', default=False)
    parser.add_argument("--enable-tb", action='store_true', default=False)
    parser.add_argument("--without-dihedral", action='store_true', default=False)

    parser.add_argument("--train-all", action='store_true', default=False)
    parser.add_argument("--use-attn", action='store_true', default=False)

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
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--rng-seed", type=int)

    args = parser.parse_args()
    assert args.task_type in ['regression', 'classification']
    print(args)

    init_distributed_mode(args)

    main(args)


if __name__ == '__main__':
    main_cli()
