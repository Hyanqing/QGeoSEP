import argparse
import io
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnPretrainedDataset
from utils import init_distributed_mode, WarmCosine
#
# import sys
# import torch
# from torch.utils.data import dataloader
# from torch.multiprocessing import reductions
# from multiprocessing.reduction import ForkingPickler
#
# default_collate_func = dataloader.default_collate
#
#
# def default_collate_override(batch):
#     dataloader._use_shared_memory = False
#     return default_collate_func(batch)
#
#
# setattr(dataloader, 'default_collate', default_collate_override)
#
#
# for t in torch._storage_classes:
#     if sys.version_info[0] == 2:
#         if t in ForkingPickler.dispatch:
#             del ForkingPickler.dispatch[t]
#     else:
#         if t in ForkingPickler._extra_reducers:
#             del ForkingPickler._extra_reducers[t]


def train(model, device, loader, optimizer, scheduler, args):
    model.train()

    loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    for step, batch in enumerate(pbar):
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
            "cm5_charges": batch.cm5_charges.to(device),
            "espc_charges": batch.espc_charges.to(device),
            "hirshfeld_charges": batch.hirshfeld_charges.to(device),
            "npa_charges": batch.npa_charges.to(device),
            "bond_orders": batch.bond_orders.to(device),
            "num_graphs": batch.num_graphs,
            "num_atoms": batch.n_atoms.to(device),
            "num_bonds": batch.n_bonds.to(device),
            "num_angles": batch.n_angles.to(device),
            "atom_batch": batch.batch.to(device),
            "masked_atom_indices": batch.masked_atom_indices.to(device),
            "masked_bond_indices": batch.masked_bond_indices.to(device),
            "masked_angle_indices": batch.masked_angle_indices.to(device),
            "masked_dihedral_indices": batch.masked_dihedral_indices.to(device)
        }

        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()

            loss, loss_dict = model(**input_params)
            loss.backward()

            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

            optimizer.step()
            scheduler.step()

            for k, v in loss_dict.items():
                loss_accum_dict[k] += v

            if step % args.log_interval == 0:
                description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                description += f" lr: {scheduler.get_last_lr()[0]:.5e}"

                pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1

    return loss_accum_dict


def evaluate(model, device, loader, args):
    model.eval()

    loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    for step, batch in enumerate(pbar):
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
            "cm5_charges": batch.cm5_charges.to(device),
            "espc_charges": batch.espc_charges.to(device),
            "hirshfeld_charges": batch.hirshfeld_charges.to(device),
            "npa_charges": batch.npa_charges.to(device),
            "bond_orders": batch.bond_orders.to(device),
            "num_graphs": batch.num_graphs,
            "num_atoms": batch.n_atoms.to(device),
            "num_bonds": batch.n_bonds.to(device),
            "num_angles": batch.n_angles.to(device),
            "atom_batch": batch.batch.to(device),
            "masked_atom_indices": batch.masked_atom_indices.to(device),
            "masked_bond_indices": batch.masked_bond_indices.to(device),
            "masked_angle_indices": batch.masked_angle_indices.to(device),
            "masked_dihedral_indices": batch.masked_dihedral_indices.to(device)
        }

        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        with torch.no_grad():
            loss, loss_dict = model(**input_params)

        for k, v in loss_dict.items():
            loss_accum_dict[k] += v

        if step % args.log_interval == 0:
            description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
            pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1

    return loss_accum_dict


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    with open(args.config_path) as f:
        config = json.load(f)

    dataset = EgeognnPretrainedDataset(
        dataset_name=args.dataset_name,
        remove_hs=args.remove_hs,
        base_path=args.dataset_base_path,
        with_provided_3d=args.with_provided_3d,
        mask_ratio=args.mask_ratio,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        use_mpi=args.use_mpi,
        force_generate=args.dataset,
    )

    if args.dataset:
        return None

    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    valid_size = (total_size - train_size) // 2
    test_size = total_size - train_size - valid_size

    print(
        {
            "train": train_size,
            "valid": valid_size,
            "test": test_size
        }
    )

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

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

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.model_ver == 'gat':
        from models.gat import EGeoGNNModel, EGEM
    else:
        from models.gin import EGeoGNNModel, EGEM

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
        "device": device
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    model_params = {
        "compound_encoder": compound_encoder,
        "pretrain_tasks": config["pretrain_tasks"],
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "use_layer_norm": False,
        "use_bn": False,
        "adc_vocab": 4
    }
    model = EGEM(**model_params).to(device)

    model_without_ddp = model
    args.disable_tqdm = False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with io.open(
            os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
    ) as tgt:
        print(args, file=tgt)

    if args.eval_from is not None and args.eval_from != '':
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["model_state_dict"]
        model_without_ddp.load_state_dict(checkpoint, strict=not args.with_provided_3d)
        print(f"load params from {args.eval_from}")

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    if args.use_adamw:
        optimizer = optim.AdamW(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )

    if not args.lr_warmup:
        scheduler = LambdaLR(optimizer, lambda x: 1.0)
    else:
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
        scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    if args.checkpoint_dir and args.enable_tb:
        tb_writer = SummaryWriter(args.checkpoint_dir)

    train_curve = []
    valid_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))

        if args.distributed:
            sampler_train.set_epoch(epoch)

        print("Training...")
        loss_dict = train(model, device, train_loader, optimizer, scheduler, args)

        print("Evaluating...")
        valid_dict = evaluate(model, device, valid_loader, args)
        test_dict = evaluate(model, device, test_loader, args)

        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")

        train_pref = loss_dict['loss']
        valid_pref = valid_dict['loss']
        test_pref = test_dict['loss']
        # print(f"Train: {train_pref} Validation: {valid_pref} Test: {test_pref}")

        train_curve.append(train_pref)
        valid_curve.append(valid_pref)
        test_curve.append(test_pref)

        if args.checkpoint_dir:
            logs = {"epoch": epoch, "Train": loss_dict, "Valid": valid_dict, "Test": test_dict}
            with io.open(
                    os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

            checkpoint = {
                "epoch": epoch,
                "compound_encoder_state_dict": model_without_ddp.compound_encoder.state_dict(),
                "model_state_dict": model_without_ddp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": args,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))
            if args.enable_tb:
                tb_writer.add_scalar("evaluation/train", train_pref, epoch)
                tb_writer.add_scalar("evaluation/valid", valid_pref, epoch)
                tb_writer.add_scalar("evaluation/test", test_pref, epoch)
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"training/{k}", v, epoch)

    best_val_epoch = np.argmin(np.array(valid_curve))  # todo

    if args.checkpoint_dir and args.enable_tb:
        tb_writer.close()
    if args.distributed:
        torch.distributed.destroy_process_group()

    print("Finished traning!")
    print(f"Best validation epoch: {best_val_epoch + 1}")
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_curve[best_val_epoch]}")


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--config-path", type=str)

    parser.add_argument("--dataset-base-path", type=str)
    parser.add_argument("--dataset-name", type=str)

    parser.add_argument("--remove-hs", action='store_true', default=False)
    parser.add_argument("--with-provided-3d", action='store_true', default=False)
    parser.add_argument("--mask-ratio", type=float, default=0.5)

    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--encoder-hidden-size", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--num-message-passing-steps", type=int, default=2)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)

    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-adamw", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-warmup", action='store_true', default=False)
    parser.add_argument("--period", type=float, default=10)

    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--enable-tb", action='store_true', default=False)
    parser.add_argument("--use-mpi", action='store_true', default=False)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--grad-norm", type=float, default=None)

    parser.add_argument("--model-ver", type=str, default='gat')

    parser.add_argument("--dataset", action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    init_distributed_mode(args)

    main(args)


if __name__ == "__main__":
    main_cli()
