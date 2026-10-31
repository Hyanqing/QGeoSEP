#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_SIZE="${PRETRAINED_SIZE:-10}"
ENCODER_EVAL_FROM="${ENCODER_EVAL_FROM:-./outs/checkpoints/pretrain/zinc22-${PRETRAINED_SIZE}-gin/checkpoint_10.pt}"
OUT_DIR="${OUT_DIR:-${PRETRAINED_SIZE}0M}"
lrs_list="${LRS_LIST:-1e-3,1e-4}"
drop_list="${DROP_LIST:-0.1}"
batch_size_list="${BATCH_SIZE_LIST:-128}"
seed_list="${SEED_LIST:-2017}"

for BATCH_SIZE in $batch_size_list; do
  for lrs in $lrs_list; do
    IFS=, read -r encoder_lr head_lr <<< "$lrs"
    for dropout_rate in $drop_list; do
      for seed in $seed_list; do
        bash "$SCRIPT_DIR/finetune-transformer_ems.sh" 0 "$ENCODER_EVAL_FROM" "$OUT_DIR" "$encoder_lr" "$head_lr" "$dropout_rate" "$BATCH_SIZE" "$seed"
      done
    done
  done
done
