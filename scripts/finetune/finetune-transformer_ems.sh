#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PYTHONUNBUFFERED=1

SEED="${8:-2017}"

MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-./config/config.json}"
ENDPOINT_STATUS_FILE="${ENDPOINT_STATUS_FILE:-./config/ems_9_3_endpoint.json}"

NAME="ems-9_3"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-./data/$NAME}"

LATENT_SIZE=32
NUM_ENCODER_LAYERS=8
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1

HIDDEN_SIZE=512
NUM_LAYERS=3
DROPOUT_RATE="${6:-0.1}"

BATCH_SIZE="${7:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BETA=0.999
WEIGHT_DECAY=1e-2

ENCODER_LR="${4:-1e-3}"
HEAD_LR="${5:-1e-4}"

EPOCHS="${EPOCHS:-120}"
LOG_INTERVAL=5

DEVICE="${DEVICE:-cuda}"

MODEL_VER='gin'

ENCODER_EVAL_FROM="${2:-./outs/checkpoints/pretrain/zinc22-10-gin/checkpoint_10.pt}"

TASK_TYPE='regression'
N_LABELS=5

DISTRIBUTED="${1:-0}"
OUT_DIR="${3:-100M}"

CHECKPOINT_DIR="./outs/checkpoints/${NAME}_transformer_3/${OUT_DIR}_${BATCH_SIZE}_${EPOCHS}/${MODEL_VER}-L${NUM_LAYERS}-HS${HIDDEN_SIZE}-D${DROPOUT_RATE}-ELR${ENCODER_LR}-LR${HEAD_LR}-${SEED}-v3"
ENDPOINTS="dv fs is Isp dt d dp hof q"

PREPROCESS_ENDPOINTS="kotori"
EXCLUDE_SMILES="kotori"

# Multi-card initialization in the Python entry point needs separate fixes.
if [[ "$DISTRIBUTED" != "0" ]]; then
  echo "Distributed fine-tuning is not supported; use argument 0." >&2
  exit 2
fi

"${PYTHON:-python}" finetune_transformer.py \
    --seed "$SEED" \
    --model-config-path "$MODEL_CONFIG_PATH" \
    --endpoint-status-file "$ENDPOINT_STATUS_FILE" \
    --dataset-base-path "$DATASET_BASE_PATH" \
    --latent-size "$LATENT_SIZE" \
    --encoder-hidden-size 256 \
    --num-encoder-layers "$NUM_ENCODER_LAYERS" \
    --dropnode-rate "$DROPNODE_RATE" \
    --encoder-dropout "$ENCODER_DROPOUT" \
    --num-message-passing-steps 2 \
    --hidden-size "$HIDDEN_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --dropout-rate "$DROPOUT_RATE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --encoder-eval-from "$ENCODER_EVAL_FROM" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --encoder_lr "$ENCODER_LR" \
    --head_lr "$HEAD_LR" \
    --beta "$BETA" \
    --weight-decay "$WEIGHT_DECAY" \
    --epochs "$EPOCHS" \
    --log-interval "$LOG_INTERVAL" \
    --model-ver "$MODEL_VER" \
    --device "$DEVICE" \
    --lr-warmup \
    --use-adamw \
    --remove-hs \
    --enable-tb \
    --endpoints $ENDPOINTS \
    --preprocess-endpoints $PREPROCESS_ENDPOINTS \
    --exclude-smiles $EXCLUDE_SMILES \
    --task-type "$TASK_TYPE" \
    --n-labels "$N_LABELS" \
    --use-attn \
    --use-auxiliary \
    --auxiliary-items oxygen_balance nitrogen_content CO_balance \
    --auxiliary-centers -324 50 1 0 90 1 -5 8 0.1 \
    --aux-encoder-mode rbf \
    --aux-aggregate-mode concat \
    --aux-fusion-mode residual_gate
