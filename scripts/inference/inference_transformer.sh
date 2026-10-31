#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PYTHONUNBUFFERED=1

SEED="${1:-2017}"

MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-./config/config.json}"
ENDPOINT_STATUS_FILE="${ENDPOINT_STATUS_FILE:-./config/ems_9_3_status.json}"

LATENT_SIZE=32
NUM_ENCODER_LAYERS=8
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1

HIDDEN_SIZE=512
NUM_LAYERS=3
DROPOUT_RATE=0.1

NUM_WORKERS="${NUM_WORKERS:-6}"
BETA=0.999
WEIGHT_DECAY=1e-2

DEVICE="${DEVICE:-cuda}"

MODEL_EVAL_FROM="${MODEL_EVAL_FROM:-./outs/checkpoints/ems_final/checkpoint_final.pt}"

TASK_TYPE='regression'

ENDPOINTS="dv fs is Isp dt d dp hof q"

SMILES_FILE="${SMILES_FILE:-./data/ems-9_3/x.csv}"
OUT_FILE="${OUT_FILE:-./outs/inference/x.csv}"

echo "Running inference on $DEVICE"
echo "=================="
mkdir -p "$(dirname "$OUT_FILE")"
"${PYTHON:-python}" inference_transformer.py \
  --seed "$SEED" \
  --task-type "$TASK_TYPE" \
  --model-config-path "$MODEL_CONFIG_PATH" \
  --endpoint-status-file "$ENDPOINT_STATUS_FILE" \
  --latent-size "$LATENT_SIZE" \
  --num-encoder-layers "$NUM_ENCODER_LAYERS" \
  --dropnode-rate "$DROPNODE_RATE" \
  --encoder-dropout "$ENCODER_DROPOUT" \
  --hidden-size "$HIDDEN_SIZE" \
  --num-layers "$NUM_LAYERS" \
  --dropout-rate "$DROPOUT_RATE" \
  --model-eval-from "$MODEL_EVAL_FROM" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --remove-hs \
  --endpoints $ENDPOINTS \
  --out-file "$OUT_FILE" \
  --smiles-file "$SMILES_FILE" \
    --use-attn \
    --use-auxiliary \
    --auxiliary-items oxygen_balance nitrogen_content CO_balance \
    --auxiliary-centers -324 50 1 0 90 1 -5 8 0.1 \
    --aux-encoder-mode rbf \
    --aux-aggregate-mode concat \
    --aux-fusion-mode residual_gate
