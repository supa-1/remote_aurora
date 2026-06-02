#!/usr/bin/env bash
set -euo pipefail

# One-command local training launcher for RTX 4060 Laptop 8GB.
# Defaults to the AuroraIG text reconstruction low-memory path.
#
# Smoke test:
#   MAX_STEPS=1 bash scripts/train_vla/run_4060_8gb_text_recon.sh
#
# Short local run:
#   bash scripts/train_vla/run_4060_8gb_text_recon.sh
#
# Override data/model if needed:
#   DATA_PATH=/path/to/training.json \
#   MODEL_NAME_OR_PATH=/path/to/pretrain-checkpoint \
#   bash scripts/train_vla/run_4060_8gb_text_recon.sh

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_MODEL="$ROOT_DIR/../Reconvla/reconvla/checkpoints/pretrain-checkpoint-10388"
DEFAULT_DATA="$HOME/myreconvla/calvin/dataset/calvin_debug_dataset_processed_json/training_r5.json"

RUN_NAME="${RUN_NAME:-4060_text_recon_$(date +%Y%m%d_%H%M%S)}"

export DATA_PATH="${DATA_PATH:-$DEFAULT_DATA}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$DEFAULT_MODEL}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints/$RUN_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TEXT_RECON_MEMORY_MODE="${TEXT_RECON_MEMORY_MODE:-cpu}"
export TRAIN_MODE="${TRAIN_MODE:-lora}"

export MAX_STEPS="${MAX_STEPS:-20}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

export FP16="${FP16:-True}"
export BF16="${BF16:-False}"
export TF32="${TF32:-False}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

export ENABLE_TEXT_RECONSTRUCTION="${ENABLE_TEXT_RECONSTRUCTION:-True}"
export TEXT_RECONSTRUCTION_WEIGHT="${TEXT_RECONSTRUCTION_WEIGHT:-0.3}"
export TEXT_RECONSTRUCTION_MAX_LENGTH="${TEXT_RECONSTRUCTION_MAX_LENGTH:-128}"
export TEXT_RECONSTRUCTION_CORRUPT_RATIO="${TEXT_RECONSTRUCTION_CORRUPT_RATIO:-0.3}"
export ENABLE_CONSISTENCY_AUX="${ENABLE_CONSISTENCY_AUX:-False}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[ERROR] DATA_PATH not found: $DATA_PATH" >&2
  echo "        Set DATA_PATH=/path/to/reconvla_train.json and run again." >&2
  exit 1
fi

if [[ ! -f "$MODEL_NAME_OR_PATH/config.json" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH does not look like a checkpoint: $MODEL_NAME_OR_PATH" >&2
  echo "        Expected config.json under MODEL_NAME_OR_PATH." >&2
  exit 1
fi

shopt -s nullglob
existing_checkpoints=("$OUTPUT_DIR"/checkpoint-*)
shopt -u nullglob

if (( ${#existing_checkpoints[@]} > 0 )) && [[ "${ALLOW_RESUME:-False}" != "True" ]]; then
  echo "[ERROR] OUTPUT_DIR already has checkpoint-* files: $OUTPUT_DIR" >&2
  echo "        Use a new OUTPUT_DIR, or set ALLOW_RESUME=True if resume is intentional." >&2
  exit 1
fi

echo "[AuroraIG 4060 8GB launcher]"
echo "  DATA_PATH=$DATA_PATH"
echo "  MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MAX_STEPS=$MAX_STEPS"
echo "  TEXT_RECON_MEMORY_MODE=$TEXT_RECON_MEMORY_MODE"
echo "  GRAD_ACC_STEPS=$GRAD_ACC_STEPS"
echo "  MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH"
echo "  ENABLE_CONSISTENCY_AUX=$ENABLE_CONSISTENCY_AUX"

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "[WARN] nvidia-smi failed; continuing because GPU status printing is optional." >&2
  fi
fi

exec bash scripts/train_vla/text_recon_only.sh
