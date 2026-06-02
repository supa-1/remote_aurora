#!/usr/bin/env bash
set -euo pipefail

# A100 server smoke test for Reconvla LoRA fine-tuning.
# Defaults avoid GPU 0 because it is often occupied on the shared server.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AURORAIG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CONDA_ENV="${CONDA_ENV:-reconvla}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

export MAX_STEPS="${MAX_STEPS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-$AURORAIG_ROOT/checkpoints/a100_lora_smoke}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

export BF16="${BF16:-True}"
export FP16="${FP16:-False}"
export TF32="${TF32:-True}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"

export ENABLE_TEXT_RECONSTRUCTION="${ENABLE_TEXT_RECONSTRUCTION:-False}"
export ENABLE_CONSISTENCY_AUX="${ENABLE_CONSISTENCY_AUX:-False}"
export RECONSTRUCT_IMAGE_NUM="${RECONSTRUCT_IMAGE_NUM:-1}"

cd "$AURORAIG_ROOT"
bash scripts/train_vla/preflight_check.sh
bash scripts/train_vla/lora_finetune.sh
