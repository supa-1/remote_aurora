#!/usr/bin/env bash
set -euo pipefail

# 文本重建 + 图像重建联合训练
# 默认走 LoRA；可用 TRAIN_MODE=full 切换到全量脚本。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_MODE="${TRAIN_MODE:-lora}"

export ENABLE_TEXT_RECONSTRUCTION="${ENABLE_TEXT_RECONSTRUCTION:-True}"
export TEXT_RECONSTRUCTION_WEIGHT="${TEXT_RECONSTRUCTION_WEIGHT:-0.3}"
export TEXT_RECONSTRUCTION_MAX_LENGTH="${TEXT_RECONSTRUCTION_MAX_LENGTH:-128}"
export TEXT_RECONSTRUCTION_CORRUPT_RATIO="${TEXT_RECONSTRUCTION_CORRUPT_RATIO:-0.3}"
export LM_HEAD_CPU_OFFLOAD="${LM_HEAD_CPU_OFFLOAD:-True}"
export LM_HEAD_CPU_DTYPE="${LM_HEAD_CPU_DTYPE:-float32}"

# 开启图像重建监督（vm_loss）：reconstruct_image_num 需为 1 或 2。
# Day16/17 排障结论：联合训练同样会经过 lm_head 投影，
# 默认开启 CPU offload 以降低 8GB 卡上的瞬时显存峰值。
export RECONSTRUCT_IMAGE_NUM="${RECONSTRUCT_IMAGE_NUM:-1}"
export RECONSTRUCT_IMAGE="${RECONSTRUCT_IMAGE:-False}"

# 本脚本聚焦重建分支，不默认叠加一致性辅助。
export ENABLE_CONSISTENCY_AUX="${ENABLE_CONSISTENCY_AUX:-False}"

case "$TRAIN_MODE" in
  lora)
    exec bash "$SCRIPT_DIR/lora_finetune.sh"
    ;;
  full)
    exec bash "$SCRIPT_DIR/full_finetune.sh"
    ;;
  *)
    echo "[ERROR] unsupported TRAIN_MODE: $TRAIN_MODE (use lora|full)" >&2
    exit 1
    ;;
esac
