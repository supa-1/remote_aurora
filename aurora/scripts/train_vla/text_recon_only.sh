#!/usr/bin/env bash
set -euo pipefail

# 仅文本重建训练（关闭图像重建）
# 默认走 LoRA；可用 TRAIN_MODE=full 切换到全量脚本。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_MODE="${TRAIN_MODE:-lora}"
TEXT_RECON_MEMORY_MODE="${TEXT_RECON_MEMORY_MODE:-cpu}"

export ENABLE_TEXT_RECONSTRUCTION="${ENABLE_TEXT_RECONSTRUCTION:-True}"
export TEXT_RECONSTRUCTION_WEIGHT="${TEXT_RECONSTRUCTION_WEIGHT:-0.3}"
export TEXT_RECONSTRUCTION_MAX_LENGTH="${TEXT_RECONSTRUCTION_MAX_LENGTH:-128}"
export TEXT_RECONSTRUCTION_CORRUPT_RATIO="${TEXT_RECONSTRUCTION_CORRUPT_RATIO:-0.3}"
export LM_HEAD_CPU_DTYPE="${LM_HEAD_CPU_DTYPE:-float32}"

case "$TEXT_RECON_MEMORY_MODE" in
  cpu|cpu_offload)
    # 本机低显存版本：lm_head 放 CPU，embedding/lm_head 不做 k-bit fp32 upcast。
    export LM_HEAD_CPU_OFFLOAD="${LM_HEAD_CPU_OFFLOAD:-True}"
    export KBIT_SKIP_LARGE_EMBEDDING_UPCAST="${KBIT_SKIP_LARGE_EMBEDDING_UPCAST:-True}"
    export KBIT_KEEP_LM_HEAD_IN_FP16="${KBIT_KEEP_LM_HEAD_IN_FP16:-False}"
    export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    ;;
  gpu|no_cpu)
    # 服务器 GPU 版本：不迁移到 CPU，保留标准 GPU/k-bit 行为。
    export LM_HEAD_CPU_OFFLOAD="${LM_HEAD_CPU_OFFLOAD:-False}"
    export KBIT_SKIP_LARGE_EMBEDDING_UPCAST="${KBIT_SKIP_LARGE_EMBEDDING_UPCAST:-False}"
    export KBIT_KEEP_LM_HEAD_IN_FP16="${KBIT_KEEP_LM_HEAD_IN_FP16:-False}"
    ;;
  peft_default)
    # 对照版本：保留 PEFT 默认 upcast 行为，8GB 显存上很可能 OOM。
    export LM_HEAD_CPU_OFFLOAD="${LM_HEAD_CPU_OFFLOAD:-False}"
    export KBIT_SKIP_LARGE_EMBEDDING_UPCAST="${KBIT_SKIP_LARGE_EMBEDDING_UPCAST:-False}"
    export KBIT_KEEP_LM_HEAD_IN_FP16="${KBIT_KEEP_LM_HEAD_IN_FP16:-False}"
    ;;
  *)
    echo "[ERROR] unsupported TEXT_RECON_MEMORY_MODE: $TEXT_RECON_MEMORY_MODE (use cpu|gpu|peft_default)" >&2
    exit 1
    ;;
esac

# cpu 模式会设置 8GB 本机低显存默认值；gpu 模式面向服务器，不强行改上下文长度或 allocator。

# 真正关闭图像相关分支（强制覆盖，避免继承外部环境变量）。
export VISION_TOWER="none"
export MM_PIXEL_DECODER="none"
export RECONSTRUCT_IMAGE_NUM="${RECONSTRUCT_IMAGE_NUM:-0}"
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
