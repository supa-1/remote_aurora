#!/usr/bin/env bash
set -euo pipefail

# AuroraIG 全量微调脚本（仿照 Reconvla 风格）
# 用法：
#   bash scripts/train_vla/full_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hpc_env.sh"
setup_hpc_env "${CONDA_ENV:-reconvla}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SEED="${SEED:-42}"
export DATA_SEED="${DATA_SEED:-$SEED}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

AURORAIG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECONVLA_ROOT="${RECONVLA_ROOT:-$AURORAIG_ROOT/reconvla}"
ASSET_ROOT="${ASSET_ROOT:-$AURORAIG_ROOT/../ReconVLA/reconvla}"
DATASET_NAME="${DATASET_NAME:-calvin_debug_dataset}"
DATA_ROOT="${DATA_ROOT:-$AURORAIG_ROOT/../calvin/dataset/process/$DATASET_NAME}"

export PYTHONPATH="$RECONVLA_ROOT:$AURORAIG_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$PYTHON_BIN" == */* ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "[ERROR] python not executable: $PYTHON_BIN" >&2
        exit 1
    fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] python not found on PATH: $PYTHON_BIN" >&2
    exit 1
fi

MODEL_VERSION="${MODEL_VERSION:-qwen_3}"
MAX_STEPS_ARG=${MAX_STEPS:+--max_steps ${MAX_STEPS}}

"$PYTHON_BIN" "$RECONVLA_ROOT/train_vla.py" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS:-16}" \
    --learning_rate "${LEARNING_RATE:-2e-5}" \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH:-$ASSET_ROOT/checkpoints/pretrain-checkpoint-10388}" \
    --output_dir "${OUTPUT_DIR:-$AURORAIG_ROOT/checkpoints/auroraig_full}" \
    --vision_tower "${VISION_TOWER:-$ASSET_ROOT/siglip-so400m-patch14-384}" \
    --version "$MODEL_VERSION" \
    --mm_pixel_decoder "${MM_PIXEL_DECODER:-$ASSET_ROOT/pretrained_vae/vae}" \
    --reconstruct_image_num "${RECONSTRUCT_IMAGE_NUM:-1}" \
    --data_path "${DATA_PATH:-$DATA_ROOT/processed_json/training_r5.json}" \
    --image_folder "${IMAGE_FOLDER:-$DATA_ROOT/processed_images/vla_processed_r5}" \
    --target_image_folder "${TARGET_IMAGE_FOLDER:-$DATA_ROOT/processed_images/vla_processed_r5}" \
    --action_stat "${ACTION_STAT:-$RECONVLA_ROOT/statistics.yaml}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE:-mlp2x_gelu}" \
    --mm_inv_projector_type "${MM_INV_PROJECTOR_TYPE:-denoiser_vit3x}" \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER:--2}" \
    --mm_use_im_start_end "${MM_USE_IM_START_END:-False}" \
    --mm_use_im_patch_token "${MM_USE_IM_PATCH_TOKEN:-False}" \
    --image_aspect_ratio "${IMAGE_ASPECT_RATIO:-pad}" \
    --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH:-True}" \
    --bf16 "${BF16:-False}" \
    --fp16 "${FP16:-True}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    ${MAX_STEPS_ARG:-} \
    --seed "$SEED" \
    --data_seed "$DATA_SEED" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --evaluation_strategy "${EVAL_STRATEGY:-no}" \
    --save_strategy "${SAVE_STRATEGY:-epoch}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --tf32 "${TF32:-False}" \
    --model_max_length "${MODEL_MAX_LENGTH:-8192}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-True}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
    --kbit_skip_large_embedding_upcast "${KBIT_SKIP_LARGE_EMBEDDING_UPCAST:-False}" \
    --kbit_keep_lm_head_in_fp16 "${KBIT_KEEP_LM_HEAD_IN_FP16:-False}" \
    --reconstruct_image "${RECONSTRUCT_IMAGE:-False}" \
    --lazy_preprocess "${LAZY_PREPROCESS:-True}" \
    --enable_text_reconstruction "${ENABLE_TEXT_RECONSTRUCTION:-False}" \
    --text_reconstruction_weight "${TEXT_RECONSTRUCTION_WEIGHT:-0.3}" \
    --text_reconstruction_max_length "${TEXT_RECONSTRUCTION_MAX_LENGTH:-128}" \
    --text_reconstruction_corrupt_ratio "${TEXT_RECONSTRUCTION_CORRUPT_RATIO:-0.3}" \
    --enable_consistency_aux "${ENABLE_CONSISTENCY_AUX:-False}" \
    --consistency_aux_weight "${CONSISTENCY_AUX_WEIGHT:-0.3}" \
    --consistency_margin "${CONSISTENCY_MARGIN:-0.2}" \
    --consistency_alpha "${CONSISTENCY_ALPHA:-0.4}" \
    --consistency_beta "${CONSISTENCY_BETA:-0.3}" \
    --consistency_gamma "${CONSISTENCY_GAMMA:-0.3}" \
    --consistency_max_length "${CONSISTENCY_MAX_LENGTH:-128}" \
    --consistency_use_pair_weights "${CONSISTENCY_USE_PAIR_WEIGHTS:-True}" \
    --consistency_min_pair_weight "${CONSISTENCY_MIN_PAIR_WEIGHT:-0.5}" \
    --consistency_type_weights_json "${CONSISTENCY_TYPE_WEIGHTS_JSON:-{\"action_polarity_flip\":1.0,\"neighbor_object_replacement\":0.9,\"direction_replacement\":0.85,\"hard_color_negative\":0.85,\"subject_object_swap\":0.85,\"spatial_replacement\":0.8,\"color_replacement\":0.75,\"easy_color_negative\":0.55,\"content_simplification\":0.65,\"other_rewrite\":0.7,\"rule_fallback\":0.6}}"
