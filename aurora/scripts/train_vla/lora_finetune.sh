#!/usr/bin/env bash
set -euo pipefail

# AuroraIG LoRA 微调脚本（仿照 Reconvla 风格）
# 用法：
#   bash scripts/train_vla/lora_finetune.sh
# 可通过环境变量覆盖关键参数。

source ~/miniconda3/bin/activate "${CONDA_ENV:-reconvla}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export SEED="${SEED:-42}"
export DATA_SEED="${DATA_SEED:-$SEED}"

AURORAIG_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RECONVLA_ROOT="${RECONVLA_ROOT:-$AURORAIG_ROOT/reconvla}"
ASSET_ROOT="${ASSET_ROOT:-$AURORAIG_ROOT/../Reconvla/reconvla}"

export PYTHONPATH="$RECONVLA_ROOT:$AURORAIG_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/supa1/miniconda3/envs/reconvla/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] python not executable: $PYTHON_BIN" >&2
    exit 1
fi

MODEL_VERSION="${MODEL_VERSION:-qwen_2}"
MAX_STEPS_ARG=${MAX_STEPS:+--max_steps ${MAX_STEPS}}

"$PYTHON_BIN" "$RECONVLA_ROOT/train_vla.py" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS:-8}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH:-$ASSET_ROOT/checkpoints/pretrain-checkpoint-10388}" \
    --output_dir "${OUTPUT_DIR:-$AURORAIG_ROOT/checkpoints/auroraig_lora}" \
    --vision_tower "${VISION_TOWER:-$ASSET_ROOT/siglip-so400m-patch14-384}" \
    --version "$MODEL_VERSION" \
    --mm_pixel_decoder "${MM_PIXEL_DECODER:-$ASSET_ROOT/pretrained_vae/vae}" \
    --reconstruct_image_num "${RECONSTRUCT_IMAGE_NUM:-1}" \
    --data_path "${DATA_PATH:-$HOME/myreconvla/calvin/dataset/calvin_debug_dataset_processed_json/training_r5.json}" \
    --image_folder "${IMAGE_FOLDER:-$HOME/myreconvla/calvin/dataset/processed_images/calvin_debug_dataset/vla_processed_r5}" \
    --target_image_folder "${TARGET_IMAGE_FOLDER:-$HOME/myreconvla/calvin/dataset/processed_images/calvin_debug_dataset/vla_processed_r5}" \
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
    --eval_strategy "${EVAL_STRATEGY:-no}" \
    --save_strategy "${SAVE_STRATEGY:-epoch}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-1}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --tf32 "${TF32:-False}" \
    --model_max_length "${MODEL_MAX_LENGTH:-32768}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-True}" \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
    --lm_head_cpu_offload "${LM_HEAD_CPU_OFFLOAD:-False}" \
    --lm_head_cpu_dtype "${LM_HEAD_CPU_DTYPE:-float32}" \
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
    --consistency_type_weights_json "${CONSISTENCY_TYPE_WEIGHTS_JSON:-{\"action_polarity_flip\":1.0,\"neighbor_object_replacement\":0.9,\"direction_replacement\":0.85,\"hard_color_negative\":0.85,\"subject_object_swap\":0.85,\"spatial_replacement\":0.8,\"color_replacement\":0.75,\"easy_color_negative\":0.55,\"content_simplification\":0.65,\"other_rewrite\":0.7,\"rule_fallback\":0.6}}" \
    --lora_enable \
    --lora_r "${LORA_R:-8}" \
    --lora_alpha "${LORA_ALPHA:-16}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --lora_bias "${LORA_BIAS:-none}" \
    --double_quant "${DOUBLE_QUANT:-True}" \
    --quant_type "${QUANT_TYPE:-nf4}" \
    --bit "${BIT:-4}"
