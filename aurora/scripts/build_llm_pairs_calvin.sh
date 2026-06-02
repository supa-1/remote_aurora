#!/usr/bin/env bash
set -euo pipefail

# One-shot runner: build LLM-based true/false instruction pairs for
# both training and validation splits from processed CALVIN JSON.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH_DEFAULT="/home/supa1/myreconvla/AuroraIG/models/use/Qwen3-4B-Instruct-2507"
PYTHON_BIN="${PYTHON_BIN:-/home/supa1/miniconda3/envs/activegaze/bin/python}"

MODEL_PATH="${AURORAIG_LLM_MODEL_PATH:-$MODEL_PATH_DEFAULT}"
JSON_ROOT="${1:-$PROJECT_ROOT/data/processed_json/calvin_debug_dataset}"
OUT_ROOT="${2:-$PROJECT_ROOT/data/consistency_pairs/calvin_debug_dataset}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT_ROOT/data/processed_images/calvin_debug_dataset/vla_processed_r5}"
MAX_LLM_NEGATIVES="${MAX_LLM_NEGATIVES:-6}"
MAX_RULE_NEGATIVES="${MAX_RULE_NEGATIVES:-0}"
MIN_PAIRS="${MIN_PAIRS:-100}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$PROJECT_ROOT/../ReconVLA/reconvla/scripts/helper/best.pt}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"

TRAIN_JSON="$JSON_ROOT/training_r5.json"
VAL_JSON="$JSON_ROOT/validation_r5.json"
TRAIN_OUT="$OUT_ROOT/training_llm_ranking.jsonl"
VAL_OUT="$OUT_ROOT/validation_llm_ranking.jsonl"
TRAIN_RAW_OUT="$OUT_ROOT/training_llm_ranking.raw.jsonl"
VAL_RAW_OUT="$OUT_ROOT/validation_llm_ranking.raw.jsonl"
TRAIN_DROPPED_OUT="$OUT_ROOT/training_llm_ranking.dropped.jsonl"
VAL_DROPPED_OUT="$OUT_ROOT/validation_llm_ranking.dropped.jsonl"
TRAIN_REVIEW_OUT="$OUT_ROOT/training_llm_ranking.review_100.jsonl"
VAL_REVIEW_OUT="$OUT_ROOT/validation_llm_ranking.review_100.jsonl"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] model path not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_JSON" ]]; then
  echo "[ERROR] missing training json: $TRAIN_JSON" >&2
  exit 1
fi

if [[ ! -f "$VAL_JSON" ]]; then
  echo "[ERROR] missing validation json: $VAL_JSON" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

export AURORAIG_LLM_MODEL_PATH="$MODEL_PATH"
export AURORAIG_LLM_FAMILY="${AURORAIG_LLM_FAMILY:-qwen3}"
export AURORAIG_LLM_LOAD_IN_4BIT="${AURORAIG_LLM_LOAD_IN_4BIT:-1}"
export AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE="${AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
export AURORAIG_LLM_OFFLOAD_BUFFERS="${AURORAIG_LLM_OFFLOAD_BUFFERS:-0}"
export AURORAIG_LLM_TEMPERATURE="${AURORAIG_LLM_TEMPERATURE:-0}"
export AURORAIG_LLM_TOP_P="${AURORAIG_LLM_TOP_P:-1.0}"
export AURORAIG_LLM_MAX_NEW_TOKENS="${AURORAIG_LLM_MAX_NEW_TOKENS:-96}"
export AURORAIG_LLM_ENABLE_SPATIAL_VOCAB_FILTER_ONLY="${AURORAIG_LLM_ENABLE_SPATIAL_VOCAB_FILTER_ONLY:-1}"
export AURORAIG_LLM_ENABLE_SUBJECT_OBJECT_VOCAB_COUNT_FILTER_ONLY="${AURORAIG_LLM_ENABLE_SUBJECT_OBJECT_VOCAB_COUNT_FILTER_ONLY:-1}"
export AURORAIG_LLM_ENABLE_NON_NEIGHBOR_RULE_FILTER="${AURORAIG_LLM_ENABLE_NON_NEIGHBOR_RULE_FILTER:-1}"
export AURORAIG_LLM_ENABLE_NEIGHBOR_RULE_FILTER="${AURORAIG_LLM_ENABLE_NEIGHBOR_RULE_FILTER:-1}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[INFO] project_root: $PROJECT_ROOT"
echo "[INFO] model_path: $AURORAIG_LLM_MODEL_PATH"
echo "[INFO] json_root: $JSON_ROOT"
echo "[INFO] out_root: $OUT_ROOT"
echo "[INFO] image_root: $IMAGE_ROOT"
echo "[INFO] max_llm_negatives: $MAX_LLM_NEGATIVES"
echo "[INFO] max_rule_negatives: $MAX_RULE_NEGATIVES"
echo "[INFO] min_pairs: $MIN_PAIRS"
echo "[INFO] python_bin: $PYTHON_BIN"
echo "[INFO] yolo: $YOLO_MODEL_PATH conf=$YOLO_CONF device=$YOLO_DEVICE"
echo "[INFO] 4bit: $AURORAIG_LLM_LOAD_IN_4BIT ($AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE)"
echo "[INFO] deterministic: temperature=$AURORAIG_LLM_TEMPERATURE top_p=$AURORAIG_LLM_TOP_P"
echo "[INFO] max_new_tokens: $AURORAIG_LLM_MAX_NEW_TOKENS"
echo "[INFO] offload_buffers: $AURORAIG_LLM_OFFLOAD_BUFFERS"
echo "[INFO] filters: spatial_only=$AURORAIG_LLM_ENABLE_SPATIAL_VOCAB_FILTER_ONLY subject_object_count=$AURORAIG_LLM_ENABLE_SUBJECT_OBJECT_VOCAB_COUNT_FILTER_ONLY non_neighbor=$AURORAIG_LLM_ENABLE_NON_NEIGHBOR_RULE_FILTER neighbor=$AURORAIG_LLM_ENABLE_NEIGHBOR_RULE_FILTER"
echo "[INFO] PYTHONPATH includes: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" scripts/build_consistency_pairs.py \
  --reconvla_json "$TRAIN_JSON" \
  --output_jsonl "$TRAIN_RAW_OUT" \
  --enable_yolo_neighbors \
  --image_root "$IMAGE_ROOT" \
  --yolo_model_path "$YOLO_MODEL_PATH" \
  --yolo_conf "$YOLO_CONF" \
  --yolo_device "$YOLO_DEVICE" \
  --max_llm_negatives "$MAX_LLM_NEGATIVES" \
  --max_rule_negatives "$MAX_RULE_NEGATIVES" \
  --min_pairs "$MIN_PAIRS" \
  --disable_rule_fallback_when_llm_empty

"$PYTHON_BIN" scripts/build_consistency_pairs.py \
  --reconvla_json "$VAL_JSON" \
  --output_jsonl "$VAL_RAW_OUT" \
  --enable_yolo_neighbors \
  --image_root "$IMAGE_ROOT" \
  --yolo_model_path "$YOLO_MODEL_PATH" \
  --yolo_conf "$YOLO_CONF" \
  --yolo_device "$YOLO_DEVICE" \
  --max_llm_negatives "$MAX_LLM_NEGATIVES" \
  --max_rule_negatives "$MAX_RULE_NEGATIVES" \
  --min_pairs "$MIN_PAIRS" \
  --disable_rule_fallback_when_llm_empty

"$PYTHON_BIN" scripts/filter_consistency_pairs.py \
  --input_jsonl "$TRAIN_RAW_OUT" \
  --output_jsonl "$TRAIN_OUT" \
  --dropped_jsonl "$TRAIN_DROPPED_OUT" \
  --review_jsonl "$TRAIN_REVIEW_OUT" \
  --review_size 100

"$PYTHON_BIN" scripts/filter_consistency_pairs.py \
  --input_jsonl "$VAL_RAW_OUT" \
  --output_jsonl "$VAL_OUT" \
  --dropped_jsonl "$VAL_DROPPED_OUT" \
  --review_jsonl "$VAL_REVIEW_OUT" \
  --review_size 100

echo "[DONE] training: $TRAIN_OUT"
echo "[DONE] validation: $VAL_OUT"
echo "[DONE] training review sample: $TRAIN_REVIEW_OUT"
echo "[DONE] validation review sample: $VAL_REVIEW_OUT"
