#!/usr/bin/env bash
set -euo pipefail

# One-shot runner: build LLM-based true/false instruction pairs for
# both training and validation splits from processed CALVIN JSON.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH_DEFAULT="$PROJECT_ROOT/models/qwen-8b"
# Run this script from the server's aurora environment. Override PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/train_vla/hpc_env.sh"
setup_hpc_env "${CONDA_ENV:-aurora}"

MODEL_PATH="${AURORAIG_LLM_MODEL_PATH:-$MODEL_PATH_DEFAULT}"
DATASET_NAME="${DATASET_NAME:-calvin_debug_dataset}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../calvin/dataset/process/$DATASET_NAME}"
JSON_ROOT="${1:-$DATA_ROOT/processed_json}"
OUT_ROOT="${2:-$DATA_ROOT/consistency_pairs}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT/processed_images/vla_processed_r5}"
MAX_LLM_NEGATIVES="${MAX_LLM_NEGATIVES:-6}"
MAX_RULE_NEGATIVES="${MAX_RULE_NEGATIVES:-2}"
MIN_PAIRS="${MIN_PAIRS:-100}"
DISABLE_RULE_FALLBACK_WHEN_LLM_EMPTY="${DISABLE_RULE_FALLBACK_WHEN_LLM_EMPTY:-0}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$PROJECT_ROOT/../ReconVLA/reconvla/scripts/helper/best.pt}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
ENABLE_YOLO_NEIGHBORS="${ENABLE_YOLO_NEIGHBORS:-1}"

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

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] python not executable: $PYTHON_BIN" >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] python not found on PATH: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

export AURORAIG_LLM_MODEL_PATH="$MODEL_PATH"
export AURORAIG_LLM_FAMILY="${AURORAIG_LLM_FAMILY:-qwen3}"
export AURORAIG_LLM_LOAD_IN_4BIT="${AURORAIG_LLM_LOAD_IN_4BIT:-0}"
if [[ "$AURORAIG_LLM_LOAD_IN_4BIT" == "1" || "$AURORAIG_LLM_LOAD_IN_4BIT" == "true" || "$AURORAIG_LLM_LOAD_IN_4BIT" == "True" ]]; then
  export AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE="${AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
fi
export AURORAIG_LLM_OFFLOAD_BUFFERS="${AURORAIG_LLM_OFFLOAD_BUFFERS:-0}"
export AURORAIG_LLM_TEMPERATURE="${AURORAIG_LLM_TEMPERATURE:-0}"
export AURORAIG_LLM_TOP_P="${AURORAIG_LLM_TOP_P:-1.0}"
export AURORAIG_LLM_MAX_NEW_TOKENS="${AURORAIG_LLM_MAX_NEW_TOKENS:-128}"
export AURORAIG_LLM_ATTN_IMPLEMENTATION="${AURORAIG_LLM_ATTN_IMPLEMENTATION:-eager}"
export AURORAIG_LLM_DISABLE_THINKING="${AURORAIG_LLM_DISABLE_THINKING:-1}"
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
echo "[INFO] yolo: enabled=$ENABLE_YOLO_NEIGHBORS path=$YOLO_MODEL_PATH conf=$YOLO_CONF device=$YOLO_DEVICE"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] 4bit: $AURORAIG_LLM_LOAD_IN_4BIT (${AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE:-disabled})"
echo "[INFO] deterministic: temperature=$AURORAIG_LLM_TEMPERATURE top_p=$AURORAIG_LLM_TOP_P"
echo "[INFO] max_new_tokens: $AURORAIG_LLM_MAX_NEW_TOKENS"
echo "[INFO] llm_attn: $AURORAIG_LLM_ATTN_IMPLEMENTATION"
echo "[INFO] disable_thinking: $AURORAIG_LLM_DISABLE_THINKING"
echo "[INFO] offload_buffers: $AURORAIG_LLM_OFFLOAD_BUFFERS"
echo "[INFO] filters: spatial_only=$AURORAIG_LLM_ENABLE_SPATIAL_VOCAB_FILTER_ONLY subject_object_count=$AURORAIG_LLM_ENABLE_SUBJECT_OBJECT_VOCAB_COUNT_FILTER_ONLY non_neighbor=$AURORAIG_LLM_ENABLE_NON_NEIGHBOR_RULE_FILTER neighbor=$AURORAIG_LLM_ENABLE_NEIGHBOR_RULE_FILTER"
echo "[INFO] PYTHONPATH includes: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

if [[ "${SKIP_LLM_SMOKE_TEST:-0}" != "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
from auroraig.interfaces.llm_client import RewriteRequest, resolve_default_llm_client

client = resolve_default_llm_client()
request = RewriteRequest(
    instruction="sweep the pink block to the right",
    n=3,
    object_candidates=["drawer", "slider"],
)
rewrites = client.rewrite_instruction_with_types(request)
print("[INFO] LLM smoke rewrites:", [(x.text, x.negative_type) for x in rewrites])
bad_markers = (
    "<think",
    "</think",
    "okay",
    "let's",
    "first,",
    "the user",
    "i need",
    "let me",
    "the task",
    "the instruction",
)
bad_rewrites = [
    x.text
    for x in rewrites
    if any(marker in x.text.lower() for marker in bad_markers)
]
if bad_rewrites:
    raise SystemExit(
        "LLM smoke test returned reasoning/explanatory text instead of rewritten instructions: "
        + repr(bad_rewrites[:3])
    )
if not rewrites:
    print("[INFO] LLM smoke diagnostics:")
    print("  client:", type(client).__name__)
    print("  enabled:", getattr(client, "enabled", lambda: False)())
    print("  model_path:", getattr(client, "model_name_or_path", ""))
    print("  model_family:", getattr(client, "model_family", ""))
    print("  device:", getattr(client, "device", ""))
    try:
        loaded = client._lazy_load()
        print("  lazy_load:", loaded)
    except Exception as exc:
        print("  lazy_load_error:", type(exc).__name__, exc)
        loaded = False
    if loaded:
        prompts = client._build_rule_prompts_with_types(request)[:6]
        for rule_type, prompt in prompts:
            try:
                raw = client._generate_text(prompt)
                extracted = client._extract_rewrite_lines(raw)
                normalized = [
                    client._normalize_rewrite(x, request.instruction)
                    for x in extracted
                ]
                valid = [
                    x for x in normalized
                    if x and client._is_valid_for_rule(rule_type, request.instruction, x, ["drawer", "slider"])
                ]
                print(f"  rule={rule_type}")
                print(f"    raw={raw!r}")
                print(f"    extracted={extracted!r}")
                print(f"    normalized={normalized!r}")
                print(f"    valid={valid!r}")
            except Exception as exc:
                print(f"  rule={rule_type} error={type(exc).__name__}: {exc}")
    raise SystemExit(
        "LLM smoke test returned no rewrites. Check Qwen path/family, CUDA memory, or filters before full generation."
    )
PY
fi

RULE_FALLBACK_FLAG=()
if [[ "$DISABLE_RULE_FALLBACK_WHEN_LLM_EMPTY" == "1" || "$DISABLE_RULE_FALLBACK_WHEN_LLM_EMPTY" == "true" || "$DISABLE_RULE_FALLBACK_WHEN_LLM_EMPTY" == "True" ]]; then
  RULE_FALLBACK_FLAG=(--disable_rule_fallback_when_llm_empty)
fi

YOLO_ARGS=()
if [[ "$ENABLE_YOLO_NEIGHBORS" == "1" || "$ENABLE_YOLO_NEIGHBORS" == "true" || "$ENABLE_YOLO_NEIGHBORS" == "True" ]]; then
  YOLO_ARGS=(
    --enable_yolo_neighbors
    --image_root "$IMAGE_ROOT"
    --yolo_model_path "$YOLO_MODEL_PATH"
    --yolo_conf "$YOLO_CONF"
    --yolo_device "$YOLO_DEVICE"
  )
fi

"$PYTHON_BIN" scripts/build_consistency_pairs.py \
  --reconvla_json "$TRAIN_JSON" \
  --output_jsonl "$TRAIN_RAW_OUT" \
  --max_llm_negatives "$MAX_LLM_NEGATIVES" \
  --max_rule_negatives "$MAX_RULE_NEGATIVES" \
  --min_pairs "$MIN_PAIRS" \
  "${YOLO_ARGS[@]}" \
  "${RULE_FALLBACK_FLAG[@]}"

"$PYTHON_BIN" scripts/build_consistency_pairs.py \
  --reconvla_json "$VAL_JSON" \
  --output_jsonl "$VAL_RAW_OUT" \
  --max_llm_negatives "$MAX_LLM_NEGATIVES" \
  --max_rule_negatives "$MAX_RULE_NEGATIVES" \
  --min_pairs "$MIN_PAIRS" \
  "${YOLO_ARGS[@]}" \
  "${RULE_FALLBACK_FLAG[@]}"

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
