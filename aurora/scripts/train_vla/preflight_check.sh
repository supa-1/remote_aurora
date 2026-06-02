#!/usr/bin/env bash
set -euo pipefail

AURORAIG_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RECONVLA_ROOT="${RECONVLA_ROOT:-$AURORAIG_ROOT/reconvla}"
ASSET_ROOT="${ASSET_ROOT:-$AURORAIG_ROOT/../ReconVLA/reconvla}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ASSET_ROOT/checkpoints/pretrain-checkpoint-10388}"
VISION_TOWER="${VISION_TOWER:-$ASSET_ROOT/siglip-so400m-patch14-384}"
MM_PIXEL_DECODER="${MM_PIXEL_DECODER:-$ASSET_ROOT/pretrained_vae/vae}"
DATA_PATH="${DATA_PATH:-$HOME/myreconvla/calvin/dataset/calvin_debug_dataset_processed_json/training_r5.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$HOME/myreconvla/calvin/dataset/processed_images/calvin_debug_dataset/vla_processed_r5}"
TARGET_IMAGE_FOLDER="${TARGET_IMAGE_FOLDER:-$HOME/myreconvla/calvin/dataset/processed_images/calvin_debug_dataset/vla_processed_r5}"
ACTION_STAT="${ACTION_STAT:-$RECONVLA_ROOT/statistics.yaml}"

check_path() {
  local p="$1"
  local name="$2"
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] Missing ${name}: ${p}"
    exit 1
  fi
  echo "[OK] ${name}: ${p}"
}

check_path "$RECONVLA_ROOT/train_vla.py" "train_vla.py"
check_path "$MODEL_NAME_OR_PATH" "MODEL_NAME_OR_PATH"
check_path "$VISION_TOWER" "VISION_TOWER"
check_path "$MM_PIXEL_DECODER" "MM_PIXEL_DECODER"
check_path "$DATA_PATH" "DATA_PATH"
check_path "$IMAGE_FOLDER" "IMAGE_FOLDER"
check_path "$TARGET_IMAGE_FOLDER" "TARGET_IMAGE_FOLDER"
check_path "$ACTION_STAT" "ACTION_STAT"

echo "[INFO] ENABLE_TEXT_RECONSTRUCTION=${ENABLE_TEXT_RECONSTRUCTION:-False}"
echo "[INFO] TEXT_RECONSTRUCTION_WEIGHT=${TEXT_RECONSTRUCTION_WEIGHT:-0.3}"
echo "[INFO] ENABLE_CONSISTENCY_AUX=${ENABLE_CONSISTENCY_AUX:-False}"
echo "[INFO] CONSISTENCY_AUX_WEIGHT=${CONSISTENCY_AUX_WEIGHT:-0.3}"

echo "Preflight passed."
