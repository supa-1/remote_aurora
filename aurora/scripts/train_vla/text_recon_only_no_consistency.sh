#!/usr/bin/env bash
set -euo pipefail

# Text reconstruction only, explicitly without consistency auxiliary loss.
# Uses the same aux JSON default as the with-consistency wrapper so paired
# ablations differ only in ENABLE_CONSISTENCY_AUX.
# Usage:
#   bash scripts/train_vla/text_recon_only_no_consistency.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AURORAIG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export AURORAIG_AUX_DATA_PATH="${AURORAIG_AUX_DATA_PATH:-$AURORAIG_ROOT/data/final_json/auroraig_train_with_aux.json}"
export DATA_PATH="${DATA_PATH:-$AURORAIG_AUX_DATA_PATH}"

export ENABLE_CONSISTENCY_AUX="False"

exec bash "$SCRIPT_DIR/text_recon_only.sh"
