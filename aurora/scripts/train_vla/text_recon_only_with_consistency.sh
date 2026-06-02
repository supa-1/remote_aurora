#!/usr/bin/env bash
set -euo pipefail

# Text reconstruction only, explicitly with consistency auxiliary loss.
# By default this uses the same checked-in local aux JSON as the no-consistency
# wrapper so paired ablations differ only in ENABLE_CONSISTENCY_AUX.
# DATA_PATH may point to any Reconvla training JSON precomputed with aux_* fields:
#   python scripts/build_precomputed_aux_data.py \
#     --input_json /path/to/reconvla_train.json \
#     --output_json /path/to/reconvla_train_with_aux.json \
#     --consistency_jsonl /path/to/training_llm_ranking.jsonl
#   DATA_PATH=/path/to/reconvla_train_with_aux.json bash scripts/train_vla/text_recon_only_with_consistency.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AURORAIG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export AURORAIG_AUX_DATA_PATH="${AURORAIG_AUX_DATA_PATH:-$AURORAIG_ROOT/data/final_json/auroraig_train_with_aux.json}"
export DATA_PATH="${DATA_PATH:-$AURORAIG_AUX_DATA_PATH}"
RESOLVED_DATA_PATH="$DATA_PATH"

if [[ -f "$RESOLVED_DATA_PATH" ]]; then
  if ! grep -q '"aux_fake_instruction_pool"' "$RESOLVED_DATA_PATH" || \
     ! grep -q '"aux_negative_type_pool"' "$RESOLVED_DATA_PATH"; then
    echo "[ERROR] DATA_PATH exists but does not contain aux_fake_instruction_pool/aux_negative_type_pool:" >&2
    echo "        $RESOLVED_DATA_PATH" >&2
    echo "        Build an aux training JSON first, for example:" >&2
    echo "        cd $AURORAIG_ROOT" >&2
    echo "        python scripts/build_precomputed_aux_data.py \\" >&2
    echo "          --input_json /path/to/reconvla_train.json \\" >&2
    echo "          --output_json /path/to/reconvla_train_with_aux.json \\" >&2
    echo "          --consistency_jsonl /path/to/training_llm_ranking.jsonl" >&2
    exit 1
  fi
fi

export ENABLE_CONSISTENCY_AUX="True"
export CONSISTENCY_USE_PAIR_WEIGHTS="${CONSISTENCY_USE_PAIR_WEIGHTS:-True}"
export CONSISTENCY_AUX_WEIGHT="${CONSISTENCY_AUX_WEIGHT:-0.3}"
export CONSISTENCY_MARGIN="${CONSISTENCY_MARGIN:-0.2}"
export CONSISTENCY_ALPHA="${CONSISTENCY_ALPHA:-0.4}"
export CONSISTENCY_BETA="${CONSISTENCY_BETA:-0.3}"
export CONSISTENCY_GAMMA="${CONSISTENCY_GAMMA:-0.3}"
export CONSISTENCY_MAX_LENGTH="${CONSISTENCY_MAX_LENGTH:-128}"

exec bash "$SCRIPT_DIR/text_recon_only.sh"
