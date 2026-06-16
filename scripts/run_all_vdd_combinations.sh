#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_vdd_combinations.sh
#   ./run_all_vdd_combinations.sh "0.9,1.0,1.1" "0.9,1.0,1.1" "0.9,1.0,1.1"
#
# Args:
#   $1 -> v_sys list (comma-separated)
#   $2 -> v_rram list (comma-separated)
#   $3 -> v_feeder list (comma-separated)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/../compilers/generate_vdd_json.py"

V_SYS="${1:-0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3}"
V_RRAM="${2:-0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3}"
V_FEEDER="${3:-0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3}"

echo "Running VDD sweep generation for all models..."
echo "v_sys    = $V_SYS"
echo "v_rram   = $V_RRAM"
echo "v_feeder = $V_FEEDER"

python3 "$GEN" --model squeezenet --v-sys "$V_SYS" --v-rram "$V_RRAM" --v-feeder "$V_FEEDER" \
  --output "$SCRIPT_DIR/../data/squeezenet_vdd_combinations.json"

python3 "$GEN" --model resnet18 --v-sys "$V_SYS" --v-rram "$V_RRAM" --v-feeder "$V_FEEDER" \
  --output "$SCRIPT_DIR/../data/resnet18_vdd_combinations.json"

python3 "$GEN" --model mobilenetv3_small --v-sys "$V_SYS" --v-rram "$V_RRAM" --v-feeder "$V_FEEDER" \
  --output "$SCRIPT_DIR/../data/mobilenetv3_small_vdd_combinations.json"

python3 "$GEN" --model mobilevit_xxs --v-sys "$V_SYS" --v-rram "$V_RRAM" --v-feeder "$V_FEEDER" \
  --output "$SCRIPT_DIR/../data/mobilevit_xxs_vdd_combinations.json"

echo "Done. JSON files written to: $SCRIPT_DIR"
