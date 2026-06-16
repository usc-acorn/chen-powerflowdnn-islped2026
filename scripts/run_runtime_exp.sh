#!/usr/bin/env bash
set -euo pipefail

# Generate the V3-V6 voltage-sweep JSON inputs used by the oracle/ILP
# comparison experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/../compilers/generate_vdd_json.py"
OUT_DIR="$SCRIPT_DIR/../data/runtime_exp"

mkdir -p "$OUT_DIR"

generate_set() {
  local suffix="$1"
  local volts="$2"

  echo "Generating $suffix with voltages: $volts"

  python3 "$GEN" --model squeezenet --v-sys "$volts" --v-rram "$volts" --v-feeder "$volts" \
    --output "$OUT_DIR/squeezenet_${suffix}.json"

  python3 "$GEN" --model resnet18 --v-sys "$volts" --v-rram "$volts" --v-feeder "$volts" \
    --output "$OUT_DIR/resnet18_${suffix}.json"

  python3 "$GEN" --model mobilenetv3_small --v-sys "$volts" --v-rram "$volts" --v-feeder "$volts" \
    --output "$OUT_DIR/mobilenetv3_small_${suffix}.json"

  python3 "$GEN" --model mobilevit_xxs --v-sys "$volts" --v-rram "$volts" --v-feeder "$volts" \
    --output "$OUT_DIR/mobilevit_xxs_${suffix}.json"
}

echo "Running exp5 granularity VDD sweep generation..."
echo "Output directory: $OUT_DIR"

generate_set "V3" "0.9,1.1,1.3"
generate_set "V4" "0.9,1.0,1.1,1.2,1.3"
generate_set "V5" "0.9,1.0,1.1,1.2,1.25,1.3"
# generate_set "V6" "0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3"

echo "Done. JSON files written to: $OUT_DIR"

python3 "$SCRIPT_DIR/../figures/eval_dp_ilp.py"
python3 "$SCRIPT_DIR/../figures/plot_compile_time_recomputed.py"
