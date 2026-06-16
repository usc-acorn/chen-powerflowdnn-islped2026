#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Quick test for model..."
python3 "$SCRIPT_DIR/../networks/run_one_layer.py"
echo "Quick test for model done!"

echo "Quick test for figure 1 and 2..."
python3 "$SCRIPT_DIR/../figures/squeezenet_no_pg_3layer_breakdown.py" --layers 0,12,16
python3 "$SCRIPT_DIR/../figures/squeezenet_power_state_scatter.py" --layer-idxs 0,12,16
echo "Quick test for figure 1 and 2 done!"

