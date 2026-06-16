#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG_DIR="$SCRIPT_DIR/../figures"

# Motivation figures.
python3 "$FIG_DIR/squeezenet_no_pg_3layer_breakdown.py" --layers 0,12,16
python3 "$FIG_DIR/squeezenet_power_state_scatter.py" --layer-idxs 0,12,16

# Timeline figure.
python3 "$FIG_DIR/plot_squeezenet_dp_vs_nominal_timeline.py"

# Main evaluation figures.
python3 "$FIG_DIR/plot_interval_energy_vs_fps.py"
python3 "$FIG_DIR/plot_squeezenet_sorted_marginal_utility.py"
python3 "$FIG_DIR/plot_multi_model.py"
python3 "$FIG_DIR/plot_energy_vs_rails.py"

# Additional sensitivity/diagnostic plots.
python3 "$FIG_DIR/plot_transition_energy_sensitivity.py"
