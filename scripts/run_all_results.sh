#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/../compilers/run_compiler_sweeps.py" --mode fps --models squeezenet 
python3 "$SCRIPT_DIR/../compilers/run_compiler_sweeps.py" --mode fps --models resnet --fps-start 7 --fps-stop 14
python3 "$SCRIPT_DIR/../compilers/run_compiler_sweeps.py" --mode fps --models mobilevit --fps-start 7 --fps-stop 15
python3 "$SCRIPT_DIR/../compilers/run_compiler_sweeps.py" --mode fps --models mobilenet --fps-start 56 --fps-stop 113
python3 "$SCRIPT_DIR/../compilers/run_compiler_sweeps.py" --mode etran
