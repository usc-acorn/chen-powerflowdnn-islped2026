import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "model"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feeder_sa_cycles_model import FeederCfg, model_layer
from energy_config import get_energy_params, get_bw_params, get_dvfs_params

def main():
    # ex: 224x224x64, 3x3 stride 1, Cout 128
    feeder = FeederCfg(
        ifmap_w=224,
        ifmap_h=224,
        ifmap_c=64,
        ker_size=3,
        word_w=8,
        stride=1,
        num_lanes=8,
    )

    dvfs = get_dvfs_params()
    bw = get_bw_params()
    e = get_energy_params()

    rep = model_layer(
        feeder=feeder,
        channel_out=128,
        dvfs=dvfs,
        bw=bw,
        e=e,
        act_bits=8,
        weight_bits=8,
        out_bits=32,
        C=8,
        overlap_weight_fetch=True,
        verbose=True,
        per_layer_pg=True,
    )

    print("=== Layer 1 ===")
    print("derived:", rep["derived"])
    print("trace_counts:", rep["trace_counts"])
    print("cycles:", rep["cycles"])
    print("times_s:", rep["times_s"])
    print("energy_j:", rep["energy_j"])
    print("perf:", rep["perf"])

    # Second layer config: 11x11x16, 3x3 kernel, stride 1, Cout 16
    feeder2 = FeederCfg(
        ifmap_w=11,
        ifmap_h=11,
        ifmap_c=16,
        ker_size=3,
        word_w=8,
        stride=1,
        num_lanes=8,
    )

    rep2 = model_layer(
        feeder=feeder2,
        channel_out=16,
        dvfs=dvfs,
        bw=bw,
        e=e,
        act_bits=8,
        weight_bits=8,
        out_bits=32,
        C=8,
        overlap_weight_fetch=True,
        verbose=True,
        per_layer_pg=True,
    )

    print("=== Layer 2 ===")
    print("derived:", rep2["derived"])
    print("trace_counts:", rep2["trace_counts"])
    print("cycles:", rep2["cycles"])
    print("times_s:", rep2["times_s"])
    print("energy_j:", rep2["energy_j"])
    print("perf:", rep2["perf"])

if __name__ == "__main__":
    main()
