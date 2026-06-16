from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

def _short_layer_name(layer_idx: int, name: str) -> str:
    name = name.replace("features.", "F")
    name = name.replace(".squeeze", "-squeeze")
    name = name.replace(".expand1x1", "-exp1x1")
    name = name.replace(".expand3x3", "-exp3x3")
    return f"L{layer_idx}:{name}"

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
    }
)

# Allow imports from parent Experiment directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "model"),
    os.path.join(_PROJECT_ROOT, "networks"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feeder_sa_cycles_model import FeederCfg, model_layer, static_v_scale, LEAKAGE_MODE_NO_PG
from energy_config import get_energy_params, get_bw_params, get_dvfs_params
from squeezenet_run import build_layers, conv_layers_from_spec


def _build_layer(layer_idx: int) -> Tuple[str, FeederCfg, int]:
    layers = conv_layers_from_spec(build_layers())
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer-idx out of range: {layer_idx}, valid [0, {len(layers)-1}]")
    name, cfg = layers[layer_idx]
    in_shape = cfg["input_shape"]
    w_shape = cfg["weight_shape"]
    stride = cfg.get("stride", 1)
    feeder = FeederCfg(
        ifmap_w=in_shape[2],
        ifmap_h=in_shape[3],
        ifmap_c=w_shape[1],
        ker_size=w_shape[2],
        word_w=8,
        stride=stride,
        num_lanes=8,
    )
    return name, feeder, w_shape[0]


def _domain_breakdown_uj(rep: Dict[str, Any], e, dvfs) -> Dict[str, Dict[str, float]]:
    b = rep["energy_j"]["buckets"]
    t_layer = rep["times_s"]["t_layer"]

    s_sys_static = static_v_scale(dvfs.volt_sys_v, e.vref)
    s_rram_static = static_v_scale(dvfs.volt_rram_v, e.vref)
    s_feeder_static = static_v_scale(dvfs.volt_feeder_v, e.vref)

    # Dynamic buckets
    e_weight_dyn = (
        b["E_dyn_rram_store"]
        + b["E_dyn_rram_read"]
        + b["E_dyn_w_tile_store"]
        + b["E_dyn_w_tile_read"]
        + b["E_dyn_weight_dma_ctrl"]
        + b.get("E_dyn_pg_ctrl", 0.0)
    )
    e_ifmap_dyn = (
        b["E_dyn_spad_store"]
        + b["E_dyn_spad_read"]
        + b["E_dyn_lane_store"]
        + b["E_dyn_lane_read"]
        + b["E_dyn_feeder_ctrl"]
    )
    e_compute_dyn = b["E_dyn_pe"] + b["E_dyn_sa_stagger"]

    # Leakage/idle buckets
    e_weight_leak = (e.p_idle_weight_domain_w * s_rram_static + b["P_rram_leak_w"]) * t_layer
    e_ifmap_leak = (e.p_idle_feeder_w * s_feeder_static + b["P_sram_leak_w"] * s_feeder_static) * t_layer
    e_compute_leak = (e.p_idle_sa_w * s_sys_static) * t_layer

    return {
        "Weight side": {"dynamic": e_weight_dyn * 1e6, "leakage": e_weight_leak * 1e6},
        "IFMAP side": {"dynamic": e_ifmap_dyn * 1e6, "leakage": e_ifmap_leak * 1e6},
        "Compute": {"dynamic": e_compute_dyn * 1e6, "leakage": e_compute_leak * 1e6},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="No-PG grouped energy breakdown for 3 SqueezeNet layers at nominal DVFS."
    )
    ap.add_argument(
        "--layers",
        default="0,1,2",
        help="Three conv layer indices, comma-separated (default: 0,1,2).",
    )
    ap.add_argument("--out", default="./outputs/figure1.pdf", help="Output plot path.")
    ap.add_argument("--show", action="store_true", help="Show interactive plot.")
    args = ap.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if len(layer_ids) != 3:
        raise ValueError("Please provide exactly 3 layer indices via --layers (e.g., 0,1,2).")

    e = get_energy_params()
    bw = get_bw_params()
    dvfs = get_dvfs_params()  # nominal point

    domains = ["Weight side", "IFMAP side", "Compute"]
    # domain_colors = {"Weight side": "#f58518", "IFMAP side": "#54a24b", "Compute": "#4c78a8"}
    domain_colors = {
        "Weight side": "#2f3e1f",  # deep olive
        "IFMAP side": "#6b8e23",   # olive drab
        "Compute": "#c3d86d",      # light olive
    }
    dyn_vals: Dict[str, List[float]] = {d: [] for d in domains}
    leak_vals: Dict[str, List[float]] = {d: [] for d in domains}
    layer_labels: List[str] = []

    for idx in layer_ids:
        name, feeder, channel_out = _build_layer(idx)
        rep = model_layer(
            feeder=feeder,
            channel_out=channel_out,
            dvfs=dvfs,
            bw=bw,
            e=e,
            leakage_mode=LEAKAGE_MODE_NO_PG,
        )
        parts = _domain_breakdown_uj(rep, e=e, dvfs=dvfs)
        layer_label = _short_layer_name(idx, name)
        layer_labels.append(layer_label)

        # print(f"{layer_label}")
        for domain in domains:
            vd = parts[domain]["dynamic"]
            vl = parts[domain]["leakage"]
            vt = vd + vl
            # print(f"  {domain:12s} total={vt:.6e} uJ  dyn={vd:.6e}  leak={vl:.6e}")
            dyn_vals[domain].append(vd)
            leak_vals[domain].append(vl)

    fig, ax = plt.subplots(figsize=(3.5, 1.1), dpi=220)
    x = list(range(len(layer_labels)))
    width = 0.2
    offsets = {"Weight side": -width, "IFMAP side": 0.0, "Compute": width}

    for d in domains:
        xpos = [xi + offsets[d] for xi in x]
        ax.bar(
            xpos,
            dyn_vals[d],
            width=width,
            color=domain_colors[d],
            edgecolor="black",
            linewidth=0.4,
            label=f"{d} (dynamic)",
            zorder=3,
        )
        ax.bar(
            xpos,
            leak_vals[d],
            width=width,
            bottom=dyn_vals[d],
            color=domain_colors[d],
            edgecolor="black",
            linewidth=0.4,
            hatch="///",
            alpha=0.35,
            label=f"{d} (leakage)",
            zorder=3,
        )

    ax.set_xlabel("Layer Name", labelpad=1)
    ax.set_ylabel(r"Energy ($\mu$J)", labelpad=0)
    ax.set_title("SqueezeNet Energy Breakdown at Nominal Voltage", fontweight="bold", pad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=0, ha="center")

    # 3x2 legend: rows are domains, columns are dynamic/leakage.
    dyn_handles = []
    dyn_labels = []
    leak_handles = []
    leak_labels = []

    for d in domains:
        dyn_handles.append(Patch(facecolor=domain_colors[d], edgecolor="black", linewidth=0.4))
        dyn_labels.append(f"{d.replace(' side', '')} dyn.")
        leak_handles.append(Patch(facecolor=domain_colors[d], edgecolor="black", linewidth=0.4, hatch="///", alpha=0.35))
        leak_labels.append(f"{d.replace(' side', '')} leak.")

    legend_handles = dyn_handles + leak_handles
    legend_labels = dyn_labels + leak_labels
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=2,
        frameon=True,
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        columnspacing=0.5,
        handlelength=1.0,
        handletextpad=0.2,
        borderpad=0,
        labelspacing=0.2,
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", pad=0, width=0.5, labelsize=8)
    ax.tick_params(axis="y", pad=0, width=0.5, labelsize=8)
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    fig.subplots_adjust(left=0.09, right=1, bottom=0.2, top=0.9)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, pad_inches=0, transparent=False, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
