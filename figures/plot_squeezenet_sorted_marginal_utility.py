from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, rcParams
from matplotlib.ticker import MaxNLocator

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "model"),
    os.path.join(_PROJECT_ROOT, "compilers"),
    os.path.join(_PROJECT_ROOT, "networks")
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from energy_config import get_bw_params, get_dvfs_params, get_energy_params
from feeder_sa_cycles_model import DVFS, FeederCfg, LEAKAGE_MODE_NO_PG, model_layer
from squeezenet_run import build_layers, conv_layers_from_spec
from vf_model import VFModel


@dataclass(frozen=True)
class LayerSpec:
    name: str
    feeder: FeederCfg
    channel_out: int


@dataclass(frozen=True)
class LayerRow:
    layer_idx: int
    layer_name: str
    layer_type: str
    v_sys: float
    v_rram: float
    v_feeder: float
    energy_nominal_ungated_j: float
    energy_selected_j: float
    delta_e_j: float
    delta_e_pct: float


def _build_squeezenet_layers() -> List[LayerSpec]:
    out: List[LayerSpec] = []
    for name, cfg in conv_layers_from_spec(build_layers()):
        in_shape = cfg["input_shape"]
        w_shape = cfg["weight_shape"]
        stride = cfg.get("stride", 1)
        padding = cfg.get("padding", 0)
        feeder = FeederCfg(
            ifmap_w=in_shape[2],
            ifmap_h=in_shape[3],
            ifmap_c=w_shape[1],
            ker_size=w_shape[2],
            word_w=8,
            stride=stride,
            padding=padding,
            num_lanes=8,
        )
        out.append(LayerSpec(name=name, feeder=feeder, channel_out=w_shape[0]))
    return out


def _infer_layer_type(layer_name: str) -> str:
    if layer_name == "features.0":
        return "stem"
    if ".squeeze" in layer_name:
        return "squeeze"
    if "expand1x1" in layer_name:
        return "exp1x1"
    if "expand3x3" in layer_name:
        return "exp3x3"
    if "classifier" in layer_name:
        return "classifier"
    return "other"


def _mk_dvfs(v_sys: float, v_rram: float, v_feeder: float, vf_sys: VFModel, vf_rram: VFModel, vf_feeder: VFModel) -> DVFS:
    return DVFS(
        freq_sys_hz=vf_sys.f_hz(v_sys),
        volt_sys_v=v_sys,
        freq_rram_hz=vf_rram.f_hz(v_rram),
        volt_rram_v=v_rram,
        freq_feeder_hz=vf_feeder.f_hz(v_feeder),
        volt_feeder_v=v_feeder,
    )


def _load_selected_path_rows(compiler_csv: str, fps: int) -> Dict[int, Dict[str, float]]:
    with open(compiler_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        selected_row = None
        for row in reader:
            if abs(float(row["fps"]) - float(fps)) < 1e-9:
                selected_row = row
                break
    if selected_row is None:
        raise ValueError(f"FPS {fps} not found in compiler CSV: {compiler_csv}")
    selected_path = json.loads(selected_row["selected_path_json"])
    selected: Dict[int, Dict[str, float]] = {}
    for item in selected_path:
        selected[int(item["layer_idx"])] = item
    return selected


def _compute_layer_rows(
    layers: List[LayerSpec],
    selected_path_rows: Dict[int, Dict[str, float]],
    nominal_v: float,
) -> List[LayerRow]:
    if len(selected_path_rows) != len(layers):
        raise ValueError(
            f"Expected {len(layers)} selected layers, got {len(selected_path_rows)}."
        )

    e = get_energy_params()
    bw = get_bw_params()
    dvfs_ref = get_dvfs_params()
    vf_sys = VFModel(v_ref=dvfs_ref.volt_sys_v, f_ref_hz=dvfs_ref.freq_sys_hz)
    vf_rram = VFModel(v_ref=dvfs_ref.volt_rram_v, f_ref_hz=dvfs_ref.freq_rram_hz)
    vf_feeder = VFModel(v_ref=dvfs_ref.volt_feeder_v, f_ref_hz=dvfs_ref.freq_feeder_hz)

    nominal_dvfs = _mk_dvfs(nominal_v, nominal_v, nominal_v, vf_sys, vf_rram, vf_feeder)
    rows: List[LayerRow] = []
    for layer_idx, layer in enumerate(layers):
        sel = selected_path_rows[layer_idx]

        nominal_rep = model_layer(
            feeder=layer.feeder,
            channel_out=layer.channel_out,
            dvfs=nominal_dvfs,
            bw=bw,
            e=e,
            leakage_mode=LEAKAGE_MODE_NO_PG,
        )

        energy_nominal = float(nominal_rep["energy_j"]["E_total"])
        energy_selected = float(sel["energy_j"])
        delta_e = energy_nominal - energy_selected
        delta_pct = 100.0 * delta_e / energy_nominal if energy_nominal > 0.0 else 0.0
        rows.append(
            LayerRow(
                layer_idx=layer_idx,
                layer_name=layer.name,
                layer_type=_infer_layer_type(layer.name),
                v_sys=float(sel["v_sys"]),
                v_rram=float(sel["v_rram"]),
                v_feeder=float(sel["v_feeder"]),
                energy_nominal_ungated_j=energy_nominal,
                energy_selected_j=energy_selected,
                delta_e_j=delta_e,
                delta_e_pct=delta_pct,
            )
        )
    return rows


def _sort_rows(rows: List[LayerRow], sort_by: str) -> List[LayerRow]:
    if sort_by == "delta_e":
        key_fn = lambda row: (row.delta_e_j, row.delta_e_pct)
    elif sort_by == "delta_e_pct":
        key_fn = lambda row: (row.delta_e_pct, row.delta_e_j)
    else:
        raise ValueError(f"Unsupported sort key: {sort_by}")
    return sorted(rows, key=key_fn, reverse=True)


def _type_palette() -> Dict[str, str]:
    return {
        "stem": "#00466b",
        "squeeze": "#006f92",
        "exp1x1": "#009ea3",
        "exp3x3": "#7ccfc3",
        "classifier": "#b9ddd0",
        "other": "#5f8f95",
    }


def _write_rows_csv(out_csv: str, rows: List[LayerRow], fps: int, sort_by: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fps",
                "sort_by",
                "sorted_rank",
                "layer_idx",
                "layer_name",
                "layer_type",
                "v_sys",
                "v_rram",
                "v_feeder",
                "energy_nominal_ungated_j",
                "energy_selected_j",
                "delta_e_j",
                "delta_e_pct",
                "cumulative_delta_e_j",
                "cumulative_delta_e_pct_of_total",
            ]
        )
        cumulative = 0.0
        total = sum(row.delta_e_j for row in rows)
        for rank, row in enumerate(rows, start=1):
            cumulative += row.delta_e_j
            cumulative_pct = 100.0 * cumulative / total if total != 0.0 else 0.0
            writer.writerow(
                [
                    fps,
                    sort_by,
                    rank,
                    row.layer_idx,
                    row.layer_name,
                    row.layer_type,
                    f"{row.v_sys:.3f}",
                    f"{row.v_rram:.3f}",
                    f"{row.v_feeder:.3f}",
                    f"{row.energy_nominal_ungated_j:.12e}",
                    f"{row.energy_selected_j:.12e}",
                    f"{row.delta_e_j:.12e}",
                    f"{row.delta_e_pct:.6f}",
                    f"{cumulative:.12e}",
                    f"{cumulative_pct:.6f}",
                ]
            )


def _plot(rows: List[LayerRow], fps: int, sort_by: str, out_plot: str) -> None:
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    palette = _type_palette()
    x = list(range(1, len(rows) + 1))
    delta_uj = [row.delta_e_j * 1e6 for row in rows]    
    total_delta = sum(row.delta_e_j for row in rows)
    cumulative_pct = []
    running = 0.0
    for row in rows:
        running += row.delta_e_j
        cumulative_pct.append(100.0 * running / total_delta if total_delta != 0.0 else 0.0)

    fig, (ax_bar, ax_line) = plt.subplots(
        1,
        2,
        figsize=(3.5, 1.15),
        dpi=220,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    bar_colors = [palette[row.layer_type] for row in rows]
    ax_bar.bar(x, delta_uj, color=bar_colors, width=0.82, edgecolor="white", linewidth=0.4)
    ax_bar.axhline(0.0, color="black", linewidth=0.8)
    ax_bar.set_ylabel("Energy Reduction ($\mu$J)", labelpad=0)
    ax_bar.yaxis.set_label_coords(-0.12, 0.4)
    ax_bar.set_title(f"(a) Per-Layer Delta E ({fps} FPS)", loc="left", fontweight="bold", pad=1)
    ax_bar.set_xlim(0.3, len(rows) + 0.7)
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax_line.plot(x, cumulative_pct, color="#1d3557", linewidth=0.5, marker="o", markersize=1)
    ax_line.set_ylabel("Cumulative (%)", labelpad=0)
    ax_line.set_title("(b) Cumulative", loc="left", fontweight="bold", pad=1)
    ax_line.set_xlim(0.3, len(rows) + 0.7)
    ax_line.set_ylim(0, max(100.0, max(cumulative_pct) * 1.05 if cumulative_pct else 100.0))
    ax_line.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.6)

    for ax in (ax_bar, ax_line):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="x", pad=0, width=0.5, labelsize=8)
        ax.tick_params(axis="y", pad=0, width=0.5, labelsize=8)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["top"].set_linewidth(0.5)
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

    legend_items = [
        patches.Patch(facecolor=color, label=layer_type)
        for layer_type, color in palette.items()
        if any(row.layer_type == layer_type for row in rows)
    ]
    ax_bar.legend(
        handles=legend_items,
        ncol=min(2, len(legend_items)),
        frameon=True,
        loc="upper right",
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        columnspacing=0.5,
        handlelength=0.8,
        handletextpad=0.2,
        borderpad=0,
        labelspacing=0.2,
    )

    fig.supxlabel("Layer Rank (sorted by marginal utility)", y=0.01, x=0.58, fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.22, top=0.82, wspace=0.3)
    fig.savefig(out_plot, dpi=220, bbox_inches="tight", pad_inches=0, transparent=False)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot sorted per-layer energy reduction for SqueezeNet at one target FPS. "
            "Bars show ΔE_i relative to nominal ungated energy, using selected per-layer "
            "states from compiler_results_squeezenet_3rails.csv."
        )
    )
    ap.add_argument("--fps", type=int, default=60, help="Target inference rate to visualize.")
    ap.add_argument(
        "--compiler-csv",
        default=os.path.join(_HERE, "../data/3rails_fps/compiler_results_squeezenet_3rails.csv"),
        help="Compiler CSV containing selected_path_json.",
    )
    ap.add_argument("--nominal-v", type=float, default=1.1, help="Nominal ungated reference voltage.")
    ap.add_argument(
        "--sort-by",
        choices=["delta_e", "delta_e_pct"],
        default="delta_e",
        help="Sorting key for the layer ranking.",
    )
    ap.add_argument(
        "--out-pdf",
        "--out-png",
        dest="out_plot",
        default=os.path.join("./outputs/figure8.pdf"),
        help="Output plot path. PDF is used by default; extension controls the saved format.",
    )
    ap.add_argument(
        "--out-csv",
        default=os.path.join("./data/figure8_marginal_utility.csv"),
        help="Output CSV path for the sorted per-layer table.",
    )
    args = ap.parse_args()

    layers = _build_squeezenet_layers()
    selected_path_rows = _load_selected_path_rows(args.compiler_csv, args.fps)
    rows = _compute_layer_rows(layers, selected_path_rows, args.nominal_v)
    rows = _sort_rows(rows, args.sort_by)

    _write_rows_csv(args.out_csv, rows, args.fps, args.sort_by)
    _plot(rows, args.fps, args.sort_by, args.out_plot)

    total_delta_mj = sum(row.delta_e_j for row in rows) * 1e3
    print(f"Saved plot: {args.out_plot}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Total energy reduction vs nominal ungated: {total_delta_mj:.6f} mJ")


if __name__ == "__main__":
    main()
