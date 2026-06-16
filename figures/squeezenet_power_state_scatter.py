from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
from matplotlib import rcParams

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "model"),
    os.path.join(_PROJECT_ROOT, "networks"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from energy_config import get_bw_params, get_dvfs_params, get_energy_params
from feeder_sa_cycles_model import (
    DVFS,
    FeederCfg,
    LEAKAGE_MODE_NO_PG,
    model_layer,
)
from squeezenet_run import build_layers, conv_layers_from_spec
from vf_model import VFModel


@dataclass(frozen=True)
class LayerSpec:
    name: str
    feeder: FeederCfg
    channel_out: int


def _parse_layer_idxs(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
            num_lanes=8,
            padding=padding,
        )
        out.append(LayerSpec(name=name, feeder=feeder, channel_out=w_shape[0]))
    return out


def _layer_name(layer_idx: int, layers: List[LayerSpec]) -> str:
    if layer_idx < 0 or layer_idx >= len(layers):
        return f"Layer {layer_idx}"
    return _short_layer_name(layer_idx, layers[layer_idx].name)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot SqueezeNet NO_PG power-state clouds and nominal points."
    )
    ap.add_argument("--layer-idxs", default="0,1,2")
    ap.add_argument("--v-list", default="0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3")
    ap.add_argument(
        "--out",
        default=os.path.join(_HERE, "../outputs", "figure2.pdf"),
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    layer_idxs = _parse_layer_idxs(args.layer_idxs)
    if not layer_idxs:
        raise RuntimeError("No layer indices selected.")

    v_list = [float(x.strip()) for x in args.v_list.split(",") if x.strip()]
    layers = _build_squeezenet_layers()

    e = get_energy_params()
    bw = get_bw_params()
    dvfs_nom = get_dvfs_params()

    vf_sys = VFModel(v_ref=dvfs_nom.volt_sys_v, f_ref_hz=dvfs_nom.freq_sys_hz)
    vf_rram = VFModel(v_ref=dvfs_nom.volt_rram_v, f_ref_hz=dvfs_nom.freq_rram_hz)
    vf_feeder = VFModel(v_ref=dvfs_nom.volt_feeder_v, f_ref_hz=dvfs_nom.freq_feeder_hz)

    subplot_colors = [
        "#2f3e1f",  # deep olive
        "#6b8e23",  # olive drab
        "#c3d86d",  # light olive
    ]

    nominal_color = "#d62728"  # red
    cols = len(layer_idxs)
    fig, axes = plt.subplots(1, cols, figsize=(3.5, 1.3), dpi=220, squeeze=False)
    axes_flat = axes.flatten()

    for i, layer_idx in enumerate(layer_idxs):
        ax = axes_flat[i]

        if layer_idx < 0 or layer_idx >= len(layers):
            ax.set_title(f"Layer {layer_idx} (invalid)", fontweight="bold")
            ax.axis("off")
            continue

        layer = layers[layer_idx]

        xs: List[float] = []
        ys: List[float] = []

        for v_sys in v_list:
            for v_rram in v_list:
                for v_feeder in v_list:
                    dvfs = DVFS(
                        freq_sys_hz=vf_sys.f_hz(v_sys),
                        volt_sys_v=v_sys,
                        freq_rram_hz=vf_rram.f_hz(v_rram),
                        volt_rram_v=v_rram,
                        freq_feeder_hz=vf_feeder.f_hz(v_feeder),
                        volt_feeder_v=v_feeder,
                    )
                    rep = model_layer(
                        feeder=layer.feeder,
                        channel_out=layer.channel_out,
                        dvfs=dvfs,
                        bw=bw,
                        e=e,
                        leakage_mode=LEAKAGE_MODE_NO_PG,
                    )
                    xs.append(1.0 / float(rep["times_s"]["t_layer"]))
                    ys.append(1e-3 / float(rep["energy_j"]["E_total"]))

        ax.scatter(
            xs,
            ys,
            s=1,
            color=subplot_colors[i],
            marker="o",
        )
        nominal_rep = model_layer(
            feeder=layer.feeder,
            channel_out=layer.channel_out,
            dvfs=dvfs_nom,
            bw=bw,
            e=e,
            leakage_mode=LEAKAGE_MODE_NO_PG,
        )
        x_nom = 1.0 / float(nominal_rep["times_s"]["t_layer"])
        y_nom = 1e-3 / float(nominal_rep["energy_j"]["E_total"])
        ax.scatter(
            [x_nom],
            [y_nom],
            s=4,
            color=nominal_color,
            # edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="V_nom",
        )

        ax.set_title(_layer_name(layer_idx, layers), fontweight="bold", pad=1, fontsize=8)
        if i == 1:
            ax.set_xlabel("Performance (task/s)", labelpad=-1)
        if i == 0:
            ax.set_ylabel("Efficiency (task/mJ)", labelpad=0)

        ax.tick_params(axis="both", labelsize=8, pad=0, width=0.5)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        
        if i == 0:
            legend = ax.legend(
                loc="best",
                frameon=True,
                fontsize=8,
                handlelength=1.0,
                handletextpad=0.1,
                borderpad=0.1,
                labelspacing=0,
            )

            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")
            frame.set_linewidth(0.1)
            frame.set_alpha(0.8)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["top"].set_linewidth(0.5)
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

    fig.subplots_adjust(left=0.09, right=1, bottom=0.25, top=0.82, wspace=0.25, hspace=0.1)

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
