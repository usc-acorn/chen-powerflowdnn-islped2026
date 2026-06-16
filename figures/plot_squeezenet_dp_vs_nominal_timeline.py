from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.unicode_minus"] = False

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

from energy_config import get_bw_params, get_dvfs_params, get_energy_params
from feeder_sa_cycles_model import DVFS, FeederCfg, LEAKAGE_MODE_NO_PG, model_layer
from squeezenet_run import build_layers, conv_layers_from_spec
from vf_model import VFModel


@dataclass(frozen=True)
class LayerSpec:
    name: str
    feeder: FeederCfg
    channel_out: int


@dataclass
class Segment:
    scenario: str
    segment_type: str
    label: str
    start_s: float
    duration_s: float
    power_w: float
    energy_j: float


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


def _read_compiler_row(path: str, target_fps: float | None = None) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in compiler CSV: {path}")
    if target_fps is None:
        return max(rows, key=lambda r: float(r["fps"]))
    for row in rows:
        if abs(float(row["fps"]) - target_fps) < 1e-9:
            return row
    raise RuntimeError(f"Target fps {target_fps} not found in compiler CSV: {path}")


def _fixed_vf_segments(
    scenario: str,
    target_fps: float,
    leakage_mode: str,
    dvfs: DVFS,
) -> List[Segment]:
    e = get_energy_params()
    bw = get_bw_params()
    layers = _build_squeezenet_layers()
    segments: List[Segment] = []
    t_cursor = 0.0

    for idx, layer in enumerate(layers):
        rep = model_layer(
            feeder=layer.feeder,
            channel_out=layer.channel_out,
            dvfs=dvfs,
            bw=bw,
            e=e,
            leakage_mode=leakage_mode,
        )
        t_layer = float(rep["times_s"]["t_layer"])
        e_layer = float(rep["energy_j"]["E_total"])
        p_layer = e_layer / t_layer if t_layer > 0 else 0.0
        segments.append(
            Segment(
                scenario=scenario,
                segment_type="layer",
                label=f"L{idx}",
                start_s=t_cursor,
                duration_s=t_layer,
                power_w=p_layer,
                energy_j=e_layer,
            )
        )
        t_cursor += t_layer

    frame_s = 1.0 / target_fps
    if t_cursor < frame_s:
        idle_power_w = 0.0
        slack_s = frame_s - t_cursor
        segments.append(
            Segment(
                scenario=scenario,
                segment_type="slack",
                label="Slack",
                start_s=t_cursor,
                duration_s=slack_s,
                power_w=idle_power_w,
                energy_j=idle_power_w * slack_s,
            )
        )
    return segments


def _nominal_segments(target_fps: float, leakage_mode: str) -> List[Segment]:
    return _fixed_vf_segments(
        scenario="Nominal",
        target_fps=target_fps,
        leakage_mode=leakage_mode,
        dvfs=get_dvfs_params(),
    )


def _dp_segments(max_row: Dict[str, str]) -> List[Segment]:
    selected_path = json.loads(max_row["selected_path_json"])
    e_trans = float(max_row["transition_energy_j"])
    l_trans = float(max_row["transition_time_s"])
    frame_s = 1.0 / float(max_row["fps"])
    segments: List[Segment] = []
    t_cursor = 0.0

    for idx, point in enumerate(selected_path):
        t_layer = float(point["time_s"])
        e_layer = float(point["energy_j"])
        p_layer = e_layer / t_layer if t_layer > 0 else 0.0
        segments.append(
            Segment(
                scenario="Power State Orchestration",
                segment_type="layer",
                label=f"L{int(point['layer_idx'])}",
                start_s=t_cursor,
                duration_s=t_layer,
                power_w=p_layer,
                energy_j=e_layer,
            )
        )
        t_cursor += t_layer

        if idx == len(selected_path) - 1:
            continue
        prev = point
        nxt = selected_path[idx + 1]
        changed_count = sum(
            1 for k in ("v_sys", "v_rram", "v_feeder") if float(prev[k]) != float(nxt[k])
        )
        if changed_count <= 0:
            continue
        t_trans = changed_count * l_trans
        e_trans_total = changed_count * e_trans
        p_trans = e_trans_total / t_trans if t_trans > 0 else 0.0
        segments.append(
            Segment(
                scenario="Power State Orchestration",
                segment_type="transition",
                label="Switch",
                start_s=t_cursor,
                duration_s=t_trans,
                power_w=p_trans,
                energy_j=e_trans_total,
            )
        )
        t_cursor += t_trans

    total_energy = float(max_row["energy_j"])
    layer_energy = sum(s.energy_j for s in segments if s.segment_type == "layer")
    trans_energy = sum(s.energy_j for s in segments if s.segment_type == "transition")
    slack_energy = max(0.0, total_energy - layer_energy - trans_energy)
    if t_cursor < frame_s:
        slack_s = frame_s - t_cursor
        slack_power = slack_energy / slack_s if slack_s > 0 else 0.0
        segments.append(
            Segment(
                scenario="Power State Orchestration",
                segment_type="slack",
                label="Slack",
                start_s=t_cursor,
                duration_s=slack_s,
                power_w=slack_power,
                energy_j=slack_energy,
            )
        )
    return segments


def _write_segments_csv(path: str, segments: List[Segment], target_fps: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_fps", f"{target_fps:.6f}"])
        w.writerow([])
        w.writerow(["scenario", "segment_type", "label", "start_s", "duration_s", "power_w", "energy_j"])
        for s in segments:
            w.writerow([s.scenario, s.segment_type, s.label, s.start_s, s.duration_s, s.power_w, s.energy_j])


def _summary_by_scenario(segments: List[Segment]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for scenario in sorted({s.scenario for s in segments}):
        pts = [s for s in segments if s.scenario == scenario]
        out[scenario] = {
            "layer": sum(s.energy_j for s in pts if s.segment_type == "layer"),
            "transition": sum(s.energy_j for s in pts if s.segment_type == "transition"),
            "slack": sum(s.energy_j for s in pts if s.segment_type == "slack"),
        }
    return out


def _layer_energy_breakdown_by_scenario(segments: List[Segment]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for scenario in sorted({s.scenario for s in segments}):
        pts = [s for s in segments if s.scenario == scenario and s.segment_type == "layer"]
        out[scenario] = {s.label: s.energy_j for s in pts}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare SqueezeNet nominal vs full-optimization per-layer power timeline at the max target rate."
    )
    ap.add_argument(
        "--compiler-csv",
        default=os.path.join(
            "./data/3rails_fps/compiler_results_squeezenet_3rails.csv"
        ),
    )
    ap.add_argument("--target-fps", type=float, default=90.0)
    ap.add_argument("--nominal-leakage-mode", default=LEAKAGE_MODE_NO_PG, choices=["no_pg", "layer_pg", "rram_pg"])
    ap.add_argument("--v-low", type=float, default=0.9)
    ap.add_argument("--v-high", type=float, default=1.3)
    ap.add_argument(
        "--out",
        default=os.path.join("./outputs/figure3.png"),
    )
    ap.add_argument(
        "--out-csv",
        default=os.path.join("./data/figure3.csv"),
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    max_row = _read_compiler_row(args.compiler_csv, args.target_fps)
    target_fps = float(max_row["fps"])

    dvfs_nom = get_dvfs_params()
    vf_sys = VFModel(v_ref=dvfs_nom.volt_sys_v, f_ref_hz=dvfs_nom.freq_sys_hz)
    vf_rram = VFModel(v_ref=dvfs_nom.volt_rram_v, f_ref_hz=dvfs_nom.freq_rram_hz)
    vf_feeder = VFModel(v_ref=dvfs_nom.volt_feeder_v, f_ref_hz=dvfs_nom.freq_feeder_hz)
    dvfs_low = DVFS(
        freq_sys_hz=vf_sys.f_hz(args.v_low),
        volt_sys_v=args.v_low,
        freq_rram_hz=vf_rram.f_hz(args.v_low),
        volt_rram_v=args.v_low,
        freq_feeder_hz=vf_feeder.f_hz(args.v_low),
        volt_feeder_v=args.v_low,
    )
    dvfs_high = DVFS(
        freq_sys_hz=vf_sys.f_hz(args.v_high),
        volt_sys_v=args.v_high,
        freq_rram_hz=vf_rram.f_hz(args.v_high),
        volt_rram_v=args.v_high,
        freq_feeder_hz=vf_feeder.f_hz(args.v_high),
        volt_feeder_v=args.v_high,
    )

    low_segments = _fixed_vf_segments("Low VF", target_fps, args.nominal_leakage_mode, dvfs_low)
    high_segments = _fixed_vf_segments("High VF", target_fps, args.nominal_leakage_mode, dvfs_high)
    nominal_segments = _nominal_segments(target_fps=target_fps, leakage_mode=args.nominal_leakage_mode)
    dp_segments = _dp_segments(max_row)
    all_segments = low_segments + high_segments + nominal_segments + dp_segments
    _write_segments_csv(args.out_csv, all_segments, target_fps)

    summary = _summary_by_scenario(all_segments)
    layer_breakdown = _layer_energy_breakdown_by_scenario(all_segments)

    fig = plt.figure(figsize=(9.0, 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.8, 0.8], height_ratios=[1, 1], wspace=0.15, hspace=0.22)
    ax_dp = fig.add_subplot(gs[0, 0])
    ax_single = fig.add_subplot(gs[1, 0], sharex=ax_dp, sharey=ax_dp)
    ax_bar = fig.add_subplot(gs[:, 1])

    layer_colors = {
        "Power State Orchestration": "#225ea8",
        "Nominal": "#7fcdbb",
        "High VF": "#fdae6b",
        "Low VF": "#fdd0a2",
    }
    transition_color = "#cb181d"
    slack_color = "#bdbdbd"

    frame_ms = 1e3 / target_fps
    max_end_ms = max((s.start_s + s.duration_s) * 1e3 for s in all_segments)
    x_view_max_ms = 12.5

    def _draw_segments(ax, scenario: str, include_only_layers: bool = False) -> None:
        segs = [s for s in all_segments if s.scenario == scenario]
        for s in segs:
            color = layer_colors[scenario]
            if s.segment_type == "transition":
                color = transition_color
            elif s.segment_type == "slack":
                color = slack_color
            if include_only_layers and s.segment_type != "layer":
                continue
            x_ms = s.start_s * 1e3 + (s.duration_s * 1e3) / 2.0
            width_ms = s.duration_s * 1e3
            ax.bar(
                x=x_ms,
                height=s.power_w * 1e3,
                width=width_ms,
                color=color,
                edgecolor="black",
                linewidth=0.35 if not include_only_layers else 0.25,
                align="center",
            )
            if (
                not include_only_layers
                and s.segment_type == "layer"
                and s.duration_s * 1e3 > 0.2
            ):
                ax.text(
                    x_ms,
                    s.power_w * 1e3 + 0.15,
                    s.label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    rotation=90,
                    clip_on=True,
                )

    _draw_segments(ax_dp, "Power State Orchestration", include_only_layers=False)
    ax_dp.axvline(frame_ms, color=transition_color, linestyle="--", linewidth=1.0)
    ax_dp.set_ylabel("Power (mW)", size=12)
    ax_dp.set_title("Power State Orchestration", fontsize=12, fontweight="bold", loc="left", y=1)
    ax_dp.grid(axis="y", alpha=0.3)

    for scenario in ["High VF", "Nominal", "Low VF"]:
        _draw_segments(ax_single, scenario, include_only_layers=True)
    ax_single.axvline(frame_ms, color=transition_color, linestyle="--", linewidth=1.0)
    ax_single.set_ylabel("Power (mW)", size=12)
    ax_single.set_title("Single Power State", fontsize=12, fontweight="bold", loc="left", y=1)
    ax_single.grid(axis="y", alpha=0.3)
    ax_single.legend(
        [
            plt.Rectangle((0, 0), 1, 1, facecolor=layer_colors["Nominal"], edgecolor="black", linewidth=0.25),
            plt.Rectangle((0, 0), 1, 1, facecolor=layer_colors["High VF"], edgecolor="black", linewidth=0.25),
            plt.Rectangle((0, 0), 1, 1, facecolor=layer_colors["Low VF"], edgecolor="black", linewidth=0.25),
        ],
        ["Nominal", "High VF", "Low VF"],
        fontsize=12,
        frameon=True,
        ncol=3,
        loc="upper left",
    )

    shared_power_max_mw = max(
        s.power_w * 1e3
        for s in all_segments
        if s.scenario in {"Power State Orchestration", "Nominal", "High VF", "Low VF"}
    )
    shared_y_top = shared_power_max_mw * 1.08 if shared_power_max_mw > 0 else 1.0
    ax_dp.set_ylim(0, shared_y_top)

    ax_single.set_xlabel("Accumulated Execution Time (ms)", size=12)
    ax_dp.label_outer()
    ax_dp.set_xlim(0, x_view_max_ms)

    for ax in (ax_dp, ax_single):
        y_top = ax.get_ylim()[1]
        ax.text(
            frame_ms + 0.05,
            y_top * 0.92,
            "Deadline",
            color=transition_color,
            fontsize=12,
            rotation=90,
            va="top",
            ha="left",
        )

    low_end_ms = max((s.start_s + s.duration_s) * 1e3 for s in all_segments if s.scenario == "Low VF")
    if low_end_ms > x_view_max_ms:
        y_top = ax_single.get_ylim()[1]
        ax_single.annotate(
            "cont...",
            xy=(x_view_max_ms - 0.18, y_top * 0.06),
            xytext=(x_view_max_ms - 1.5, y_top * 0.06),
            arrowprops=dict(arrowstyle="->", linewidth=1.0),
            # color=layer_colors["Low VF"],
            fontsize=12,
            ha="left",
            va="center",
        )

    scenario_order = ["Power State Orchestration", "Nominal", "High VF", "Low VF"]
    x = list(range(len(scenario_order)))
    layer_labels = [f"L{i}" for i in range(len(_build_squeezenet_layers()))]
    bottoms = [0.0 for _ in scenario_order]
    for i, layer_label in enumerate(layer_labels):
        vals = [layer_breakdown.get(s, {}).get(layer_label, 0.0) * 1e3 for s in scenario_order]
        ax_bar.bar(
            x,
            vals,
            bottom=bottoms,
            color=[layer_colors[s] for s in scenario_order],
            edgecolor="black",
            linewidth=0.2,
            width=0.55,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    trans_vals = [summary[s]["transition"] * 1e3 for s in scenario_order]
    ax_bar.bar(x, trans_vals, bottom=bottoms, color=transition_color, edgecolor="black", linewidth=0.35, label="Transitions", width=0.30)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Orch.", "Nom.", "High VF", "Low VF"], rotation=20, fontsize=12)
    ax_bar.set_ylabel("Total Energy (mJ)")
    ax_bar.set_title("Accumulated Energy", fontsize=12, fontweight="bold")
    ax_bar.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_bar.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_bar.grid(axis="y", alpha=0.3)
    # ax_bar.legend(fontsize=8, frameon=True)
    for xi, total_layers in zip(x, bottoms):
        ax_bar.text(xi, 0.002, "L0", ha="center", va="bottom", fontsize=12)
        ax_bar.text(xi, total_layers - 0.002, f"L{len(layer_labels)-1}", ha="center", va="top", fontsize=12)

    fig.suptitle(
        f"SqueezeNet Graph Execution at Target Rate ({int(round(target_fps))} inference/s)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(left=0.055, right=0.985)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Saved plot: {args.out}")
    print(f"Saved CSV: {args.out_csv}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
