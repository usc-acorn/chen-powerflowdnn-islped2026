from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.unicode_minus"] = False

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
from feeder_sa_cycles_model import (
    DVFS,
    FeederCfg,
    LEAKAGE_MODE_LAYER_PG,
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


@dataclass(frozen=True)
class LayerMetric:
    energy_j: float
    time_s: float
    idle_power_w: float


LEVELS = ["low", "nom", "high"]


def _build_layers() -> List[LayerSpec]:
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
            padding=padding
        )
        out.append(LayerSpec(name=name, feeder=feeder, channel_out=w_shape[0]))
    return out


def _mk_dvfs(v: float, vf_sys: VFModel, vf_rram: VFModel, vf_feeder: VFModel) -> DVFS:
    return DVFS(
        freq_sys_hz=vf_sys.f_hz(v),
        volt_sys_v=v,
        freq_rram_hz=vf_rram.f_hz(v),
        volt_rram_v=v,
        freq_feeder_hz=vf_feeder.f_hz(v),
        volt_feeder_v=v,
    )


def _transition_count(level_idxs: List[int]) -> int:
    return sum(1 for i in range(1, len(level_idxs)) if level_idxs[i] != level_idxs[i - 1])


def _schedule_totals(
    level_idxs: List[int],
    mode: str,
    metrics: Dict[str, Dict[str, List[LayerMetric]]],
    transition_s: float,
    transition_energy_j: float,
) -> Tuple[float, float, float]:
    level_keys = [LEVELS[i] for i in level_idxs]
    layer_metrics = [metrics[mode][lvl][i] for i, lvl in enumerate(level_keys)]
    total_e = sum(m.energy_j for m in layer_metrics)
    total_t = sum(m.time_s for m in layer_metrics)
    transition_count = 0
    for i in range(1, len(level_idxs)):
        if level_idxs[i] == level_idxs[i - 1]:
            continue
        transition_count += 1
    total_e += transition_count * transition_energy_j
    total_t += transition_count * transition_s
    # Use idle power at the final selected layer level for post-inference slack period.
    p_idle_post = layer_metrics[-1].idle_power_w
    return total_e, total_t, p_idle_post

def _build_discrete_levels(v_low: float, v_high: float, count: int) -> List[float]:
    if count < 2:
        raise ValueError("Discrete level count must be >= 2")
    step = (v_high - v_low) / float(count - 1)
    return [v_low + i * step for i in range(count)]


def _state_switch_count(
    prev_state: Tuple[float, float, float],
    next_state: Tuple[float, float, float],
) -> int:
    return sum(1 for a, b in zip(prev_state, next_state) if a != b)


def _schedule_totals_3d(
    state_idxs: List[int],
    state_space: List[Tuple[float, float, float]],
    layer_state_metrics: List[List[LayerMetric]],
    transition_s: float,
    transition_energy_j: float,
) -> Tuple[float, float, float]:
    chosen = [layer_state_metrics[i][s] for i, s in enumerate(state_idxs)]
    total_e = sum(m.energy_j for m in chosen)
    total_t = sum(m.time_s for m in chosen)
    transition_count = 0
    for i in range(1, len(state_idxs)):
        transition_count += _state_switch_count(
            state_space[state_idxs[i - 1]],
            state_space[state_idxs[i]],
        )
    total_e += transition_count * transition_energy_j
    total_t += transition_count * transition_s
    p_idle_post = chosen[-1].idle_power_w
    return total_e, total_t, p_idle_post


def _greedy_schedule_for_fps_3d(
    fps_target: float,
    state_space: List[Tuple[float, float, float]],
    layer_state_metrics: List[List[LayerMetric]],
    transition_s: float,
    transition_energy_j: float,
) -> Optional[Tuple[float, float, float, List[int]]]:
    budget_s = 1.0 / fps_target
    n_layers = len(layer_state_metrics)
    n_states = len(layer_state_metrics[0])
    state_idxs = [0] * n_layers  # start from (low, low, low) for all layers
    total_e, total_t, p_idle_post = _schedule_totals_3d(
        state_idxs, state_space, layer_state_metrics, transition_s, transition_energy_j
    )
    if total_t <= budget_s:
        return total_e, total_t, p_idle_post, state_idxs[:]

    while total_t > budget_s:
        best = None
        for i in range(n_layers):
            cur = layer_state_metrics[i][state_idxs[i]]
            for s in range(n_states):
                if s == state_idxs[i]:
                    continue
                nxt = layer_state_metrics[i][s]
                dt = cur.time_s - nxt.time_s
                if dt <= 0.0:
                    continue
                de = nxt.energy_j - cur.energy_j
                score = dt / max(de, 1e-21)
                e2 = total_e + de
                t2 = total_t - dt
                # Transition-aware greedy: account for BOTH neighboring edges affected by
                # changing layer i: (i-1 -> i) and (i -> i+1).
                old_state = state_space[state_idxs[i]]
                new_state = state_space[s]
                delta_switches = 0

                if i > 0:
                    prev_state = state_space[state_idxs[i - 1]]
                    old_prev = _state_switch_count(prev_state, old_state)
                    new_prev = _state_switch_count(prev_state, new_state)
                    delta_switches += (new_prev - old_prev)

                if i < (n_layers - 1):
                    next_state = state_space[state_idxs[i + 1]]
                    old_next = _state_switch_count(old_state, next_state)
                    new_next = _state_switch_count(new_state, next_state)
                    delta_switches += (new_next - old_next)

                de_total = de + delta_switches * transition_energy_j
                dt_total = dt - delta_switches * transition_s  # effective execution-time gain

                if dt_total <= 0.0:
                    continue

                score = dt_total / max(de_total, 1e-21)
                e2 = total_e + de_total
                t2 = total_t - dt_total
                p2 = nxt.idle_power_w if i == (n_layers - 1) else p_idle_post
                if best is None or score > best[0]:
                    best = (score, i, s, e2, t2, p2)
        if best is None:
            return None
        _, i_best, s_best, total_e, total_t, p_idle_post = best
        state_idxs[i_best] = s_best
    return total_e, total_t, p_idle_post, state_idxs[:]


def _energy_per_interval_j(
    total_energy_j: float,
    total_time_s: float,
    idle_power_post_w: float,
    fps_target: float,
    p_sleep_base: float,
    transition_s: float,
    transition_energy_j: float,
) -> float:
    frame_s = 1.0 / fps_target
    slack_s = max(0.0, frame_s - total_time_s)
    stay_on_energy_j = total_energy_j + idle_power_post_w * slack_s

    sleep_switch_count = 6
    sleep_transition_time_s = sleep_switch_count * transition_s
    sleep_transition_energy_j = sleep_switch_count * transition_energy_j
    if slack_s < sleep_transition_time_s:
        return stay_on_energy_j

    sleep_time_s = slack_s - sleep_transition_time_s
    sleep_energy_j = total_energy_j + sleep_transition_energy_j + p_sleep_base * sleep_time_s
    return min(stay_on_energy_j, sleep_energy_j)


def _load_compiler_points(csv_path: str) -> List[Tuple[float, float]]:
    if not csv_path or not os.path.isfile(csv_path):
        return []
    pts: List[Tuple[float, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fps = float(row["fps"])
                if "energy_j" in row and row["energy_j"]:
                    energy_j = float(row["energy_j"])
                elif "avg_power_w" in row and row["avg_power_w"]:
                    energy_j = float(row["avg_power_w"]) / fps
                else:
                    continue
                pts.append((fps, energy_j))
            except Exception:
                continue
    pts.sort(key=lambda x: x[0])
    return pts

def _selection_switch_count_3d(
    state_idxs: List[int],
    state_space: List[Tuple[float, float, float]],
) -> int:
    total = 0
    for i in range(1, len(state_idxs)):
        total += _state_switch_count(
            state_space[state_idxs[i - 1]],
            state_space[state_idxs[i]],
        )
    return total


def main() -> None:
    # Hard-coded defaults
    FPS_MIN = 1
    FPS_MAX = 100
    FPS_STEP = 1
    V_LOW = 0.9
    V_NOM = 1.1
    V_HIGH = 1.3
    SWEEP_LEVEL_COUNTS = [3]
    TRANSITION_NS = 5.0
    P_SLEEP_BASE = 0.0

    layers = _build_layers()
    e = get_energy_params()
    bw = get_bw_params()
    dvfs_ref = get_dvfs_params()
    vf_sys = VFModel(v_ref=dvfs_ref.volt_sys_v, f_ref_hz=dvfs_ref.freq_sys_hz)
    vf_rram = VFModel(v_ref=dvfs_ref.volt_rram_v, f_ref_hz=dvfs_ref.freq_rram_hz)
    vf_feeder = VFModel(v_ref=dvfs_ref.volt_feeder_v, f_ref_hz=dvfs_ref.freq_feeder_hz)

    v_by_level = {"low": V_LOW, "nom": V_NOM, "high": V_HIGH}
    modes = {"always_on": LEAKAGE_MODE_NO_PG, "always_on_gating": "rram_pg"}

    metrics: Dict[str, Dict[str, List[LayerMetric]]] = {
        "always_on": {k: [] for k in LEVELS},
        "always_on_gating": {k: [] for k in LEVELS},
    }
    metrics_3d: Dict[str, Dict[int, List[List[LayerMetric]]]] = {
        "always_on": {},
        "always_on_gating": {},
    }
    sweep_states_by_count: Dict[int, List[Tuple[float, float, float]]] = {}

    for mode_key, leakage_mode in modes.items():
        for level in LEVELS:
            v = v_by_level[level]
            dvfs = _mk_dvfs(v, vf_sys, vf_rram, vf_feeder)
            per_level: List[LayerMetric] = []
            for layer in layers:
                rep = model_layer(
                    feeder=layer.feeder,
                    channel_out=layer.channel_out,
                    dvfs=dvfs,
                    bw=bw,
                    e=e,
                    leakage_mode=leakage_mode,
                )
                t_layer = rep["times_s"]["t_layer"]
                e_idle = rep["energy_j"]["E_idle"]
                p_idle = (e_idle / t_layer) if t_layer > 0 else 0.0
                per_level.append(
                    LayerMetric(
                        energy_j=rep["energy_j"]["E_total"],
                        time_s=t_layer,
                        idle_power_w=p_idle,
                    )
                )
            metrics[mode_key][level] = per_level

    for mode_key, leakage_mode in modes.items():
        for c in SWEEP_LEVEL_COUNTS:
            level_vals = _build_discrete_levels(V_LOW, V_HIGH, c)
            if c not in sweep_states_by_count:
                sweep_states_by_count[c] = [
                    (v_sys, v_rram, v_feeder)
                    for v_sys in level_vals
                    for v_rram in level_vals
                    for v_feeder in level_vals
                ]
            per_layer_states: List[List[LayerMetric]] = []
            for layer in layers:
                states: List[LayerMetric] = []
                for v_sys, v_rram, v_feeder in sweep_states_by_count[c]:
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
                        leakage_mode=leakage_mode,
                    )
                    t_layer = rep["times_s"]["t_layer"]
                    e_idle = rep["energy_j"]["E_idle"]
                    p_idle = (e_idle / t_layer) if t_layer > 0 else 0.0
                    states.append(
                        LayerMetric(
                            energy_j=rep["energy_j"]["E_total"],
                            time_s=t_layer,
                            idle_power_w=p_idle,
                        )
                    )
                per_layer_states.append(states)
            metrics_3d[mode_key][c] = per_layer_states

    transition_s = TRANSITION_NS * 1e-9

    fps_targets = list(range(FPS_MIN, FPS_MAX + 1, FPS_STEP))

    y_always_on: List[float] = []
    y_always_on_gating: List[float] = []
    y_greedy_3d: Dict[int, List[float]] = {c: [] for c in SWEEP_LEVEL_COUNTS}
    y_greedy_3d_gating: Dict[int, List[float]] = {c: [] for c in SWEEP_LEVEL_COUNTS}
    greedy_selections: Dict[Tuple[str, int, int], Optional[List[int]]] = {}
    greedy_switch_counts: Dict[Tuple[str, int, int], Optional[int]] = {}

    e_ao, t_ao, p_idle_ao = _schedule_totals(
        [1] * len(layers), "always_on", metrics, transition_s=0.0, transition_energy_j=0.0
    )
    e_aog, t_aog, p_idle_aog = _schedule_totals(
        [1] * len(layers), "always_on_gating", metrics, transition_s=0.0, transition_energy_j=0.0
    )

    for fps in fps_targets:
        budget = 1.0 / fps

        if t_ao <= budget:
            y_always_on.append(
                _energy_per_interval_j(
                    e_ao, t_ao, p_idle_ao, fps, P_SLEEP_BASE, transition_s, 0.0
                )
            )
        else:
            y_always_on.append(float("nan"))

        if t_aog <= budget:
            y_always_on_gating.append(
                _energy_per_interval_j(
                    e_aog, t_aog, p_idle_aog, fps, P_SLEEP_BASE, transition_s, 0.0
                )
            )
        else:
            y_always_on_gating.append(float("nan"))

        for c in SWEEP_LEVEL_COUNTS:
            sol_3d = _greedy_schedule_for_fps_3d(
                fps,
                sweep_states_by_count[c],
                metrics_3d["always_on"][c],
                transition_s,
                0.0,
            )
            if sol_3d is None:
                y_greedy_3d[c].append(float("nan"))
                greedy_selections[("always_on", c, fps)] = None
                greedy_switch_counts[("always_on", c, fps)] = None
            else:
                y_greedy_3d[c].append(
                    _energy_per_interval_j(
                        sol_3d[0], sol_3d[1], sol_3d[2], fps, P_SLEEP_BASE, transition_s, 0.0
                    )
                )
                greedy_selections[("always_on", c, fps)] = sol_3d[3]
                greedy_switch_counts[("always_on", c, fps)] = _selection_switch_count_3d(
                    sol_3d[3], sweep_states_by_count[c]
                )

            sol_3d_g = _greedy_schedule_for_fps_3d(
                fps,
                sweep_states_by_count[c],
                metrics_3d["always_on_gating"][c],
                transition_s,
                0.0,
            )
            if sol_3d_g is None:
                y_greedy_3d_gating[c].append(float("nan"))
                greedy_selections[("always_on_gating", c, fps)] = None
                greedy_switch_counts[("always_on_gating", c, fps)] = None
            else:
                y_greedy_3d_gating[c].append(
                    _energy_per_interval_j(
                        sol_3d_g[0], sol_3d_g[1], sol_3d_g[2], fps, P_SLEEP_BASE, transition_s, 0.0
                    )
                )
                greedy_selections[("always_on_gating", c, fps)] = sol_3d_g[3]
                greedy_switch_counts[("always_on_gating", c, fps)] = _selection_switch_count_3d(
                    sol_3d_g[3], sweep_states_by_count[c]
                )

    fig, ax = plt.subplots(figsize=(3.5, 1), dpi=220)
    fig.subplots_adjust(left=0.1, right=0.977, bottom=0.2, top=0.75)
    fig.suptitle(
        "Interval Energy vs Inference/s Target",
        fontsize=8,
        fontweight="bold",
        y=0.98,
        x=0.5335,
    )

    color_always = "#a1dab4"
    color_greedy = "#41b6c4"

    ax.plot(
        fps_targets,
        [v * 1e3 if math.isfinite(v) else v for v in y_always_on],
        linewidth=1.0,
        linestyle="-",
        color=color_always,
        label="Baseline",
    )
    ax.plot(
        fps_targets,
        [v * 1e3 if math.isfinite(v) else v for v in y_always_on_gating],
        linewidth=1.0,
        linestyle="--",
        color=color_always,
        label="+Gating",
    )
    for c in sorted(SWEEP_LEVEL_COUNTS):
        ax.plot(
            fps_targets,
            [v * 1e3 if math.isfinite(v) else v for v in y_greedy_3d[c]],
            linewidth=1.0,
            linestyle="-",
            color=color_greedy,
            label=f"+Greedy",
            # label=f"+Greedy ({c} level)",
        )
        ax.plot(
            fps_targets,
            [v * 1e3 if math.isfinite(v) else v for v in y_greedy_3d_gating[c]],
            linewidth=1.0,
            linestyle=":",
            color=color_greedy,
            label=f"+Gating+Greedy",
            # label=f"+Greedy+Gating ({c} levels)",
        )
        compiler_pts = _load_compiler_points(
            os.path.join(
                _PROJECT_ROOT,
                "data",
                "3rails_fps",
                "compiler_results_squeezenet_3rails.csv",
            )
        )
    if compiler_pts:
        x_comp = [x for x, _ in compiler_pts]
        y_comp = [y * 1e3 for _, y in compiler_pts]
        ax.plot(
            x_comp,
            y_comp,
            linewidth=1,
            linestyle="-",
            color="#2c7fb8",
            label="Solver",
            # label="Solver (searched 3 rails)",
        )
    else:
        print("No compiler points found to plot.")
    ax.set_xlabel("Target Inference Rate (inference/s)", fontsize=8, labelpad=-1)
    ax.set_ylabel("Inter. Energy(mJ)", fontsize=8, labelpad=2)
    # ax.yaxis.set_label_coords(-0.02, 0.4)
    # ax.grid(axis="y", alpha=0.30)
    ax.set_xlim(0, 100)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.9),
        fontsize=8,
        frameon=True,
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        ncol=5,
        columnspacing=0.5,
        handlelength=1,
        handletextpad=0.1,
        borderpad=0,
        labelspacing=0.2,
    )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    # ax.set_ylim(0.1, 0.3)
    ax.set_yticks([0.15, 0.2, 0.25, 0.3])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", pad=0, width=0.5, labelsize=8)
    ax.tick_params(axis="y", pad=0, width=0.5, labelsize=8)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)


    out_path = os.path.join(_HERE, "../outputs", "figure5.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, pad_inches=0, transparent=False, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    # if plt.get_backend().lower() != "agg":
    #     plt.show()
    # else:
    #     plt.close(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
