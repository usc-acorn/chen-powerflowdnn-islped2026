from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

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
from feeder_sa_cycles_model import (
    DVFS,
    LEAKAGE_MODE_RRAM_PG,
    LEAKAGE_MODE_NO_PG,
    MemoryBankCfg,
    model_layer,
    model_matmul_simple,
)
from generate_vdd_json import (
    LayerSpec,
    _build_mobilevit_xxs_layers,
    _build_mobilenet_v3_small_layers,
    _build_resnet18_layers,
    _build_squeezenet_layers,
    _build_resnet18_mem_banks,
    _build_mobilevit_xxs_mem_banks,
    _build_mobilenet_v3_small_mem_banks,
    _derive_mem_banks_from_tensors
)
from vf_model import VFModel


@dataclass(frozen=True)
class LayerMetric:
    energy_j: float
    time_s: float
    idle_power_w: float


@dataclass(frozen=True)
class ModelCfg:
    key: str
    label: str
    compiler_csv: str


@dataclass(frozen=True)
class PreparedModel:
    state_space: List[Tuple[float, float, float]]
    nominal_metrics: Dict[str, List[LayerMetric]]
    greedy_metrics: Dict[str, List[List[LayerMetric]]]


def _build_discrete_levels(v_low: float, v_high: float, count: int) -> List[float]:
    if count < 2:
        raise ValueError("Discrete level count must be >= 2")
    step = (v_high - v_low) / float(count - 1)
    return [v_low + i * step for i in range(count)]


def _mk_dvfs(
    v_sys: float,
    v_rram: float,
    v_feeder: float,
    vf_sys: VFModel,
    vf_rram: VFModel,
    vf_feeder: VFModel,
) -> DVFS:
    return DVFS(
        freq_sys_hz=vf_sys.f_hz(v_sys),
        volt_sys_v=v_sys,
        freq_rram_hz=vf_rram.f_hz(v_rram),
        volt_rram_v=v_rram,
        freq_feeder_hz=vf_feeder.f_hz(v_feeder),
        volt_feeder_v=v_feeder,
    )


def _state_switch_count(
    prev_state: Tuple[float, float, float],
    next_state: Tuple[float, float, float],
) -> int:
    return sum(1 for a, b in zip(prev_state, next_state) if a != b)


def _schedule_totals_fixed(
    layer_metrics: List[LayerMetric],
) -> Tuple[float, float, float]:
    total_e = sum(m.energy_j for m in layer_metrics)
    total_t = sum(m.time_s for m in layer_metrics)
    p_idle_post = layer_metrics[-1].idle_power_w if layer_metrics else 0.0
    return total_e, total_t, p_idle_post


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


def _load_compiler_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                float(row["fps"])
            except Exception:
                continue
            rows.append(row)
    rows.sort(key=lambda r: float(r["fps"]))
    return rows


def _infer_transition_params(rows: List[Dict[str, str]]) -> Tuple[float, float]:
    e_vals = set()
    t_vals = set()
    for row in rows:
        try:
            e_vals.add(float(row.get("transition_energy_j", "")))
        except Exception:
            pass
        try:
            t_vals.add(float(row.get("transition_time_s", "")))
        except Exception:
            pass
    transition_energy_j = next(iter(e_vals)) if len(e_vals) == 1 else 0.0
    transition_s = next(iter(t_vals)) if len(t_vals) == 1 else 5e-9
    return transition_energy_j, transition_s


def _select_max_fps(rows: List[Dict[str, str]]) -> float:
    return max(float(r["fps"]) for r in rows)


def _select_relative_fps(rows: List[Dict[str, str]], ratio: float) -> float:
    max_fps = _select_max_fps(rows)
    target_fps = max_fps * ratio
    return min((float(r["fps"]) for r in rows), key=lambda fps: abs(fps - target_fps))


def _compiler_energy_at_fps(rows: List[Dict[str, str]], fps_target: float) -> float:
    best = min(rows, key=lambda r: abs(float(r["fps"]) - fps_target))
    if best.get("energy_j"):
        return float(best["energy_j"])
    fps = float(best["fps"])
    return float(best["avg_power_w"]) / fps


def _build_model_layers(model_key: str) -> List[LayerSpec]:
    if model_key == "squeezenet":
        return _build_squeezenet_layers()
    if model_key == "resnet18":
        return _build_resnet18_layers()
    if model_key == "mobilenetv3_small":
        return _build_mobilenet_v3_small_layers(include_se=False, depthwise_1ch=True)
    if model_key == "mobilevit_xxs":
        return _build_mobilevit_xxs_layers(
            include_transformer=True,
            include_attn_matmul=True,
            include_head=False,
            depthwise_1ch=True,
        )
    raise ValueError(f"Unsupported model: {model_key}")

def _build_model_mem(model_key: str):
    if model_key == "squeezenet":
        return MemoryBankCfg()
    if model_key == "resnet18":
        return _build_resnet18_mem_banks()
    if model_key == "mobilenetv3_small":
        return _build_mobilenet_v3_small_mem_banks(include_se=False)
    if model_key == "mobilevit_xxs":
        return _build_mobilevit_xxs_mem_banks(
            include_transformer=True,
            include_attn_matmul=True,
            include_head=False,
        )
    return MemoryBankCfg()

    raise ValueError(f"Unsupported model: {model_key}")

def _eval_layer_metric(
    layer: LayerSpec,
    dvfs: DVFS,
    bw,
    e,
    leakage_mode: str,
    mem_banks: MemoryBankCfg,
) -> LayerMetric:
    if layer.kind == "conv":
        if layer.feeder is None:
            raise ValueError("conv layer missing feeder")
        rep = model_layer(
            feeder=layer.feeder,
            channel_out=layer.channel_out,
            dvfs=dvfs,
            bw=bw,
            e=e,
            act_bits=8,
            weight_bits=8,
            out_bits=32,
            C=8,
            overlap_weight_fetch=True,
            k_chunk_size=layer.k_chunk_size,
            verbose=False,
            leakage_mode=leakage_mode,
            mem_banks=mem_banks,
        )
    else:
        if layer.m is None or layer.k is None or layer.n is None:
            raise ValueError("matmul layer missing m/k/n")
        rep = model_matmul_simple(
            m=layer.m,
            k=layer.k,
            n=layer.n,
            dvfs=dvfs,
            bw=bw,
            e=e,
            act_bits=8,
            weight_bits=8,
            out_bits=32,
            C=8,
            overlap_weight_fetch=True,
            leakage_mode=leakage_mode,
            mem_banks=mem_banks,
        )
    e_total = rep["energy_j"]["E_total"] * layer.repeats
    t_layer = rep["times_s"]["t_layer"] * layer.repeats
    e_idle = rep["energy_j"]["E_idle"] * layer.repeats
    p_idle = (e_idle / t_layer) if t_layer > 0 else 0.0
    return LayerMetric(energy_j=e_total, time_s=t_layer, idle_power_w=p_idle)


def _prepare_model_metrics(
    model_key: str,
    v_low: float,
    v_nom: float,
    v_high: float,
    level_count: int,
) -> PreparedModel:
    layers = _build_model_layers(model_key)
    e = get_energy_params()
    bw = get_bw_params()
    dvfs_ref = get_dvfs_params()
    vf_sys = VFModel(v_ref=dvfs_ref.volt_sys_v, f_ref_hz=dvfs_ref.freq_sys_hz)
    vf_rram = VFModel(v_ref=dvfs_ref.volt_rram_v, f_ref_hz=dvfs_ref.freq_rram_hz)
    vf_feeder = VFModel(v_ref=dvfs_ref.volt_feeder_v, f_ref_hz=dvfs_ref.freq_feeder_hz)
    # model based memory bank
    mem_banks = _build_model_mem(model_key)

    modes = {
        "baseline": LEAKAGE_MODE_NO_PG,
        "baseline_gating": LEAKAGE_MODE_RRAM_PG,
    }
    nominal_metrics: Dict[str, List[LayerMetric]] = {}
    greedy_metrics: Dict[str, List[List[LayerMetric]]] = {}

    level_vals = _build_discrete_levels(v_low, v_high, level_count)
    state_space = [
        (v_sys, v_rram, v_feeder)
        for v_sys in level_vals
        for v_rram in level_vals
        for v_feeder in level_vals
    ]

    for method_key, leakage_mode in modes.items():
        dvfs_nom = _mk_dvfs(v_nom, v_nom, v_nom, vf_sys, vf_rram, vf_feeder)
        nominal_metrics[method_key] = [
            _eval_layer_metric(layer, dvfs_nom, bw, e, leakage_mode, mem_banks)
            for layer in layers
        ]

        per_layer_states: List[List[LayerMetric]] = []
        for layer in layers:
            states = []
            for v_sys, v_rram, v_feeder in state_space:
                dvfs = _mk_dvfs(v_sys, v_rram, v_feeder, vf_sys, vf_rram, vf_feeder)
                states.append(_eval_layer_metric(layer, dvfs, bw, e, leakage_mode, mem_banks))
            per_layer_states.append(states)
        greedy_metrics[method_key] = per_layer_states

    return PreparedModel(
        state_space=state_space,
        nominal_metrics=nominal_metrics,
        greedy_metrics=greedy_metrics,
    )


def _compute_method_energies(
    prepared: PreparedModel,
    fps_target: float,
    transition_energy_j: float,
    transition_s: float,
    p_sleep_base: float,
) -> Dict[str, float]:
    nominal_metrics = prepared.nominal_metrics
    greedy_metrics = prepared.greedy_metrics
    state_space = prepared.state_space

    out: Dict[str, float] = {}
    for method_key in ("baseline", "baseline_gating"):
        total_e, total_t, p_idle_post = _schedule_totals_fixed(nominal_metrics[method_key])
        out[method_key] = _energy_per_interval_j(
            total_e,
            total_t,
            p_idle_post,
            fps_target,
            p_sleep_base,
            transition_s,
            transition_energy_j,
        ) if total_t <= (1.0 / fps_target) else float("nan")

    sol_g = _greedy_schedule_for_fps_3d(
        fps_target, state_space, greedy_metrics["baseline"], transition_s, transition_energy_j
    )
    out["greedy"] = _energy_per_interval_j(
        sol_g[0], sol_g[1], sol_g[2], fps_target, p_sleep_base, transition_s, transition_energy_j
    ) if sol_g is not None else float("nan")

    sol_gg = _greedy_schedule_for_fps_3d(
        fps_target, state_space, greedy_metrics["baseline_gating"], transition_s, transition_energy_j
    )
    out["greedy_gating"] = _energy_per_interval_j(
        sol_gg[0], sol_gg[1], sol_gg[2], fps_target, p_sleep_base, transition_s, transition_energy_j
    ) if sol_gg is not None else float("nan")

    return out


def _feasible_fps_rows_for_baseline(
    rows: List[Dict[str, str]],
    nominal_metrics: List[LayerMetric],
) -> List[Dict[str, str]]:
    _, total_t, _ = _schedule_totals_fixed(nominal_metrics)
    return [row for row in rows if total_t <= (1.0 / float(row["fps"]))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-model normalized interval-energy bar plot at the maximum baseline-feasible FPS."
    )
    ap.add_argument("--v-low", type=float, default=0.9)
    ap.add_argument("--v-nom", type=float, default=1.1)
    ap.add_argument("--v-high", type=float, default=1.3)
    ap.add_argument("--level-count", type=int, default=3, help="Discrete levels per domain for greedy.")
    ap.add_argument("--p-sleep-base", type=float, default=0.0)
    ap.add_argument("--relaxed-ratio", type=float, default=0.5, help="Relative relaxed point within the baseline-feasible range.")
    ap.add_argument(
        "--normalize-to",
        choices=["baseline", "baseline_gating"],
        default="baseline",
        help="Baseline used for normalization and FPS-region selection.",
    )
    ap.add_argument(
        "--out-csv",
        default=os.path.join(_HERE, "../data", "figure6.csv"),
    )
    ap.add_argument(
        "--out-pdf",
        "--out-png",
        dest="out_pdf",
        default=os.path.join(_HERE, "../outputs", "figure6.pdf"),
        help="Output plot path. PDF is used by default; extension controls the saved format.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)), exist_ok=True)

    models = [
        ModelCfg(
            key="squeezenet",
            label="SqueezeNet",
            compiler_csv= "./data/3rails_fps/compiler_results_squeezenet_3rails.csv"
        ),
        ModelCfg(
            key="resnet18",
            label="ResNet18",
            compiler_csv= "./data/3rails_fps/compiler_results_resnet_3rails.csv"

        ),
        ModelCfg(
            key="mobilevit_xxs",
            label="MobileViT",
            compiler_csv= "./data/3rails_fps/compiler_results_mobilevit_3rails.csv"

        ),
        ModelCfg(
            key="mobilenetv3_small",
            label="MobileNetV3",
            compiler_csv= "./data/3rails_fps/compiler_results_mobilenet_3rails.csv"
        ),
    ]

    summary_rows: List[Dict[str, float | str]] = []
    method_keys = ["baseline", "baseline_gating", "greedy", "greedy_gating", "ours"]
    method_labels = {
        "baseline": "Baseline",
        "baseline_gating": "+Gating",
        "greedy": "+Greedy",
        "greedy_gating": "+Gating+Greedy",
        "ours": "Solver",
    }
    colors = {
        "baseline": "#c7e9b4",
        "baseline_gating": "#7fcdbb",
        "greedy": "#41b6c4",
        "greedy_gating": "#1d91c0",
        "ours": "#225ea8",
    }

    for model in models:
        rows = _load_compiler_rows(model.compiler_csv)
        if not rows:
            continue
        transition_energy_j, transition_s = _infer_transition_params(rows)
        prepared = _prepare_model_metrics(
            model_key=model.key,
            v_low=args.v_low,
            v_nom=args.v_nom,
            v_high=args.v_high,
            level_count=args.level_count,
        )
        feasible_rows = _feasible_fps_rows_for_baseline(rows, prepared.nominal_metrics[args.normalize_to])
        if not feasible_rows:
            continue
        for operating_point, selected_fps in (
            ("relaxed", _select_relative_fps(feasible_rows, args.relaxed_ratio)),
            ("max", _select_max_fps(feasible_rows)),
        ):
            energies = _compute_method_energies(
                prepared=prepared,
                fps_target=selected_fps,
                transition_energy_j=transition_energy_j,
                transition_s=transition_s,
                p_sleep_base=args.p_sleep_base,
            )
            energies["ours"] = _compiler_energy_at_fps(rows, selected_fps)
            base_energy = energies[args.normalize_to]
            if not (base_energy == base_energy and base_energy > 0.0):
                continue
            summary_rows.append(
                {
                    "model": model.label,
                    "operating_point": operating_point,
                    "selected_fps": selected_fps,
                    "normalization_baseline": args.normalize_to,
                    "baseline_energy_j": energies["baseline"],
                    "baseline_gating_energy_j": energies["baseline_gating"],
                    "greedy_energy_j": energies["greedy"],
                    "greedy_gating_energy_j": energies["greedy_gating"],
                    "ours_energy_j": energies["ours"],
                    "baseline_norm": energies["baseline"] / base_energy,
                    "baseline_gating_norm": energies["baseline_gating"] / base_energy,
                    "greedy_norm": energies["greedy"] / base_energy,
                    "greedy_gating_norm": energies["greedy_gating"] / base_energy,
                    "ours_norm": energies["ours"] / base_energy,
                }
            )

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["selection_rule", "relaxed_and_max_baseline_feasible_fps"])
        w.writerow(["relaxed_ratio", args.relaxed_ratio])
        w.writerow(["normalize_to", args.normalize_to])
        w.writerow(["metric", "normalized_interval_energy"])
        w.writerow([])
        w.writerow(
            [
                "model",
                "operating_point",
                "selected_fps",
                "normalization_baseline",
                "baseline_energy_j",
                "baseline_gating_energy_j",
                "greedy_energy_j",
                "greedy_gating_energy_j",
                "ours_energy_j",
                "baseline_norm",
                "baseline_gating_norm",
                "greedy_norm",
                "greedy_gating_norm",
                "ours_norm",
            ]
        )
        for row in summary_rows:
            w.writerow(
                [
                    row["model"],
                    row["operating_point"],
                    row["selected_fps"],
                    row["normalization_baseline"],
                    row["baseline_energy_j"],
                    row["baseline_gating_energy_j"],
                    row["greedy_energy_j"],
                    row["greedy_gating_energy_j"],
                    row["ours_energy_j"],
                    row["baseline_norm"],
                    row["baseline_gating_norm"],
                    row["greedy_norm"],
                    row["greedy_gating_norm"],
                    row["ours_norm"],
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.0), dpi=220, sharey=True)
    width = 0.15
    offsets = {
        "baseline": -2 * width,
        "baseline_gating": -1 * width,
        "greedy": 0.0,
        "greedy_gating": width,
        "ours": 2 * width,
    }
    subplot_titles = {
        "relaxed": f"Relaxed FPS ({int(args.relaxed_ratio * 100)}%)",
        "max": "Max Baseline Feasible FPS",
    }
    tick_label = {
        "SqueezeNet": "SqNet",
        "ResNet18": "ResNet",
        "MobileViT": "MViT",
        "MobileNetV3": "MNetV3",
    }

    for ax, operating_point in zip(axes, ("max", "relaxed")):
        point_rows = [row for row in summary_rows if row["operating_point"] == operating_point]
        x_models = list(range(len(point_rows)))
        for method in method_keys:
            vals_norm = [float(row[f"{method}_norm"]) for row in point_rows]
            ax.bar(
                [x + offsets[method] for x in x_models],
                vals_norm,
                width=width,
                color=colors[method],
                edgecolor="black",
                linewidth=0.4,
                label=method_labels[method],
            )
        x_labels = [tick_label.get(str(row["model"]), str(row["model"])) for row in point_rows]
        ax.set_xticks(x_models)
        # ax.set_xticklabels(x_labels)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        ax.set_title(subplot_titles[operating_point], fontweight="bold", pad=-4)
        
        ax.axhline(1.0, color="#253494", linewidth=0.6, linestyle="--")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", pad=-2, width=0, labelsize=8)
        ax.tick_params(axis="y", pad=0, width=0.5, labelsize=8)
        ax.spines["top"].set_linewidth(0.5)
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

    axes[0].set_ylabel("Norm. Inter. Energy", labelpad=0)
    axes[0].yaxis.set_label_coords(-0.2, 0.6)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles and labels:
        desired = [
            "Baseline",
            "+Gating",
            "+Greedy",
            "+Gating+Greedy",
            "Solver",
        ]
        by_label = {label: handle for handle, label in zip(handles, labels)}
        labels = [label for label in desired if label in by_label]
        handles = [by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        fontsize=rcParams["legend.fontsize"],
        ncol=5,
        columnspacing=0.25,
        handlelength=0.7,
        handletextpad=0.1,
        borderpad=0,
        labelspacing=0.2,
    )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.13, top=0.75, wspace=0.06)
    fig.savefig(args.out_pdf, dpi=220, pad_inches=0, transparent=False, bbox_inches="tight")

    # if plt.get_backend().lower() != "agg":
    #     plt.show()
    # else:
    #     plt.close(fig)

    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved plot: {args.out_pdf}")
    plt.close(fig)

if __name__ == "__main__":
    main()
