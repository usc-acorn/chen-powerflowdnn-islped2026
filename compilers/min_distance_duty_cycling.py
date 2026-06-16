from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SolveResult:
    energy: float
    active_time: float
    idle_time: float
    sleep_time: float
    fps: float
    rail_switch_count: int
    sleep_rail_switch_count: int
    used_sleep_mode: bool
    used_v_sys_rails: int
    used_v_sys_values: Tuple[float, ...]
    selected_path: Tuple[Tuple[float, ...], ...] = ()
    method: str = "min_distance_dp_new"


def _load_layers(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    layers_dict = data.get("layers", {})
    try:
        sorted_keys = sorted(layers_dict.keys(), key=lambda x: int(x))
    except Exception:
        sorted_keys = sorted(layers_dict.keys())

    layers_list = [layers_dict[k] for k in sorted_keys]
    combos_list = [layer.get("combinations", layer) for layer in layers_list]
    v_sys_candidates = data.get("meta", {}).get("v_sys_candidates", [])
    if not v_sys_candidates:
        vals = set()
        for combos in combos_list:
            for state in combos.values():
                vals.add(float(state["v_sys"]))
        v_sys_candidates = sorted(vals)
    return combos_list, [float(v) for v in v_sys_candidates]


def _state_matches_domain_constraints(
    state: Dict[str, float],
    max_unique_domains: Optional[int],
    exact_unique_domains: Optional[int],
    required_equalities: Optional[Sequence[Tuple[str, str]]],
    allowed_voltage_sets: Optional[Sequence[Sequence[float]]],
) -> bool:
    v_sys = float(state["v_sys"])
    v_rram = float(state["v_rram"])
    v_feeder = float(state["v_feeder"])
    unique_voltages = {v_sys, v_rram, v_feeder}
    num_domains = len(unique_voltages)

    if exact_unique_domains is not None and num_domains != exact_unique_domains:
        return False
    if max_unique_domains is not None and num_domains > max_unique_domains:
        return False

    if required_equalities:
        for lhs, rhs in required_equalities:
            if float(state[lhs]) != float(state[rhs]):
                return False

    if allowed_voltage_sets:
        normalized_sets = {tuple(sorted(float(v) for v in vals)) for vals in allowed_voltage_sets}
        if tuple(sorted(unique_voltages)) not in normalized_sets:
            return False

    return True


def _build_layer_arrays(
    combos_list: Sequence[Dict[str, Dict[str, float]]],
    allowed_v_sys: Sequence[float],
    max_unique_domains: Optional[int] = None,
    exact_unique_domains: Optional[int] = None,
    required_equalities: Optional[Sequence[Tuple[str, str]]] = None,
    allowed_voltage_sets: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[List[Dict[str, np.ndarray]]]:
    allowed = set(float(v) for v in allowed_v_sys)
    out: List[Dict[str, np.ndarray]] = []

    for combos in combos_list:
        states = [
            c
            for c in combos.values()
            if float(c["v_sys"]) in allowed
            and float(c["v_rram"]) in allowed
            and float(c["v_feeder"]) in allowed
            and _state_matches_domain_constraints(
                c,
                max_unique_domains=max_unique_domains,
                exact_unique_domains=exact_unique_domains,
                required_equalities=required_equalities,
                allowed_voltage_sets=allowed_voltage_sets,
            )
        ]
        if not states:
            return None

        out.append(
            {
                "energy": np.asarray([float(s["energy_j"]) for s in states], dtype=np.float64),
                "time": np.asarray([float(s["time_s"]) for s in states], dtype=np.float64),
                "leak": np.asarray([float(s["Leakage_w"]) for s in states], dtype=np.float64),
                "v_sys": np.asarray([float(s["v_sys"]) for s in states], dtype=np.float64),
                "v_rram": np.asarray([float(s["v_rram"]) for s in states], dtype=np.float64),
                "v_feeder": np.asarray([float(s["v_feeder"]) for s in states], dtype=np.float64),
            }
        )
    return out


def _run_dp_weighted(
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    lam: float,
    e_trans_unit: float,
    l_trans_unit: float,
) -> Tuple[List[np.ndarray], int]:
    backptr: List[np.ndarray] = []
    cost_prev = layer_arrays[0]["energy"] + lam * layer_arrays[0]["time"]
    backptr.append(np.full(cost_prev.shape[0], -1, dtype=np.int32))

    for layer_idx in range(1, len(layer_arrays)):
        prev = layer_arrays[layer_idx - 1]
        cur = layer_arrays[layer_idx]

        changed_count = (
            (prev["v_sys"][:, None] != cur["v_sys"][None, :]).astype(np.float64)
            + (prev["v_rram"][:, None] != cur["v_rram"][None, :]).astype(np.float64)
            + (prev["v_feeder"][:, None] != cur["v_feeder"][None, :]).astype(np.float64)
        )
        trans = changed_count * (e_trans_unit + lam * l_trans_unit)

        transition_cost = cost_prev[:, None] + trans
        arg = np.argmin(transition_cost, axis=0).astype(np.int32)
        best = transition_cost[arg, np.arange(transition_cost.shape[1])]

        cost_prev = best + cur["energy"] + lam * cur["time"]
        backptr.append(arg)

    final_state = int(np.argmin(cost_prev))
    return backptr, final_state


def _reconstruct_path(
    backptr: Sequence[np.ndarray],
    final_state: int,
) -> List[int]:
    path = [0] * len(backptr)
    path[-1] = final_state
    for layer_idx in range(len(backptr) - 1, 0, -1):
        path[layer_idx - 1] = int(backptr[layer_idx][path[layer_idx]])
    return path


def _eval_path(
    path: Sequence[int],
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    t_target: float,
    p_sleep_base: float,
    e_trans_unit: float,
    l_trans_unit: float,
) -> Tuple[float, float, float, float, int, int, bool]:
    active_energy = 0.0
    active_time = 0.0
    rail_switch_count = 0

    for layer_idx, state_idx in enumerate(path):
        state_idx = int(state_idx)
        active_energy += float(layer_arrays[layer_idx]["energy"][state_idx])
        active_time += float(layer_arrays[layer_idx]["time"][state_idx])
        if layer_idx > 0:
            prev_state_idx = int(path[layer_idx - 1])
            changed_count = (
                int(layer_arrays[layer_idx - 1]["v_sys"][prev_state_idx] != layer_arrays[layer_idx]["v_sys"][state_idx])
                + int(layer_arrays[layer_idx - 1]["v_rram"][prev_state_idx] != layer_arrays[layer_idx]["v_rram"][state_idx])
                + int(layer_arrays[layer_idx - 1]["v_feeder"][prev_state_idx] != layer_arrays[layer_idx]["v_feeder"][state_idx])
            )
            if changed_count:
                rail_switch_count += changed_count
                active_energy += changed_count * e_trans_unit
                active_time += changed_count * l_trans_unit

    if active_time > t_target:
        return float("inf"), active_time, 0.0, 0.0, rail_switch_count, 0, False

    last_leak_w = float(layer_arrays[-1]["leak"][path[-1]])
    raw_slack = t_target - active_time

    best_total_energy = active_energy + last_leak_w * raw_slack
    best_active_time = active_time
    best_idle_time = raw_slack
    best_sleep_time = 0.0
    best_sleep_switch_count = 0
    used_sleep_mode = False

    sleep_switch_count = 6
    sleep_transition_time = sleep_switch_count * l_trans_unit
    sleep_transition_energy = sleep_switch_count * e_trans_unit
    if raw_slack >= sleep_transition_time:
        sleep_time = raw_slack - sleep_transition_time
        sleep_total_energy = active_energy + sleep_transition_energy + p_sleep_base * sleep_time
        if sleep_total_energy < best_total_energy:
            best_total_energy = sleep_total_energy
            best_active_time = active_time + sleep_transition_time
            best_idle_time = 0.0
            best_sleep_time = sleep_time
            best_sleep_switch_count = sleep_switch_count
            used_sleep_mode = True

    return (
        best_total_energy,
        best_active_time,
        best_idle_time,
        best_sleep_time,
        rail_switch_count,
        best_sleep_switch_count,
        used_sleep_mode,
    )


def _path_to_records(
    path: Sequence[int],
    layer_arrays: Sequence[Dict[str, np.ndarray]],
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for layer_idx, state_idx in enumerate(path):
        state_idx = int(state_idx)
        records.append(
            {
                "layer_idx": layer_idx,
                "state_idx": state_idx,
                "v_sys": float(layer_arrays[layer_idx]["v_sys"][state_idx]),
                "v_rram": float(layer_arrays[layer_idx]["v_rram"][state_idx]),
                "v_feeder": float(layer_arrays[layer_idx]["v_feeder"][state_idx]),
                "energy_j": float(layer_arrays[layer_idx]["energy"][state_idx]),
                "time_s": float(layer_arrays[layer_idx]["time"][state_idx]),
                "leakage_w": float(layer_arrays[layer_idx]["leak"][state_idx]),
            }
        )
    return records


def _voltage_set(layer: Dict[str, np.ndarray], state_idx: int) -> frozenset[float]:
    return frozenset(
        (
            float(layer["v_sys"][state_idx]),
            float(layer["v_rram"][state_idx]),
            float(layer["v_feeder"][state_idx]),
        )
    )


def _structure_dominates(
    layer: Dict[str, np.ndarray],
    keep_idx: int,
    drop_idx: int,
) -> bool:
    keep_set = _voltage_set(layer, keep_idx)
    drop_set = _voltage_set(layer, drop_idx)
    if not keep_set.issubset(drop_set):
        return False

    keep_energy = float(layer["energy"][keep_idx])
    drop_energy = float(layer["energy"][drop_idx])
    keep_time = float(layer["time"][keep_idx])
    drop_time = float(layer["time"][drop_idx])
    keep_leak = float(layer["leak"][keep_idx])
    drop_leak = float(layer["leak"][drop_idx])

    no_worse = (
        keep_energy <= drop_energy
        and keep_time <= drop_time
        and keep_leak <= drop_leak
    )
    strictly_better = (
        keep_energy < drop_energy
        or keep_time < drop_time
        or keep_leak < drop_leak
    )
    return no_worse and strictly_better


def _prune_combos_structure_aware(
    combos: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    keys = list(combos.keys())
    n_states = len(keys)
    if n_states <= 1:
        return dict(combos), {"before": n_states, "after": n_states, "pruned": 0}

    layer = {
        "energy": np.asarray([float(combos[k]["energy_j"]) for k in keys], dtype=np.float64),
        "time": np.asarray([float(combos[k]["time_s"]) for k in keys], dtype=np.float64),
        "leak": np.asarray([float(combos[k]["Leakage_w"]) for k in keys], dtype=np.float64),
        "v_sys": np.asarray([float(combos[k]["v_sys"]) for k in keys], dtype=np.float64),
        "v_rram": np.asarray([float(combos[k]["v_rram"]) for k in keys], dtype=np.float64),
        "v_feeder": np.asarray([float(combos[k]["v_feeder"]) for k in keys], dtype=np.float64),
    }

    survivors = np.ones(n_states, dtype=bool)
    for drop_idx in range(n_states):
        if not survivors[drop_idx]:
            continue
        for keep_idx in range(n_states):
            if keep_idx == drop_idx or not survivors[keep_idx]:
                continue
            if _structure_dominates(layer, keep_idx, drop_idx):
                survivors[drop_idx] = False
                break

    kept = {key: combos[key] for idx, key in enumerate(keys) if survivors[idx]}
    after = int(np.sum(survivors))
    return kept, {"before": n_states, "after": after, "pruned": n_states - after}


def _apply_global_structure_pruning(
    combos_list: Sequence[Dict[str, Dict[str, float]]],
) -> Tuple[List[Dict[str, Dict[str, float]]], Dict[str, object]]:
    pruned_combos: List[Dict[str, Dict[str, float]]] = []
    per_layer: List[Dict[str, int]] = []
    total_before = 0
    total_after = 0

    for combos in combos_list:
        pruned_layer, stats = _prune_combos_structure_aware(combos)
        pruned_combos.append(pruned_layer)
        per_layer.append(stats)
        total_before += stats["before"]
        total_after += stats["after"]

    return pruned_combos, {
        "per_layer": per_layer,
        "total_before": total_before,
        "total_after": total_after,
        "total_pruned": total_before - total_after,
    }


def _run_lambda_candidate(
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    lam: float,
    t_target: float,
    p_sleep_base: float,
    e_trans_unit: float,
    l_trans_unit: float,
) -> Tuple[List[int], Tuple[float, float, float, float, int, int, bool]]:
    backptr, final_state = _run_dp_weighted(layer_arrays, lam, e_trans_unit, l_trans_unit)
    path = _reconstruct_path(backptr, final_state)
    metrics = _eval_path(
        path,
        layer_arrays,
        t_target,
        p_sleep_base,
        e_trans_unit,
        l_trans_unit,
    )
    return path, metrics


def _changed_count_between(
    left_layer: Dict[str, np.ndarray],
    left_state_idx: int,
    right_layer: Dict[str, np.ndarray],
    right_state_idx: int,
) -> int:
    return (
        int(left_layer["v_sys"][left_state_idx] != right_layer["v_sys"][right_state_idx])
        + int(left_layer["v_rram"][left_state_idx] != right_layer["v_rram"][right_state_idx])
        + int(left_layer["v_feeder"][left_state_idx] != right_layer["v_feeder"][right_state_idx])
    )


def _active_path_totals(
    path: Sequence[int],
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    e_trans_unit: float,
    l_trans_unit: float,
) -> Tuple[float, float, int]:
    active_energy = 0.0
    active_time = 0.0
    rail_switch_count = 0

    for layer_idx, state_idx in enumerate(path):
        layer = layer_arrays[layer_idx]
        state_idx = int(state_idx)
        active_energy += float(layer["energy"][state_idx])
        active_time += float(layer["time"][state_idx])
        if layer_idx > 0:
            prev_layer = layer_arrays[layer_idx - 1]
            prev_state_idx = int(path[layer_idx - 1])
            changed_count = _changed_count_between(prev_layer, prev_state_idx, layer, state_idx)
            if changed_count:
                rail_switch_count += changed_count
                active_energy += changed_count * e_trans_unit
                active_time += changed_count * l_trans_unit

    return active_energy, active_time, rail_switch_count


def _finish_path_metrics(
    active_energy: float,
    active_time: float,
    rail_switch_count: int,
    last_leak_w: float,
    t_target: float,
    p_sleep_base: float,
    e_trans_unit: float,
    l_trans_unit: float,
) -> Tuple[float, float, float, float, int, int, bool]:
    if active_time > t_target:
        return float("inf"), active_time, 0.0, 0.0, rail_switch_count, 0, False

    raw_slack = t_target - active_time
    stay_total_energy = active_energy + last_leak_w * raw_slack
    best_total_energy = stay_total_energy
    best_active_time = active_time
    best_idle_time = raw_slack
    best_sleep_time = 0.0
    best_sleep_switch_count = 0
    used_sleep_mode = False

    sleep_switch_count = 6
    sleep_transition_time = sleep_switch_count * l_trans_unit
    sleep_transition_energy = sleep_switch_count * e_trans_unit
    if raw_slack >= sleep_transition_time:
        sleep_time = raw_slack - sleep_transition_time
        sleep_total_energy = active_energy + sleep_transition_energy + p_sleep_base * sleep_time
        if sleep_total_energy < best_total_energy:
            best_total_energy = sleep_total_energy
            best_active_time = active_time + sleep_transition_time
            best_idle_time = 0.0
            best_sleep_time = sleep_time
            best_sleep_switch_count = sleep_switch_count
            used_sleep_mode = True

    return (
        best_total_energy,
        best_active_time,
        best_idle_time,
        best_sleep_time,
        rail_switch_count,
        best_sleep_switch_count,
        used_sleep_mode,
    )


def _single_layer_delta(
    path: Sequence[int],
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    layer_idx: int,
    new_state_idx: int,
    e_trans_unit: float,
    l_trans_unit: float,
) -> Tuple[float, float, int]:
    old_state_idx = int(path[layer_idx])
    if old_state_idx == new_state_idx:
        return 0.0, 0.0, 0

    layer = layer_arrays[layer_idx]
    delta_energy = float(layer["energy"][new_state_idx] - layer["energy"][old_state_idx])
    delta_time = float(layer["time"][new_state_idx] - layer["time"][old_state_idx])
    delta_switch_count = 0

    if layer_idx > 0:
        prev_layer = layer_arrays[layer_idx - 1]
        prev_state_idx = int(path[layer_idx - 1])
        old_changed = _changed_count_between(prev_layer, prev_state_idx, layer, old_state_idx)
        new_changed = _changed_count_between(prev_layer, prev_state_idx, layer, new_state_idx)
        delta_changed = new_changed - old_changed
        delta_switch_count += delta_changed
        delta_energy += delta_changed * e_trans_unit
        delta_time += delta_changed * l_trans_unit

    if layer_idx < len(path) - 1:
        next_layer = layer_arrays[layer_idx + 1]
        next_state_idx = int(path[layer_idx + 1])
        old_changed = _changed_count_between(layer, old_state_idx, next_layer, next_state_idx)
        new_changed = _changed_count_between(layer, new_state_idx, next_layer, next_state_idx)
        delta_changed = new_changed - old_changed
        delta_switch_count += delta_changed
        delta_energy += delta_changed * e_trans_unit
        delta_time += delta_changed * l_trans_unit

    return delta_energy, delta_time, delta_switch_count


def _local_refine_path(
    path: Sequence[int],
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    t_target: float,
    p_sleep_base: float,
    e_trans_unit: float,
    l_trans_unit: float,
    max_passes: int = 8,
    min_improvement_j: float = 1e-18,
) -> Tuple[List[int], Tuple[float, float, float, float, int, int, bool], int]:
    cur_path = [int(v) for v in path]
    active_energy, active_time, rail_switch_count = _active_path_totals(
        cur_path,
        layer_arrays,
        e_trans_unit,
        l_trans_unit,
    )
    cur_metrics = _finish_path_metrics(
        active_energy,
        active_time,
        rail_switch_count,
        float(layer_arrays[-1]["leak"][cur_path[-1]]),
        t_target,
        p_sleep_base,
        e_trans_unit,
        l_trans_unit,
    )
    if not np.isfinite(cur_metrics[0]):
        return cur_path, cur_metrics, 0

    move_count = 0
    for _ in range(max_passes):
        best_move: Optional[Tuple[int, int, float, float, int]] = None
        best_metrics = cur_metrics

        for layer_idx, layer in enumerate(layer_arrays):
            old_state_idx = cur_path[layer_idx]
            n_states = int(layer["energy"].shape[0])
            for cand_state_idx in range(n_states):
                if cand_state_idx == old_state_idx:
                    continue
                delta_energy, delta_time, delta_switch_count = _single_layer_delta(
                    cur_path,
                    layer_arrays,
                    layer_idx,
                    cand_state_idx,
                    e_trans_unit,
                    l_trans_unit,
                )
                trial_active_energy = active_energy + delta_energy
                trial_active_time = active_time + delta_time
                trial_switch_count = rail_switch_count + delta_switch_count
                last_leak_w = float(layer_arrays[-1]["leak"][cur_path[-1]])
                if layer_idx == len(layer_arrays) - 1:
                    last_leak_w = float(layer_arrays[-1]["leak"][cand_state_idx])
                trial_metrics = _finish_path_metrics(
                    trial_active_energy,
                    trial_active_time,
                    trial_switch_count,
                    last_leak_w,
                    t_target,
                    p_sleep_base,
                    e_trans_unit,
                    l_trans_unit,
                )
                if not np.isfinite(trial_metrics[0]):
                    continue
                if trial_metrics[0] < best_metrics[0] - min_improvement_j:
                    best_move = (
                        layer_idx,
                        cand_state_idx,
                        delta_energy,
                        delta_time,
                        delta_switch_count,
                    )
                    best_metrics = trial_metrics

        if best_move is None:
            break
        layer_idx, cand_state_idx, delta_energy, delta_time, delta_switch_count = best_move
        cur_path[layer_idx] = cand_state_idx
        active_energy += delta_energy
        active_time += delta_time
        rail_switch_count += delta_switch_count
        cur_metrics = best_metrics
        move_count += 1

    return cur_path, cur_metrics, move_count


def _solve_for_rail_subset(
    layer_arrays: Sequence[Dict[str, np.ndarray]],
    t_target: float,
    p_sleep_base: float,
    e_trans_unit: float,
    l_trans_unit: float,
    max_lambda: float = 1e8,
    bsearch_steps: int = 28,
    local_refinement: bool = True,
    max_refine_passes: int = 8,
    refine_candidate_count: int = 10,
) -> Optional[Dict[str, object]]:
    candidate_paths: List[Tuple[List[int], Tuple[float, float, float, float, int, int, bool], float]] = []
    trial_lambdas: List[float] = []

    def add_candidate(lam: float) -> bool:
        trial_lambdas.append(float(lam))
        path, metrics = _run_lambda_candidate(
            layer_arrays,
            lam,
            t_target,
            p_sleep_base,
            e_trans_unit,
            l_trans_unit,
        )
        if np.isfinite(metrics[0]):
            candidate_paths.append((path, metrics, float(lam)))
            return True
        return False

    p0, metrics0 = _run_lambda_candidate(
        layer_arrays,
        0.0,
        t_target,
        p_sleep_base,
        e_trans_unit,
        l_trans_unit,
    )
    trial_lambdas.append(0.0)
    if np.isfinite(metrics0[0]):
        candidate_paths.append((p0, metrics0, 0.0))

    lo = 0.0
    hi = 1.0
    feasible_hi = False
    while hi <= max_lambda:
        if add_candidate(hi):
            feasible_hi = True
            break
        lo = hi
        hi *= 2.0

    if feasible_hi:
        for _ in range(bsearch_steps):
            mid = 0.5 * (lo + hi)
            if add_candidate(mid):
                hi = mid
            else:
                lo = mid

    if not candidate_paths:
        return None

    candidate_paths.sort(key=lambda item: item[1][0])

    best_path = candidate_paths[0][0]
    best_metrics = candidate_paths[0][1]
    best_lambda = candidate_paths[0][2]
    total_refine_moves = 0
    refined_candidate_count = 0

    if local_refinement and refine_candidate_count != 0:
        if refine_candidate_count < 0:
            refinement_candidates = candidate_paths
        else:
            refinement_candidates = candidate_paths[:refine_candidate_count]
    else:
        refinement_candidates = []

    for path, metrics, lam in refinement_candidates:
        cand_path = path
        cand_metrics = metrics
        refine_moves = 0
        cand_path, cand_metrics, refine_moves = _local_refine_path(
            path,
            layer_arrays,
            t_target,
            p_sleep_base,
            e_trans_unit,
            l_trans_unit,
            max_passes=max_refine_passes,
        )
        refined_candidate_count += 1
        if cand_metrics[0] < best_metrics[0]:
            best_path = cand_path
            best_metrics = cand_metrics
            best_lambda = lam
            total_refine_moves = refine_moves

    return {
        "energy": best_metrics[0],
        "active_time": best_metrics[1],
        "idle_time": best_metrics[2],
        "sleep_time": best_metrics[3],
        "rail_switch_count": best_metrics[4],
        "sleep_rail_switch_count": best_metrics[5],
        "used_sleep_mode": best_metrics[6],
        "selected_path": _path_to_records(best_path, layer_arrays),
        "best_lambda": best_lambda,
        "trial_lambdas": sorted(set(trial_lambdas)),
        "local_refinement_moves": total_refine_moves,
        "refined_candidate_count": refined_candidate_count,
    }


def _selected_path_tuple(selected_path: Sequence[Dict[str, float]]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(
        (
            float(p["layer_idx"]),
            float(p["state_idx"]),
            float(p["v_sys"]),
            float(p["v_rram"]),
            float(p["v_feeder"]),
            float(p["energy_j"]),
            float(p["time_s"]),
            float(p["leakage_w"]),
        )
        for p in selected_path
    )


def _result_to_dict(
    best_global: SolveResult,
    best_result: Dict[str, object],
    max_unique_domains: Optional[int],
    exact_unique_domains: Optional[int],
    pruning_stats: Dict[str, object],
    prune_runtime_s: float,
) -> Dict[str, object]:
    return {
        "energy": best_global.energy,
        "active_time": best_global.active_time,
        "idle_time": best_global.idle_time,
        "sleep_time": best_global.sleep_time,
        "fps": best_global.fps,
        "rail_switch_count": best_global.rail_switch_count,
        "sleep_rail_switch_count": best_global.sleep_rail_switch_count,
        "total_rail_switch_count": best_global.rail_switch_count + best_global.sleep_rail_switch_count,
        "used_sleep_mode": best_global.used_sleep_mode,
        "used_v_sys_rails": best_global.used_v_sys_rails,
        "used_v_sys_values": list(best_global.used_v_sys_values),
        "selected_path": [
            {
                "layer_idx": int(p[0]),
                "state_idx": int(p[1]),
                "v_sys": p[2],
                "v_rram": p[3],
                "v_feeder": p[4],
                "energy_j": p[5],
                "time_s": p[6],
                "leakage_w": p[7],
            }
            for p in best_global.selected_path
        ],
        "shared_rail_constraint": True,
        "max_unique_domains": max_unique_domains,
        "exact_unique_domains": exact_unique_domains,
        "method": best_global.method,
        "best_lambda": float(best_result.get("best_lambda", 0.0)),
        "trial_lambdas": list(best_result.get("trial_lambdas", [])),
        "local_refinement_moves": int(best_result.get("local_refinement_moves", 0)),
        "refined_candidate_count": int(best_result.get("refined_candidate_count", 0)),
        "pruning_stats": pruning_stats,
        "prune_runtime_s": prune_runtime_s,
        "solve_runtime_s": float(best_result.get("solve_runtime_s", 0.0)),
        "total_runtime_s": prune_runtime_s + float(best_result.get("solve_runtime_s", 0.0)),
    }


def solve_scheduling_with_rail_limit(
    file_path,
    t_target,
    p_sleep_base,
    max_v_sys_rails=5,
    max_unique_domains=3,
    exact_unique_domains=None,
    required_equalities=None,
    allowed_voltage_sets=None,
    e_trans_unit=0.0000,
    l_trans_unit=5e-9,
    structure_pruning: bool = True,
    local_refinement: bool = True,
    max_refine_passes: int = 8,
    refine_candidate_count: int = 10,
):
    combos_list, v_sys_candidates = _load_layers(file_path)
    if max_v_sys_rails <= 0:
        return None

    pruning_stats: Dict[str, object] = {
        "per_layer": [],
        "total_before": sum(len(combos) for combos in combos_list),
        "total_after": sum(len(combos) for combos in combos_list),
        "total_pruned": 0,
    }
    prune_runtime_s = 0.0
    if structure_pruning:
        prune_start = time.perf_counter()
        combos_list, pruning_stats = _apply_global_structure_pruning(combos_list)
        prune_runtime_s = time.perf_counter() - prune_start

    best_global: Optional[SolveResult] = None
    best_result: Optional[Dict[str, object]] = None

    max_r = min(max_v_sys_rails, len(v_sys_candidates))
    rail_subsets: List[Tuple[float, ...]] = []
    for r in range(1, max_r + 1):
        rail_subsets.extend(itertools.combinations(v_sys_candidates, r))

    method = "min_distance_dp_new"
    if structure_pruning:
        method += "_struct_prune"
    if local_refinement:
        method += "_local_refine"

    for subset in rail_subsets:
        layer_arrays = _build_layer_arrays(
            combos_list,
            subset,
            max_unique_domains=max_unique_domains,
            exact_unique_domains=exact_unique_domains,
            required_equalities=required_equalities,
            allowed_voltage_sets=allowed_voltage_sets,
        )
        if layer_arrays is None:
            continue

        solve_start = time.perf_counter()
        sol = _solve_for_rail_subset(
            layer_arrays=layer_arrays,
            t_target=t_target,
            p_sleep_base=p_sleep_base,
            e_trans_unit=e_trans_unit,
            l_trans_unit=l_trans_unit,
            local_refinement=local_refinement,
            max_refine_passes=max_refine_passes,
            refine_candidate_count=refine_candidate_count,
        )
        solve_runtime_s = time.perf_counter() - solve_start
        if sol is None:
            continue

        selected_path = _selected_path_tuple(sol.get("selected_path", []))
        cand = SolveResult(
            energy=float(sol["energy"]),
            active_time=float(sol["active_time"]),
            idle_time=float(sol["idle_time"]),
            sleep_time=float(sol["sleep_time"]),
            fps=1.0 / t_target,
            rail_switch_count=int(sol["rail_switch_count"]),
            sleep_rail_switch_count=int(sol["sleep_rail_switch_count"]),
            used_sleep_mode=bool(sol["used_sleep_mode"]),
            used_v_sys_rails=len(subset),
            used_v_sys_values=tuple(float(v) for v in subset),
            selected_path=selected_path,
            method=method,
        )
        if best_global is None or cand.energy < best_global.energy:
            best_global = cand
            best_result = dict(sol)
            best_result["solve_runtime_s"] = solve_runtime_s

    if best_global is None or best_result is None:
        return None

    return _result_to_dict(
        best_global=best_global,
        best_result=best_result,
        max_unique_domains=max_unique_domains,
        exact_unique_domains=exact_unique_domains,
        pruning_stats=pruning_stats,
        prune_runtime_s=prune_runtime_s,
    )
