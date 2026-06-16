import os
import sys
import time
import re
import json
from math import comb
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_COMPILER_DIR = os.path.join(_PROJECT_ROOT, "compilers")
for _p in (
    _PROJECT_ROOT,
    _COMPILER_DIR,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ilp_duty_cycling as ilp
import min_distance_duty_cycling as dp

# --- CONFIGURATION ---
FPS_CONFIG = {
    "mobilenetv3": 15.0,
    "mobilevit": 10.0,
    "resnet18": 10.0,
    "squeezenet": 50.0,
    "default": 33.3
}
MAX_V_SYS_RAILS = 3
P_SLEEP_BASE = 0.00
E_TRANS_UNIT = 0.0
L_TRANS_UNIT = 5e-9
DP_STRUCTURE_PRUNING = True
DATA_DIR = os.path.join(_PROJECT_ROOT, "./data", "runtime_exp") #FIXME
OUT_CSV = os.path.join(_PROJECT_ROOT, "./data", "figure9_compile_time.csv")
SCHEDULE_DIFF_DIR = os.path.join(_PROJECT_ROOT, "./data", "schedule_diffs")

def get_fps_for_model(file_name):
    name_lower = file_name.lower()
    for key, fps in FPS_CONFIG.items():
        if key in name_lower:
            return fps
    return FPS_CONFIG["default"]


def strip_model_suffix(name):
    if not isinstance(name, str): return name
    name = re.sub(r'\.json$|\.jso$|\.txt$|\.csv$', '', name, flags=re.IGNORECASE)
    return name


def load_problem_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    layers = payload.get("layers", {})
    v_sys = meta.get("v_sys_candidates", [])
    v_rram = meta.get("v_rram_candidates", [])
    v_feeder = meta.get("v_feeder_candidates", [])
    n_rails = len(v_sys)
    max_subset_size = min(MAX_V_SYS_RAILS, n_rails)
    subset_count = sum(comb(n_rails, r) for r in range(1, max_subset_size + 1))
    state_space = len(v_sys) * len(v_rram) * len(v_feeder)
    layer_count = len(layers)

    return {
        "Subsets": subset_count,
        "Layers": layer_count,
        "State_Space": state_space,
        "Effective_State_Graph_Size": layer_count * state_space,
    }


def transition_count(prev_state, cur_state):
    if prev_state is None or cur_state is None:
        return 0
    return int(float(prev_state["v_sys"]) != float(cur_state["v_sys"])) + \
        int(float(prev_state["v_rram"]) != float(cur_state["v_rram"])) + \
        int(float(prev_state["v_feeder"]) != float(cur_state["v_feeder"]))


def voltage_diff_domains(dp_state, ilp_state):
    domains = []
    for domain in ("v_sys", "v_rram", "v_feeder"):
        if float(dp_state[domain]) != float(ilp_state[domain]):
            domains.append(domain)
    return domains


def write_schedule_diff_csv(file_name, fps, method_label, res_dp, res_ilp):
    dp_path = res_dp.get("selected_path", [])
    ilp_path = res_ilp.get("selected_path", [])
    if not dp_path or not ilp_path:
        return "", 0, -1

    os.makedirs(SCHEDULE_DIFF_DIR, exist_ok=True)
    rows = []
    cum_energy_delta = 0.0
    cum_time_delta = 0.0
    cum_dp_transitions = 0
    cum_ilp_transitions = 0

    n_layers = min(len(dp_path), len(ilp_path))
    for layer_idx in range(n_layers):
        dp_state = dp_path[layer_idx]
        ilp_state = ilp_path[layer_idx]
        dp_prev = None if layer_idx == 0 else dp_path[layer_idx - 1]
        ilp_prev = None if layer_idx == 0 else ilp_path[layer_idx - 1]

        dp_transition_count = transition_count(dp_prev, dp_state)
        ilp_transition_count = transition_count(ilp_prev, ilp_state)
        cum_dp_transitions += dp_transition_count
        cum_ilp_transitions += ilp_transition_count

        layer_energy_delta = float(dp_state["energy_j"]) - float(ilp_state["energy_j"])
        layer_time_delta = float(dp_state["time_s"]) - float(ilp_state["time_s"])
        cum_energy_delta += layer_energy_delta
        cum_time_delta += layer_time_delta
        diff_domains = voltage_diff_domains(dp_state, ilp_state)

        rows.append(
            {
                "file_name": file_name,
                "fps": fps,
                "layer_idx": layer_idx,
                "voltage_tuple_matches": len(diff_domains) == 0,
                "changed_domains_vs_ilp": "|".join(diff_domains),
                "dp_state_idx": dp_state.get("state_idx", ""),
                "ilp_state_idx": ilp_state.get("state_idx", ""),
                "ilp_state_key": ilp_state.get("state_key", ""),
                "dp_v_sys": float(dp_state["v_sys"]),
                "ilp_v_sys": float(ilp_state["v_sys"]),
                "dp_v_rram": float(dp_state["v_rram"]),
                "ilp_v_rram": float(ilp_state["v_rram"]),
                "dp_v_feeder": float(dp_state["v_feeder"]),
                "ilp_v_feeder": float(ilp_state["v_feeder"]),
                "dp_layer_energy_j": float(dp_state["energy_j"]),
                "ilp_layer_energy_j": float(ilp_state["energy_j"]),
                "layer_energy_delta_j": layer_energy_delta,
                "cumulative_layer_energy_delta_j": cum_energy_delta,
                "dp_layer_time_s": float(dp_state["time_s"]),
                "ilp_layer_time_s": float(ilp_state["time_s"]),
                "layer_time_delta_s": layer_time_delta,
                "cumulative_layer_time_delta_s": cum_time_delta,
                "dp_leakage_w": float(dp_state["leakage_w"]),
                "ilp_leakage_w": float(ilp_state["leakage_w"]),
                "dp_transition_count_from_prev": dp_transition_count,
                "ilp_transition_count_from_prev": ilp_transition_count,
                "transition_count_delta_from_prev": dp_transition_count - ilp_transition_count,
                "cumulative_dp_transition_count": cum_dp_transitions,
                "cumulative_ilp_transition_count": cum_ilp_transitions,
            }
        )

    out_path = os.path.join(
        SCHEDULE_DIFF_DIR,
        f"{strip_model_suffix(file_name)}_{method_label}_schedule_diff.csv",
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    diff_mask = ~df["voltage_tuple_matches"]
    layer_diff_count = int(diff_mask.sum())
    first_diff_layer = int(df.loc[diff_mask, "layer_idx"].iloc[0]) if layer_diff_count else -1
    return out_path, layer_diff_count, first_diff_layer


def solve_dp_variant(label, path, t_target, local_refinement, structure_pruning=DP_STRUCTURE_PRUNING):
    start = time.perf_counter()
    try:
        res = dp.solve_scheduling_with_rail_limit(
            path,
            t_target=t_target,
            p_sleep_base=P_SLEEP_BASE,
            max_v_sys_rails=MAX_V_SYS_RAILS,
            e_trans_unit=E_TRANS_UNIT,
            l_trans_unit=L_TRANS_UNIT,
            structure_pruning=structure_pruning,
            local_refinement=local_refinement,
        )
    except Exception as exc:
        print(f"{label} failed for {os.path.basename(path)}: {exc}")
        return None, time.perf_counter() - start
    return res, time.perf_counter() - start


def solve_ilp_oracle(path, t_target):
    start = time.perf_counter()
    try:
        res = ilp.solve_scheduling_with_rail_limit(
            path,
            t_target=t_target,
            p_sleep_base=P_SLEEP_BASE,
            max_v_sys_rails=MAX_V_SYS_RAILS,
            e_trans_unit=E_TRANS_UNIT,
            l_trans_unit=L_TRANS_UNIT,
            solver_msg=0,
        )
    except Exception as exc:
        print(f"ILP failed for {os.path.basename(path)}: {exc}")
        return None, time.perf_counter() - start
    return res, time.perf_counter() - start


def pct_gap(candidate_energy, oracle_energy):
    if oracle_energy == 0:
        return np.nan
    return ((candidate_energy - oracle_energy) / oracle_energy) * 100.0


def oracle_gap_or_nan(raw_gap):
    if raw_gap is None or not np.isfinite(raw_gap):
        return np.nan, False
    if raw_gap < 0.0:
        return np.nan, True
    return raw_gap, False


def run_comprehensive_benchmark():
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return pd.DataFrame()


    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")],
                   key=lambda x: int(x.split('_V')[1].split('.json')[0]) if '_V' in x else 0)

    all_data = []
    print(f"\n>>> Running benchmarks for {len(files)} files...\n")

    for file_name in files:
        path = os.path.join(data_dir, file_name)
        problem_meta = load_problem_metadata(path)
        target_fps = get_fps_for_model(file_name)
        t_target = 1.0 / target_fps

        res_dp_no_refine, dp_no_refine_runtime_s = solve_dp_variant(
            "DP without refinement",
            path,
            t_target,
            local_refinement=False,
            structure_pruning=False,
        )
        res_dp_refine, dp_refine_runtime_s = solve_dp_variant(
            "DP with refinement",
            path,
            t_target,
            local_refinement=True,
            structure_pruning=False,
        )

        res_dp_no_refine_prune, dp_no_refine_prune_runtime_s = solve_dp_variant(
            "DP without refinement with pruning",
            path,
            t_target,
            local_refinement=False,
            structure_pruning=True,
        )
        res_dp_refine_prune, dp_refine__prun_runtime_s = solve_dp_variant(
            "DP with refinement with pruning",
            path,
            t_target,
            local_refinement=True,
            structure_pruning=True,
        )        
        # verification on the pruning
        prune_no_refine_match = np.nan
        prune_refine_match = np.nan

        if res_dp_no_refine and res_dp_no_refine_prune:
            prune_no_refine_match = np.isclose(
                res_dp_no_refine["energy"],
                res_dp_no_refine_prune["energy"],
                rtol=1e-9,
                atol=1e-12,
            )

        if res_dp_refine and res_dp_refine_prune:
            prune_refine_match = np.isclose(
                res_dp_refine["energy"],
                res_dp_refine_prune["energy"],
                rtol=1e-9,
                atol=1e-12,
            )
        dp_no_refine_prune_delta = np.nan
        dp_refine_prune_delta = np.nan

        if res_dp_no_refine and res_dp_no_refine_prune:
            dp_no_refine_prune_delta = (
                res_dp_no_refine_prune["energy"]
                - res_dp_no_refine["energy"]
            )

        if res_dp_refine and res_dp_refine_prune:
            dp_refine_prune_delta = (
                res_dp_refine_prune["energy"]
                - res_dp_refine["energy"]
            )
        res_ilp, ilp_runtime_s = solve_ilp_oracle(path, t_target)

        if res_ilp or res_dp_no_refine or res_dp_refine:
            e_ilp = np.nan if res_ilp is None else res_ilp['energy']
            dp_no_refine_gap = np.nan
            dp_refine_gap = np.nan
            dp_no_refine_oracle_gap = np.nan
            dp_refine_oracle_gap = np.nan
            excluded_from_oracle_gap = False
            dp_no_refine_layer_diff_count = np.nan
            dp_refine_layer_diff_count = np.nan
            dp_no_refine_schedule_diff_csv = ""
            dp_refine_schedule_diff_csv = ""

            if res_dp_no_refine and res_ilp:
                dp_no_refine_gap = pct_gap(res_dp_no_refine["energy"], e_ilp)
                dp_no_refine_schedule_diff_csv, dp_no_refine_layer_diff_count, _ = write_schedule_diff_csv(
                    file_name,
                    target_fps,
                    "dp_no_refine",
                    res_dp_no_refine,
                    res_ilp,
                )

            if res_dp_refine and res_ilp:
                dp_refine_gap = pct_gap(res_dp_refine["energy"], e_ilp)
                dp_refine_schedule_diff_csv, dp_refine_layer_diff_count, _ = write_schedule_diff_csv(
                    file_name,
                    target_fps,
                    "dp_refine",
                    res_dp_refine,
                    res_ilp,
                )

            dp_no_refine_oracle_gap, no_refine_negative_gap = oracle_gap_or_nan(dp_no_refine_gap)
            dp_refine_oracle_gap, refine_negative_gap = oracle_gap_or_nan(dp_refine_gap)
            excluded_from_oracle_gap = no_refine_negative_gap or refine_negative_gap
            ilp_solver_status = "" if res_ilp is None else str(res_ilp.get("solver_status", "Solved"))
            ilp_status = "Failed"
            oracle_note = ""
            if res_ilp:
                ilp_status = ilp_solver_status
                if excluded_from_oracle_gap:
                    ilp_status = "TimeLimit/Feasible"
                    oracle_note = (
                        "Excluded from oracle-gap averages: DP energy is lower than "
                        "the ILP incumbent, so ILP likely hit the time limit and is "
                        "not a proven oracle for this case."
                    )

            all_data.append({
                "Model": strip_model_suffix(file_name),
                "File_Name": file_name,
                "FPS": target_fps,
                **problem_meta,
                "DP_No_Refine_Energy_(J)": np.nan if res_dp_no_refine is None else res_dp_no_refine["energy"],
                "DP_Refine_Energy_(J)": np.nan if res_dp_refine is None else res_dp_refine["energy"],
                "DP_No_Refine_Pruned_Energy_(J)": np.nan if res_dp_no_refine_prune is None else res_dp_no_refine_prune["energy"],
                "DP_Refine_Pruned_Energy_(J)": np.nan if res_dp_refine_prune is None else res_dp_refine_prune["energy"],
                "DP_No_Refine_Pruning_Delta_(J)":dp_no_refine_prune_delta,
                "DP_Refine_Pruning_Delta_(J)":dp_refine_prune_delta,
                "DP_No_Refine_Pruning_Match":prune_no_refine_match,
                "DP_Refine_Pruning_Match":prune_refine_match,
                "ILP_Energy_(J)": e_ilp,
                "DP_No_Refine_Gap_(%)": dp_no_refine_gap,
                "DP_Refine_Gap_(%)": dp_refine_gap,
                "DP_No_Refine_Oracle_Gap_(%)": dp_no_refine_oracle_gap,
                "DP_Refine_Oracle_Gap_(%)": dp_refine_oracle_gap,
                "DP_No_Refine_Active_Time_(s)": np.nan if res_dp_no_refine is None else res_dp_no_refine.get("active_time", np.nan),
                "DP_Refine_Active_Time_(s)": np.nan if res_dp_refine is None else res_dp_refine.get("active_time", np.nan),
                "DP_No_Refine_Idle_Time_(s)": np.nan if res_dp_no_refine is None else res_dp_no_refine.get("idle_time", np.nan),
                "DP_Refine_Idle_Time_(s)": np.nan if res_dp_refine is None else res_dp_refine.get("idle_time", np.nan),
                "ILP_Active_Time_(s)": np.nan if res_ilp is None else res_ilp.get("active_time", np.nan),
                "ILP_Idle_Time_(s)": np.nan if res_ilp is None else res_ilp.get("idle_time", np.nan),
                "DP_No_Refine_Runtime_(s)": dp_no_refine_runtime_s,
                "DP_Refine_Runtime_(s)": dp_refine_runtime_s,
                "ILP_Runtime_(s)": np.nan if res_ilp is None else ilp_runtime_s,
                "ILP_Attempt_Runtime_(s)": ilp_runtime_s,
                "ILP_Status": ilp_status,
                "Oracle_Comparison_Note": oracle_note,
                "DP_No_Refine_Layer_Diffs": dp_no_refine_layer_diff_count,
                "DP_Refine_Layer_Diffs": dp_refine_layer_diff_count,
                "DP_Refine_Local_Moves": np.nan if res_dp_refine is None else res_dp_refine.get("local_refinement_moves", np.nan),
                "DP_No_Refine_Schedule_Diff_CSV": dp_no_refine_schedule_diff_csv,
                "DP_Refine_Schedule_Diff_CSV": dp_refine_schedule_diff_csv,
            })
            print(
                f"Done: {file_name} | "
                f"no_refine_gap={dp_no_refine_gap:.6f}% | refine_gap={dp_refine_gap:.6f}% | "
                f"no_refine_layers={dp_no_refine_layer_diff_count} | refine_layers={dp_refine_layer_diff_count} | "
                f"ilp_status={ilp_status}"
            )
        else:
            print(f"Failed: {file_name} (missing ILP oracle or both DP results)")

    return pd.DataFrame(all_data)


def print_full_report(df):
    if df.empty:
        print("No data to display.")
        return


    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    print("\n" + "=" * 120)
    print(f"{'DP vs ILP SOLVED-CASE DIAGNOSTIC REPORT':^120}")
    print("=" * 120)


    formatters = {
        'DP_No_Refine_Energy_(J)': '{:.8e}'.format,
        'DP_Refine_Energy_(J)': '{:.8e}'.format,
        'ILP_Energy_(J)': '{:.8e}'.format,
        'DP_No_Refine_Gap_(%)': '{:,.6f}%'.format,
        'DP_Refine_Gap_(%)': '{:,.6f}%'.format,
        'DP_No_Refine_Oracle_Gap_(%)': '{:,.6f}%'.format,
        'DP_Refine_Oracle_Gap_(%)': '{:,.6f}%'.format,
        'FPS': '{:.1f}'.format,
        'DP_No_Refine_Active_Time_(s)': '{:.8e}'.format,
        'DP_Refine_Active_Time_(s)': '{:.8e}'.format,
        'ILP_Active_Time_(s)': '{:.8e}'.format,
        'DP_No_Refine_Runtime_(s)': '{:.3f}'.format,
        'DP_Refine_Runtime_(s)': '{:.3f}'.format,
        'ILP_Idle_Time_(s)': '{:.8e}'.format,
    }

    display_cols = [
        "File_Name",
        "FPS",
        "DP_No_Refine_Energy_(J)",
        "DP_Refine_Energy_(J)",
        "ILP_Energy_(J)",
        "DP_No_Refine_Gap_(%)",
        "DP_Refine_Gap_(%)",
        "DP_No_Refine_Oracle_Gap_(%)",
        "DP_Refine_Oracle_Gap_(%)",
        "ILP_Status",
        "DP_No_Refine_Active_Time_(s)",
        "DP_Refine_Active_Time_(s)",
        "ILP_Active_Time_(s)",
        "DP_No_Refine_Layer_Diffs",
        "DP_Refine_Layer_Diffs",
        "DP_Refine_Local_Moves",
    ]
    print(df[display_cols].to_string(index=False, formatters=formatters))

    print("-" * 120)


    common = df[df["DP_No_Refine_Gap_(%)"].notna() & df["DP_Refine_Gap_(%)"].notna()].copy()
    oracle_common = df[
        df["DP_No_Refine_Oracle_Gap_(%)"].notna()
        & df["DP_Refine_Oracle_Gap_(%)"].notna()
    ].copy()
    excluded = df[df["Oracle_Comparison_Note"].fillna("").astype(str) != ""].copy()
    no_refine_abs_gap = oracle_common["DP_No_Refine_Oracle_Gap_(%)"].abs()
    refine_abs_gap = oracle_common["DP_Refine_Oracle_Gap_(%)"].abs()
    no_refine_worst_idx = no_refine_abs_gap.idxmax() if not no_refine_abs_gap.empty else None
    refine_worst_idx = refine_abs_gap.idxmax() if not refine_abs_gap.empty else None

    print("DIAGNOSTIC SUMMARY:")
    print(" - Inputs            : p_sleep_base=0.0, e_trans_unit=0.0, l_trans_unit=5e-9")
    print(f" - DP Setup          : structure_pruning={DP_STRUCTURE_PRUNING}")
    print(f" - ILP Attempts      : {len(df)}")
    print(f" - Common DP Cases   : {len(common)}")
    print(f" - Oracle Gap Cases  : {len(oracle_common)}")
    print(f" - Excluded Timeouts : {len(excluded)} negative DP-vs-ILP gap case(s)")
    if not oracle_common.empty:
        print(f" - No-Refine Avg Gap : {oracle_common['DP_No_Refine_Oracle_Gap_(%)'].mean():.8f}% signed, {no_refine_abs_gap.mean():.8f}% abs")
        print(f" - Refine Avg Gap    : {oracle_common['DP_Refine_Oracle_Gap_(%)'].mean():.8f}% signed, {refine_abs_gap.mean():.8f}% abs")
        print(f" - No-Refine Worst   : {no_refine_abs_gap.loc[no_refine_worst_idx]:.8f}% ({df.loc[no_refine_worst_idx, 'File_Name']})")
        print(f" - Refine Worst      : {refine_abs_gap.loc[refine_worst_idx]:.8f}% ({df.loc[refine_worst_idx, 'File_Name']})")
    if not excluded.empty:
        print(" - Timeout Note      : negative gaps are excluded because ILP returned a time-limited incumbent, not a proven oracle")
        for _, row in excluded.iterrows():
            print(
                f"   * {row['File_Name']}: raw no-refine gap={row['DP_No_Refine_Gap_(%)']:.6f}%, "
                f"raw refine gap={row['DP_Refine_Gap_(%)']:.6f}%, "
                f"ILP runtime={row['ILP_Attempt_Runtime_(s)']:.3f}s"
            )
    print(" - Layer_Diffs       : count of layers where DP and ILP voltage tuples differ")
    print("=" * 120)


if __name__ == "__main__":
    results_df = run_comprehensive_benchmark()
    if not results_df.empty:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        results_df.to_csv(OUT_CSV, index=False)
        print(f"Saved compile-time data: {OUT_CSV}")
    print_full_report(results_df)
