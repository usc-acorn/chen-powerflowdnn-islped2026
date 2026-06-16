import json
from pulp import *


def _voltage_key(value):
    return f"{float(value):.12g}"


def _changed_count(prev_state, cur_state):
    return (
        int(float(prev_state['v_sys']) != float(cur_state['v_sys']))
        + int(float(prev_state['v_rram']) != float(cur_state['v_rram']))
        + int(float(prev_state['v_feeder']) != float(cur_state['v_feeder']))
    )


def _selected_path_records(combos_list, x):
    records = []
    for layer_idx, combos in enumerate(combos_list):
        selected_key = None
        selected_value = -1.0
        for state_key in combos.keys():
            raw_value = value(x[(layer_idx, state_key)])
            raw_value = 0.0 if raw_value is None else float(raw_value)
            if raw_value > selected_value:
                selected_key = state_key
                selected_value = raw_value

        if selected_key is None:
            continue

        state = combos[selected_key]
        try:
            state_idx = int(selected_key)
        except Exception:
            state_idx = -1
        records.append(
            {
                "layer_idx": layer_idx,
                "state_idx": state_idx,
                "state_key": str(selected_key),
                "v_sys": float(state["v_sys"]),
                "v_rram": float(state["v_rram"]),
                "v_feeder": float(state["v_feeder"]),
                "energy_j": float(state["energy_j"]),
                "time_s": float(state["time_s"]),
                "leakage_w": float(state["Leakage_w"]),
            }
        )
    return records


def _path_rail_switch_count(records):
    count = 0
    for prev, cur in zip(records, records[1:]):
        count += (
            int(prev["v_sys"] != cur["v_sys"])
            + int(prev["v_rram"] != cur["v_rram"])
            + int(prev["v_feeder"] != cur["v_feeder"])
        )
    return count


def solve_scheduling_with_rail_limit(file_path, t_target, p_sleep_base,
                                     max_v_sys_rails=5,
                                     e_trans_unit=0.0000,
                                     l_trans_unit=5e-9,
                                     solver_msg=1):
    print(f"[Step 1] Loading Large JSON...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    layers_dict = data.get('layers', {})
    try:
        sorted_keys = sorted(layers_dict.keys(), key=lambda x: int(x))
    except Exception:
        sorted_keys = sorted(layers_dict.keys())

    layers_list = [layers_dict[k] for k in sorted_keys]
    combos_list = [layer.get('combinations', layer) for layer in layers_list]
    num_layers = len(combos_list)
    v_sys_candidates = data.get('meta', {}).get('v_sys_candidates', [])
    if not v_sys_candidates:
        vals = set()
        for combos in combos_list:
            for s in combos.values():
                vals.add(float(s['v_sys']))
        v_sys_candidates = sorted(vals)
    energy_objective_scale = 1e12

    prob = LpProblem("Large_Scale_DNN_Optimization", LpMinimize)

    # --- Decision Variables ---
    x = {}
    print(f"[Step 2] Creating Variables for {num_layers} layers...")
    for i, combos in enumerate(combos_list):
        for s in combos.keys():
            x[(i, s)] = LpVariable(f"x_L{i}_{s}", cat='Binary')
        if i % 10 == 0: print(f"  - Layer {i} variables generated.")

    y = {}
    for i in range(1, num_layers):
        states_prev = list(combos_list[i - 1].keys())
        states_curr = list(combos_list[i].keys())
        for sp in states_prev:
            for sc in states_curr:
                y[(i, sp, sc)] = LpVariable(f"y_L{i}_{sp}_{sc}", cat='Binary')

    z = {_voltage_key(v): LpVariable(f"use_v_{_voltage_key(v)}", cat='Binary') for v in v_sys_candidates}
    t_idle = LpVariable("T_idle", lowBound=0)

    # Dynamic Idle variables
    last_layer_idx = num_layers - 1
    last_layer_states = list(combos_list[last_layer_idx].keys())
    sleep_switch_count = 6
    sleep_transition_time = sleep_switch_count * l_trans_unit
    sleep_transition_energy = sleep_switch_count * e_trans_unit
    t_idle_stay_split = {s: LpVariable(f"T_idle_stay_split_{s}", lowBound=0) for s in last_layer_states}
    t_sleep_split = {s: LpVariable(f"T_sleep_split_{s}", lowBound=0) for s in last_layer_states}
    use_sleep = {s: LpVariable(f"use_sleep_{s}", cat='Binary') for s in last_layer_states}

    # --- Dynamic Idle Constraints ---
    prob += (
        lpSum([t_idle_stay_split[s] + t_sleep_split[s] + sleep_transition_time * use_sleep[s] for s in last_layer_states])
        == t_idle
    )
    M = t_target
    for s in last_layer_states:
        prob += t_idle_stay_split[s] <= M * x[(last_layer_idx, s)]
        prob += t_sleep_split[s] <= M * use_sleep[s]
        prob += use_sleep[s] <= x[(last_layer_idx, s)]

    # --- Objective ---
    e_comp = lpSum(
        [energy_objective_scale * combos_list[i][s]['energy_j'] * x[(i, s)] for i in range(num_layers) for s in combos_list[i].keys()])

    e_trans_terms = []
    for i in range(1, num_layers):
        for sp in combos_list[i - 1].keys():
            for sc in combos_list[i].keys():
                changed_count = _changed_count(combos_list[i - 1][sp], combos_list[i][sc])
                if changed_count:
                    e_trans_terms.append(changed_count * energy_objective_scale * e_trans_unit * y[(i, sp, sc)])

    e_idle_dynamic = lpSum([
        energy_objective_scale
        * (
            float(combos_list[last_layer_idx][s]['Leakage_w']) * t_idle_stay_split[s]
            + float(p_sleep_base) * t_sleep_split[s]
            + sleep_transition_energy * use_sleep[s]
        )
        for s in last_layer_states
    ])
    prob += e_comp + lpSum(e_trans_terms) + e_idle_dynamic

    # --- Constraints ---
    print(f"[Step 3] Setting Constraints...")
    for i in range(num_layers):
        prob += lpSum([x[(i, s)] for s in combos_list[i].keys()]) == 1

    for i in range(1, num_layers):
        states_p = list(combos_list[i - 1].keys())
        states_c = list(combos_list[i].keys())
        for sc in states_c:
            prob += lpSum([y[(i, sp, sc)] for sp in states_p]) == x[(i, sc)]
        for sp in states_p:
            prob += lpSum([y[(i, sp, sc)] for sc in states_c]) == x[(i, sp)]

    for i in range(num_layers):
        for s, state in combos_list[i].items():
            for v in {state['v_sys'], state['v_rram'], state['v_feeder']}:
                key = _voltage_key(v)
                if key in z:
                    prob += z[key] >= x[(i, s)]
                else:
                    prob += x[(i, s)] == 0
    prob += lpSum(z.values()) <= max_v_sys_rails

    l_comp = lpSum([combos_list[i][s]['time_s'] * x[(i, s)] for i in range(num_layers) for s in combos_list[i].keys()])
    l_trans_terms = []
    for i in range(1, num_layers):
        for sp in combos_list[i - 1].keys():
            for sc in combos_list[i].keys():
                changed_count = _changed_count(combos_list[i - 1][sp], combos_list[i][sc])
                if changed_count:
                    l_trans_terms.append(changed_count * l_trans_unit * y[(i, sp, sc)])

    prob += l_comp + lpSum(l_trans_terms) + t_idle == t_target

    # --- Solver with Time Limit and Message Logs ---
    print(f"[Step 4] Calling ILP solver (Target FPS: {1 / t_target:.2f})")
    # msg=1 allows you to see the actual progress of the solver in the terminal
    # timeLimit=300 sets a 5-minute cap to prevent infinite hanging
    status = prob.solve(PULP_CBC_CMD(msg=solver_msg, timeLimit=300, threads=4))

    if LpStatus[status] in ['Optimal', 'Feasible']:
        final_energy = value(prob.objective) / energy_objective_scale
        raw_idle = value(t_idle)
        final_idle = sum(float(value(t_idle_stay_split[s]) or 0.0) for s in last_layer_states)
        final_sleep = sum(float(value(t_sleep_split[s]) or 0.0) for s in last_layer_states)
        final_sleep_switch_count = int(round(sum(float(value(use_sleep[s]) or 0.0) for s in last_layer_states))) * sleep_switch_count
        selected_path = _selected_path_records(combos_list, x)
        rail_switch_count = _path_rail_switch_count(selected_path)
        print(f"SUCCESS: Result found. Energy: {final_energy:.6f} J")

        return {
            "energy": final_energy,
            "active_time": (t_target - raw_idle + final_sleep_switch_count * l_trans_unit),
            "idle_time": final_idle,
            "sleep_time": final_sleep,
            "rail_switch_count": rail_switch_count,
            "sleep_rail_switch_count": final_sleep_switch_count,
            "total_rail_switch_count": rail_switch_count + final_sleep_switch_count,
            "used_sleep_mode": bool(final_sleep_switch_count),
            "fps": 1.0 / t_target,
            "solver_status": LpStatus[status],
            "method": "ilp",
            "selected_path": selected_path,
        }
    else:
        print("Solver failed to find a solution within time limit.")
        return None
