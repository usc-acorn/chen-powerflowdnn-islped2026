from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import min_distance_duty_cycling as dp


HERE = Path(__file__).resolve().parent
VDD_DIR = HERE.parent / "data"
OUT_ROOT = HERE.parent / "data"

MODEL_CFGS: Dict[str, Dict[str, object]] = {
    "squeezenet": {
        "json": VDD_DIR / "squeezenet_vdd_combinations.json",
        "etran_fps": 83,
    },
    "resnet": {
        "json": VDD_DIR / "resnet18_vdd_combinations.json",
        "etran_fps": 16,
    },
    "mobilevit": {
        "json": VDD_DIR / "mobilevit_xxs_vdd_combinations.json",
        "etran_fps": 15,
    },
    "mobilenet": {
        "json": VDD_DIR / "mobilenetv3_small_vdd_combinations.json",
        "etran_fps": 129,
    },
}

ETRANS_LIST = [0.0, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]


def _parse_models(text: str) -> List[str]:
    models = [m.strip() for m in text.split(",") if m.strip()]
    invalid = [m for m in models if m not in MODEL_CFGS]
    if invalid:
        raise ValueError(f"Unsupported model key(s): {invalid}. Choose from {sorted(MODEL_CFGS)}")
    return models


def _fps_range(start: int, stop: int) -> Iterable[int]:
    return range(start, stop + 1)


def _print_solve_summary(label: str, res: Optional[Dict[str, object]], runtime_s: float) -> None:
    if res is None:
        print(f"{label} infeasible after {runtime_s:.3f}s", flush=True)
        return

    lambda_count = len(res.get("trial_lambdas", []))
    print(
        f"{label} done in {runtime_s:.3f}s | "
        f"energy={float(res['energy']):.4e} J | "
        f"active={float(res['active_time']):.4e}s | "
        f"moves={int(res.get('local_refinement_moves', 0))} | "
        f"lambdas={lambda_count} | "
        f"rails={res.get('used_v_sys_values', [])}",
        flush=True,
    )


def _solve_point(
    file_path: Path,
    fps: float,
    e_trans_unit: float,
    l_trans_unit_s: float,
    max_v_sys_rails: int,
    p_sleep_base: float,
    max_unique_domains: int,
    exact_unique_domains,
    required_equalities,
    allowed_voltage_sets,
):
    t_target = 1.0 / fps
    return dp.solve_scheduling_with_rail_limit(
        file_path=str(file_path),
        t_target=t_target,
        p_sleep_base=p_sleep_base,
        max_v_sys_rails=max_v_sys_rails,
        max_unique_domains=max_unique_domains,
        exact_unique_domains=exact_unique_domains,
        required_equalities=required_equalities,
        allowed_voltage_sets=allowed_voltage_sets,
        e_trans_unit=e_trans_unit,
        l_trans_unit=l_trans_unit_s,
    )


def _result_row(res: Dict[str, object], fps: float, e_trans_unit: float, l_trans_unit_s: float) -> List[object]:
    avg_power_w = float(res["energy"]) * fps
    rails_txt = "|".join(f"{v:.3f}" for v in res.get("used_v_sys_values", []))
    selected_path_json = json.dumps(res.get("selected_path", []), separators=(",", ":"))
    return [
        f"{fps:.6f}",
        f"{e_trans_unit:.12e}",
        int(round(l_trans_unit_s * 1e9)),
        f"{l_trans_unit_s:.12e}",
        f"{avg_power_w:.12e}",
        f"{float(res['energy']):.12e}",
        f"{float(res['active_time']):.12e}",
        f"{float(res['idle_time']):.12e}",
        f"{float(res.get('sleep_time', 0.0)):.12e}",
        int(res.get("rail_switch_count", -1)),
        int(res.get("sleep_rail_switch_count", -1)),
        int(res.get("total_rail_switch_count", -1)),
        int(bool(res.get("used_sleep_mode", False))),
        int(res.get("used_v_sys_rails", -1)),
        rails_txt,
        str(res.get("method", "")),
        selected_path_json,
    ]


def _write_csv(path: Path, rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "fps",
                "transition_energy_j",
                "transition_time_ns",
                "transition_time_s",
                "avg_power_w",
                "energy_j",
                "active_time_s",
                "idle_time_s",
                "sleep_time_s",
                "rail_switch_count",
                "sleep_rail_switch_count",
                "total_rail_switch_count",
                "used_sleep_mode",
                "used_v_sys_rails",
                "used_v_sys_values",
                "method",
                "selected_path_json",
            ]
        )
        w.writerows(rows)


def run_fps_sweep(
    model_key: str,
    fps_start: int,
    fps_stop: int,
    e_trans_unit: float,
    l_trans_unit_s: float,
    max_v_sys_rails: int,
    p_sleep_base: float,
) -> Path:
    cfg = MODEL_CFGS[model_key]
    file_path = Path(cfg["json"])
    rows: List[List[object]] = []

    for fps in _fps_range(fps_start, fps_stop):
        label = f"[fps] model={model_key} fps={fps}"
        print(f"{label} solving", flush=True)
        start = time.perf_counter()
        res = _solve_point(
            file_path=file_path,
            fps=float(fps),
            e_trans_unit=e_trans_unit,
            l_trans_unit_s=l_trans_unit_s,
            max_v_sys_rails=max_v_sys_rails,
            p_sleep_base=p_sleep_base,
            max_unique_domains=3,
            exact_unique_domains=None,
            required_equalities=None,
            allowed_voltage_sets=None,
        )
        runtime_s = time.perf_counter() - start
        _print_solve_summary(label, res, runtime_s)
        if res is None:
            break
        rows.append(_result_row(res, float(fps), e_trans_unit, l_trans_unit_s))

    out_csv = OUT_ROOT / "3rails_fps" / f"compiler_results_{model_key}_3rails.csv"
    _write_csv(out_csv, rows)
    return out_csv


def run_etran_sweep(
    model_key: str,
    l_trans_unit_s: float,
    max_v_sys_rails: int,
    p_sleep_base: float,
) -> Path:
    cfg = MODEL_CFGS[model_key]
    file_path = Path(cfg["json"])
    fps = float(cfg["etran_fps"])
    rows: List[List[object]] = []

    for e_trans_unit in ETRANS_LIST:
        label = f"[etran] model={model_key} fps={fps:.0f} e_trans={e_trans_unit:.1e}"
        print(f"{label} solving", flush=True)
        start = time.perf_counter()
        res = _solve_point(
            file_path=file_path,
            fps=fps,
            e_trans_unit=e_trans_unit,
            l_trans_unit_s=l_trans_unit_s,
            max_v_sys_rails=max_v_sys_rails,
            p_sleep_base=p_sleep_base,
            max_unique_domains=3,
            exact_unique_domains=None,
            required_equalities=None,
            allowed_voltage_sets=None,
        )
        runtime_s = time.perf_counter() - start
        _print_solve_summary(label, res, runtime_s)
        if res is None:
            break
        rows.append(_result_row(res, fps, e_trans_unit, l_trans_unit_s))

    out_csv = OUT_ROOT / "3rail_etrans" / f"compiler_results_{model_key}_3rails.csv"
    _write_csv(out_csv, rows)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Run compiler duty-cycling sweeps for FPS and transition energy.")
    ap.add_argument("--models", default="squeezenet,resnet,mobilevit,mobilenet")
    ap.add_argument("--mode", choices=["fps", "etran", "both"], default="both")
    ap.add_argument("--fps-start", type=int, default=1)
    ap.add_argument("--fps-stop", type=int, default=149)
    ap.add_argument("--fps-transition-energy", type=float, default=1e-10)
    ap.add_argument("--transition-time-ns", type=float, default=15.0)
    ap.add_argument("--max-v-sys-rails", type=int, default=3)
    ap.add_argument("--p-sleep-base", type=float, default=0.0)
    args = ap.parse_args()

    model_keys = _parse_models(args.models)
    l_trans_unit_s = args.transition_time_ns * 1e-9

    for model_key in model_keys:
        if args.mode in ("fps", "both"):
            out_csv = run_fps_sweep(
                model_key=model_key,
                fps_start=args.fps_start,
                fps_stop=args.fps_stop,
                e_trans_unit=args.fps_transition_energy,
                l_trans_unit_s=l_trans_unit_s,
                max_v_sys_rails=args.max_v_sys_rails,
                p_sleep_base=args.p_sleep_base,
            )
            print(f"Saved FPS sweep: {out_csv}")

        if args.mode in ("etran", "both"):
            out_csv = run_etran_sweep(
                model_key=model_key,
                l_trans_unit_s=l_trans_unit_s,
                max_v_sys_rails=args.max_v_sys_rails,
                p_sleep_base=args.p_sleep_base,
            )
            print(f"Saved transition-energy sweep: {out_csv}")


if __name__ == "__main__":
    main()
