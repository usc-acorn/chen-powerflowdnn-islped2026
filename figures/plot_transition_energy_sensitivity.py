from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.unicode_minus"] = False

DISPLAY_NAMES = {
    "resnet": "ResNet-18",
    "squeezenet": "SqueezeNet",
    "mobilenet": "MobileNet",
    "mobilevit": "MobileViT",
}


def _read_compiler_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Plot rail switch count vs transition energy.")
    ap.add_argument(
        "--compiler-dir",
        default=os.path.join(here, "..", "data", "3rail_etrans"),
        help="Directory with compiler_results_<model>_3rails.csv files.",
    )
    ap.add_argument("--models", default="resnet,squeezenet,mobilenet,mobilevit")
    ap.add_argument(
        "--out-csv",
        default=os.path.join(here, "../data", "figureX_trans_sensitivity.csv"),
    )
    ap.add_argument(
        "--out-png",
        default=os.path.join(here, "../outputs", "figureX_trans_sensitivity.png"),
    )
    args = ap.parse_args()

    out_csv_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)
    out_png_dir = os.path.dirname(os.path.abspath(args.out_png))
    if out_png_dir:
        os.makedirs(out_png_dir, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows_out: List[List[object]] = []
    by_model: Dict[str, List[Dict[str, float]]] = {m: [] for m in models}

    for model in models:
        csv_path = os.path.join(args.compiler_dir, f"compiler_results_{model}_3rails.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        rows = _read_compiler_rows(csv_path)
        if not rows:
            continue

        if "rail_switch_count" not in rows[0]:
            raise KeyError(
                f"CSV missing 'rail_switch_count': {csv_path}. "
                "Regenerate the duty-cycling CSVs after updating main.py."
            )

        pts = []
        for r in sorted(rows, key=lambda row: float(row["transition_energy_j"])):
            pt = {
                "fps_target": float(r["fps"]),
                "transition_energy_j": float(r["transition_energy_j"]),
                "rail_switch_count": float(r["rail_switch_count"]),
                "avg_power_mw": float(r["avg_power_w"]) * 1e3,
            }
            pts.append(pt)
            rows_out.append(
                [
                    model,
                    pt["fps_target"],
                    pt["transition_energy_j"],
                    pt["rail_switch_count"],
                    pt["avg_power_mw"],
                ]
            )
        by_model[model] = pts

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "fps_target",
                "transition_energy_j",
                "rail_switch_count",
                "avg_power_mw",
            ]
        )
        for r in rows_out:
            w.writerow(r)

    fig, ax = plt.subplots(figsize=(6,2.5))
    cmap = mpl.colormaps["tab20"]
    palette = cmap(np.linspace(0, 1, max(len(models), 1)))
    colors = {model: palette[i % len(palette)] for i, model in enumerate(models)}
    for model in models:
        pts = by_model[model]
        if not pts:
            continue
        pts = [p for p in pts if p["transition_energy_j"] > 0.0]
        if not pts:
            continue
        fps_target = int(round(pts[0]["fps_target"]))
        label = f"{model} ({fps_target} FPS)"
        label = DISPLAY_NAMES.get(model, model)

        x = [p["transition_energy_j"] * 1e9 for p in pts]
        y = [p["rail_switch_count"] for p in pts]
        ax.step(
            x,
            y,
            where="post",
            linewidth=2.0,
            color=colors.get(model, None),
            label=label,
        )
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            linewidth=0.0,
            color=colors.get(model, None),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Transition Energy per Switch (nJ) in log scale")
    ax.set_ylabel("Voltage Rail Switching Count")
    ax.set_title("Voltage Rail Switching Count vs Transition Energy", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18, hspace=0.8)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved plot: {args.out_png}")


if __name__ == "__main__":
    main()
