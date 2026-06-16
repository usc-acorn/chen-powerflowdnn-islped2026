import sys
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

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

from compilers.min_distance_duty_cycling import (
    _load_layers,
    _build_layer_arrays,
    _solve_for_rail_subset,
    solve_scheduling_with_rail_limit
)


def _solution_energy(sol):
    if isinstance(sol, dict):
        return float(sol["energy"])
    return float(sol[0])


def main():
    out_csv = os.path.join(_PROJECT_ROOT,"data/figure_energy_vs_rails.csv")
    out_pdf = os.path.join(_PROJECT_ROOT, "outputs/figure7.pdf")

    models = [
        {"name": "SqueezeNet", "path": "./data/squeezenet_vdd_combinations.json", "fps": 60},
        {"name": "ResNet-18", "path": "./data/resnet18_vdd_combinations.json", "fps": 20},
        {"name": "MobileViT-XXS", "path": "./data/mobilevit_xxs_vdd_combinations.json", "fps": 15},
        {"name": "MobileNetV3-Small", "path": "./data/mobilenetv3_small_vdd_combinations.json", "fps": 100}
    ]

    rail_counts = [1, 2, 3, 4, 5]
    cached_rows = []
    if  os.path.isfile(out_csv):
        with open(out_csv, "r", encoding="utf-8") as f:
            cached_rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 1.5), dpi=220, sharey=False)
    axes = axes.flatten()
    rows_to_write = []

    for idx, model in enumerate(models):
        file_path = str(Path(model["path"]).resolve())
        model_name = model["name"]
        target_fps = model["fps"]
        ax = axes[idx]

        print(f"\n" + "=" * 80)
        print(f">>> Processing {model_name} @ {target_fps} FPS")
        print(f"{'Rails':<6} | {'Type':<10} | {'Selected VDDs':<40} | {'Energy (J)':<15}")
        print("-" * 80)

        combos_list, v_sys_candidates = _load_layers(file_path)
        v_min, v_max = min(v_sys_candidates), max(v_sys_candidates)

        results_searched = {}
        results_fixed = {}
        model_cached = [r for r in cached_rows if r.get("model") == model_name]
        if model_cached:
            for row in model_cached:
                num_rails = int(row["num_rails"])
                energy_j = float(row["energy_j"])
                if row["type"] == "searched":
                    results_searched[num_rails] = energy_j
                    print(f"{num_rails:<6} | {'Searched':<10} | {row['selected_vdds']:<40} | {energy_j:.6e}")
                else:
                    results_fixed[num_rails] = energy_j
                    print(f"{'':<6} | {'Fixed':<10} | {row['selected_vdds']:<40} | {energy_j:.6e}")
        else:
            for num_rails in rail_counts:
                t_target = 1.0 / target_fps

                # [1] Ours-Searched
                res_s = solve_scheduling_with_rail_limit(file_path, t_target, 0.001, num_rails)
                if res_s:
                    results_searched[num_rails] = res_s['energy']
                    vdd_print = [round(v, 2) for v in sorted(res_s['used_v_sys_values'])]
                    vdd_txt = str(vdd_print)
                    print(f"{num_rails:<6} | {'Searched':<10} | {vdd_txt:<40} | {res_s['energy']:.6e}")
                    rows_to_write.append([model_name, target_fps, num_rails, "searched", res_s["energy"], vdd_txt])

                # [2] Ours-Fixed
                if num_rails == 1:
                    mid_idx = len(v_sys_candidates) // 2
                    fixed_v_list = [v_sys_candidates[mid_idx]]
                else:
                    raw_fixed = np.linspace(v_min, v_max, num_rails)
                    fixed_v_list = sorted(list(set([min(v_sys_candidates, key=lambda x: abs(x - v)) for v in raw_fixed])))

                layer_arrays = _build_layer_arrays(combos_list, fixed_v_list)
                sol_f = (
                    _solve_for_rail_subset(
                        layer_arrays=layer_arrays,
                        t_target=t_target,
                        p_sleep_base=0.0,
                        e_trans_unit=0.0,
                        l_trans_unit=5e-9,
                    )
                    if layer_arrays
                    else None
                )

                # Rail 1 Infeasible
                if num_rails == 1 and sol_f is None:
                    fixed_v_list = [v_max]
                    layer_arrays = _build_layer_arrays(combos_list, fixed_v_list)
                    sol_f = _solve_for_rail_subset(
                        layer_arrays=layer_arrays,
                        t_target=t_target,
                        p_sleep_base=0.0,
                        e_trans_unit=0.0,
                        l_trans_unit=5e-9,
                    )

                if sol_f:
                    fixed_energy = _solution_energy(sol_f)
                    results_fixed[num_rails] = fixed_energy
                    vdd_txt = str([round(v, 2) for v in fixed_v_list])
                    print(f"{'':<6} | {'Fixed':<10} | {vdd_txt:<40} | {fixed_energy:.6e}")
                    rows_to_write.append([model_name, target_fps, num_rails, "fixed", fixed_energy, vdd_txt])


        s_rails = sorted(results_searched.keys())
        ax.plot(s_rails, [results_searched[r] * 1e3 for r in s_rails],
                marker='o', markersize=3.5, linestyle='-', color='#022453', linewidth=1.0, label='Optimized')

        f_rails = sorted(results_fixed.keys())
        ax.plot(f_rails, [results_fixed[r] * 1e3 for r in f_rails],
                marker='D', markersize=3.2, linestyle='--', color='#21A599', linewidth=1.0, label='Even')

        ax.set_title(f"({chr(97 + idx)}) {model['name']}", loc="left", fontweight="bold", pad=1)
        ax.set_xticks(rail_counts)
        ax.tick_params(axis="x", pad=0, width=0.5, labelsize=8)
        ax.tick_params(axis="y", pad=0, width=0.5, labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["top"].set_linewidth(0.5)
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if results_searched or results_fixed:
            all_vals = [v * 1e3 for v in list(results_searched.values()) + list(results_fixed.values())]
            ax.set_ylim(min(all_vals) * 0.94, max(all_vals) * 1.06)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.42, 0.9),
        frameon=True,
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        ncol=1,
        columnspacing=0.6,
        handlelength=1.0,
        handletextpad=0.2,
        borderpad=0,
        labelspacing=0.2,
    )
    fig.supxlabel("Number of Supply Rails", fontsize=8, y=0.0)
    fig.supylabel("Interval Energy (mJ)", fontsize=8, x=0.01)

    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.17, top=0.9, wspace=0.24, hspace=0.6)
    
    fig.savefig(out_pdf, dpi=220, pad_inches=0, transparent=False, bbox_inches="tight")
    plt.close(fig)

    if rows_to_write:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "target_fps", "num_rails", "type", "energy_j", "selected_vdds"])
            w.writerows(rows_to_write)

    print(f"Saved PDF: {out_pdf}")
    print(f"Using CSV: {out_csv}")

if __name__ == "__main__":
    main()
