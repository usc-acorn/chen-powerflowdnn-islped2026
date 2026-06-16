import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


HERE = Path(__file__).resolve()
COMPILE_TIME_CSV = HERE.parent / ".." / "data" / "figure9_compile_time.csv"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)

def strip_model_suffix(name):
    if not isinstance(name, str):
        return name
    name = re.sub(r"\.json$|\.jso$|\.txt$|\.csv$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"([_\-\s]V)\d.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_V\d+$", "", name, flags=re.IGNORECASE)
    return name


def compute_effective_layered_state_graph_size(num_layers, state_space):
    return int(num_layers) * int(state_space)


def load_compile_time_dataframe(csv_path=COMPILE_TIME_CSV):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required_cols = [
        "Model",
        "Subsets",
        "Layers",
        "State_Space",
        "Effective_State_Graph_Size",
        "DP_No_Refine_Runtime_(s)",
        "DP_Refine_Runtime_(s)",
        "ILP_Runtime_(s)",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in {csv_path}: {missing}")

    df = df.rename(
        columns={
            "DP_No_Refine_Runtime_(s)": "DP_Time",
            "DP_Refine_Runtime_(s)": "New_DP_Time",
            "DP_Refine_Local_Moves": "New_DP_Local_Moves",
            "ILP_Runtime_(s)": "ILP_Time",
        }
    )
    if "New_DP_Local_Moves" not in df:
        df["New_DP_Local_Moves"] = np.nan
    if "ILP_Status" not in df:
        df["ILP_Status"] = np.where(df["ILP_Time"].isna(), "Failed", "Solved")
    df["BaseModel"] = df["Model"].apply(strip_model_suffix)
    return df.sort_values("Effective_State_Graph_Size").reset_index(drop=True)


def build_dataframe(compile_time_csv=COMPILE_TIME_CSV):
    measured_df = load_compile_time_dataframe(compile_time_csv)
    if measured_df is not None:
        return measured_df

    rows = [
        ("resnet18_V3", 7, 20, 27, 0.0579, 0.6133),
        ("squeezenet_V3", 7, 26, 27, 0.0645, 0.7910),
        ("mobilenetv3_V3", 7, 34, 27, 0.0103, 1.1479),
        ("mobilevit_V3", 7, 89, 27, 0.2211, 2.8700),
        ("resnet18_V4", 14, 20, 64, 0.1200, 3.5123),
        ("squeezenet_V4", 14, 26, 64, 0.1170, 4.6887),
        ("mobilenetv3_V4", 14, 34, 64, 0.0248, 6.3134),
        ("resnet18_V5", 25, 20, 125, 0.2123, 16.4875),
        ("mobilevit_V4", 14, 89, 64, 0.4061, 18.0828),
        ("squeezenet_V5", 25, 26, 125, 0.1958, 26.2080),
        ("mobilenetv3_V5", 25, 34, 125, 0.0700, 29.5091),
        ("resnet18_V6", 41, 20, 216, 0.3472, 103.4723),
        ("squeezenet_V6", 41, 26, 216, 0.3104, 98.0630),
        ("mobilevit_V5", 25, 89, 125, 0.6833, 120.2049),
        ("mobilenetv3_V6", 41, 34, 216, 0.1108, 118.5290),
        ("mobilevit_V6", 41, 89, 216, 1.0816, np.nan),
    ]

    df = pd.DataFrame(
        rows,
        columns=["Model", "Subsets", "Layers", "State_Space", "DP_Time", "ILP_Time"],
    )
    df["Effective_State_Graph_Size"] = df.apply(
        lambda row: compute_effective_layered_state_graph_size(row["Layers"], row["State_Space"]),
        axis=1,
    )
    df["ILP_Status"] = np.where(df["ILP_Time"].isna(), "OOM", "Solved")
    df["BaseModel"] = df["Model"].apply(strip_model_suffix)
    df["New_DP_Time"] = np.nan
    df["New_DP_Local_Moves"] = np.nan
    return df.sort_values("Effective_State_Graph_Size").reset_index(drop=True)


def build_mobilevit_case_study_dataframe():
    df = build_dataframe()
    return df[df["BaseModel"].str.lower() == "mobilevit"].reset_index(drop=True)


def plot_dp_ilp_grouped_by_model(
    df,
    figsize=(3.5, 1.1),
    dpi=220,
    save_path=None,
    split_at_first_failed=True,
):
    if df is None or df.empty:
        raise ValueError("Empty dataframe provided.")

    df = df.copy()
    df["DP_Time"] = pd.to_numeric(df["DP_Time"], errors="coerce")
    df["New_DP_Time"] = pd.to_numeric(df["New_DP_Time"], errors="coerce")
    df["ILP_Time"] = pd.to_numeric(df["ILP_Time"], errors="coerce")
    df["Effective_State_Graph_Size"] = pd.to_numeric(df["Effective_State_Graph_Size"], errors="coerce")
    df = df.sort_values("Effective_State_Graph_Size").reset_index(drop=True)

    base_models = list(df["BaseModel"].unique())
    cmap = mpl.colormaps["tab20"]
    colors = cmap(np.linspace(0, 1, max(len(base_models), 1)))
    base_color = {bm: colors[i % len(colors)] for i, bm in enumerate(base_models)}

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0)

    for bm in base_models:
        sub = df[df["BaseModel"] == bm].sort_values("Effective_State_Graph_Size")
        if sub.empty:
            continue
        col = base_color[bm]

        ax.plot(
            sub["Effective_State_Graph_Size"],
            sub["DP_Time"],
            linestyle="-",
            marker="o",
            markersize=4.5,
            linewidth=1,
            color="tab:blue",
            markeredgecolor="tab:blue",
            markerfacecolor=col,
            zorder=4,
        )
        if bm.lower().startswith("mobilevit"):
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{row['DP_Time']:.2f}s",
                    (row["Effective_State_Graph_Size"], row["DP_Time"]),
                    xytext=(0, -7),
                    textcoords="offset points",
                    ha="center",
                    color="tab:blue",
                    fontsize=7,
                )

        sub_new_dp = sub[sub["New_DP_Time"].notna()].sort_values("Effective_State_Graph_Size")
        if not sub_new_dp.empty:
            ax.plot(
                sub_new_dp["Effective_State_Graph_Size"],
                sub_new_dp["New_DP_Time"],
                linestyle="-.",
                marker="^",
                markersize=4.5,
                linewidth=1,
                color="#d95f02",
                markeredgecolor="#d95f02",
                markerfacecolor=col,
                zorder=5,
            )

        sub_ilp = sub[sub["ILP_Time"].notna()].sort_values("Effective_State_Graph_Size")
        if not sub_ilp.empty:
            ax.plot(
                sub_ilp["Effective_State_Graph_Size"],
                sub_ilp["ILP_Time"],
                linestyle="--",
                marker="s",
                markersize=4.5,
                linewidth=1,
                color="#16a085",
                markeredgecolor="#16a085",
                markerfacecolor=col,
                zorder=3,
            )

    refine_points = df[df["New_DP_Time"].notna()].sort_values("Effective_State_Graph_Size")
    if not refine_points.empty:
        last_refine = refine_points.iloc[-1]
        ax.annotate(
            f"{last_refine['New_DP_Time']:.2f}s",
            (last_refine["Effective_State_Graph_Size"], last_refine["New_DP_Time"]),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#d95f02",
            fontsize=7,
            fontweight="bold",
            zorder=7,
        )

    x_min = df["Effective_State_Graph_Size"].min()
    x_max = df["Effective_State_Graph_Size"].max()
    pad = max(50.0, 0.08 * (x_max - x_min))
    ax.set_xlim(x_min - pad, x_max + pad)

    dp_min = df["DP_Time"].min(skipna=True) if df["DP_Time"].notna().any() else 0.0
    dp_max = df["DP_Time"].max(skipna=True) if df["DP_Time"].notna().any() else 1.0
    new_dp_min = df["New_DP_Time"].min(skipna=True) if df["New_DP_Time"].notna().any() else dp_min
    new_dp_max = df["New_DP_Time"].max(skipna=True) if df["New_DP_Time"].notna().any() else 0.0
    ilp_max = df["ILP_Time"].max(skipna=True) if df["ILP_Time"].notna().any() else 0.0
    y_max = max(dp_max, new_dp_max, ilp_max)
    # bottom_candidates = [min(dp_min, new_dp_min) * 0.5 - 0.01, -0.15 * y_max, -0.05]
    bottom = -70
    top = y_max * 1.15 if y_max > 0 else 0.1
    ax.set_ylim(bottom, top)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylabel("Solver Run Time (sec)", labelpad=0)
    ax.yaxis.set_label_coords(-0.08, 0.4)

    ax.set_xlabel("Effective Layered State-Graph Size ($10^3$ units)", labelpad=-1)
    ax.set_title("Compile Run Time: DP vs ILP", fontweight="bold", pad=-1)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e3:.1f}"))
    
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", pad=0, width=0.5)
    ax.tick_params(axis="y", pad=0, width=0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    marker_leg = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            marker="o",
            color="tab:blue",
            label="DP",
            markerfacecolor="tab:blue",
            markeredgecolor="tab:blue",
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            linestyle="-.",
            marker="^",
            color="#d95f02",
            label="DP+R",
            markerfacecolor="#d95f02",
            markeredgecolor="#d95f02",
            markersize=4,
        ),
        Line2D([0], [0], linestyle="--", marker="s", color="#16a085", label="ILP ", markerfacecolor="#16a085", markeredgecolor="#16a085", markersize=4.5),
    ]
    legend_label = {
        "resnet18": "ResNet-18",
        "squeezenet": "SqueezeNet",
        "mobilenetv3_small": "MobileNet",
        "mobilenetv3": "MobileNet",
        "mobilevit_xxs": "MobileViT",
        "mobilevit": "MobileViT",
    }

    model_leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=legend_label.get(bm.lower(), bm),
            markerfacecolor=base_color[bm],
            markersize=4.5,
        )
        for bm in base_models
    ]

    legend_rows = max(len(model_leg), len(marker_leg))
    blank_leg = Line2D([], [], linestyle="none", marker="", color="none", label="")
    solver_column = marker_leg + [blank_leg] * (legend_rows - len(marker_leg))
    model_column = model_leg + [blank_leg] * (legend_rows - len(model_leg))

    combined_leg = ax.legend(
        handles=model_column + solver_column,
        loc="upper right",
        bbox_to_anchor=(0.97, 1.13),
        frameon=True,
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        fontsize=mpl.rcParams["legend.fontsize"],
        ncol=2,
        columnspacing=0.5,
        handlelength=1,
        handletextpad=0.11,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, pad_inches=0, transparent=False)
        print("Saved:", save_path)
    # if plt.get_backend().lower() != "agg":
    #     plt.show()
    # else:
    #     plt.close(fig)
    plt.close(fig)

if __name__ == "__main__":
    dataframe = build_dataframe()
    print(
        dataframe[
            [
                "Model",
                "Layers",
                "State_Space",
                "Effective_State_Graph_Size",
                "DP_Time",
                "New_DP_Time",
                "New_DP_Local_Moves",
                "ILP_Time",
                "ILP_Status",
            ]
        ].to_string(index=False)
    )
    plot_dp_ilp_grouped_by_model(
        dataframe,
        save_path="./outputs/figure9.pdf",
    )
