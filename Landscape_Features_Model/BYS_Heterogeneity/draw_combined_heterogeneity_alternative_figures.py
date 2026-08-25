from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


RESULT_DIR = Path(__file__).resolve().parent / "Result"
DATA_PATH = RESULT_DIR / "Combined_Face_Landscape_Effect_vs_Heterogeneity_Data.csv"

FACETED_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Faceted_Quadrant.png"
BUBBLE_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Bubble_Matrix.png"
LOLLIPOP_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Heterogeneity_Lollipop.png"

FACE_COLOR = "#88A0CB"
LANDSCAPE_COLOR = "#ECA7C0"
FACE_DARK = "#5875A7"
LANDSCAPE_DARK = "#C96D96"
ERROR_COLOR = "#747980"
TEXT_COLOR = "#24292F"
GRID_COLOR = "#E7EAF0"
ZERO_COLOR = "#747980"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Combined data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = {
        "image_group",
        "feature",
        "plot_label",
        "fixed_mean",
        "fixed_hdi_3",
        "fixed_hdi_97",
        "heterogeneity_mean",
        "heterogeneity_hdi_3",
        "heterogeneity_hdi_97",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {sorted(missing)}")
    return df


def group_color(group: str) -> str:
    return FACE_COLOR if group == "Face" else LANDSCAPE_COLOR


def group_dark(group: str) -> str:
    return FACE_DARK if group == "Face" else LANDSCAPE_DARK


def padded_limits(values_low: pd.Series, values_high: pd.Series, pad_ratio: float) -> tuple[float, float]:
    low = float(values_low.min())
    high = float(values_high.max())
    pad = (high - low) * pad_ratio
    return low - pad, high + pad


def annotate_points(ax: plt.Axes, df: pd.DataFrame, *, fontsize: float = 8.2) -> None:
    # A deterministic radial offset pattern keeps labels legible without adding
    # an extra dependency such as adjustText.
    offsets = [
        (8, 8),
        (8, -9),
        (-8, 9),
        (-8, -9),
        (12, 0),
        (-12, 0),
    ]
    sorted_df = df.sort_values("heterogeneity_mean", ascending=False).reset_index(drop=True)
    for idx, row in sorted_df.iterrows():
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            str(row["plot_label"]),
            xy=(row["fixed_mean"], row["heterogeneity_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left" if dx >= 0 else "right",
            va="center",
            color=TEXT_COLOR,
            arrowprops=dict(arrowstyle="-", color="#9AA0A6", lw=0.45, shrinkA=0, shrinkB=4),
            zorder=5,
        )


def draw_faceted_quadrant(df: pd.DataFrame) -> None:
    x_min, x_max = padded_limits(df["fixed_hdi_3"], df["fixed_hdi_97"], 0.055)
    y_min, y_max = padded_limits(df["heterogeneity_hdi_3"], df["heterogeneity_hdi_97"], 0.055)
    y_min = max(0.0, y_min)
    y_threshold = float(df["heterogeneity_mean"].median())

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 9.4), dpi=450, sharex=True, sharey=True)
    for ax, group in zip(axes, ["Face", "Landscape"]):
        sub = df.loc[df["image_group"] == group].copy()
        color = group_color(group)
        dark = group_dark(group)

        xerr = [sub["fixed_mean"] - sub["fixed_hdi_3"], sub["fixed_hdi_97"] - sub["fixed_mean"]]
        yerr = [
            sub["heterogeneity_mean"] - sub["heterogeneity_hdi_3"],
            sub["heterogeneity_hdi_97"] - sub["heterogeneity_mean"],
        ]
        ax.errorbar(
            sub["fixed_mean"],
            sub["heterogeneity_mean"],
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=0.95,
            capsize=2.4,
            capthick=0.85,
            alpha=0.58,
            zorder=1,
        )
        ax.scatter(
            sub["fixed_mean"],
            sub["heterogeneity_mean"],
            s=72,
            color=color,
            edgecolor="#202020",
            linewidth=0.68,
            zorder=3,
        )
        annotate_points(ax, sub, fontsize=8.0)

        ax.axvline(0.0, color=ZERO_COLOR, linestyle=(0, (4, 4)), linewidth=1.05, zorder=0)
        ax.axhline(y_threshold, color=ZERO_COLOR, linestyle=(0, (4, 4)), linewidth=0.95, zorder=0)
        ax.grid(color=GRID_COLOR, linewidth=0.82)
        ax.set_axisbelow(True)
        ax.set_title(f"{group} features", fontsize=13.5, fontweight="bold", color=dark, pad=8)
        ax.tick_params(labelsize=10.2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Between-rater heterogeneity SD", fontsize=12)
    axes[1].set_ylabel("Between-rater heterogeneity SD", fontsize=12)
    axes[1].set_xlabel("Population-level effect on aesthetic rating (posterior mean)", fontsize=12)
    fig.suptitle("Average Effect Versus Individual Disagreement", fontsize=16, fontweight="bold", y=0.988)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    fig.savefig(FACETED_OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_bubble_matrix(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["feature_label"] = plot_df["plot_label"] + "  "
    plot_df = plot_df.sort_values(["image_group", "heterogeneity_mean"], ascending=[True, False]).reset_index(drop=True)
    plot_df["x"] = np.arange(len(plot_df))
    y_map = {"Face": 1.0, "Landscape": 0.0}
    plot_df["y"] = plot_df["image_group"].map(y_map)

    effect_limit = float(np.nanmax(np.abs(plot_df[["fixed_hdi_3", "fixed_hdi_97", "fixed_mean"]].to_numpy())))
    norm = mpl.colors.TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "effect_blue_white_pink",
        ["#5F79B4", "#F4F6FA", "#D86F9B"],
        N=256,
    )
    size_min, size_max = 120, 960
    h_min = float(plot_df["heterogeneity_mean"].min())
    h_max = float(plot_df["heterogeneity_mean"].max())
    sizes = size_min + (plot_df["heterogeneity_mean"] - h_min) / max(h_max - h_min, 1e-12) * (size_max - size_min)

    fig, ax = plt.subplots(figsize=(12.8, 4.8), dpi=450)
    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=sizes,
        c=plot_df["fixed_mean"],
        cmap=cmap,
        norm=norm,
        edgecolor="#262626",
        linewidth=0.72,
        alpha=0.94,
        zorder=3,
    )

    for _, row in plot_df.iterrows():
        ax.plot(
            [row["x"], row["x"]],
            [row["y"] - 0.22, row["y"] + 0.22],
            color=group_color(row["image_group"]),
            linewidth=2.8,
            alpha=0.22,
            solid_capstyle="round",
            zorder=1,
        )

    ax.set_xticks(plot_df["x"])
    ax.set_xticklabels(plot_df["feature_label"], rotation=50, ha="right", fontsize=8.7)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["Face", "Landscape"], fontsize=12, fontweight="bold")
    ax.set_ylim(-0.62, 1.62)
    ax.set_xlim(-0.8, len(plot_df) - 0.2)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.72, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title("Effect Direction and Heterogeneity Magnitude Across Features", fontsize=15.2, fontweight="bold", pad=11)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.028)
    cbar.set_label("Population-level effect", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9.4)
    cbar.outline.set_linewidth(0.8)

    handles = [
        plt.scatter([], [], s=size_min, facecolor="#E9ECF2", edgecolor="#262626", label=f"Low heterogeneity ({h_min:.2f})"),
        plt.scatter([], [], s=(size_min + size_max) / 2, facecolor="#E9ECF2", edgecolor="#262626", label="Medium heterogeneity"),
        plt.scatter([], [], s=size_max, facecolor="#E9ECF2", edgecolor="#262626", label=f"High heterogeneity ({h_max:.2f})"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.43),
        fontsize=8.7,
        ncol=3,
        labelspacing=1.0,
        columnspacing=1.8,
        handletextpad=1.0,
    )

    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.42, top=0.88)
    fig.savefig(BUBBLE_OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_heterogeneity_lollipop(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("heterogeneity_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8.4, 9.2), dpi=450)
    for idx, row in plot_df.iterrows():
        color = group_color(row["image_group"])
        ax.hlines(y[idx], 0, row["heterogeneity_mean"], color=color, linewidth=3.8, alpha=0.38, zorder=1)
        ax.errorbar(
            row["heterogeneity_mean"],
            y[idx],
            xerr=np.array(
                [
                    [row["heterogeneity_mean"] - row["heterogeneity_hdi_3"]],
                    [row["heterogeneity_hdi_97"] - row["heterogeneity_mean"]],
                ]
            ),
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=1.05,
            capsize=2.4,
            capthick=0.9,
            alpha=0.62,
            zorder=2,
        )
        ax.scatter(
            row["heterogeneity_mean"],
            y[idx],
            s=74,
            marker="o" if row["image_group"] == "Face" else "D",
            color=color,
            edgecolor="#202020",
            linewidth=0.68,
            zorder=3,
        )

    labels = [f"{row.plot_label}  " for row in plot_df.itertuples(index=False)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.1)
    for tick, group in zip(ax.get_yticklabels(), plot_df["image_group"]):
        tick.set_color(group_dark(group))
        tick.set_fontweight("bold")

    ax.set_xlabel("Between-rater heterogeneity SD", fontsize=12.2)
    ax.set_title("Ranking of Feature-Level Individual Disagreement", fontsize=15.2, fontweight="bold", pad=11)
    ax.set_xlim(0, float(plot_df["heterogeneity_hdi_97"].max()) * 1.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.82)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10.4)

    handles = [
        Line2D([0], [0], marker="o", color=FACE_COLOR, markerfacecolor=FACE_COLOR, markeredgecolor="#202020", linewidth=3, label="Face features"),
        Line2D([0], [0], marker="D", color=LANDSCAPE_COLOR, markerfacecolor=LANDSCAPE_COLOR, markeredgecolor="#202020", linewidth=3, label="Landscape features"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9.4)

    fig.tight_layout()
    fig.savefig(LOLLIPOP_OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    df = load_data()
    draw_faceted_quadrant(df)
    draw_bubble_matrix(df)
    draw_heterogeneity_lollipop(df)
    print(f"Saved: {FACETED_OUTPUT}")
    print(f"Saved: {BUBBLE_OUTPUT}")
    print(f"Saved: {LOLLIPOP_OUTPUT}")


if __name__ == "__main__":
    main()
