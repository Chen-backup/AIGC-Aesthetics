from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "Result"
FACE_SUMMARY_PATH = ROOT_DIR / "BYS_Ultimate_Heterogeneity_Figures" / "Ultimate_Heterogeneity_14Feature_Summary.csv"
LANDSCAPE_SUMMARY_PATH = RESULT_DIR / "Ultimate_Heterogeneity_Leaderboard.csv"

PNG_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Effect_vs_Heterogeneity.png"
PDF_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Effect_vs_Heterogeneity.pdf"
CSV_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Effect_vs_Heterogeneity_Data.csv"

FACE_COLOR = "#88A0CB"
LANDSCAPE_COLOR = "#ECA7C0"
ERROR_COLOR = "#6F7378"
TEXT_COLOR = "#24292F"
GRID_COLOR = "#E6E9EF"

LANDSCAPE_LABELS = {
    "warm_cool_balance": "Warm-cool",
    "horizon_y_norm": "Horizon",
    "depth_gradient_mean": "Depth gradient",
    "artificial_ratio": "Artificial",
    "saturation_mean": "Saturation",
    "thirds_brightness_mean": "Thirds brightness",
    "line_strength": "Line strength",
    "depth_std": "Depth variation",
    "semantic_diversity": "Semantic diversity",
    "left_right_balance": "Left-right balance",
}

FACE_LABEL_OFFSETS = {
    "upper_lower_ratio": (10, 6),
    "face_hw_ratio": (-12, 8),
    "mouth_nose_ratio": (10, 6),
    "eye_y_ratio": (-30, -15),
    "eye_face_w_ratio": (-42, 15),
    "edge_density": (-15, 16),
    "mouth_face_w_ratio": (10, -12),
    "face_clarity": (-18, -12),
    "le_nose_re_angle": (-14, 10),
    "face_brightness": (8, 7),
    "face_contrast": (10, 1),
    "saturation": (-10, -13),
    "total_symmetry": (12, -9),
    "three_courts_balance": (13, -18),
}

LANDSCAPE_LABEL_OFFSETS = {
    "saturation_mean": (11, -7),
    "line_strength": (10, 12),
    "depth_gradient_mean": (8, 8),
    "warm_cool_balance": (8, 8),
    "horizon_y_norm": (-32, -14),
    "artificial_ratio": (8, 17),
    "left_right_balance": (10, -10),
    "semantic_diversity": (8, -18),
    "thirds_brightness_mean": (8, 14),
    "depth_std": (8, 8),
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_combined_summary() -> pd.DataFrame:
    face = pd.read_csv(FACE_SUMMARY_PATH).copy()
    face["image_group"] = "Face"
    face["plot_label"] = face["label"]

    landscape = pd.read_csv(LANDSCAPE_SUMMARY_PATH).copy()
    landscape["image_group"] = "Landscape"
    landscape["plot_label"] = landscape["feature"].map(LANDSCAPE_LABELS).fillna(
        landscape["feature"].str.replace("_", " ", regex=False).str.title()
    )

    columns = [
        "image_group",
        "feature",
        "plot_label",
        "heterogeneity_mean",
        "heterogeneity_hdi_3",
        "heterogeneity_hdi_97",
        "fixed_mean",
        "fixed_hdi_3",
        "fixed_hdi_97",
    ]
    combined = pd.concat([face[columns], landscape[columns]], ignore_index=True)
    return combined.sort_values(["image_group", "heterogeneity_mean"], ascending=[True, False]).reset_index(drop=True)


def draw_error_points(ax: plt.Axes, df: pd.DataFrame, *, group: str, color: str, marker: str) -> None:
    group_df = df.loc[df["image_group"] == group].copy()
    xerr = [
        group_df["fixed_mean"] - group_df["fixed_hdi_3"],
        group_df["fixed_hdi_97"] - group_df["fixed_mean"],
    ]
    yerr = [
        group_df["heterogeneity_mean"] - group_df["heterogeneity_hdi_3"],
        group_df["heterogeneity_hdi_97"] - group_df["heterogeneity_mean"],
    ]
    ax.errorbar(
        group_df["fixed_mean"],
        group_df["heterogeneity_mean"],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor=ERROR_COLOR,
        elinewidth=1.0,
        capsize=2.6,
        capthick=0.95,
        alpha=0.62,
        zorder=1,
    )
    ax.scatter(
        group_df["fixed_mean"],
        group_df["heterogeneity_mean"],
        s=76 if group == "Landscape" else 64,
        marker=marker,
        color=color,
        edgecolor="#222222",
        linewidth=0.72,
        alpha=0.96,
        zorder=3,
    )


def annotate_group(ax: plt.Axes, df: pd.DataFrame, *, group: str) -> None:
    offsets = FACE_LABEL_OFFSETS if group == "Face" else LANDSCAPE_LABEL_OFFSETS
    group_df = df.loc[df["image_group"] == group]
    for _, row in group_df.iterrows():
        offset = offsets.get(str(row["feature"]), (8, 6))
        ax.annotate(
            str(row["plot_label"]),
            xy=(row["fixed_mean"], row["heterogeneity_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.0 if group == "Landscape" else 7.2,
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            color=TEXT_COLOR,
            arrowprops=dict(arrowstyle="-", color="#90949A", lw=0.52, shrinkA=0, shrinkB=5),
            zorder=4,
        )


def draw_combined_figure() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    df = load_combined_summary()
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    x_low = float(df["fixed_hdi_3"].min())
    x_high = float(df["fixed_hdi_97"].max())
    y_low = max(0.0, float(df["heterogeneity_hdi_3"].min()) - 0.03)
    y_high = float(df["heterogeneity_hdi_97"].max()) + 0.045
    x_pad = (x_high - x_low) * 0.045

    y_threshold = float(df["heterogeneity_mean"].median())

    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=450)
    fig.patch.set_facecolor("white")

    ax.axvline(0.0, color="#6F7378", linestyle=(0, (4, 4)), linewidth=1.15, zorder=0)
    ax.axhline(y_threshold, color="#6F7378", linestyle=(0, (4, 4)), linewidth=1.05, zorder=0)

    draw_error_points(ax, df, group="Face", color=FACE_COLOR, marker="o")
    draw_error_points(ax, df, group="Landscape", color=LANDSCAPE_COLOR, marker="D")
    annotate_group(ax, df, group="Face")
    annotate_group(ax, df, group="Landscape")

    ax.text(
        0.018,
        0.965,
        "weak average effect\nhigh disagreement",
        transform=ax.transAxes,
        fontsize=8.3,
        ha="left",
        va="top",
        color="#5F646A",
    )
    ax.text(
        0.925,
        0.965,
        "strong average effect\nhigh disagreement",
        transform=ax.transAxes,
        fontsize=8.3,
        ha="right",
        va="top",
        color="#5F646A",
    )

    ax.set_xlim(x_low - x_pad, x_high + x_pad)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("Population-level effect on aesthetic rating (posterior mean)", fontsize=12.5)
    ax.set_ylabel("Between-rater heterogeneity SD", fontsize=12.5)
    ax.set_title(
        "Average Aesthetic Effect Versus Individual Disagreement",
        fontsize=15.5,
        fontweight="bold",
        pad=13,
    )
    ax.grid(color=GRID_COLOR, linewidth=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10.8, length=4.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.18)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FACE_COLOR, markeredgecolor="#222222", markersize=7.2, label="Face features"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=LANDSCAPE_COLOR, markeredgecolor="#222222", markersize=7.2, label="Landscape features"),
        Line2D([0], [0], color="#6F7378", linestyle=(0, (4, 4)), linewidth=1.1, label="Reference lines"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.2, handlelength=2.4)

    fig.tight_layout()
    fig.savefig(PNG_OUTPUT, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved PDF: {PDF_OUTPUT}")


if __name__ == "__main__":
    draw_combined_figure()
