from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "Result"

FACE_PREF_PATH = ROOT_DIR / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"
LANDSCAPE_PREF_PATH = RESULT_DIR / "Rater_10D_Preferences.csv"

PNG_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Consensus_Centered_Parallel_Coordinates.png"
DATA_OUTPUT = RESULT_DIR / "Combined_Face_Landscape_Consensus_Centered_Parallel_Coordinates_Data.csv"

FACE_COLOR = "#88A0CB"
LANDSCAPE_COLOR = "#ECA7C0"
FACE_LINE_COLOR = "#6C83B0"
LANDSCAPE_LINE_COLOR = "#D58CAD"
MEDIAN_COLOR = "#1E2430"
BASELINE_COLOR = "#8C3D3D"
GRID_COLOR = "#E8ECF2"
HIGH_SD_SHADE_COLOR = "#C82423"
LOW_SD_SHADE_COLOR = "#9CBF8C"

Y_MIN = -1.1
Y_MAX = 1.1
X_SPACING = 0.52
CENTER_GAP = X_SPACING / 2

LANDSCAPE_FEATURES = [
    "warm_cool_balance",
    "horizon_y_norm",
    "depth_gradient_mean",
    "artificial_ratio",
    "saturation_mean",
    "thirds_brightness_mean",
    "line_strength",
    "depth_std",
    "semantic_diversity",
    "left_right_balance",
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_face_preferences() -> pd.DataFrame:
    df = pd.read_csv(FACE_PREF_PATH, index_col=0)
    excluded = {"Cluster", "tSNE_1", "tSNE_2", "Final_Cluster"}
    feature_columns = [col for col in df.columns if col not in excluded]
    return df[feature_columns].apply(pd.to_numeric, errors="coerce")


def load_landscape_preferences() -> pd.DataFrame:
    df = pd.read_csv(LANDSCAPE_PREF_PATH)
    missing = [feature for feature in LANDSCAPE_FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing landscape preference columns: {missing}")
    return df[LANDSCAPE_FEATURES].apply(pd.to_numeric, errors="coerce")


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "heterogeneity_sd": df.std(),
            "q05": df.quantile(0.05),
            "q25": df.quantile(0.25),
            "median": df.quantile(0.50),
            "q75": df.quantile(0.75),
            "q95": df.quantile(0.95),
        }
    )


def draw_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: np.ndarray,
    color: str,
    line_color: str,
    label_prefix: str,
) -> None:
    values = df.to_numpy(dtype=float)
    q05 = df.quantile(0.05).to_numpy(dtype=float)
    q50 = df.quantile(0.50).to_numpy(dtype=float)
    q95 = df.quantile(0.95).to_numpy(dtype=float)

    for row in values:
        ax.plot(x, row, color=line_color, alpha=0.028, lw=0.62, zorder=1)

    ax.fill_between(x, q05, q95, color=color, alpha=0.34, linewidth=0, zorder=2)
    ax.plot(
        x,
        q50,
        color=MEDIAN_COLOR,
        lw=2.25,
        marker="o",
        markersize=3.9,
        markerfacecolor="#FFFFFF",
        markeredgecolor=MEDIAN_COLOR,
        markeredgewidth=1.0,
        zorder=5,
        label=f"{label_prefix} median",
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    face_df = load_face_preferences()
    landscape_df = load_landscape_preferences()

    face_sd = face_df.std()
    landscape_sd = landscape_df.std()

    landscape_order = landscape_sd.sort_values(ascending=True).index.tolist()
    face_order = face_sd.sort_values(ascending=False).index.tolist()

    landscape_sorted = landscape_df[landscape_order]
    face_sorted = face_df[face_order]

    landscape_x = -CENTER_GAP - np.arange(len(landscape_order) - 1, -1, -1) * X_SPACING
    face_x = CENTER_GAP + np.arange(len(face_order)) * X_SPACING

    fig, ax = plt.subplots(figsize=(16.0, 4.5), dpi=450)

    combined_sd = pd.concat([landscape_sd, face_sd])
    low_sd_threshold = combined_sd.quantile(0.25)
    high_sd_threshold = combined_sd.quantile(0.75)

    for xi, feature in zip(landscape_x, landscape_order):
        feature_sd = landscape_sd[feature]
        if feature_sd >= high_sd_threshold:
            ax.axvspan(xi - X_SPACING * 0.42, xi + X_SPACING * 0.42, color=HIGH_SD_SHADE_COLOR, alpha=0.09, linewidth=0, zorder=0)
        elif feature_sd <= low_sd_threshold:
            ax.axvspan(xi - X_SPACING * 0.42, xi + X_SPACING * 0.42, color=LOW_SD_SHADE_COLOR, alpha=0.13, linewidth=0, zorder=0)

    for xi, feature in zip(face_x, face_order):
        feature_sd = face_sd[feature]
        if feature_sd >= high_sd_threshold:
            ax.axvspan(xi - X_SPACING * 0.42, xi + X_SPACING * 0.42, color=HIGH_SD_SHADE_COLOR, alpha=0.09, linewidth=0, zorder=0)
        elif feature_sd <= low_sd_threshold:
            ax.axvspan(xi - X_SPACING * 0.42, xi + X_SPACING * 0.42, color=LOW_SD_SHADE_COLOR, alpha=0.13, linewidth=0, zorder=0)

    draw_group(ax, landscape_sorted, landscape_x, LANDSCAPE_COLOR, LANDSCAPE_LINE_COLOR, "Landscape")
    draw_group(ax, face_sorted, face_x, FACE_COLOR, FACE_LINE_COLOR, "Face")

    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.7, alpha=0.9, zorder=4)

    ax.text(
        -0.34,
        Y_MAX - 0.18,
        "Landscape",
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=LANDSCAPE_LINE_COLOR,
    )
    ax.text(
        0.34,
        Y_MAX - 0.18,
        "Face",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=FACE_LINE_COLOR,
    )
    all_x = np.concatenate([landscape_x, face_x])
    all_labels = landscape_order + face_order
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=48, ha="right", fontsize=8.2, fontweight="bold")
    ax.set_xlim(landscape_x.min(), face_x.max())
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Preference Slope (BLUPs)", fontsize=12, fontweight="bold")
    fig.suptitle(
        "Consensus-Centered Preference Heterogeneity Across Landscape and Face Features",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for xi in all_x:
        ax.axvline(xi, color="#D8DEE8", linestyle=":", lw=0.72, alpha=0.72, zorder=0)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.72, alpha=0.95)
    ax.set_axisbelow(True)

    legend_handles = [
        mpl.lines.Line2D([0], [0], color=LANDSCAPE_LINE_COLOR, lw=1.15, alpha=0.42, label="Landscape raters"),
        mpl.lines.Line2D([0], [0], color=FACE_LINE_COLOR, lw=1.15, alpha=0.42, label="Face raters"),
        mpl.patches.Patch(facecolor=LANDSCAPE_COLOR, edgecolor="none", alpha=0.34, label="Landscape 5-95%"),
        mpl.patches.Patch(facecolor=FACE_COLOR, edgecolor="none", alpha=0.34, label="Face 5-95%"),
        mpl.patches.Patch(facecolor=HIGH_SD_SHADE_COLOR, edgecolor="none", alpha=0.09, label="High SD"),
        mpl.patches.Patch(facecolor=LOW_SD_SHADE_COLOR, edgecolor="none", alpha=0.13, label="Low SD"),
        mpl.lines.Line2D(
            [0],
            [0],
            color=MEDIAN_COLOR,
            lw=2.25,
            marker="o",
            markersize=3.9,
            markerfacecolor="#FFFFFF",
            markeredgecolor=MEDIAN_COLOR,
            label="Median",
        ),
        mpl.lines.Line2D([0], [0], color=BASELINE_COLOR, lw=1.7, linestyle="--", label="Neutral baseline"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
        fontsize=8.2,
        ncol=8,
        columnspacing=0.75,
        handlelength=1.45,
    )

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.36, top=0.62)
    fig.savefig(PNG_OUTPUT, dpi=450, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    landscape_summary = compute_summary(landscape_sorted).reset_index(names="feature")
    landscape_summary.insert(0, "image_type", "Landscape")
    landscape_summary.insert(1, "x_position", landscape_x)

    face_summary = compute_summary(face_sorted).reset_index(names="feature")
    face_summary.insert(0, "image_type", "Face")
    face_summary.insert(1, "x_position", face_x)

    pd.concat([landscape_summary, face_summary], ignore_index=True).to_csv(
        DATA_OUTPUT, index=False, encoding="utf-8-sig"
    )

    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved data: {DATA_OUTPUT}")


if __name__ == "__main__":
    main()
