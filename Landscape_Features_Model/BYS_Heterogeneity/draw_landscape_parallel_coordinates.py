from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent / "Result"
PREFERENCES_PATH = RESULT_DIR / "Rater_10D_Preferences.csv"
PNG_OUTPUT = RESULT_DIR / "Landscape_Parallel_Coordinates_10Features.png"

FEATURE_ORDER = [
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

FEATURE_LABELS = {
    "warm_cool_balance": "warm_cool_balance",
    "horizon_y_norm": "horizon_y_norm",
    "depth_gradient_mean": "depth_gradient_mean",
    "artificial_ratio": "artificial_ratio",
    "saturation_mean": "saturation_mean",
    "thirds_brightness_mean": "thirds_brightness_mean",
    "line_strength": "line_strength",
    "depth_std": "depth_std",
    "semantic_diversity": "semantic_diversity",
    "left_right_balance": "left_right_balance",
}

MAIN_COLOR = "#ECA7C0"
LINE_COLOR = "#C981A9"
ENVELOPE_COLOR = "#F2C6D7"
BASELINE_COLOR = "#A23B3B"
MEDIAN_COLOR = "#202020"
GRID_COLOR = "#E5E9EF"
VERTICAL_GRID_COLOR = "#E7D4DE"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_preferences() -> pd.DataFrame:
    if not PREFERENCES_PATH.exists():
        raise FileNotFoundError(f"Preference matrix not found: {PREFERENCES_PATH}")
    df = pd.read_csv(PREFERENCES_PATH)
    missing = [feature for feature in FEATURE_ORDER if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_ORDER].copy()


def draw_parallel_coordinates() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    df_raw = load_preferences()

    x = np.arange(df_raw.shape[1])
    values = df_raw.to_numpy(dtype=float)
    q05 = df_raw.quantile(0.05).to_numpy(dtype=float)
    q50 = df_raw.quantile(0.50).to_numpy(dtype=float)
    q95 = df_raw.quantile(0.95).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.2), dpi=450)
    for row in values:
        ax.plot(x, row, color=LINE_COLOR, alpha=0.050, lw=0.78, zorder=1)

    ax.fill_between(x, q05, q95, color=ENVELOPE_COLOR, alpha=0.32, linewidth=0, zorder=2)
    ax.plot(x, q50, color=MEDIAN_COLOR, lw=1.75, marker="o", markersize=3.4, zorder=4, label="Median")
    ax.axhline(
        0,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.85,
        alpha=0.98,
        label="Neutral Baseline (0)",
        zorder=3,
    )

    flat_values = values.reshape(-1)
    y_min = float(np.quantile(flat_values, 0.003))
    y_max = float(np.quantile(flat_values, 0.997))
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
        rotation=45,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlim(0, len(FEATURE_ORDER) - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Parallel Coordinates of {len(df_raw)} Raters across 10 Landscape Features",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Preference Slope (BLUPs)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for xi in x:
        ax.axvline(xi, color=VERTICAL_GRID_COLOR, linestyle=":", lw=0.8, alpha=0.85, zorder=0)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.95)
    ax.set_axisbelow(True)

    legend_handles = [
        mpl.lines.Line2D([0], [0], color=LINE_COLOR, lw=1.1, alpha=0.55, label="Individual raters"),
        mpl.patches.Patch(facecolor=ENVELOPE_COLOR, edgecolor="none", alpha=0.32, label="5-95% envelope"),
        mpl.lines.Line2D([0], [0], color=MEDIAN_COLOR, lw=1.75, marker="o", markersize=3.4, label="Median"),
        mpl.lines.Line2D([0], [0], color=BASELINE_COLOR, lw=1.85, linestyle="--", label="Neutral Baseline (0)"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        fontsize=8.4,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.8,
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.29, top=0.89)
    fig.savefig(PNG_OUTPUT, dpi=450, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    draw_parallel_coordinates()
