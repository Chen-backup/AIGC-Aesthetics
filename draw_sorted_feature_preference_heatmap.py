from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT_DIR = Path(__file__).resolve().parent
FACE_FIGURE_DIR = ROOT_DIR / "Picture_fig3"
sys.path.insert(0, str(FACE_FIGURE_DIR))

import draw_fig3_01D_donut_feature_means_sd_effect_values as effect_cmap_source  # noqa: E402

INPUT_PATH = ROOT_DIR / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"
OUTPUT_DIR = ROOT_DIR / "BYS_Heterogeneity_Evidence"
PNG_OUTPUT = OUTPUT_DIR / "Plot5_ColumnSorted_Preference_Heatmap.png"
CSV_OUTPUT = OUTPUT_DIR / "Plot5_ColumnSorted_Preference_Heatmap_Data.csv"

EXCLUDE_COLUMNS = {"Cluster", "tSNE_1", "tSNE_2", "Final_Cluster", "rater__factor_dim", "rater"}

FEATURE_LABELS = {
    "face_hw_ratio": "Face H/W",
    "eye_face_w_ratio": "Eye/face W",
    "mouth_face_w_ratio": "Mouth/face W",
    "three_courts_balance": "Three-courts",
    "upper_lower_ratio": "Upper/lower",
    "eye_y_ratio": "Eye Y",
    "total_symmetry": "Symmetry",
    "le_nose_re_angle": "Nose angle",
    "mouth_nose_ratio": "Mouth/nose",
    "face_brightness": "Brightness",
    "face_contrast": "Contrast",
    "face_clarity": "Clarity",
    "saturation": "Saturation",
    "edge_density": "Edge density",
}

FIG_SIZE = (8.8, 7.4)
DPI = 450

# =========================
# Color controls
# =========================
# Set to a Matplotlib colormap name, e.g. "coolwarm", "RdBu_r", or "seismic".
# Set to None to use the switches below instead.
COLORMAP_NAME = "coolwarm"

# Darken the selected Matplotlib colormap by blending every color toward black.
# 0.00 keeps the original colormap; 0.15 means 15% darker.
COLORMAP_DARKEN_AMOUNT = 0.00

# True: use the same colormap as
# Landscape_Features_Model/Figures_Landscape/Posterior_effect_value_colorbar.png.
# False: use MANUAL_CMAP_COLORS below.
USE_POSTERIOR_EFFECT_COLORBAR = False

# Manual colorbar colors, used only when USE_POSTERIOR_EFFECT_COLORBAR = False.
# Order: strong negative, weak negative, weak positive, strong positive.
# Blue means preferring lower feature values; red means preferring higher values.
MANUAL_CMAP_COLORS = ["#F9ECEA", "#EEC6C3", "#DA7875", "#C93F37"]

COLORBAR_LABEL = (
    "Preference slope (BLUPs)\n"
    "blue = prefers lower values, red = prefers higher values"
)


def get_preference_cmap() -> mpl.colors.Colormap:
    if COLORMAP_NAME is not None:
        cmap = mpl.colormaps[COLORMAP_NAME]
        if COLORMAP_DARKEN_AMOUNT <= 0:
            return cmap
        colors = cmap(np.linspace(0, 1, 256))
        colors[:, :3] *= 1.0 - COLORMAP_DARKEN_AMOUNT
        return mpl.colors.ListedColormap(colors, name=f"{COLORMAP_NAME}_darkened")
    if USE_POSTERIOR_EFFECT_COLORBAR:
        return effect_cmap_source.make_effect_cmap()
    return mpl.colors.LinearSegmentedColormap.from_list(
        "manual_preference_slope_cmap",
        MANUAL_CMAP_COLORS,
        N=256,
    )


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_preferences() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Preference matrix not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    feature_columns = [column for column in df.columns if column not in EXCLUDE_COLUMNS]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {INPUT_PATH}")
    return df[feature_columns].apply(pd.to_numeric, errors="coerce")


def column_sorted_preferences(df: pd.DataFrame) -> pd.DataFrame:
    sorted_columns = {}
    for feature in df.columns:
        values = df[feature].dropna().sort_values(ascending=False).to_numpy(dtype=float)
        sorted_columns[feature] = values

    min_length = min(len(values) for values in sorted_columns.values())
    return pd.DataFrame({feature: values[:min_length] for feature, values in sorted_columns.items()})


def draw_heatmap() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    raw = load_preferences()
    sorted_df = column_sorted_preferences(raw)
    sorted_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    values = sorted_df.to_numpy(dtype=float)
    robust_limit = float(np.nanquantile(np.abs(values), 0.995))
    norm = TwoSlopeNorm(vmin=-robust_limit, vcenter=0.0, vmax=robust_limit)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=get_preference_cmap(),
        norm=norm,
    )

    labels = [FEATURE_LABELS.get(feature, feature.replace("_", " ").title()) for feature in sorted_df.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight="bold")

    n_raters = len(sorted_df)
    y_ticks = [0, n_raters // 4, n_raters // 2, 3 * n_raters // 4, n_raters - 1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(tick + 1) for tick in y_ticks], fontsize=9.5)
    ax.set_ylabel("Raters sorted within each feature\n(high preference to low preference)", fontsize=11.5, fontweight="bold")
    ax.set_title(
        f"Column-Sorted Preference Distributions Across {len(labels)} Aesthetic Features",
        fontsize=15.5,
        fontweight="bold",
        pad=11,
    )

    for boundary in np.arange(0.5, len(labels), 1):
        ax.axvline(boundary, color="white", linewidth=0.55, alpha=0.85)
    ax.axhline(n_raters / 2 - 0.5, color="#262626", linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.45)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    cbar = fig.colorbar(image, ax=ax, pad=0.018, fraction=0.032)
    cbar.set_label(COLORBAR_LABEL, fontsize=10.2)
    cbar.ax.tick_params(labelsize=9.2, width=0.8, length=3.6)
    cbar.outline.set_linewidth(0.85)

    fig.subplots_adjust(left=0.13, right=0.90, top=0.90, bottom=0.22)
    fig.savefig(PNG_OUTPUT, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    draw_heatmap()
