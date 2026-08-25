from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "Result"
CURVE_DATA = RESULT_DIR / "GAMM_NonLinear_Expected_Curves_4Features_data.csv"
RAW_FEATURE_DATA = BASE_DIR.parent / "BYS_interpretable_Features_Model" / "landscape_interpretable_features.csv"

FEATURES = [
    "horizon_y_norm",
    "saturation_mean",
    "artificial_ratio",
    "semantic_diversity",
]

PANEL_SIZE = (5.2, 4.4)
CURVE_LINE_COLOR = "#DC1F1E"
CURVE_BAND_COLOR = "#DCC7D9"
GRID_COLOR = "#E6EAF0"
RUG_QUANTILE_COUNT = 20


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_curve_panel(
    curve_df: pd.DataFrame,
    raw_features: pd.DataFrame,
    feature: str,
    output_path: Path,
    image_points: pd.DataFrame | None = None,
) -> None:
    panel = curve_df.loc[curve_df["feature"] == feature].sort_values("x_real")
    if panel.empty:
        raise ValueError(f"Curve data not found for feature: {feature}")

    x = panel["x_real"].to_numpy(dtype=float)
    y = panel["expected_score"].to_numpy(dtype=float)
    lower = panel["ci_lower"].to_numpy(dtype=float)
    upper = panel["ci_upper"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=400)
    ax.fill_between(x, lower, upper, color=CURVE_BAND_COLOR, alpha=0.68, linewidth=0, zorder=1)
    if image_points is not None:
        ax.scatter(
            image_points[feature],
            image_points["rating_mean"],
            s=19,
            color="#D889A8",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.52,
            zorder=1.5,
        )
    ax.plot(x, y, color=CURVE_LINE_COLOR, linewidth=2.8, zorder=2)

    y_min = float(np.nanmin(lower))
    y_max = float(np.nanmax(upper))
    if image_points is not None:
        y_min = min(y_min, float(image_points["rating_mean"].min()))
        y_max = max(y_max, float(image_points["rating_mean"].max()))
    rug_height = (y_max - y_min) * 0.035
    observed_values = pd.to_numeric(raw_features[feature], errors="coerce").dropna().to_numpy(dtype=float)
    rug_values = np.quantile(
        observed_values,
        np.linspace(0.05, 0.95, RUG_QUANTILE_COUNT),
    )
    for value in rug_values:
        ax.vlines(value, y_min, y_min + rug_height, color="#7B8794", linewidth=0.7, alpha=0.8, zorder=3)

    y_pad = max((y_max - y_min) * 0.08, 0.02)
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=24, width=3.5, length=7.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")
    for spine in ax.spines.values():
        spine.set_linewidth(3.5)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.subplots_adjust(left=0.16, right=0.985, top=0.985, bottom=0.14)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    if not CURVE_DATA.exists():
        raise FileNotFoundError(f"Curve data not found: {CURVE_DATA}")
    if not RAW_FEATURE_DATA.exists():
        raise FileNotFoundError(f"Raw landscape feature data not found: {RAW_FEATURE_DATA}")

    set_style()
    curve_df = pd.read_csv(CURVE_DATA)
    raw_features = pd.read_csv(RAW_FEATURE_DATA)
    for index, feature in enumerate(FEATURES, start=2):
        output_path = RESULT_DIR / f"Fig3_{index:02d}_{feature}_textless.png"
        draw_curve_panel(curve_df, raw_features, feature, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
