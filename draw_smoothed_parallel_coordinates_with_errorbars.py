from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent

FACE_DATA = ROOT / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"
FACE_OUT = ROOT / "BYS_Heterogeneity_Evidence" / "Plot1_Parallel_Coordinates_Smoothed_ErrorBars.png"
FACE_TEXTLESS_OUT = ROOT / "BYS_Heterogeneity_Evidence" / "Plot1_Parallel_Coordinates_Smoothed_ErrorBars_textless.png"
FACE_TEXTLESS_PDF_OUT = ROOT / "BYS_Heterogeneity_Evidence" / "Plot1_Parallel_Coordinates_Smoothed_ErrorBars_textless.pdf"
FACE_SUMMARY = ROOT / "BYS_Heterogeneity_Evidence" / "Plot1_Parallel_Coordinates_Smoothed_ErrorBars_Data.csv"

LANDSCAPE_DATA = ROOT / "Landscape_Features_Model" / "BYS_Heterogeneity" / "Result" / "Rater_10D_Preferences.csv"
LANDSCAPE_OUT = (
    ROOT
    / "Landscape_Features_Model"
    / "BYS_Heterogeneity"
    / "Result"
    / "Landscape_Parallel_Coordinates_10Features_Smoothed_ErrorBars.png"
)
LANDSCAPE_TEXTLESS_OUT = (
    ROOT
    / "Landscape_Features_Model"
    / "BYS_Heterogeneity"
    / "Result"
    / "Landscape_Parallel_Coordinates_10Features_Smoothed_ErrorBars_textless.png"
)
LANDSCAPE_TEXTLESS_PDF_OUT = (
    ROOT
    / "Landscape_Features_Model"
    / "BYS_Heterogeneity"
    / "Result"
    / "Landscape_Parallel_Coordinates_10Features_Smoothed_ErrorBars_textless.pdf"
)
LANDSCAPE_SUMMARY = (
    ROOT
    / "Landscape_Features_Model"
    / "BYS_Heterogeneity"
    / "Result"
    / "Landscape_Parallel_Coordinates_10Features_Smoothed_ErrorBars_Data.csv"
)


FACE_FEATURES = [
    "face_hw_ratio",
    "eye_face_w_ratio",
    "mouth_face_w_ratio",
    "three_courts_balance",
    "upper_lower_ratio",
    "eye_y_ratio",
    "total_symmetry",
    "le_nose_re_angle",
    "mouth_nose_ratio",
    "face_brightness",
    "face_contrast",
    "face_clarity",
    "saturation",
    "edge_density",
]

FACE_LABELS = [
    "Face\nH/W",
    "Eye\nWidth",
    "Mouth\nWidth",
    "Three\nCourts",
    "Upper\nLower",
    "Eye\nY",
    "Total\nSymmetry",
    "Eye-Nose\nAngle",
    "Mouth\nNose",
    "Brightness",
    "Contrast",
    "Clarity",
    "Saturation",
    "Edge\nDensity",
]

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

LANDSCAPE_LABELS = [
    "Warm/Cool\nBalance",
    "Horizon-Y\nPosition",
    "Depth\nGradient",
    "Artificial\nRatio",
    "Saturation",
    "Thirds\nBrightness",
    "Line\nStrength",
    "Depth\nSD",
    "Semantic\nDiversity",
    "Left-Right\nBalance",
]


def _safe_pchip(x, y, dense_x):
    return PchipInterpolator(x, y)(dense_x)


def _plot_smoothed_parallel(
    data_path,
    features,
    labels,
    output_path,
    summary_path,
    color,
    envelope_color,
    title,
    figure_size,
    line_alpha,
    line_width,
    show_text=True,
    pdf_output_path=None,
):
    df = pd.read_csv(data_path)
    values = df[features].astype(float)

    x = np.arange(len(features), dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 700)

    q05 = values.quantile(0.05).to_numpy()
    q50 = values.quantile(0.50).to_numpy()
    q95 = values.quantile(0.95).to_numpy()
    mean = values.mean().to_numpy()
    variance = values.var(ddof=1).to_numpy()
    sd = np.sqrt(variance)

    summary = pd.DataFrame(
        {
            "feature": features,
            "label": [label.replace("\n", " ") for label in labels],
            "mean": mean,
            "variance": variance,
            "sd": sd,
            "q05": q05,
            "median": q50,
            "q95": q95,
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    y_min = min(values.quantile(0.01).min(), np.min(mean - sd), np.min(q05))
    y_max = max(values.quantile(0.99).max(), np.max(mean + sd), np.max(q95))
    y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 0.1

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=figure_size, dpi=300)
    ax.set_facecolor("#FFFFFF")

    for xi in x:
        ax.axvline(xi, color="#DADADA", lw=0.9, alpha=0.85, zorder=0)
    ax.axhline(0, color="#666666", lw=1.0, ls="--", alpha=0.55, zorder=1)

    # Draw individual preference profiles as smooth, faint curves.
    for row in values.to_numpy():
        ax.plot(
            dense_x,
            _safe_pchip(x, row, dense_x),
            color=color,
            alpha=line_alpha,
            lw=line_width,
            solid_capstyle="round",
            zorder=2,
        )

    q05_dense = _safe_pchip(x, q05, dense_x)
    q50_dense = _safe_pchip(x, q50, dense_x)
    q95_dense = _safe_pchip(x, q95, dense_x)

    ax.fill_between(
        dense_x,
        q05_dense,
        q95_dense,
        color=envelope_color,
        alpha=0.18,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        dense_x,
        q50_dense,
        color="#222222",
        lw=2.0,
        alpha=0.92,
        solid_capstyle="round",
        zorder=5,
        label="Median",
    )

    ax.errorbar(
        x,
        mean,
        yerr=sd,
        fmt="o",
        color="#202020",
        ecolor="#202020",
        elinewidth=2.0,
        capsize=4.5,
        capthick=2.0,
        markersize=4.5,
        markerfacecolor="#FFFFFF",
        markeredgewidth=1.7,
        zorder=7,
        label="Mean ± SD",
    )

    ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks(x)
    if show_text:
        ax.set_xticklabels(labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=10, width=1.2, length=4)
        ax.tick_params(axis="x", width=1.2, length=4, pad=8)
        ax.set_ylabel("Preference Slope (BLUPs)", fontsize=13)
        ax.set_title(title, fontsize=16, pad=15)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", width=1.8, length=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_text:
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)
    else:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    if show_text:
        ax.legend(frameon=False, loc="upper right", fontsize=10, handlelength=2.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    if pdf_output_path is not None:
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    if pdf_output_path is not None:
        print(f"Saved PDF: {pdf_output_path}")
    print(f"Saved summary: {summary_path}")


def main():
    _plot_smoothed_parallel(
        data_path=FACE_DATA,
        features=FACE_FEATURES,
        labels=FACE_LABELS,
        output_path=FACE_OUT,
        summary_path=FACE_SUMMARY,
        color="#88A0CB",
        envelope_color="#88A0CB",
        title="Smoothed Face Preference Heterogeneity",
        figure_size=(13.5, 5.4),
        line_alpha=0.040,
        line_width=0.65,
    )
    _plot_smoothed_parallel(
        data_path=FACE_DATA,
        features=FACE_FEATURES,
        labels=FACE_LABELS,
        output_path=FACE_TEXTLESS_OUT,
        summary_path=FACE_SUMMARY,
        color="#88A0CB",
        envelope_color="#88A0CB",
        title="",
        figure_size=(13.5, 5.4),
        line_alpha=0.040,
        line_width=0.65,
        show_text=False,
        pdf_output_path=FACE_TEXTLESS_PDF_OUT,
    )

    _plot_smoothed_parallel(
        data_path=LANDSCAPE_DATA,
        features=LANDSCAPE_FEATURES,
        labels=LANDSCAPE_LABELS,
        output_path=LANDSCAPE_OUT,
        summary_path=LANDSCAPE_SUMMARY,
        color="#ECA7C0",
        envelope_color="#ECA7C0",
        title="Smoothed Landscape Preference Heterogeneity",
        figure_size=(11.5, 5.4),
        line_alpha=0.040,
        line_width=0.65,
    )
    _plot_smoothed_parallel(
        data_path=LANDSCAPE_DATA,
        features=LANDSCAPE_FEATURES,
        labels=LANDSCAPE_LABELS,
        output_path=LANDSCAPE_TEXTLESS_OUT,
        summary_path=LANDSCAPE_SUMMARY,
        color="#ECA7C0",
        envelope_color="#ECA7C0",
        title="",
        figure_size=(11.5, 5.4),
        line_alpha=0.040,
        line_width=0.65,
        show_text=False,
        pdf_output_path=LANDSCAPE_TEXTLESS_PDF_OUT,
    )


if __name__ == "__main__":
    main()
