from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
FACE_FIGURE_DIR = ROOT_DIR / "Picture_fig3"
sys.path.insert(0, str(FACE_FIGURE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import draw_fig3_01D_donut_feature_means_sd_effect_values as base  # noqa: E402
import draw_combined_face_landscape_fig3_01F_donut_no_radar_270deg as source  # noqa: E402


MAIN_PNG_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_360deg_main.png"
MAIN_PDF_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_360deg_main.pdf"
MAIN_SVG_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_360deg_main.svg"
COLORBAR_PNG_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_360deg_colorbar_textless.png"
COLORBAR_PDF_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_360deg_colorbar_textless.pdf"
CSV_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_360deg_data.csv"
MAIN_FIG_SIZE = (12.8, 12.8)
MAIN_AX_RECT = [0.015, 0.015, 0.97, 0.97]
COLORBAR_FIG_SIZE = (0.9, 4.6)

# Outer feature-name ring font size. Adjust this value without changing the
# effect-value ring text size.
FEATURE_LABEL_SIZE = 22.0
EFFECT_VALUE_LABEL_SIZE = 22.0

# Effect-value ring gradients. Set the inner and outer colors independently
# for face and landscape sectors when tuning the palette.
FACE_EFFECT_VALUE_INNER_COLOR = "#B5C2D7"
FACE_EFFECT_VALUE_OUTER_COLOR = "#879FC7"
LANDSCAPE_EFFECT_VALUE_INNER_COLOR = "#EED9E1"
LANDSCAPE_EFFECT_VALUE_OUTER_COLOR = "#E7A5BD"

LABEL_LINE_BREAKS = {
    "Eye-nose angle": "Eye-nose\nangle",
    "Upper-lower": "Upper-\nlower",
    "Three-courts": "Three-\ncourts",
    "Mouth-nose": "Mouth-\nnose",
    "Mouth-face": "Mouth-\nface",
    "Total symmetry": "Total\nsymmetry",
    "Edge density": "Edge\ndensity",
    "Eye vertical": "Eye\nvertical",
    "Eye-face": "Eye-\nface",
    "Height-width": "Height-\nwidth",
    "Depth variation": "Depth\nvariation",
    "Left-right balance": "Left-right\nbalance",
    "Thirds brightness": "Thirds\nbrightness",
    "Semantic diversity": "Semantic\ndiversity",
    "Depth gradient": "Depth\ngradient",
    "Line strength": "Line\nstrength",
}


def draw_full_ring_lines(ax: plt.Axes, theta_full: np.ndarray) -> None:
    rings = [
        source.BAR_OUTER_R,
        base.EFFECT_VALUE_INNER_R,
        base.EFFECT_VALUE_OUTER_R,
        base.SCATTER_INNER_R,
        base.SCATTER_OUTER_R,
        base.LABEL_INNER_R,
        base.LABEL_OUTER_R,
    ]
    for radius in sorted({round(float(radius), 6) for radius in rings}):
        ax.plot(theta_full, np.full_like(theta_full, radius), color=base.RING_LINE_COLOR, lw=1.05, zorder=8)


def draw_radial_effect_value_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df) -> None:
    width = sector_width * (1.0 - base.SECTOR_GAP_RATIO)
    ring_height = base.EFFECT_VALUE_OUTER_R - base.EFFECT_VALUE_INNER_R
    for angle, group in zip(theta, plot_df["feature_group"]):
        if group == "Face":
            inner_color = FACE_EFFECT_VALUE_INNER_COLOR
            outer_color = FACE_EFFECT_VALUE_OUTER_COLOR
        else:
            inner_color = LANDSCAPE_EFFECT_VALUE_INNER_COLOR
            outer_color = LANDSCAPE_EFFECT_VALUE_OUTER_COLOR
        base.draw_gradient_bars(
            ax,
            np.array([angle]),
            np.array([ring_height]),
            width=width,
            bottom=base.EFFECT_VALUE_INNER_R,
            inner_color=inner_color,
            outer_color=outer_color,
            alpha=1.0,
            edgecolor="#D5DCE6",
            linewidth=0.55,
            zorder=2,
        )

    text_r = (base.EFFECT_VALUE_INNER_R + base.EFFECT_VALUE_OUTER_R) / 2
    for angle, value, stars in zip(
        theta,
        plot_df["effect_value"].to_numpy(dtype=float),
        plot_df["significance"].fillna("").astype(str),
    ):
        rotation, ha = base.text_rotation(float(angle))
        label = f"{value:.2f}"
        if stars:
            # Keep significance marks on the center-facing side after text
            # rotation flips labels on the left half of the circle.
            angle_normalized = float(angle) % (2 * np.pi)
            if np.pi / 2 < angle_normalized < 3 * np.pi / 2:
                label = f"{stars}\n{label}"
            else:
                label += f"\n{stars}"
        ax.text(
            angle,
            text_r,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=EFFECT_VALUE_LABEL_SIZE,
            fontweight="normal",
            color="#2F3437",
            zorder=9,
        )


def draw_group_label_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df) -> None:
    width = sector_width * (1.0 - base.SECTOR_GAP_RATIO)
    colors = [
        source.FACE_LABEL_COLOR if group == "Face" else source.LANDSCAPE_LABEL_COLOR
        for group in plot_df["feature_group"]
    ]
    ax.bar(
        theta,
        np.full(len(theta), base.LABEL_OUTER_R - base.LABEL_INNER_R),
        width=width,
        bottom=base.LABEL_INNER_R,
        color=colors,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.88,
        zorder=2,
    )
    label_r = (base.LABEL_INNER_R + base.LABEL_OUTER_R) / 2
    for angle, label in zip(theta, plot_df["short_label"]):
        rotation, ha = base.text_rotation(float(angle))
        ax.text(
            angle,
            label_r,
            str(label),
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=FEATURE_LABEL_SIZE,
            fontweight="normal",
            color="#24292E",
            zorder=10,
        )


def prepare_plot_data(plot_df):
    plot_df = plot_df.copy()
    plot_df["short_label"] = plot_df["short_label"].replace(LABEL_LINE_BREAKS)
    n_features = len(plot_df)
    sector_width = 2 * np.pi / n_features
    theta = np.linspace(0, 2 * np.pi, n_features, endpoint=False) + sector_width / 2
    theta_full = np.linspace(0, 2 * np.pi, 1200)
    return plot_df, sector_width, theta, theta_full


def draw_main_chart(plot_df, values_by_feature: dict[str, np.ndarray]) -> None:
    base.set_style()
    mpl.rcParams["font.weight"] = "normal"
    mpl.rcParams["axes.labelweight"] = "normal"
    source.COMBINED_LABEL_SIZE = FEATURE_LABEL_SIZE
    base.LABEL_SIZE = FEATURE_LABEL_SIZE
    base.EFFECT_VALUE_SIZE = EFFECT_VALUE_LABEL_SIZE

    plot_df, sector_width, theta, theta_full = prepare_plot_data(plot_df)
    all_values = np.concatenate([values_by_feature[row.source_feature] for row in plot_df.itertuples(index=False)])
    max_abs_scatter = float(np.quantile(np.abs(all_values), 0.995))
    norm = TwoSlopeNorm(vmin=-base.EFFECT_COLOR_LIMIT, vcenter=0.0, vmax=base.EFFECT_COLOR_LIMIT)
    cmap = base.make_effect_cmap()

    fig = plt.figure(figsize=MAIN_FIG_SIZE, dpi=base.DPI, facecolor="white")
    ax = fig.add_axes(MAIN_AX_RECT, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, base.OUTER_LIMIT)
    ax.set_axis_off()

    base.draw_background_disc(ax, theta, sector_width)
    source.draw_source_gradient_bars(ax, theta, sector_width, plot_df)
    draw_radial_effect_value_ring(ax, theta, sector_width, plot_df)
    base.draw_distribution_scatter(
        ax,
        theta,
        sector_width,
        plot_df,
        values_by_feature,
        norm,
        cmap,
        max_abs_scatter,
    )
    draw_group_label_ring(ax, theta, sector_width, plot_df)
    base.draw_radial_lines(ax, theta, sector_width)
    draw_full_ring_lines(ax, theta_full)

    fig.savefig(MAIN_PNG_OUTPUT, dpi=base.DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(MAIN_PDF_OUTPUT, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(MAIN_SVG_OUTPUT, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def draw_textless_colorbar() -> None:
    base.set_style()
    norm = TwoSlopeNorm(vmin=-base.EFFECT_COLOR_LIMIT, vcenter=0.0, vmax=base.EFFECT_COLOR_LIMIT)
    cmap = base.make_effect_cmap()

    fig = plt.figure(figsize=COLORBAR_FIG_SIZE, dpi=base.DPI, facecolor="white")
    cax = fig.add_axes([0.34, 0.04, 0.32, 0.92])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("")
    colorbar.ax.set_yticklabels([])
    colorbar.ax.tick_params(labelsize=0, width=2.2, length=9)
    colorbar.outline.set_linewidth(2.2)

    fig.savefig(COLORBAR_PNG_OUTPUT, dpi=base.DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(COLORBAR_PDF_OUTPUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main() -> None:
    plot_df, values_by_feature = source.load_inputs()
    plot_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_main_chart(plot_df, values_by_feature)
    draw_textless_colorbar()
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved main PNG: {MAIN_PNG_OUTPUT}")
    print(f"Saved main PDF: {MAIN_PDF_OUTPUT}")
    print(f"Saved main SVG: {MAIN_SVG_OUTPUT}")
    print(f"Saved colorbar PNG: {COLORBAR_PNG_OUTPUT}")
    print(f"Saved colorbar PDF: {COLORBAR_PDF_OUTPUT}")


if __name__ == "__main__":
    main()
