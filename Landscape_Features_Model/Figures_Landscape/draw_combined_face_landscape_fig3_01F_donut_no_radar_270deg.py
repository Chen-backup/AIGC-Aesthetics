from __future__ import annotations

from pathlib import Path
import sys

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
LANDSCAPE_DIR = SCRIPT_DIR.parent
ROOT_DIR = LANDSCAPE_DIR.parent
FACE_FIGURE_DIR = ROOT_DIR / "Picture_fig3"
sys.path.insert(0, str(FACE_FIGURE_DIR))

import draw_fig3_01D_donut_feature_means_sd_effect_values as base  # noqa: E402
import draw_fig3_01E_donut_feature_means_sd_effect_values_270deg as fig01e  # noqa: E402


FACE_DATA = FACE_FIGURE_DIR / "Fig3_01F_donut_feature_abs_effect_bars_270deg_data.csv"
FACE_TRACE = ROOT_DIR / "BYS_interpretable_model_result" / "full_model_trace.nc"
LANDSCAPE_DATA = SCRIPT_DIR / "Fig3_01F_landscape_donut_feature_abs_effect_bars_270deg_data.csv"
LANDSCAPE_TRACE = LANDSCAPE_DIR / "BYS_interpretable_Features_Model" / "Result" / "full_model_trace.nc"

PNG_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_270deg.png"
CSV_OUTPUT = SCRIPT_DIR / "Fig3_01F_combined_face_landscape_donut_no_radar_270deg_data.csv"

FACE_LABEL_COLOR = "#7894C6"
LANDSCAPE_LABEL_COLOR = "#EB9CB9"
FACE_BAR_INNER_COLOR = "#4B68A5"
FACE_BAR_OUTER_COLOR = "#C5E1F5"
LANDSCAPE_BAR_INNER_COLOR = "#E47392"
LANDSCAPE_BAR_OUTER_COLOR = "#F9D8DF"

COMBINED_LABEL_SIZE = 7.1

# Inner absolute-effect bar layer controls.
# BAR_INNER_R: move the bar starting point toward or away from the center.
# BAR_OUTER_R: reserve more or less radial space for the bars.
# BAR_HEIGHT_SCALE: change the maximum visible bar length within that space.
# BAR_GAP_RATIO: increase this value to make each bar narrower.
# BAR_ALPHA: lower this value to make the gradient bars more transparent.
BAR_INNER_R = 0.1
BAR_OUTER_R = base.SD_BAR_OUTER_R
BAR_HEIGHT_SCALE = 0.95
BAR_GAP_RATIO = base.SD_BAR_GAP_RATIO
BAR_ALPHA = 0.95
ABS_EFFECT_DISPLAY_MAX = 1.0


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    face_df = pd.read_csv(FACE_DATA).assign(feature_group="Face")
    landscape_df = pd.read_csv(LANDSCAPE_DATA).assign(feature_group="Landscape")
    plot_df = pd.concat([face_df, landscape_df], ignore_index=True)

    face_trace = az.from_netcdf(FACE_TRACE)
    landscape_trace = az.from_netcdf(LANDSCAPE_TRACE)
    values_by_feature: dict[str, np.ndarray] = {}
    for row in plot_df.itertuples(index=False):
        trace = face_trace if row.feature_group == "Face" else landscape_trace
        values_by_feature[row.source_feature] = np.asarray(
            trace.posterior[row.source_feature].values,
            dtype=float,
        ).reshape(-1)
    return plot_df, values_by_feature


def draw_source_gradient_bars(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df: pd.DataFrame) -> None:
    ring_height = (BAR_OUTER_R - BAR_INNER_R) * BAR_HEIGHT_SCALE
    heights = np.clip(plot_df["effect_value"].abs().to_numpy(dtype=float) / ABS_EFFECT_DISPLAY_MAX, 0, 1) * ring_height
    width = sector_width * (1.0 - BAR_GAP_RATIO)

    for angle, height, group in zip(theta, heights, plot_df["feature_group"]):
        if group == "Face":
            inner_color, outer_color = FACE_BAR_INNER_COLOR, FACE_BAR_OUTER_COLOR
        else:
            inner_color, outer_color = LANDSCAPE_BAR_INNER_COLOR, LANDSCAPE_BAR_OUTER_COLOR
        base.draw_gradient_bars(
            ax,
            np.array([angle]),
            np.array([height]),
            width=width,
            bottom=BAR_INNER_R,
            inner_color=inner_color,
            outer_color=outer_color,
            alpha=BAR_ALPHA,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )


def draw_ring_lines_without_radar_separator(ax: plt.Axes, theta_arc: np.ndarray) -> None:
    # The radar layer was removed, so omit its old outer boundary at radius 0.300.
    rings = [
        BAR_OUTER_R,
        base.EFFECT_VALUE_INNER_R,
        base.EFFECT_VALUE_OUTER_R,
        base.SCATTER_INNER_R,
        base.SCATTER_OUTER_R,
        base.LABEL_INNER_R,
        base.LABEL_OUTER_R,
    ]
    for radius in sorted({round(float(radius), 6) for radius in rings}):
        ax.plot(theta_arc, np.full_like(theta_arc, radius), color=base.RING_LINE_COLOR, lw=1.05, zorder=8)


def draw_group_label_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df: pd.DataFrame) -> None:
    width = sector_width * (1.0 - base.SECTOR_GAP_RATIO)
    colors = [
        FACE_LABEL_COLOR if group == "Face" else LANDSCAPE_LABEL_COLOR
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
            fontsize=COMBINED_LABEL_SIZE,
            fontweight="bold",
            color="#24292E",
            zorder=10,
        )


def draw_chart(plot_df: pd.DataFrame, values_by_feature: dict[str, np.ndarray]) -> None:
    base.set_style()
    fig01e.configure_base_style()
    base.LABEL_SIZE = COMBINED_LABEL_SIZE

    n_features = len(plot_df)
    sector_width = fig01e.ARC_SPAN / n_features
    theta = fig01e.ARC_START - np.linspace(0, fig01e.ARC_SPAN, n_features, endpoint=False) - sector_width / 2
    theta_arc = np.linspace(fig01e.ARC_START, fig01e.ARC_END, 900)
    all_values = np.concatenate([values_by_feature[row.source_feature] for row in plot_df.itertuples(index=False)])
    max_abs_scatter = float(np.quantile(np.abs(all_values), 0.995))
    norm = TwoSlopeNorm(vmin=-base.EFFECT_COLOR_LIMIT, vcenter=0.0, vmax=base.EFFECT_COLOR_LIMIT)
    cmap = base.make_effect_cmap()

    fig = plt.figure(figsize=base.FIG_SIZE, dpi=base.DPI, facecolor="white")
    ax = fig.add_axes(fig01e.AX_RECT, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, base.OUTER_LIMIT)
    ax.set_axis_off()

    base.draw_background_disc(ax, theta, sector_width)
    draw_source_gradient_bars(ax, theta, sector_width, plot_df)
    base.draw_effect_value_ring(ax, theta, sector_width, plot_df)
    fig01e.draw_distribution_scatter_270(
        ax,
        theta,
        sector_width,
        theta_arc,
        plot_df,
        values_by_feature,
        norm,
        cmap,
        max_abs_scatter,
    )
    draw_group_label_ring(ax, theta, sector_width, plot_df)
    fig01e.draw_radial_lines_270(ax, theta, sector_width)
    draw_ring_lines_without_radar_separator(ax, theta_arc)

    cax = fig.add_axes(fig01e.COLORBAR_RECT)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label(
        "Posterior effect value",
        fontsize=fig01e.COLORBAR_LABEL_SIZE,
        fontweight="bold",
        labelpad=8,
    )
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(labelsize=fig01e.COLORBAR_TICK_SIZE, width=1.0, length=4)
    colorbar.outline.set_linewidth(0.9)

    fig.savefig(PNG_OUTPUT, dpi=base.DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    plot_df, values_by_feature = load_inputs()
    plot_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_chart(plot_df, values_by_feature)
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
