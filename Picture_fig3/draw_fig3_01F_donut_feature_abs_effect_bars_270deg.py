from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import draw_fig3_01D_donut_feature_means_sd_effect_values as base
import draw_fig3_01E_donut_feature_means_sd_effect_values_270deg as fig01e


PNG_OUTPUT = base.SCRIPT_DIR / "Fig3_01F_donut_feature_abs_effect_bars_270deg.png"
CSV_OUTPUT = base.SCRIPT_DIR / "Fig3_01F_donut_feature_abs_effect_bars_270deg_data.csv"


# =========================
# |effect_value| 柱状层调节区
# =========================
# 柱长映射范围：0 表示没有效应，1.0 约等于当前最大的效应值。
# 如果想让柱子整体更长，可把 ABS_EFFECT_DISPLAY_MAX 调小，例如 0.8。
# 如果想让柱子整体更短，可把 ABS_EFFECT_DISPLAY_MAX 调大，例如 1.2。
ABS_EFFECT_DISPLAY_MIN = 0.0
ABS_EFFECT_DISPLAY_MAX = 1.0

# 让柱子不要顶到外层；调小会整体更短，调大更接近外边界。
ABS_EFFECT_BAR_HEIGHT_SCALE = 0.90


def draw_abs_effect_bars(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df) -> None:
    ring_height = (base.SD_BAR_OUTER_R - base.SD_BAR_BASE_R) * ABS_EFFECT_BAR_HEIGHT_SCALE
    abs_effect = plot_df["effect_value"].abs().to_numpy(dtype=float)
    scaled = (abs_effect - ABS_EFFECT_DISPLAY_MIN) / (ABS_EFFECT_DISPLAY_MAX - ABS_EFFECT_DISPLAY_MIN)
    heights = np.clip(scaled, 0, 1) * ring_height
    width = sector_width * (1.0 - base.SD_BAR_GAP_RATIO)

    base.draw_gradient_bars(
        ax,
        theta,
        heights,
        width=width,
        bottom=base.SD_BAR_BASE_R,
        inner_color=base.SD_BAR_GRADIENT_INNER_COLOR,
        outer_color=base.SD_BAR_GRADIENT_OUTER_COLOR,
        alpha=base.SD_BAR_ALPHA,
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )


def draw_chart_abs_effect_bars(plot_df, values_by_feature: dict[str, np.ndarray]) -> None:
    base.set_style()
    fig01e.configure_base_style()

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
    fig01e.draw_sample_mean_radar_270(ax, theta, theta_arc, plot_df)
    draw_abs_effect_bars(ax, theta, sector_width, plot_df)
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
    base.draw_label_ring(ax, theta, sector_width, plot_df)
    fig01e.draw_radial_lines_270(ax, theta, sector_width)
    fig01e.draw_ring_lines_270(ax, theta_arc)

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
    values_by_feature = base.load_posterior_values()
    plot_df = base.build_plot_table(values_by_feature)
    plot_df = (
        plot_df.assign(abs_effect_value=plot_df["effect_value"].abs())
        .sort_values("abs_effect_value", ascending=False)
        .reset_index(drop=True)
    )
    plot_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_chart_abs_effect_bars(plot_df, values_by_feature)
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
