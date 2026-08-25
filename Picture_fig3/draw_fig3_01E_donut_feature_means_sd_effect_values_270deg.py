from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import draw_fig3_01D_donut_feature_means_sd_effect_values as base


PNG_OUTPUT = base.SCRIPT_DIR / "Fig3_01E_donut_feature_means_sd_effect_values_270deg.png"
CSV_OUTPUT = base.SCRIPT_DIR / "Fig3_01E_donut_feature_means_sd_effect_values_270deg_data.csv"


# =========================
# 270度版本主要调节区
# =========================
# 留出第一象限空白：从正上方开始，按视觉逆时针方向画 270°。
ARC_START = 0.0
ARC_SPAN = 3 * np.pi / 2
ARC_END = ARC_START - ARC_SPAN

# 色条位置：这里已经放到左侧。四个数分别是 [左, 下, 宽, 高]。
COLORBAR_RECT = [0.055, 0.31, 0.024, 0.36]

# 主图位置。左侧要给色条留空间，所以整体稍微右移。
AX_RECT = [0.13, 0.035, 0.82, 0.91]

# 270度后每个扇区变窄，字号和点大小略微自适应缩小，避免拥挤。
LABEL_SIZE = 9.8
EFFECT_VALUE_SIZE = 7.0
SCATTER_SIZE = 7.8
COLORBAR_TICK_SIZE = 10.5
COLORBAR_LABEL_SIZE = 10.5


def configure_base_style() -> None:
    """复用01D的绘图函数，但覆盖本版本需要的版式参数。"""
    base.AX_RECT = AX_RECT
    base.LABEL_SIZE = LABEL_SIZE
    base.EFFECT_VALUE_SIZE = EFFECT_VALUE_SIZE
    base.SCATTER_SIZE = SCATTER_SIZE
    base.COLORBAR_TICK_SIZE = COLORBAR_TICK_SIZE
    base.COLORBAR_LABEL_SIZE = COLORBAR_LABEL_SIZE


def draw_ring_lines_270(ax: plt.Axes, theta_arc: np.ndarray) -> None:
    rings = [
        base.RADAR_MAX_R,
        base.SD_BAR_INNER_R,
        base.SD_BAR_OUTER_R,
        base.EFFECT_VALUE_INNER_R,
        base.EFFECT_VALUE_OUTER_R,
        base.SCATTER_INNER_R,
        base.SCATTER_OUTER_R,
        base.LABEL_INNER_R,
        base.LABEL_OUTER_R,
    ]
    for radius in sorted({round(float(radius), 6) for radius in rings}):
        ax.plot(theta_arc, np.full_like(theta_arc, radius), color=base.RING_LINE_COLOR, lw=1.05, zorder=8)


def draw_radial_lines_270(ax: plt.Axes, theta: np.ndarray, sector_width: float) -> None:
    boundaries = list(theta + sector_width / 2)
    boundaries.append(theta[-1] - sector_width / 2)
    for boundary in boundaries:
        ax.plot([boundary, boundary], [0, base.LABEL_OUTER_R], color=base.RADIAL_LINE_COLOR, lw=0.80, zorder=7)


def draw_sample_mean_radar_270(ax: plt.Axes, theta: np.ndarray, theta_arc: np.ndarray, plot_df) -> None:
    mean_values = np.clip(
        plot_df["signed_sample_mean"].to_numpy(dtype=float),
        base.RADAR_MIN_VALUE,
        base.RADAR_MAX_VALUE,
    )
    radar_span = base.RADAR_MAX_VALUE - base.RADAR_MIN_VALUE
    radar_draw_height = base.RADAR_MAX_R - base.RADAR_MIN_DISPLAY_R
    radii = base.RADAR_MIN_DISPLAY_R + (mean_values - base.RADAR_MIN_VALUE) / radar_span * radar_draw_height
    zero_r = base.RADAR_MIN_DISPLAY_R + (0.0 - base.RADAR_MIN_VALUE) / radar_span * radar_draw_height

    for tick in base.RADAR_TICKS:
        tick_r = base.RADAR_MIN_DISPLAY_R + (tick - base.RADAR_MIN_VALUE) / radar_span * radar_draw_height
        is_zero = np.isclose(tick, 0.0)
        ax.plot(
            theta_arc,
            np.full_like(theta_arc, tick_r),
            color=base.RADAR_ZERO_LINE_COLOR if is_zero else "#DDE5EF",
            lw=1.05 if is_zero else 0.8,
            linestyle=(0, (3.2, 2.2)) if is_zero else "solid",
            zorder=2 if is_zero else 1,
        )
        label_r = tick_r - base.RADAR_OUTER_TICK_LABEL_INSET if np.isclose(tick, max(base.RADAR_TICKS)) else tick_r
        ax.text(
            ARC_START - ARC_SPAN * 0.42,
            label_r,
            f"{tick:.1f}" if not is_zero else "0",
            ha="left",
            va="center",
            fontsize=6.3,
            fontweight="bold",
            color="#56616C",
            zorder=9,
        )

    # 270度缺口版本不连接首尾折线；填色从断口两端连到原点闭合。
    fill_theta = np.r_[theta[0], theta, theta[-1]]
    fill_radii = np.r_[0.0, radii, 0.0]
    ax.fill(fill_theta, fill_radii, color=base.RADAR_FILL_COLOR, alpha=base.RADAR_FILL_ALPHA, zorder=3)
    ax.plot(theta, radii, color=base.RADAR_LINE_COLOR, lw=1.85, zorder=5)
    ax.scatter(theta, radii, s=14, color=base.RADAR_LINE_COLOR, edgecolor="white", linewidth=0.55, zorder=6)


def draw_distribution_scatter_270(
    ax: plt.Axes,
    theta: np.ndarray,
    sector_width: float,
    theta_arc: np.ndarray,
    plot_df,
    values_by_feature: dict[str, np.ndarray],
    norm: TwoSlopeNorm,
    cmap: mpl.colors.Colormap,
    max_abs_scatter: float,
) -> None:
    rng = np.random.default_rng(20260525)
    scatter_height = base.SCATTER_OUTER_R - base.SCATTER_INNER_R
    usable_height = max(scatter_height - 2 * base.SCATTER_RADIAL_PADDING, scatter_height * 0.60)
    zero_r = base.SCATTER_INNER_R + base.SCATTER_RADIAL_PADDING + usable_height / 2

    ax.plot(
        theta_arc,
        np.full_like(theta_arc, zero_r),
        color="#7D8792",
        lw=1.05,
        linestyle=(0, (3.2, 2.2)),
        zorder=3,
    )

    for angle, row in zip(theta, plot_df.itertuples(index=False)):
        values = values_by_feature[row.source_feature]
        if len(values) > base.SCATTER_POINTS_PER_FEATURE:
            values = rng.choice(values, size=base.SCATTER_POINTS_PER_FEATURE, replace=False)

        clipped = np.clip(values, -max_abs_scatter, max_abs_scatter)
        radii = (
            base.SCATTER_INNER_R
            + base.SCATTER_RADIAL_PADDING
            + (clipped + max_abs_scatter) / (2 * max_abs_scatter) * usable_height
        )
        jitter = rng.uniform(
            -sector_width * base.SCATTER_JITTER_RATIO,
            sector_width * base.SCATTER_JITTER_RATIO,
            size=len(values),
        )

        ax.scatter(
            angle + jitter,
            radii,
            s=base.SCATTER_SIZE,
            color=base.darken_rgba(cmap(norm(values))),
            edgecolors="none",
            zorder=4,
        )


def draw_chart_270(plot_df, values_by_feature: dict[str, np.ndarray]) -> None:
    base.set_style()
    configure_base_style()

    n_features = len(plot_df)
    sector_width = ARC_SPAN / n_features
    theta = ARC_START - np.linspace(0, ARC_SPAN, n_features, endpoint=False) - sector_width / 2
    theta_arc = np.linspace(ARC_START, ARC_END, 900)

    all_values = np.concatenate([values_by_feature[row.source_feature] for row in plot_df.itertuples(index=False)])
    max_abs_scatter = float(np.quantile(np.abs(all_values), 0.995))
    norm = TwoSlopeNorm(vmin=-base.EFFECT_COLOR_LIMIT, vcenter=0.0, vmax=base.EFFECT_COLOR_LIMIT)
    cmap = base.make_effect_cmap()

    fig = plt.figure(figsize=base.FIG_SIZE, dpi=base.DPI, facecolor="white")
    ax = fig.add_axes(AX_RECT, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, base.OUTER_LIMIT)
    ax.set_axis_off()

    base.draw_background_disc(ax, theta, sector_width)
    draw_sample_mean_radar_270(ax, theta, theta_arc, plot_df)
    base.draw_sample_sd_bars(ax, theta, sector_width, plot_df)
    base.draw_effect_value_ring(ax, theta, sector_width, plot_df)
    draw_distribution_scatter_270(ax, theta, sector_width, theta_arc, plot_df, values_by_feature, norm, cmap, max_abs_scatter)
    base.draw_label_ring(ax, theta, sector_width, plot_df)
    draw_radial_lines_270(ax, theta, sector_width)
    draw_ring_lines_270(ax, theta_arc)

    cax = fig.add_axes(COLORBAR_RECT)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("Posterior effect value", fontsize=COLORBAR_LABEL_SIZE, fontweight="bold", labelpad=8)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, width=1.0, length=4)
    colorbar.outline.set_linewidth(0.9)

    fig.savefig(PNG_OUTPUT, dpi=base.DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    values_by_feature = base.load_posterior_values()
    plot_df = base.build_plot_table(values_by_feature)
    plot_df = (
        plot_df.assign(abs_effect_value=plot_df["effect_value"].abs())
        .sort_values("abs_effect_value", ascending=False)
        .drop(columns="abs_effect_value")
        .reset_index(drop=True)
    )
    plot_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_chart_270(plot_df, values_by_feature)
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
