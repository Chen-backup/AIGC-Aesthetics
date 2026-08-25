from __future__ import annotations

from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
sys.path.append(str(SCRIPT_DIR))

from draw_fig3_01C_donut_feature_effects import (  # noqa: E402
    FACE_FEATURES_PATH,
    FEATURE_ORDER_BY_IMPORTANCE_RANK,
    FEATURES,
    INTERPRETABLE_FEATURES_PATH,
    TRACE_PATH,
    closed,
    darken_rgba,
    significance_label,
    text_rotation,
)


PNG_OUTPUT = SCRIPT_DIR / "Fig3_01D_donut_feature_means_sd_effect_values.png"
CSV_OUTPUT = SCRIPT_DIR / "Fig3_01D_donut_feature_means_sd_effect_values_data.csv"


# =========================
# 主要版式调节区
# =========================
FIG_SIZE = (10.2, 9.6)
DPI = 300
AX_RECT = [0.035, 0.035, 0.86, 0.93]

# 最内层雷达图：展示 min-max 归一化后、以中点为 0 的样本均值。
RADAR_MIN_VALUE = -0.55
RADAR_MAX_VALUE = 0.55
RADAR_TICKS = [-0.50, -0.30, 0.00, 0.30, 0.50]
RADAR_MIN_DISPLAY_R = 0.042
RADAR_MAX_R = 0.200
RADAR_LINE_COLOR = "#689FED"
RADAR_FILL_COLOR = "#D1F9FB"
RADAR_FILL_ALPHA = 0.50
RADAR_ZERO_LINE_COLOR = "#6F7C88"
RADAR_OUTER_TICK_LABEL_INSET = 0.014

# 第二层：保持原第二层空间不变，用柱状环展示 min-max 归一化后的样本标准差。
SD_BAR_INNER_R = RADAR_MAX_R
SD_BAR_OUTER_R = 0.360
# 标准差扇形柱的起点和最大高度；调小 SD_BAR_HEIGHT_SCALE 可让整体柱子更短。
SD_BAR_BASE_R = SD_BAR_INNER_R + 0.010
SD_BAR_HEIGHT_SCALE = 0.76
SD_BAR_GAP_RATIO = 0.14
# 标准差柱子的颜色在这里改，例如 "#B8A1D9"、"#87A8D6"、"#D9A35F"。
SD_BAR_COLOR = "#C8E5DF"
SD_BAR_GRADIENT_INNER_COLOR = "#BEE1DB"
SD_BAR_GRADIENT_OUTER_COLOR = "#DAE7DD"
SD_BAR_ALPHA = 0.92

# 第三层：显示 effect value，也就是后验效应量均值。
EFFECT_VALUE_INNER_R = SD_BAR_OUTER_R
EFFECT_VALUE_OUTER_R = 0.45
EFFECT_VALUE_SIZE = 20
EFFECT_VALUE_RING_COLOR = "#B4D2C6"
EFFECT_VALUE_GRADIENT_INNER_COLOR = "#CAE0D4"
EFFECT_VALUE_GRADIENT_OUTER_COLOR = "#B4D2C6"

# 第四层：后验效应量样本散点。
SCATTER_INNER_R = EFFECT_VALUE_OUTER_R
SCATTER_OUTER_R = 0.675
SCATTER_POINTS_PER_FEATURE = 250
SCATTER_JITTER_RATIO = 0.40
SCATTER_RADIAL_PADDING = 0.02
SCATTER_SIZE = 10.0
SCATTER_ALPHA = 0.82

# 最外层：特征名称标签。
LABEL_INNER_R = SCATTER_OUTER_R
LABEL_OUTER_R = 0.765
LABEL_SIZE = 30.0
LABEL_RING_COLOR = "#B6D8C8"
LABEL_GRADIENT_INNER_COLOR = "#C6D8CD"
LABEL_GRADIENT_OUTER_COLOR = "#CAE9D9"

OUTER_LIMIT = 0.785
SECTOR_GAP_RATIO = 0.012
RING_LINE_COLOR = "#B9C3D0"
RADIAL_LINE_COLOR = "#D8DEE8"
BACKGROUND_COLORS = ["#F7F8F3", "#F7F8F3"]
EFFECT_COLOR_LIMIT = 1.0
COLORBAR_TICK_SIZE = 16
COLORBAR_LABEL_SIZE = 16


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_effect_cmap() -> LinearSegmentedColormap:
    # 色条仍然用于散点层的后验 effect value：蓝=负，红=正。
    return LinearSegmentedColormap.from_list(
        "brick_red_to_grey_blue",
        ["#3659BA", "#9BA2E0", "#EEA799", "#E0233C"],
        N=256,
    )


def blend_colors(inner_color: str, outer_color: str, ratio: float, alpha: float = 1.0) -> tuple[float, float, float, float]:
    inner = np.array(mpl.colors.to_rgba(inner_color))
    outer = np.array(mpl.colors.to_rgba(outer_color))
    color = inner * (1.0 - ratio) + outer * ratio
    color[3] *= alpha
    return tuple(color)


def draw_gradient_bars(
    ax: plt.Axes,
    theta: np.ndarray,
    heights: np.ndarray,
    width: float,
    bottom: float,
    inner_color: str,
    outer_color: str,
    *,
    alpha: float = 1.0,
    steps: int = 28,
    edgecolor: str = "white",
    linewidth: float = 0.55,
    zorder: int = 2,
) -> None:
    """Draw polar bars with a radial inner-to-outer color gradient."""
    heights = np.asarray(heights, dtype=float)
    for step in range(steps):
        ratio0 = step / steps
        ratio1 = (step + 1) / steps
        segment_bottom = bottom + heights * ratio0
        segment_height = heights * (ratio1 - ratio0)
        ax.bar(
            theta,
            segment_height,
            width=width,
            bottom=segment_bottom,
            color=blend_colors(inner_color, outer_color, (ratio0 + ratio1) / 2, alpha),
            edgecolor="none",
            linewidth=0,
            zorder=zorder,
        )

    ax.bar(
        theta,
        heights,
        width=width,
        bottom=bottom,
        color=(1, 1, 1, 0),
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder + 0.2,
    )


def load_posterior_values() -> dict[str, np.ndarray]:
    trace = az.from_netcdf(TRACE_PATH)
    return {
        item["code"]: np.asarray(trace.posterior[item["code"]].values, dtype=float).reshape(-1)
        for item in FEATURES
    }


def build_plot_table(values_by_feature: dict[str, np.ndarray]) -> pd.DataFrame:
    raw_features = pd.read_csv(INTERPRETABLE_FEATURES_PATH)
    rows: list[dict[str, float | str]] = []

    for item in FEATURES:
        code = item["code"]
        raw = pd.to_numeric(raw_features[code], errors="coerce").dropna().to_numpy(dtype=float)
        raw_min = float(np.min(raw))
        raw_max = float(np.max(raw))
        if np.isclose(raw_max, raw_min):
            normalized = np.zeros_like(raw)
        else:
            normalized = (raw - raw_min) / (raw_max - raw_min)
        normalized_mean = float(np.mean(normalized))

        posterior = values_by_feature[code]
        hdi_low, hdi_high = az.hdi(posterior, hdi_prob=0.95)
        stars, p_direction = significance_label(posterior)

        rows.append(
            {
                "source_feature": code,
                "feature_name": item["label"],
                "short_label": item["short_label"],
                "raw_sample_mean": float(np.mean(raw)),
                "raw_sample_sd": float(np.std(raw, ddof=1)),
                "normalized_sample_mean": normalized_mean,
                "signed_sample_mean": normalized_mean * 2.0 - 1.0,
                "normalized_sample_sd": float(np.std(normalized, ddof=1)),
                "effect_value": float(np.mean(posterior)),
                "effect_sd": float(np.std(posterior, ddof=1)),
                "hdi_2.5%": float(hdi_low),
                "hdi_97.5%": float(hdi_high),
                "p_direction": p_direction,
                "significance": stars,
            }
        )

    df = pd.DataFrame(rows)
    df["importance_rank"] = df["effect_value"].abs().rank(method="first", ascending=False).astype(int)
    order_lookup = {rank: order for order, rank in enumerate(FEATURE_ORDER_BY_IMPORTANCE_RANK)}
    if set(order_lookup) != set(df["importance_rank"]):
        raise ValueError("FEATURE_ORDER_BY_IMPORTANCE_RANK must contain every importance rank exactly once.")
    df["plot_order"] = df["importance_rank"].map(order_lookup)
    return df.sort_values("plot_order").drop(columns="plot_order").reset_index(drop=True)


def draw_background_disc(ax: plt.Axes, theta: np.ndarray, sector_width: float) -> None:
    colors = [BACKGROUND_COLORS[idx % 2] for idx in range(len(theta))]
    ax.bar(
        theta,
        np.full(len(theta), LABEL_OUTER_R),
        width=sector_width,
        bottom=0,
        color=colors,
        edgecolor="none",
        alpha=0.72,
        zorder=0,
    )


def draw_ring_lines(ax: plt.Axes, theta_full: np.ndarray) -> None:
    rings = [
        RADAR_MAX_R,
        SD_BAR_INNER_R,
        SD_BAR_OUTER_R,
        EFFECT_VALUE_INNER_R,
        EFFECT_VALUE_OUTER_R,
        SCATTER_INNER_R,
        SCATTER_OUTER_R,
        LABEL_INNER_R,
        LABEL_OUTER_R,
    ]
    for radius in sorted({round(float(radius), 6) for radius in rings}):
        ax.plot(theta_full, np.full_like(theta_full, radius), color=RING_LINE_COLOR, lw=1.05, zorder=8)


def draw_radial_lines(ax: plt.Axes, theta: np.ndarray, sector_width: float) -> None:
    for boundary in theta - sector_width / 2:
        ax.plot([boundary, boundary], [0, LABEL_OUTER_R], color=RADIAL_LINE_COLOR, lw=0.80, zorder=7)


def draw_sample_mean_radar(ax: plt.Axes, theta: np.ndarray, plot_df: pd.DataFrame) -> None:
    mean_values = np.clip(
        plot_df["signed_sample_mean"].to_numpy(dtype=float),
        RADAR_MIN_VALUE,
        RADAR_MAX_VALUE,
    )
    radar_span = RADAR_MAX_VALUE - RADAR_MIN_VALUE
    radar_draw_height = RADAR_MAX_R - RADAR_MIN_DISPLAY_R
    radii = RADAR_MIN_DISPLAY_R + (mean_values - RADAR_MIN_VALUE) / radar_span * radar_draw_height
    theta_full = np.linspace(0, 2 * np.pi, 720)

    for tick in RADAR_TICKS:
        tick_r = RADAR_MIN_DISPLAY_R + (tick - RADAR_MIN_VALUE) / radar_span * radar_draw_height
        is_zero = np.isclose(tick, 0.0)
        ax.plot(
            theta_full,
            np.full_like(theta_full, tick_r),
            color=RADAR_ZERO_LINE_COLOR if is_zero else "#DDE5EF",
            lw=1.05 if is_zero else 0.8,
            linestyle=(0, (3.2, 2.2)) if is_zero else "solid",
            zorder=2 if is_zero else 1,
        )
        label_r = tick_r - RADAR_OUTER_TICK_LABEL_INSET if np.isclose(tick, max(RADAR_TICKS)) else tick_r
        ax.text(
            np.deg2rad(84),
            label_r,
            f"{tick:.1f}" if not is_zero else "0",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="#56616C",
            zorder=9,
        )

    ax.fill(closed(theta), closed(radii), color=RADAR_FILL_COLOR, alpha=RADAR_FILL_ALPHA, zorder=3)
    ax.plot(closed(theta), closed(radii), color=RADAR_LINE_COLOR, lw=1.85, zorder=5)
    ax.scatter(theta, radii, s=14, color=RADAR_LINE_COLOR, edgecolor="white", linewidth=0.55, zorder=6)


def draw_sample_sd_bars(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df: pd.DataFrame) -> None:
    ring_height = (SD_BAR_OUTER_R - SD_BAR_BASE_R) * SD_BAR_HEIGHT_SCALE
    max_sd = max(float(plot_df["normalized_sample_sd"].max()), 1e-9)
    heights = plot_df["normalized_sample_sd"].to_numpy(dtype=float) / max_sd * ring_height
    width = sector_width * (1.0 - SD_BAR_GAP_RATIO)

    draw_gradient_bars(
        ax,
        theta,
        heights,
        width=width,
        bottom=SD_BAR_BASE_R,
        inner_color=SD_BAR_GRADIENT_INNER_COLOR,
        outer_color=SD_BAR_GRADIENT_OUTER_COLOR,
        alpha=SD_BAR_ALPHA,
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )


def draw_effect_value_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df: pd.DataFrame) -> None:
    width = sector_width * (1.0 - SECTOR_GAP_RATIO)
    draw_gradient_bars(
        ax,
        theta,
        np.full(len(theta), EFFECT_VALUE_OUTER_R - EFFECT_VALUE_INNER_R),
        width=width,
        bottom=EFFECT_VALUE_INNER_R,
        inner_color=EFFECT_VALUE_GRADIENT_INNER_COLOR,
        outer_color=EFFECT_VALUE_GRADIENT_OUTER_COLOR,
        alpha=1.0,
        edgecolor="#D5DCE6",
        linewidth=0.55,
        zorder=2,
    )

    text_r = (EFFECT_VALUE_INNER_R + EFFECT_VALUE_OUTER_R) / 2
    for angle, value, stars in zip(
        theta,
        plot_df["effect_value"].to_numpy(dtype=float),
        plot_df["significance"].fillna("").astype(str),
    ):
        ax.text(
            angle,
            text_r,
            f"{value:+.2f}{stars}",
            ha="center",
            va="center",
            fontsize=EFFECT_VALUE_SIZE,
            fontweight="bold",
            color="#2F3437",
            zorder=9,
        )


def draw_distribution_scatter(
    ax: plt.Axes,
    theta: np.ndarray,
    sector_width: float,
    plot_df: pd.DataFrame,
    values_by_feature: dict[str, np.ndarray],
    norm: TwoSlopeNorm,
    cmap: mpl.colors.Colormap,
    max_abs_scatter: float,
) -> None:
    rng = np.random.default_rng(20260525)
    scatter_height = SCATTER_OUTER_R - SCATTER_INNER_R
    usable_height = max(scatter_height - 2 * SCATTER_RADIAL_PADDING, scatter_height * 0.60)
    zero_r = SCATTER_INNER_R + SCATTER_RADIAL_PADDING + usable_height / 2
    theta_full = np.linspace(0, 2 * np.pi, 900)

    ax.plot(
        theta_full,
        np.full_like(theta_full, zero_r),
        color="#7D8792",
        lw=1.05,
        linestyle=(0, (3.2, 2.2)),
        zorder=3,
    )

    for angle, row in zip(theta, plot_df.itertuples(index=False)):
        values = values_by_feature[row.source_feature]
        if len(values) > SCATTER_POINTS_PER_FEATURE:
            values = rng.choice(values, size=SCATTER_POINTS_PER_FEATURE, replace=False)

        clipped = np.clip(values, -max_abs_scatter, max_abs_scatter)
        radii = (
            SCATTER_INNER_R
            + SCATTER_RADIAL_PADDING
            + (clipped + max_abs_scatter) / (2 * max_abs_scatter) * usable_height
        )
        jitter = rng.uniform(
            -sector_width * SCATTER_JITTER_RATIO,
            sector_width * SCATTER_JITTER_RATIO,
            size=len(values),
        )

        ax.scatter(
            angle + jitter,
            radii,
            s=SCATTER_SIZE,
            color=darken_rgba(cmap(norm(values))),
            edgecolors="none",
            zorder=4,
        )


def draw_label_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, plot_df: pd.DataFrame) -> None:
    width = sector_width * (1.0 - SECTOR_GAP_RATIO)
    draw_gradient_bars(
        ax,
        theta,
        np.full(len(theta), LABEL_OUTER_R - LABEL_INNER_R),
        width=width,
        bottom=LABEL_INNER_R,
        inner_color=LABEL_GRADIENT_INNER_COLOR,
        outer_color=LABEL_GRADIENT_OUTER_COLOR,
        alpha=1.0,
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )

    label_r = (LABEL_INNER_R + LABEL_OUTER_R) / 2
    for angle, label in zip(theta, plot_df["short_label"]):
        rotation, ha = text_rotation(float(angle))
        ax.text(
            angle,
            label_r,
            str(label),
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            color="#24292E",
            zorder=10,
        )


def draw_chart(plot_df: pd.DataFrame, values_by_feature: dict[str, np.ndarray]) -> None:
    set_style()
    n_features = len(plot_df)
    sector_width = 2 * np.pi / n_features
    theta = np.linspace(0, 2 * np.pi, n_features, endpoint=False) + sector_width / 2
    theta_full = np.linspace(0, 2 * np.pi, 1200)

    all_values = np.concatenate([values_by_feature[row.source_feature] for row in plot_df.itertuples(index=False)])
    max_abs_scatter = float(np.quantile(np.abs(all_values), 0.995))
    norm = TwoSlopeNorm(vmin=-EFFECT_COLOR_LIMIT, vcenter=0.0, vmax=EFFECT_COLOR_LIMIT)
    cmap = make_effect_cmap()

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes(AX_RECT, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, OUTER_LIMIT)
    ax.set_axis_off()

    draw_background_disc(ax, theta, sector_width)
    draw_sample_mean_radar(ax, theta, plot_df)
    draw_sample_sd_bars(ax, theta, sector_width, plot_df)
    draw_effect_value_ring(ax, theta, sector_width, plot_df)
    draw_distribution_scatter(ax, theta, sector_width, plot_df, values_by_feature, norm, cmap, max_abs_scatter)
    draw_label_ring(ax, theta, sector_width, plot_df)
    draw_radial_lines(ax, theta, sector_width)
    draw_ring_lines(ax, theta_full)

    cax = fig.add_axes([0.905, 0.31, 0.024, 0.36])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("Posterior effect value", fontsize=COLORBAR_LABEL_SIZE, fontweight="bold", labelpad=8)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, width=1.0, length=4)
    colorbar.outline.set_linewidth(0.9)

    fig.savefig(PNG_OUTPUT, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    values_by_feature = load_posterior_values()
    plot_df = build_plot_table(values_by_feature)
    plot_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_chart(plot_df, values_by_feature)

    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
