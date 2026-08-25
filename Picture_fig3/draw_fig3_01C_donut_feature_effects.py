from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

INTERPRETABLE_FEATURES_PATH = ROOT_DIR / "interpretable_face_features.csv"
FACE_FEATURES_PATH = ROOT_DIR / "face_features.csv"
TRACE_PATH = ROOT_DIR / "BYS_interpretable_model_result" / "full_model_trace.nc"

PNG_OUTPUT = OUTPUT_DIR / "Fig3_01C_donut_feature_effects.png"
CSV_OUTPUT = OUTPUT_DIR / "Fig3_01C_donut_feature_effects_data.csv"


# =========================
# Main visual controls
# =========================
FIG_SIZE = (10.2, 9.6)
DPI = 300
AX_RECT = [0.035, 0.035, 0.86, 0.93]

RADAR_MAX_VALUE = 0.70
RADAR_TICKS = [0.20, 0.40, 0.60]
RADAR_MAX_R = 0.200

EFFECT_INNER_R = RADAR_MAX_R
EFFECT_OUTER_R = 0.425

RANK_INNER_R = EFFECT_OUTER_R
RANK_OUTER_R = 0.500

SCATTER_INNER_R = RANK_OUTER_R
SCATTER_OUTER_R = 0.700
SCATTER_POINTS_PER_FEATURE = 150
SCATTER_JITTER_RATIO = 0.40
SCATTER_RADIAL_PADDING = 0.042
SCATTER_SIZE = 9.0
SCATTER_ALPHA = 0.82

LABEL_INNER_R = SCATTER_OUTER_R
LABEL_OUTER_R = 0.750
LABEL_SIZE = 10.4

OUTER_LIMIT = 0.895
SECTOR_GAP_RATIO = 0.012
RING_LINE_COLOR = "#B9C3D0"
RADIAL_LINE_COLOR = "#D8DEE8"
LABEL_RING_COLOR = "#9BA1A1"
BACKGROUND_COLORS = ["#C0C7C6", "#E1E5EB"]
EFFECT_COLOR_LIMIT = 1.0
COLORBAR_TICK_SIZE = 10.5
COLORBAR_LABEL_SIZE = 10.5

# 从最上方开始，按顺时针方向排列的“重要性排名”顺序；想换顺序只改这里。
FEATURE_ORDER_BY_IMPORTANCE_RANK = [5, 1, 11, 10, 6, 13, 7, 14, 2, 9, 12, 3, 8, 4]


FEATURES = [
    {"code": "face_hw_ratio", "label": "Facial height-to-width ratio", "short_label": "Height-width"},
    {"code": "eye_face_w_ratio", "label": "Eye-to-face width ratio", "short_label": "Eye-face"},
    {"code": "mouth_face_w_ratio", "label": "Mouth-face width ratio", "short_label": "Mouth-face"},
    {"code": "three_courts_balance", "label": "Three-courts facial balance", "short_label": "Three-courts"},
    {"code": "upper_lower_ratio", "label": "Upper-lower face ratio", "short_label": "Upper-lower"},
    {"code": "eye_y_ratio", "label": "Vertical eye position", "short_label": "Eye vertical"},
    {"code": "total_symmetry", "label": "Overall facial symmetry", "short_label": "Total symmetry"},
    {"code": "le_nose_re_angle", "label": "Left eye-nose-right eye angle", "short_label": "Eye-nose angle"},
    {"code": "mouth_nose_ratio", "label": "Mouth-nose distance ratio", "short_label": "Mouth-nose"},
    {"code": "face_brightness", "label": "Facial brightness", "short_label": "Brightness"},
    {"code": "face_contrast", "label": "Facial contrast", "short_label": "Contrast"},
    {"code": "face_clarity", "label": "Facial clarity", "short_label": "Clarity"},
    {"code": "saturation", "label": "Color saturation", "short_label": "Saturation"},
    {"code": "edge_density", "label": "Edge density", "short_label": "Edge density"},
]


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


def validate_input_tables() -> dict[str, int]:
    interpretable = pd.read_csv(INTERPRETABLE_FEATURES_PATH)
    face_features = pd.read_csv(FACE_FEATURES_PATH)
    feature_codes = [item["code"] for item in FEATURES]

    missing = [feature for feature in feature_codes if feature not in interpretable.columns]
    if missing:
        raise ValueError(f"Missing interpretable feature columns: {missing}")

    if "image_name" not in interpretable.columns or "image_name" not in face_features.columns:
        raise ValueError("Both feature CSV files must contain an image_name column.")

    overlap = set(interpretable["image_name"]).intersection(set(face_features["image_name"]))
    if not overlap:
        raise ValueError("No overlapping image_name records between the two feature CSV files.")

    return {
        "interpretable_images": int(interpretable["image_name"].nunique()),
        "face_feature_images": int(face_features["image_name"].nunique()),
        "overlap_images": int(len(overlap)),
    }


def load_posterior_values() -> dict[str, np.ndarray]:
    trace = az.from_netcdf(TRACE_PATH)
    values_by_feature: dict[str, np.ndarray] = {}

    for item in FEATURES:
        code = item["code"]
        if code not in trace.posterior:
            raise ValueError(f"Feature {code!r} was not found in {TRACE_PATH}.")
        values_by_feature[code] = np.asarray(trace.posterior[code].values, dtype=float).reshape(-1)

    return values_by_feature


def significance_label(values: np.ndarray) -> tuple[str, float]:
    prob_positive = float(np.mean(values > 0))
    p_direction = max(prob_positive, 1.0 - prob_positive)
    if p_direction >= 0.995:
        return "***", p_direction
    if p_direction >= 0.975:
        return "**", p_direction
    if p_direction >= 0.95:
        return "*", p_direction
    return "", p_direction


def build_effect_table(values_by_feature: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for item in FEATURES:
        code = item["code"]
        values = values_by_feature[code]
        hdi_low, hdi_high = az.hdi(values, hdi_prob=0.95)
        stars, p_direction = significance_label(values)

        rows.append(
            {
                "source_feature": code,
                "feature_name": item["label"],
                "short_label": item["short_label"],
                "effect_size": float(np.mean(values)),
                "effect_sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "hdi_2.5%": float(hdi_low),
                "hdi_97.5%": float(hdi_high),
                "p_direction": p_direction,
                "significance": stars,
            }
        )

    df = pd.DataFrame(rows).sort_values("effect_size", ascending=False).reset_index(drop=True)
    df["importance_rank"] = df["effect_size"].abs().rank(method="first", ascending=False).astype(int)
    order_lookup = {rank: order for order, rank in enumerate(FEATURE_ORDER_BY_IMPORTANCE_RANK)}
    if set(order_lookup) != set(df["importance_rank"]):
        raise ValueError("FEATURE_ORDER_BY_IMPORTANCE_RANK must contain every importance rank exactly once.")
    df["plot_order"] = df["importance_rank"].map(order_lookup)
    df = df.sort_values("plot_order").drop(columns="plot_order").reset_index(drop=True)
    return df


def closed(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[0]]


def darken_rgba(colors: np.ndarray, factor: float = 0.72) -> np.ndarray:
    dark = np.array(colors, copy=True)
    dark[:, :3] = np.clip(dark[:, :3] * factor, 0, 1)
    dark[:, 3] = SCATTER_ALPHA
    return dark


def make_effect_cmap() -> LinearSegmentedColormap:
    """Darker RdBu-like palette for effect size layers and colorbar."""
    return LinearSegmentedColormap.from_list(
        "brick_red_to_grey_blue",
        ["#3A5F8B", "#D4E0F0", "#F7D4C9", "#C43A31"],
        N=256,
    )


def text_rotation(theta: float) -> tuple[float, str]:
    # Tangential orientation: text is parallel to the circular ring.
    angle = ((-np.degrees(theta) + 180.0) % 360.0) - 180.0
    if angle < -90:
        return angle + 180.0, "center"
    if angle > 90:
        return angle - 180.0, "center"
    return angle, "center"


def draw_ring_lines(ax: plt.Axes, theta_full: np.ndarray) -> None:
    rings = [
        RADAR_MAX_R,
        EFFECT_INNER_R,
        EFFECT_OUTER_R,
        RANK_INNER_R,
        RANK_OUTER_R,
        SCATTER_INNER_R,
        SCATTER_OUTER_R,
        LABEL_INNER_R,
        LABEL_OUTER_R,
    ]
    unique_rings = sorted({round(float(radius), 6) for radius in rings})
    for radius in unique_rings:
        ax.plot(theta_full, np.full_like(theta_full, radius), color=RING_LINE_COLOR, lw=1.05, zorder=8)


def draw_background_disc(ax: plt.Axes, theta: np.ndarray, sector_width: float) -> None:
    colors = [BACKGROUND_COLORS[idx % 2] for idx in range(len(theta))]
    ax.bar(
        theta,
        np.full(len(theta), LABEL_OUTER_R),
        width=sector_width,
        bottom=0,
        color=colors,
        edgecolor="none",
        linewidth=0,
        alpha=0.72,
        zorder=0,
    )


def draw_radial_lines(ax: plt.Axes, theta: np.ndarray, sector_width: float) -> None:
    for boundary in theta - sector_width / 2:
        ax.plot(
            [boundary, boundary],
            [0, LABEL_OUTER_R],
            color=RADIAL_LINE_COLOR,
            lw=0.80,
            zorder=7,
        )


def draw_sd_radar(ax: plt.Axes, theta: np.ndarray, effect_df: pd.DataFrame) -> None:
    std_values = np.clip(effect_df["effect_sd"].to_numpy(dtype=float), 0, RADAR_MAX_VALUE)
    radii = std_values / RADAR_MAX_VALUE * RADAR_MAX_R
    theta_full = np.linspace(0, 2 * np.pi, 720)
    for tick in RADAR_TICKS:
        tick_r = tick / RADAR_MAX_VALUE * RADAR_MAX_R
        ax.plot(theta_full, np.full_like(theta_full, tick_r), color="#DDE5EF", lw=0.8, zorder=1)
        ax.text(
            np.deg2rad(84),
            tick_r,
            f"{tick:.2f}",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="#56616C",
            zorder=9,
        )
    ax.fill(closed(theta), closed(radii), color="#87A8D6", alpha=0.50, zorder=3)
    ax.plot(closed(theta), closed(radii), color="#3A5F8B", lw=1.85, zorder=5)
    ax.scatter(theta, radii, s=14, color="#3A5F8B", edgecolor="white", linewidth=0.55, zorder=6)


def draw_effect_ring(
    ax: plt.Axes,
    theta: np.ndarray,
    sector_width: float,
    effect_df: pd.DataFrame,
    norm: TwoSlopeNorm,
    cmap: mpl.colors.Colormap,
) -> None:
    width = sector_width * (1.0 - SECTOR_GAP_RATIO)
    colors = [cmap(norm(value)) for value in effect_df["effect_size"].to_numpy(dtype=float)]
    ax.bar(
        theta,
        np.full(len(theta), EFFECT_OUTER_R - EFFECT_INNER_R),
        width=width,
        bottom=EFFECT_INNER_R,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )


def draw_rank_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, effect_df: pd.DataFrame) -> None:
    width = sector_width * (1.0 - SECTOR_GAP_RATIO)
    colors = [BACKGROUND_COLORS[idx % 2] for idx in range(len(theta))]
    ax.bar(
        theta,
        np.full(len(theta), RANK_OUTER_R - RANK_INNER_R),
        width=width,
        bottom=RANK_INNER_R,
        color=colors,
        edgecolor="#D5DCE6",
        linewidth=0.55,
        zorder=2,
    )
    rank_r = (RANK_INNER_R + RANK_OUTER_R) / 2
    for angle, rank in zip(theta, effect_df["importance_rank"].to_numpy(dtype=int)):
        ax.text(
            angle,
            rank_r,
            str(rank),
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#2F3437",
            zorder=9,
        )


def draw_distribution_scatter(
    ax: plt.Axes,
    theta: np.ndarray,
    sector_width: float,
    effect_df: pd.DataFrame,
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

    for angle, row in zip(theta, effect_df.itertuples(index=False)):
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
        colors = darken_rgba(cmap(norm(values)))

        ax.scatter(
            angle + jitter,
            radii,
            s=SCATTER_SIZE,
            color=colors,
            edgecolors="none",
            zorder=4,
        )


def draw_label_ring(ax: plt.Axes, theta: np.ndarray, sector_width: float, effect_df: pd.DataFrame) -> None:
    width = sector_width * (1.0 - SECTOR_GAP_RATIO)
    ax.bar(
        theta,
        np.full(len(theta), LABEL_OUTER_R - LABEL_INNER_R),
        width=width,
        bottom=LABEL_INNER_R,
        color=LABEL_RING_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )
    label_r = (LABEL_INNER_R + LABEL_OUTER_R) / 2
    for angle, label in zip(theta, effect_df["short_label"]):
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
            color="#242E2D",
            zorder=10,
        )


def draw_multilayer_chart(effect_df: pd.DataFrame, values_by_feature: dict[str, np.ndarray]) -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_features = len(effect_df)
    sector_width = 2 * np.pi / n_features
    theta = np.linspace(0, 2 * np.pi, n_features, endpoint=False) + sector_width / 2
    theta_full = np.linspace(0, 2 * np.pi, 1200)

    all_values = np.concatenate([values_by_feature[row.source_feature] for row in effect_df.itertuples(index=False)])
    max_abs_scatter = float(np.quantile(np.abs(all_values), 0.995))
    max_abs = max(EFFECT_COLOR_LIMIT, 1e-6)

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = make_effect_cmap()

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes(AX_RECT, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, OUTER_LIMIT)
    ax.set_axis_off()

    draw_background_disc(ax, theta, sector_width)
    draw_sd_radar(ax, theta, effect_df)
    draw_effect_ring(ax, theta, sector_width, effect_df, norm, cmap)
    draw_rank_ring(ax, theta, sector_width, effect_df)
    draw_distribution_scatter(ax, theta, sector_width, effect_df, values_by_feature, norm, cmap, max_abs_scatter)
    draw_label_ring(ax, theta, sector_width, effect_df)
    draw_radial_lines(ax, theta, sector_width)
    draw_ring_lines(ax, theta_full)

    cax = fig.add_axes([0.905, 0.31, 0.024, 0.36])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("Standardized effect size", fontsize=COLORBAR_LABEL_SIZE, fontweight="bold", labelpad=8)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, width=1.0, length=4)
    colorbar.outline.set_linewidth(0.9)

    fig.savefig(PNG_OUTPUT, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    diagnostics = validate_input_tables()
    values_by_feature = load_posterior_values()
    effect_df = build_effect_table(values_by_feature)
    effect_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_multilayer_chart(effect_df, values_by_feature)

    print(f"Input diagnostics: {diagnostics}")
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
