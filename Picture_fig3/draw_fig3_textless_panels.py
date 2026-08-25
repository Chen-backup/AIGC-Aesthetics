from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "Picture_code" / "picture"
OUTPUT_DIR = ROOT_DIR / "Picture_fig3"

EFFECT_SUMMARY = SOURCE_DIR / "Interpretable_14D_Boxline_Data.csv"
EFFECT_SAMPLES = SOURCE_DIR / "Interpretable_14D_Boxline_Samples.csv"
GAMM_DATA = SOURCE_DIR / "GAMM_NonLinear_Expected_Curves_Top6_data.csv"
MODEL_SUMMARY = SOURCE_DIR / "image_variance_explained_3models_bar_data.csv"
MODEL_SAMPLES = SOURCE_DIR / "image_variance_explained_3models_boxline_samples.csv"

STYLE = {
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "tick_size": 15,
    "axis_width": 2.2,
    "grid_width": 0.8,
    "curve_line_width": 1.35,
    "ridge_height": 0.65,
    "ridge_line_width": 0.95,
    "model_box_width": 0.50,
}

GAMM_STYLE = {
    "tick_size": 24,        # Fig3_02-Fig3_07 的刻度值字号
    "axis_width": 3.5,      # Fig3_02-Fig3_07 的坐标轴和刻度线粗细
    "tick_length": 7.0,     # Fig3_02-Fig3_07 的刻度线长度
    "curve_line_width": 2.8,  # Fig3_02-Fig3_07 的蓝色曲线粗细
}

COLORS = {
    "ridge_fill": "#F5B15B",
    "ridge_line": "#E47C20",
    "ridge_hdi": "#E47C20",
    "mean": "#222222",
    "zero": "#D94F45",
    "curve_line": "#1F6EAA",
    "curve_band": "#8DB9E2",
    "axis": "#2F3437",
    "grid": "#E6EAF0",
    "model_line": "#262626",
    "model_hdi": "#575757",
}

MODEL_ORDER = ["StyleGAN", "InsightFace", "DINOv2"]
MODEL_COLORS = {
    "StyleGAN": "#8399C5",
    "InsightFace": "#527CAE",
    "DINOv2": "#285F99",
}

PANEL_SIZE = (5.2, 4.4)
RIDGE_SIZE = (5.8, 5.9)
MODEL_SIZE = (15, 6)
RADIAL_SIZE = (6.4, 6.4)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": STYLE["font_family"],
            "font.serif": STYLE["font_serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": STYLE["axis_width"],
            "xtick.major.width": STYLE["axis_width"],
            "ytick.major.width": STYLE["axis_width"],
            "xtick.major.size": 5.8,
            "ytick.major.size": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def bold_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def normal_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")


def clear_text(ax: plt.Axes, keep_y_ticks: bool = True) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if not keep_y_ticks:
        ax.set_yticklabels([])
    bold_tick_labels(ax)


def probability_stars(probability: float) -> str:
    if probability >= 0.995:
        return "***"
    if probability >= 0.975:
        return "**"
    if probability >= 0.95:
        return "*"
    return "n.s."


def draw_density_effects(summary: pd.DataFrame, samples: pd.DataFrame, output_path: Path) -> None:
    df = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    order = df["feature"].tolist()

    all_values = samples.loc[samples["feature"].isin(order), "effect"].to_numpy(dtype=float)
    x_min, x_max = np.quantile(all_values, [0.002, 0.998])
    x_pad = (x_max - x_min) * 0.10
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 700)

    fig, ax = plt.subplots(figsize=RIDGE_SIZE, dpi=400)
    for idx, row in df.iterrows():
        values = samples.loc[samples["feature"] == row["feature"], "effect"].to_numpy(dtype=float)
        density = gaussian_kde(values)(x_grid)
        density = density / max(float(density.max()), 1e-12) * STYLE["ridge_height"]
        baseline = float(idx)

        ax.fill_between(x_grid, baseline, baseline - density, color=COLORS["ridge_fill"], alpha=0.58, linewidth=0, zorder=2)
        ax.plot(x_grid, baseline - density, color=COLORS["ridge_line"], lw=STYLE["ridge_line_width"], zorder=3)
        ax.hlines(baseline, x_grid.min(), x_grid.max(), color="#F0D2B4", lw=0.7, zorder=1)
        ax.hlines(baseline - 0.04, row["hdi_3"], row["hdi_97"], color=COLORS["ridge_hdi"], lw=1.4, zorder=4)
        ax.scatter(row["mean"], baseline - 0.04, s=18, color=COLORS["mean"], edgecolor="white", linewidth=0.45, zorder=5)

    ax.axvline(0, color=COLORS["zero"], lw=0.9, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(len(df) - 0.35, -0.72)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=STYLE["tick_size"])
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
    ax.tick_params(axis="x", width=2.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color=COLORS["grid"], lw=STYLE["grid_width"])
    ax.set_axisbelow(True)
    clear_text(ax, keep_y_ticks=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.985, bottom=0.13)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def effect_to_radius(values: np.ndarray | float, effect_min: float, effect_max: float, r_min: float, r_max: float) -> np.ndarray | float:
    return r_min + (np.asarray(values) - effect_min) / (effect_max - effect_min) * (r_max - r_min)


def setup_radial_axis(ax: plt.Axes, effect_min: float, effect_max: float, r_min: float, r_max: float) -> None:
    tick_values = np.array([-1.6, -0.8, 0.0, 0.8, 1.6, 2.4])
    tick_values = tick_values[(tick_values >= effect_min) & (tick_values <= effect_max)]
    tick_radii = effect_to_radius(tick_values, effect_min, effect_max, r_min, r_max)
    ax.set_ylim(0.0, r_max + 0.06)
    ax.set_xticks([])
    ax.set_yticks(tick_radii)
    ax.set_yticklabels([f"{value:.1f}" for value in tick_values], fontsize=7.5, fontweight="bold", color="#5E6670")
    ax.set_rlabel_position(86)
    ax.grid(color="#E8ECF2", lw=0.8)
    ax.spines["polar"].set_linewidth(2.0)
    ax.spines["polar"].set_color(COLORS["axis"])


def draw_radial_posterior_violin(summary: pd.DataFrame, samples: pd.DataFrame, output_path: Path) -> None:
    df = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    n_features = len(df)
    theta_centers = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    sector_width = 2 * np.pi / n_features

    all_values = samples.loc[samples["feature"].isin(df["feature"]), "effect"].to_numpy(dtype=float)
    effect_min, effect_max = np.quantile(all_values, [0.002, 0.998])
    effect_pad = (effect_max - effect_min) * 0.08
    effect_min -= effect_pad
    effect_max += effect_pad
    effect_grid = np.linspace(effect_min, effect_max, 560)

    r_min, r_max = 0.24, 1.00
    radii = effect_to_radius(effect_grid, effect_min, effect_max, r_min, r_max)

    fig, ax = plt.subplots(figsize=RADIAL_SIZE, dpi=420, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for theta, (_, row) in zip(theta_centers, df.iterrows()):
        values = samples.loc[samples["feature"] == row["feature"], "effect"].to_numpy(dtype=float)
        density = gaussian_kde(values)(effect_grid)
        density = density / max(float(density.max()), 1e-12) * sector_width * 0.42
        visible = density > sector_width * 0.006

        left = theta - density[visible]
        right = theta + density[visible]
        r = radii[visible]
        ax.fill(
            np.concatenate([left, right[::-1]]),
            np.concatenate([r, r[::-1]]),
            color=COLORS["ridge_fill"],
            alpha=0.55,
            linewidth=0,
            zorder=2,
        )
        ax.plot(left, r, color=COLORS["ridge_line"], lw=1.15, zorder=3)
        ax.plot(right, r, color=COLORS["ridge_line"], lw=1.15, zorder=3)
        ax.plot(
            [theta, theta],
            [
                effect_to_radius(float(row["hdi_3"]), effect_min, effect_max, r_min, r_max),
                effect_to_radius(float(row["hdi_97"]), effect_min, effect_max, r_min, r_max),
            ],
            color=COLORS["ridge_hdi"],
            lw=1.6,
            zorder=4,
        )
        ax.scatter(
            [theta],
            [effect_to_radius(float(row["mean"]), effect_min, effect_max, r_min, r_max)],
            s=28,
            color=COLORS["mean"],
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )

    theta_full = np.linspace(0, 2 * np.pi, 720)
    zero_r = effect_to_radius(0.0, effect_min, effect_max, r_min, r_max)
    ax.plot(theta_full, np.full_like(theta_full, zero_r), color=COLORS["zero"], lw=1.2, linestyle=(0, (3, 3)), zorder=1)
    setup_radial_axis(ax, effect_min, effect_max, r_min, r_max)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_radial_effect_summary(summary: pd.DataFrame, output_path: Path) -> None:
    df = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    n_features = len(df)
    theta = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    sector_width = 2 * np.pi / n_features

    max_abs = float(np.nanmax(np.abs(df[["hdi_3", "hdi_97"]].to_numpy(dtype=float))))
    effect_min, effect_max = -max_abs * 1.08, max_abs * 1.08
    r_min, r_max = 0.22, 1.00
    zero_r = effect_to_radius(0.0, effect_min, effect_max, r_min, r_max)

    fig, ax = plt.subplots(figsize=RADIAL_SIZE, dpi=420, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for angle, (_, row) in zip(theta, df.iterrows()):
        mean = float(row["mean"])
        color = "#E47C20" if mean >= 0 else "#2F78B7"
        mean_r = effect_to_radius(mean, effect_min, effect_max, r_min, r_max)
        low_r = effect_to_radius(float(row["hdi_3"]), effect_min, effect_max, r_min, r_max)
        high_r = effect_to_radius(float(row["hdi_97"]), effect_min, effect_max, r_min, r_max)

        bottom = min(zero_r, mean_r)
        height = abs(mean_r - zero_r)
        ax.bar(
            angle,
            height,
            width=sector_width * 0.72,
            bottom=bottom,
            color=color,
            alpha=0.72,
            edgecolor="#2F3437",
            linewidth=0.65,
            zorder=2,
        )
        ax.plot([angle, angle], [low_r, high_r], color=COLORS["axis"], lw=1.45, zorder=4)
        ax.scatter([angle], [mean_r], s=26, color=COLORS["mean"], edgecolor="white", linewidth=0.55, zorder=5)

    theta_full = np.linspace(0, 2 * np.pi, 720)
    ax.plot(theta_full, np.full_like(theta_full, zero_r), color=COLORS["zero"], lw=1.35, linestyle=(0, (3, 3)), zorder=1)
    setup_radial_axis(ax, effect_min, effect_max, r_min, r_max)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_gamm_curve(
    curve_df: pd.DataFrame,
    feature_name: str,
    output_path: Path,
    image_points: pd.DataFrame | None = None,
) -> None:
    panel = curve_df[(curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] >= 0)].sort_values("curve_index")
    rug = curve_df[(curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] < 0)].sort_values("rug_index")

    x = panel["x"].to_numpy(dtype=float)
    y = panel["mean_expected_score"].to_numpy(dtype=float)
    lower = panel["lower_95"].to_numpy(dtype=float)
    upper = panel["upper_95"].to_numpy(dtype=float)
    y_min = float(np.nanmin(lower))
    y_max = float(np.nanmax(upper))
    if image_points is not None:
        y_min = min(y_min, float(image_points["rating_mean"].min()))
        y_max = max(y_max, float(image_points["rating_mean"].max()))

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=400)
    ax.fill_between(x, lower, upper, color=COLORS["curve_band"], alpha=0.42, linewidth=0, zorder=1)
    if image_points is not None:
        ax.scatter(
            image_points[feature_name],
            image_points["rating_mean"],
            s=19,
            color="#668DB8",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.52,
            zorder=1.5,
        )
    ax.plot(x, y, color=COLORS["curve_line"], lw=GAMM_STYLE["curve_line_width"], zorder=2)

    if not rug.empty:
        rug_height = (y_max - y_min) * 0.035
        for value in rug["x"].to_numpy(dtype=float):
            ax.vlines(value, y_min, y_min + rug_height, color="#7B8794", lw=0.7, alpha=0.8, zorder=3)

    y_pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.tick_params(
        axis="both",
        labelsize=GAMM_STYLE["tick_size"],
        width=GAMM_STYLE["axis_width"],
        length=GAMM_STYLE["tick_length"],
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=COLORS["grid"], lw=STYLE["grid_width"])
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(GAMM_STYLE["axis_width"])
    clear_text(ax)
    normal_tick_labels(ax)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.985, bottom=0.14)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def draw_model_boxline(summary: pd.DataFrame, samples: pd.DataFrame, output_path: Path) -> None:
    rng = np.random.default_rng(20260523)
    df = summary.set_index("Model").loc[MODEL_ORDER].reset_index()
    positions = np.arange(len(MODEL_ORDER))
    grouped = [
        samples.loc[samples["Model"] == model, "ExplainedVariance"].to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]

    fig, ax = plt.subplots(figsize=MODEL_SIZE, dpi=400)
    box = ax.boxplot(
        grouped,
        positions=positions,
        vert=False,
        widths=STYLE["model_box_width"],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLORS["model_line"], linewidth=1.65),
        whiskerprops=dict(color="#454545", linewidth=1.35),
        capprops=dict(color="#454545", linewidth=1.35),
        boxprops=dict(linewidth=1.35, color="#262626"),
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.42)

    for pos, model, values in zip(positions, MODEL_ORDER, grouped):
        display_values = rng.choice(values, size=min(115, len(values)), replace=False)
        jitter = rng.normal(0, 0.075, size=len(display_values))
        ax.scatter(display_values, np.full(len(display_values), pos) + jitter, s=13, color=MODEL_COLORS[model], edgecolor="#1F1F1F", linewidth=0.32, alpha=0.76, zorder=3)

    means = df["Mean"].to_numpy(dtype=float)
    ax.plot(means, positions, color=COLORS["model_line"], lw=2.15, marker="o", markersize=5.8, zorder=5)

    for pos, (_, row) in zip(positions, df.iterrows()):
        mean = float(row["Mean"])
        low = float(row["HDI_3%"])
        high = float(row["HDI_97%"])
        ax.errorbar(mean, pos, xerr=np.array([[mean - low], [high - mean]], dtype=float), fmt="none", ecolor=COLORS["model_hdi"], elinewidth=1.35, capsize=3.8, capthick=1.15, zorder=4)

    x_max_data = float(max(df["HDI_97%"].max(), samples["ExplainedVariance"].quantile(0.995)))

    x_min = float(min(samples["ExplainedVariance"].quantile(0.002), df["HDI_3%"].min()))
    x_max = 6.25
    ax.set_xlim(4.2, 5.8)
    ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.95)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=STYLE["tick_size"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="x", color=COLORS["grid"], lw=STYLE["grid_width"])
    ax.set_axisbelow(True)
    clear_text(ax, keep_y_ticks=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.985, bottom=0.16)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effect_summary = pd.read_csv(EFFECT_SUMMARY)
    effect_samples = pd.read_csv(EFFECT_SAMPLES)
    gamm = pd.read_csv(GAMM_DATA)
    model_summary = pd.read_csv(MODEL_SUMMARY)
    model_samples = pd.read_csv(MODEL_SAMPLES)

    outputs = []
    ridge_output = OUTPUT_DIR / "Fig3_01_posterior_distributions_textless.png"
    draw_density_effects(effect_summary, effect_samples, ridge_output)
    outputs.append(ridge_output)

    radial_violin_output = OUTPUT_DIR / "Fig3_01A_radial_posterior_violin_textless.png"
    draw_radial_posterior_violin(effect_summary, effect_samples, radial_violin_output)
    outputs.append(radial_violin_output)

    radial_summary_output = OUTPUT_DIR / "Fig3_01B_radial_effect_summary_textless.png"
    draw_radial_effect_summary(effect_summary, radial_summary_output)
    outputs.append(radial_summary_output)

    curve_features = (
        gamm.loc[gamm["curve_index"] >= 0, ["feature_order", "feature_name"]]
        .drop_duplicates()
        .sort_values("feature_order")
        .reset_index(drop=True)
    )
    for idx, row in curve_features.iterrows():
        output = OUTPUT_DIR / f"Fig3_{idx + 2:02d}_{row['feature_name']}_textless.png"
        draw_gamm_curve(gamm, str(row["feature_name"]), output)
        outputs.append(output)

    model_output = OUTPUT_DIR / "Fig3_08_image_variance_textless.png"
    draw_model_boxline(model_summary, model_samples, model_output)
    outputs.append(model_output)

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
