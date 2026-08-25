from __future__ import annotations

import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Picture_code" / "picture"

EFFECT_SUMMARY = OUTPUT_DIR / "Interpretable_14D_Boxline_Data.csv"
EFFECT_SAMPLES = OUTPUT_DIR / "Interpretable_14D_Boxline_Samples.csv"
GAMM_DATA = OUTPUT_DIR / "GAMM_NonLinear_Expected_Curves_Top6_data.csv"
MODEL_SUMMARY = OUTPUT_DIR / "image_variance_explained_3models_bar_data.csv"
MODEL_SAMPLES = OUTPUT_DIR / "image_variance_explained_3models_boxline_samples.csv"

PNG_OUTPUT = OUTPUT_DIR / "Interpretable_Summary_Publication_Redesign.png"


# =========================
# 可调参数区：字号、间距、图例位置、线条粗细都集中在这里。
# =========================

STYLE = {
    # 字体：论文图常用 Times New Roman；若系统没有，会回退到 Times / DejaVu Serif。
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "title_size": 18,          # 左侧和底部面板标题字号
    "curve_group_title_size": 18,  # 右侧 2x3 曲线区大标题字号
    "curve_title_size": 14,    # 右侧六个曲线小标题字号
    "axis_label_size": 12,     # 坐标轴标题字号
    "tick_size": 15,            # 坐标轴刻度字号
    "feature_label_size": 13,   # 左侧 14 个特征名字号
    "legend_size": 11,          # 图例字号
    "axis_width": 2.2,          # 坐标轴粗细
    "grid_width": 0.8,          # 网格线粗细
    "curve_line_width": 1.35,    # GAMM 曲线粗细
    "ridge_height": 0.65,        # 左侧密度脊线高度；越大越饱满
    "ridge_line_width": 0.95,    # 左侧密度轮廓线粗细
    "model_box_width": 0.50,     # 底部横向箱线图高度
}

LAYOUT = {
    # figsize 控制整张图比例；第一个数越大越宽，第二个数越大越高。
    "figsize": (14.8, 10.4),
    "dpi": 450,
    # left/right/top/bottom 控制整张图四周留白。
    "left": 0.140,
    "right": 0.985,
    "top": 0.920,
    "bottom": 0.075,
    # wspace 控制左侧密度图与右侧曲线区的水平间距；hspace 控制上方模块与底部模块间距。
    "wspace": 0.20,
    "hspace": 0.35,
    # 左右两大列宽度比例：调小第一个数、调大第二个数，可让右侧 2x3 曲线更宽。
    "outer_width_ratios": (0.35, 0.70),
    # 第一行高度, 第二行高度, 底部模型图高度。
    "outer_height_ratios": (0.85, 0.85, 0.80),
    # 右侧六个 GAMM 曲线之间的间距；数值越小越紧凑。
    "curve_wspace": 0.20,
    "curve_hspace": 0.55,
    "curve_top_spacer": 0.001,
}

POSITIONS = {
    # 底部横向图图例位置；第一个数控制左右，第二个数控制上下。
    "model_legend_anchor": (0.01, 0.97),
    # 右侧 2x3 曲线区大标题离曲线区顶部的距离；越大标题越靠上。
    "curve_group_title_pad": 0.050,
    "ridge_star_right_pad": 0.025,
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
    "text": "#1F2328",
    "model_line": "#262626",
    "model_hdi": "#575757",
}

MODEL_ORDER = ["StyleGAN", "InsightFace", "DINOv2"]
MODEL_COLORS = {
    "StyleGAN": "#6BAED6",
    "InsightFace": "#D9A441",
    "DINOv2": "#7A9A01",
}


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


def wrap_label(label: str, width: int = 25) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def probability_stars(probability: float) -> str:
    if probability >= 0.995:
        return "***"
    if probability >= 0.975:
        return "**"
    if probability >= 0.95:
        return "*"
    return "n.s."


def draw_density_effects(ax: plt.Axes, summary: pd.DataFrame, samples: pd.DataFrame) -> None:
    df = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    order = df["feature"].tolist()

    all_values = samples.loc[samples["feature"].isin(order), "effect"].to_numpy(dtype=float)
    x_min, x_max = np.quantile(all_values, [0.002, 0.998])
    x_pad = (x_max - x_min) * 0.10
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 700)

    for idx, row in df.iterrows():
        values = samples.loc[samples["feature"] == row["feature"], "effect"].to_numpy(dtype=float)
        kde = gaussian_kde(values)
        density = kde(x_grid)
        density = density / max(float(density.max()), 1e-12) * STYLE["ridge_height"]

        baseline = float(idx)
        ax.fill_between(
            x_grid,
            baseline,
            baseline - density,
            color=COLORS["ridge_fill"],
            alpha=0.58,
            linewidth=0,
            zorder=2,
        )
        ax.plot(x_grid, baseline - density, color=COLORS["ridge_line"], lw=STYLE["ridge_line_width"], zorder=3)
        ax.hlines(baseline, x_grid.min(), x_grid.max(), color="#F0D2B4", lw=0.7, zorder=1)

        hdi_low = float(row["hdi_3"])
        hdi_high = float(row["hdi_97"])
        mean = float(row["mean"])
        ax.hlines(baseline - 0.04, hdi_low, hdi_high, color=COLORS["ridge_hdi"], lw=1.4, zorder=4)
        ax.scatter(mean, baseline - 0.04, s=18, color=COLORS["mean"], edgecolor="white", linewidth=0.45, zorder=5)

        stars = str(row.get("stars", ""))
        if stars and stars != "nan":
            x_span = x_grid.max() - x_grid.min()
            ax.text(
                x_grid.max() - x_span * POSITIONS["ridge_star_right_pad"],
                baseline - 0.08,
                stars,
                ha="right",
                va="center",
                fontsize=STYLE["tick_size"] + 0.6,
                fontweight="bold",
                color=COLORS["text"],
            )

    ax.axvline(0, color=COLORS["zero"], lw=0.9, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(len(df) - 0.35, -0.72)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([wrap_label(label) for label in df["label"]], fontsize=STYLE["feature_label_size"])
    ax.set_xlabel("Standardized effect size (posterior distribution)", fontsize=STYLE["axis_label_size"], fontweight="bold")
    ax.set_title("Posterior distributions of 14 interpretable features", fontsize=STYLE["title_size"], fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=STYLE["tick_size"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color=COLORS["grid"], lw=STYLE["grid_width"])
    ax.set_axisbelow(True)


def draw_gamm_curve(ax: plt.Axes, curve_df: pd.DataFrame, feature_name: str, feature_label: str, show_ylabel: bool) -> None:
    panel = curve_df[(curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] >= 0)].sort_values("curve_index")
    rug = curve_df[(curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] < 0)].sort_values("rug_index")

    x = panel["x"].to_numpy(dtype=float)
    y = panel["mean_expected_score"].to_numpy(dtype=float)
    lower = panel["lower_95"].to_numpy(dtype=float)
    upper = panel["upper_95"].to_numpy(dtype=float)

    ax.fill_between(x, lower, upper, color=COLORS["curve_band"], alpha=0.42, linewidth=0, zorder=1)
    ax.plot(x, y, color=COLORS["curve_line"], lw=STYLE["curve_line_width"], zorder=2)

    if not rug.empty:
        ymin, ymax = float(np.nanmin(lower)), float(np.nanmax(upper))
        rug_height = (ymax - ymin) * 0.035
        for value in rug["x"].to_numpy(dtype=float):
            ax.vlines(value, ymin, ymin + rug_height, color="#7B8794", lw=0.7, alpha=0.8, zorder=3)

    y_min = float(np.nanmin(lower))
    y_max = float(np.nanmax(upper))
    y_pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))

    ax.set_title(feature_label, fontsize=STYLE["curve_title_size"], fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=COLORS["grid"], lw=STYLE["grid_width"])
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel("Expected aesthetic score", fontsize=STYLE["axis_label_size"], fontweight="bold")


def draw_horizontal_bracket(ax: plt.Axes, y1: float, y2: float, x: float, label: str, tick: float) -> None:
    ax.plot([x - tick, x, x, x - tick], [y1, y1, y2, y2], color="#555555", lw=0.85, clip_on=False)
    ax.text(
        x + tick * 0.25,
        (y1 + y2) / 2,
        label,
        ha="left",
        va="center",
        fontsize=STYLE["legend_size"],
        fontweight="bold",
        color=COLORS["text"],
        linespacing=0.85,
    )


def draw_model_boxline_horizontal(ax: plt.Axes, summary: pd.DataFrame, samples: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260523)
    df = summary.set_index("Model").loc[MODEL_ORDER].reset_index()
    positions = np.arange(len(MODEL_ORDER))
    grouped = [
        samples.loc[samples["Model"] == model, "ExplainedVariance"].to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]

    box = ax.boxplot(
        grouped,
        positions=positions,
        vert=False,
        widths=STYLE["model_box_width"],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLORS["model_line"], linewidth=1.05),
        whiskerprops=dict(color="#5B5B5B", linewidth=0.85),
        capprops=dict(color="#5B5B5B", linewidth=0.85),
        boxprops=dict(linewidth=0.90, color="#333333"),
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.42)

    for pos, model, values in zip(positions, MODEL_ORDER, grouped):
        display_values = rng.choice(values, size=min(115, len(values)), replace=False)
        jitter = rng.normal(0, 0.075, size=len(display_values))
        ax.scatter(
            display_values,
            np.full(len(display_values), pos) + jitter,
            s=10,
            color=MODEL_COLORS[model],
            edgecolor="#222222",
            linewidth=0.18,
            alpha=0.68,
            zorder=3,
        )

    means = df["Mean"].to_numpy(dtype=float)
    ax.plot(means, positions, color=COLORS["model_line"], lw=1.25, marker="o", markersize=4.2, zorder=5)

    for pos, (_, row) in zip(positions, df.iterrows()):
        mean = float(row["Mean"])
        low = float(row["HDI_3%"])
        high = float(row["HDI_97%"])
        ax.errorbar(
            mean,
            pos,
            xerr=np.array([[mean - low], [high - mean]], dtype=float),
            fmt="none",
            ecolor=COLORS["model_hdi"],
            elinewidth=0.85,
            capsize=2.8,
            capthick=0.75,
            zorder=4,
        )
        ax.text(
            high + 0.035,
            pos,
            f"{mean:.2f}",
            ha="left",
            va="center",
            fontsize=STYLE["tick_size"],
            color=COLORS["text"],
        )

    samples_by_model = {
        model: samples.loc[samples["Model"] == model, "ExplainedVariance"].to_numpy(dtype=float)
        for model in MODEL_ORDER
    }
    x_max_data = float(max(df["HDI_97%"].max(), samples["ExplainedVariance"].quantile(0.995)))
    bracket_gap = 0.125
    bracket_tick = 0.045
    for level, (left_idx, right_idx) in enumerate(combinations(range(len(MODEL_ORDER)), 2)):
        left_model = MODEL_ORDER[left_idx]
        right_model = MODEL_ORDER[right_idx]
        left_values = samples_by_model[left_model]
        right_values = samples_by_model[right_model]
        n = min(len(left_values), len(right_values))
        prob = float(np.mean(right_values[:n] > left_values[:n]))
        diff = float(df.loc[df["Model"] == right_model, "Mean"].iloc[0] - df.loc[df["Model"] == left_model, "Mean"].iloc[0])
        draw_horizontal_bracket(
            ax,
            y1=float(left_idx),
            y2=float(right_idx),
            x=x_max_data + bracket_gap * (level + 1),
            label=f"+{diff:.2f}\n{probability_stars(prob)}",
            tick=bracket_tick,
        )

    x_min = float(min(samples["ExplainedVariance"].quantile(0.002), df["HDI_3%"].min()))
    x_max = x_max_data + bracket_gap * 4.2
    ax.set_xlim(x_min - 0.12, x_max)
    # 这里给上方图例留出空白；第二个数越小，顶部空白越大。
    ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.95)
    ax.set_yticks(positions)
    ax.set_yticklabels(MODEL_ORDER, fontsize=STYLE["tick_size"], fontweight="bold")
    ax.set_xlabel("Image variance explained (sigma^2)", fontsize=STYLE["axis_label_size"], fontweight="bold")
    ax.set_title("Image variance explained by pretrained models", fontsize=STYLE["title_size"], fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=STYLE["tick_size"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="x", color=COLORS["grid"], lw=STYLE["grid_width"])
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], color=COLORS["model_line"], marker="o", lw=1.25, markersize=4.2, label="Posterior mean"),
        Line2D([0], [0], color=COLORS["model_hdi"], lw=0.95, label="94% HDI"),
        Patch(facecolor="#B9D6EA", edgecolor="#333333", alpha=0.45, label="IQR"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=POSITIONS["model_legend_anchor"],
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=1.7,
        borderaxespad=0,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_summary = pd.read_csv(EFFECT_SUMMARY)
    effect_samples = pd.read_csv(EFFECT_SAMPLES)
    gamm = pd.read_csv(GAMM_DATA)
    model_summary = pd.read_csv(MODEL_SUMMARY)
    model_samples = pd.read_csv(MODEL_SAMPLES)
    return effect_summary, effect_samples, gamm, model_summary, model_samples


def draw_curve_group_title(fig: plt.Figure, curve_axes: list[plt.Axes]) -> None:
    if not curve_axes:
        return
    left = min(ax.get_position().x0 for ax in curve_axes)
    right = max(ax.get_position().x1 for ax in curve_axes)
    top = max(ax.get_position().y1 for ax in curve_axes)
    fig.text(
        (left + right) / 2,
        top + POSITIONS["curve_group_title_pad"],
        "Nonlinear associations for top interpretable features",
        ha="center",
        va="bottom",
        fontsize=STYLE["curve_group_title_size"],
        fontweight="bold",
        color=COLORS["text"],
    )


def draw_figure() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effect_summary, effect_samples, gamm, model_summary, model_samples = load_data()

    fig = plt.figure(figsize=LAYOUT["figsize"], dpi=LAYOUT["dpi"])
    outer = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=LAYOUT["outer_width_ratios"],
        height_ratios=LAYOUT["outer_height_ratios"],
        wspace=LAYOUT["wspace"],
        hspace=LAYOUT["hspace"],
    )

    ax_effect = fig.add_subplot(outer[0:2, 0])
    curve_grid = outer[0:2, 1].subgridspec(
        3,
        3,
        height_ratios=(LAYOUT["curve_top_spacer"], 1.0, 1.0),
        wspace=LAYOUT["curve_wspace"],
        hspace=LAYOUT["curve_hspace"],
    )
    ax_model = fig.add_subplot(outer[2, :])

    draw_density_effects(ax_effect, effect_summary, effect_samples)

    curve_features = (
        gamm.loc[gamm["curve_index"] >= 0, ["feature_order", "feature_name", "feature_label"]]
        .drop_duplicates()
        .sort_values("feature_order")
        .reset_index(drop=True)
    )
    curve_axes: list[plt.Axes] = []
    for idx, row in curve_features.iterrows():
        ax = fig.add_subplot(curve_grid[idx // 3 + 1, idx % 3])
        curve_axes.append(ax)
        draw_gamm_curve(
            ax,
            gamm,
            feature_name=str(row["feature_name"]),
            feature_label=str(row["feature_label"]),
            show_ylabel=(idx % 3 == 0),
        )

    draw_model_boxline_horizontal(ax_model, model_summary, model_samples)

    fig.subplots_adjust(
        left=LAYOUT["left"],
        right=LAYOUT["right"],
        top=LAYOUT["top"],
        bottom=LAYOUT["bottom"],
    )
    draw_curve_group_title(fig, curve_axes)

    fig.savefig(PNG_OUTPUT, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    draw_figure()
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
