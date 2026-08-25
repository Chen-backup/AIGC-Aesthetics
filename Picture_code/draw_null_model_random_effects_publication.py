from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Picture_code" / "picture"
SUMMARY_PATH = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Summary.csv"

PNG_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Publication.png"
CSV_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Publication_Data.csv"
RATER_DIST_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Rater_Distribution.png"
IMAGE_DIST_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Image_Distribution.png"

COLORS = {
    "rater": "#C84C61",
    "image": "#2F78B7",
    "axis": "#333333",
    "grid": "#E7E9ED",
    "omitted": "#A8ADB5",
    "zero": "#777777",
    "text": "#222222",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.35,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 9,
            "ytick.major.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_entity_id(term: str) -> int:
    match = re.search(r"\[(\d+)\]$", term)
    return int(match.group(1)) if match else -1


def load_effects(prefix: str, group_name: str) -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH, index_col=0)
    mask = df.index.to_series().str.startswith(prefix)
    effects = df.loc[mask, ["mean", "hdi_3%", "hdi_97%"]].copy()
    effects = effects.reset_index(names="term")
    effects["id"] = effects["term"].map(parse_entity_id)
    effects["group"] = group_name
    effects["significant"] = (effects["hdi_3%"] > 0) | (effects["hdi_97%"] < 0)
    effects["abs_mean"] = effects["mean"].abs()
    return effects.sort_values("mean").reset_index(drop=True)


def draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    center_y: float,
    color: str,
    x_grid: np.ndarray,
    width: float = 0.30,
) -> None:
    kde = gaussian_kde(values)
    density = kde(x_grid)
    density = density / max(float(density.max()), 1e-12) * width
    visible = density > width * 0.006
    ax.fill_between(
        x_grid[visible],
        center_y - density[visible],
        center_y + density[visible],
        color=color,
        alpha=0.28,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x_grid[visible], center_y + density[visible], color=color, lw=1.2, zorder=2)
    ax.plot(x_grid[visible], center_y - density[visible], color=color, lw=1.2, zorder=2)


def draw_distribution_panel(ax: plt.Axes, effects: pd.DataFrame, y: float, color: str, label: str, x_grid: np.ndarray) -> None:
    values = effects["mean"].to_numpy(dtype=float)
    draw_half_violin(ax, values, y, color, x_grid)

    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    p05, p95 = np.quantile(values, [0.05, 0.95])
    ax.hlines(y, p05, p95, color=color, lw=4.2, alpha=0.34, zorder=3)
    ax.hlines(y, q1, q3, color="#222222", lw=2.1, zorder=4)
    ax.scatter([median], [y], s=34, color="#222222", zorder=5)
    ax.scatter([values.mean()], [y], s=38, facecolor="white", edgecolor="#222222", linewidth=0.8, zorder=5)

    sig_count = int(effects["significant"].sum())
    text_y = y + 0.43
    x_span = float(x_grid.max() - x_grid.min())
    ax.text(
        x_grid.max() - x_span * 0.03,
        text_y,
        f"n={len(effects)}, HDI excludes 0: {sig_count}",
        ha="right",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color=COLORS["text"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.2),
    )
    if label:
        ax.text(
            x_grid.min() + x_span * 0.035,
            text_y,
            label,
            ha="left",
            va="center",
            fontsize=10.6,
            fontweight="bold",
            color=COLORS["text"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.2),
        )


def draw_tail_panel(ax: plt.Axes, effects: pd.DataFrame, color: str, title: str, tail_n: int = 10) -> None:
    low_tail = effects.nsmallest(tail_n, "mean")
    high_tail = effects.nlargest(tail_n, "mean")
    tail = pd.concat([low_tail, high_tail], ignore_index=True).sort_values("mean").reset_index(drop=True)
    omitted = effects.drop(index=low_tail.index.union(high_tail.index))
    y = np.arange(len(tail), dtype=float)
    y[tail_n:] += 1.25

    ax.errorbar(
        tail["mean"],
        y,
        xerr=[tail["mean"] - tail["hdi_3%"], tail["hdi_97%"] - tail["mean"]],
        fmt="none",
        ecolor="#646464",
        elinewidth=0.9,
        capsize=2.0,
        capthick=0.75,
        zorder=1,
    )
    ax.scatter(
        tail["mean"],
        y,
        s=28,
        color=color,
        edgecolor="#222222",
        linewidth=0.45,
        zorder=3,
    )
    ax.axvline(0, color=COLORS["zero"], lw=0.85, linestyle=(0, (4, 4)))
    ax.set_yticks(y)
    ax.set_yticklabels([str(i) for i in tail["id"]], fontsize=8.4, fontweight="bold")
    ax.set_title(title, fontsize=10.4, fontweight="bold", pad=7)
    ax.tick_params(axis="x", labelsize=8.8)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.tick_params(axis="y", pad=6)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.65)
    ax.set_axisbelow(True)

    break_y = tail_n - 0.5 + 0.62
    x0, x1 = ax.get_xlim()
    x_span = x1 - x0
    ax.axhspan(break_y - 0.38, break_y + 0.38, color="white", zorder=2.5)

    if len(omitted) > 0:
        omitted_low = float(omitted["mean"].min())
        omitted_high = float(omitted["mean"].max())
        ax.plot(
            [omitted_low, omitted_high],
            [break_y, break_y],
            color=COLORS["omitted"],
            lw=7.0,
            alpha=0.72,
            solid_capstyle="round",
            zorder=5,
        )
        middle_values = omitted["mean"].to_numpy(dtype=float)
        display_count = min(55, len(middle_values))
        if display_count > 0:
            quantile_positions = np.linspace(0.02, 0.98, display_count)
            display_values = np.quantile(middle_values, quantile_positions)
            ax.scatter(
                display_values,
                np.full(display_count, break_y),
                s=7.0,
                color="#6F7680",
                alpha=0.62,
                edgecolor="none",
                zorder=6,
            )
        ax.text(
            (omitted_low + omitted_high) / 2,
            break_y + 0.38,
            f"{len(omitted)} omitted middle samples",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color="#4B5563",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.8),
            zorder=7,
        )

    ax.plot([x0 + x_span * 0.02, x0 + x_span * 0.09], [break_y - 0.12, break_y + 0.12], color=COLORS["axis"], lw=1.5, clip_on=False, zorder=5)
    ax.plot([x0 + x_span * 0.02, x0 + x_span * 0.09], [break_y + 0.12, break_y + 0.36], color=COLORS["axis"], lw=1.5, clip_on=False, zorder=5)
    ax.text(
        x0 + x_span * 0.50,
        break_y - 0.34,
        "...",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=COLORS["axis"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.2),
        zorder=6,
    )
    ax.set_ylim(-0.75, y[-1] + 0.75)


def format_signed_tick(value: float, _pos: float) -> str:
    if abs(value) < 1e-10:
        return "0"
    text = f"{abs(value):.1f}".rstrip("0").rstrip(".")
    return f"+{text}" if value > 0 else f"-{text}"


def compute_distribution_grid(effects: pd.DataFrame) -> np.ndarray:
    values = effects["mean"].to_numpy(dtype=float)
    x_min = float(np.quantile(values, 0.003))
    x_max = float(np.quantile(values, 0.997))
    pad = (x_max - x_min) * 0.18
    if pad == 0:
        pad = 0.35
    return np.linspace(x_min - pad, x_max + pad, 700)


def style_split_distribution_axis(ax: plt.Axes, x_grid: np.ndarray, ylabel: str, xlabel: str, title: str) -> None:
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(-0.78, 0.78)
    ax.set_yticks([])
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold", labelpad=36)
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold", labelpad=16, linespacing=1.05)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18, linespacing=1.15)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.95)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(3.0)
    ax.spines["bottom"].set_linewidth(3.0)
    ax.tick_params(axis="x", labelsize=14, pad=5)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(format_signed_tick))


def draw_split_distribution_figure(
    effects: pd.DataFrame,
    color: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    set_style()
    x_grid = compute_distribution_grid(effects)

    fig, ax = plt.subplots(figsize=(2000 / 300, 2000 / 300), dpi=300)
    draw_distribution_panel(ax, effects, 0.0, color, "", x_grid)

    ax.axvline(0, color=COLORS["axis"], lw=2.1)
    style_split_distribution_axis(
        ax,
        x_grid,
        ylabel=ylabel,
        xlabel="Posterior Random Effect\n(Mean with 95% HDI)",
        title=title,
    )

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markeredgecolor="#222222", markersize=8, label="Median"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#222222", markeredgewidth=1.2, markersize=8, label="Mean"),
        mpl.lines.Line2D([0], [0], color="#222222", lw=3.6, label="IQR"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=11,
        ncol=3,
        handlelength=1.8,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.22, top=0.83)
    fig.savefig(out_path)
    plt.close(fig)


def draw_publication_figure() -> None:
    set_style()
    rater = load_effects("1|rater[", "Rater")
    image = load_effects("1|image[", "Image")
    pd.concat([rater, image], ignore_index=True).to_csv(CSV_OUTPUT, index=False)

    all_values = pd.concat([rater["mean"], image["mean"]]).to_numpy(dtype=float)
    x_min = float(np.quantile(all_values, 0.003))
    x_max = float(np.quantile(all_values, 0.997))
    pad = (x_max - x_min) * 0.15
    x_grid = np.linspace(x_min - pad, x_max + pad, 700)

    fig = plt.figure(figsize=(9.4, 6.8), dpi=450)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.0, 1.22], height_ratios=[1, 1], wspace=0.28, hspace=0.42)

    ax_dist = fig.add_subplot(gs[:, 0])
    ax_rater_tail = fig.add_subplot(gs[0, 1])
    ax_image_tail = fig.add_subplot(gs[1, 1])

    draw_distribution_panel(ax_dist, rater, 1.12, COLORS["rater"], "Rater random intercepts", x_grid)
    draw_distribution_panel(ax_dist, image, 0.0, COLORS["image"], "Image random intercepts", x_grid)

    ax_dist.axvline(0, color=COLORS["zero"], linestyle=(0, (4, 4)), lw=1.0)
    ax_dist.set_xlim(x_grid.min(), x_grid.max())
    ax_dist.set_ylim(-0.60, 1.72)
    ax_dist.set_yticks([0, 1.12])
    ax_dist.set_yticklabels(["Image", "Rater"], fontsize=11.0, fontweight="bold", rotation=90, va="center")
    ax_dist.tick_params(axis="y", pad=20)
    ax_dist.set_xlabel("Posterior random intercept (mean)", fontsize=11.2, fontweight="bold", labelpad=8)
    ax_dist.set_title("Null Model Random Effects: Distributions and Extremes", fontsize=13.6, fontweight="bold", pad=12)
    ax_dist.grid(axis="x", color=COLORS["grid"], lw=0.75)
    ax_dist.set_axisbelow(True)
    ax_dist.tick_params(axis="x", labelsize=10.0)
    for tick in ax_dist.get_xticklabels():
        tick.set_fontweight("bold")

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markeredgecolor="#222222", markersize=5, label="Median"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#222222", markersize=5, label="Mean"),
        mpl.lines.Line2D([0], [0], color="#222222", lw=2.0, label="IQR"),
    ]
    ax_dist.legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=9.2, ncol=3, handlelength=1.5, columnspacing=1.0)

    draw_tail_panel(ax_rater_tail, rater, COLORS["rater"], "Most extreme raters", tail_n=8)
    draw_tail_panel(ax_image_tail, image, COLORS["image"], "Most extreme images", tail_n=8)
    ax_image_tail.set_xlabel("Posterior mean with 94% HDI", fontsize=9.8, fontweight="bold", labelpad=7)

    fig.savefig(PNG_OUTPUT, bbox_inches="tight")
    plt.close(fig)

    draw_split_distribution_figure(
        rater,
        COLORS["rater"],
        ylabel="Rater index",
        title="Null Model Distribution Plot:\nRater Random Effects",
        out_path=RATER_DIST_OUTPUT,
    )
    draw_split_distribution_figure(
        image,
        COLORS["image"],
        ylabel="Image index",
        title="Null Model Distribution Plot:\nImage Random Effects",
        out_path=IMAGE_DIST_OUTPUT,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_publication_figure()
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved figure: {RATER_DIST_OUTPUT}")
    print(f"Saved figure: {IMAGE_DIST_OUTPUT}")
    print(f"Saved data: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
