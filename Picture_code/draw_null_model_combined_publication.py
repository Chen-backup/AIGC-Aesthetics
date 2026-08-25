from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Picture_code" / "picture"
SUMMARY_PATH = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Summary.csv"

PNG_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Combined_2x2.png"
TEXTLESS_RATER_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Rater_Caterpillar_Textless.png"
TEXTLESS_IMAGE_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Image_Caterpillar_Textless.png"
TEXTLESS_DISTRIBUTION_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Distributions_Textless.png"
NOTICK_RATER_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Rater_Caterpillar_Textless_NoTickNumbers.png"
NOTICK_IMAGE_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Image_Caterpillar_Textless_NoTickNumbers.png"
NOTICK_DISTRIBUTION_OUTPUT = OUTPUT_DIR / "Null_Random_Effects_Distributions_Textless_NoTickNumbers.png"


# Layout knobs: adjust these first if you want to fine-tune the final row figure.
FIGURE_SIZE = (17.2, 5.7)
TEXTLESS_FIGURE_SIZE = (5.8, 5.2)
PANEL_WIDTH_RATIOS = (1.05, 1.05, 0.95)
SUBPLOT_ADJUST = {
    "left": 0.055,
    "right": 0.985,
    "top": 0.875,
    "bottom": 0.175,
    "wspace": 0.30,
}

FONT_SIZES = {
    "title": 18,
    "axis_label": 14,
    "tick": 15,
    "legend": 10.5,
    "note": 10.5,
}

# Controls the two annotation lines in the combined distribution panel.
# x is in data units; y is the vertical position used by the distribution panel.
# Increase x_fraction to move right, decrease it to move left.
# Increase y_offset to move the label upward relative to its distribution.
DISTRIBUTION_NOTE_POSITIONS = {
    "x_fraction": 0.98,
    "rater_y_offset": 0.36,
    "image_y_offset": 0.36,
}

COLORS = {
    "rater": "#C84C61",
    "image": "#2F78B7",
    "axis": "#2A2A2A",
    "grid": "#E7EBF0",
    "text": "#202020",
    "interval": "#B8C0CC",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 3.0,
            "xtick.major.width": 2.2,
            "ytick.major.width": 2.2,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def bold_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def parse_entity_id(term: str) -> int:
    match = re.search(r"\[(\d+)\]$", term)
    return int(match.group(1)) if match else -1


def load_effects(prefix: str) -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH, index_col=0)
    mask = df.index.to_series().str.startswith(prefix)
    effects = df.loc[mask, ["mean", "hdi_3%", "hdi_97%"]].copy()
    effects = effects.reset_index(names="term")
    effects["id"] = effects["term"].map(parse_entity_id)
    effects["significant"] = (effects["hdi_3%"] > 0) | (effects["hdi_97%"] < 0)
    return effects


def format_signed_tick(value: float, _pos: float) -> str:
    if abs(value) < 1e-10:
        return "0"
    text = f"{abs(value):.1f}".rstrip("0").rstrip(".")
    return f"+{text}" if value > 0 else f"-{text}"


def compute_limits(df: pd.DataFrame) -> tuple[float, float]:
    min_val = float(df["hdi_3%"].min())
    max_val = float(df["hdi_97%"].max())
    pad = (max_val - min_val) * 0.06
    if pad == 0:
        pad = 0.3
    return min_val - pad, max_val + pad


def compute_distribution_grid(*effects_frames: pd.DataFrame) -> np.ndarray:
    values = np.concatenate([df["mean"].to_numpy(dtype=float) for df in effects_frames])
    x_min = float(np.quantile(values, 0.003))
    x_max = float(np.quantile(values, 0.997))
    pad = (x_max - x_min) * 0.18
    if pad == 0:
        pad = 0.35
    return np.linspace(x_min - pad, x_max + pad, 700)


def draw_caterpillar(ax: plt.Axes, effects: pd.DataFrame, color: str, y_label: str) -> None:
    ordered = effects.sort_values("mean", ascending=False, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    y = np.arange(1, n + 1)
    x_min, x_max = compute_limits(ordered)

    line_width = 0.42 if n > 500 else 0.85
    point_size = 2.5 if n > 500 else 7.0

    ax.hlines(
        y,
        ordered["hdi_3%"].to_numpy(dtype=float),
        ordered["hdi_97%"].to_numpy(dtype=float),
        color=color,
        lw=line_width,
        alpha=0.95,
        zorder=2,
    )
    ax.scatter(
        ordered["mean"],
        y,
        s=point_size,
        color=color,
        edgecolors="none",
        zorder=3,
    )
    ax.plot(ordered["mean"], y, color=color, lw=1.05, alpha=0.9, zorder=2.5)

    ax.axvline(0, color=COLORS["axis"], lw=1.25, zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(n, 1)
    ax.set_ylabel(y_label, fontsize=FONT_SIZES["axis_label"], fontweight="bold", labelpad=8)
    ax.set_xlabel("Posterior random effect", fontsize=FONT_SIZES["axis_label"], fontweight="bold", labelpad=5)

    yticks = np.linspace(1, n, 6)
    ax.set_yticks(np.unique(np.round(yticks).astype(int)))
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(format_signed_tick))
    bold_tick_labels(ax)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.75)
    ax.grid(axis="y", color="#F4F6F8", lw=0.55)
    ax.set_axisbelow(True)


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
        alpha=0.27,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x_grid[visible], center_y + density[visible], color=color, lw=1.15, zorder=2)
    ax.plot(x_grid[visible], center_y - density[visible], color=color, lw=1.15, zorder=2)


def draw_distribution_row(
    ax: plt.Axes,
    effects: pd.DataFrame,
    color: str,
    y_pos: float,
    x_grid: np.ndarray,
) -> tuple[int, int]:
    values = effects["mean"].to_numpy(dtype=float)
    draw_half_violin(ax, values, y_pos, color, x_grid, width=0.28)

    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    p05, p95 = np.quantile(values, [0.05, 0.95])
    mean = float(values.mean())

    ax.hlines(y_pos, p05, p95, color=color, lw=8.0, alpha=0.24, zorder=3)
    ax.hlines(y_pos, q1, q3, color=COLORS["axis"], lw=3.5, zorder=4)
    ax.scatter([median], [y_pos], s=54, color=COLORS["axis"], zorder=5)
    ax.scatter([mean], [y_pos], s=54, facecolor="white", edgecolor=COLORS["axis"], linewidth=1.35, zorder=5)
    return len(effects), int(effects["significant"].sum())


def draw_combined_distribution(ax: plt.Axes, rater: pd.DataFrame, image: pd.DataFrame) -> None:
    x_grid = compute_distribution_grid(rater, image)
    rater_n, rater_sig = draw_distribution_row(ax, rater, COLORS["rater"], 1.0, x_grid)
    image_n, image_sig = draw_distribution_row(ax, image, COLORS["image"], 0.0, x_grid)

    ax.axvline(0, color=COLORS["axis"], lw=1.25, zorder=1)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["Rater", "Image"], fontsize=FONT_SIZES["tick"], fontweight="bold")
    ax.set_xlabel("Posterior random effect", fontsize=FONT_SIZES["axis_label"], fontweight="bold", labelpad=5)
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(format_signed_tick))
    bold_tick_labels(ax)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.75)
    ax.set_axisbelow(True)

    note_x = x_grid.min() + (x_grid.max() - x_grid.min()) * DISTRIBUTION_NOTE_POSITIONS["x_fraction"]
    note_bbox = dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2)
    ax.text(
        note_x,
        1.0 + DISTRIBUTION_NOTE_POSITIONS["rater_y_offset"],
        f"Rater: n={rater_n}, HDI excludes 0: {rater_sig}",
        ha="right",
        va="center",
        fontsize=FONT_SIZES["note"],
        fontweight="bold",
        color=COLORS["text"],
        bbox=note_bbox,
    )
    ax.text(
        note_x,
        0.0 + DISTRIBUTION_NOTE_POSITIONS["image_y_offset"],
        f"Image: n={image_n}, HDI excludes 0: {image_sig}",
        ha="right",
        va="center",
        fontsize=FONT_SIZES["note"],
        fontweight="bold",
        color=COLORS["text"],
        bbox=note_bbox,
    )

    legend_handles = [
        Line2D([0], [0], color=COLORS["rater"], lw=5.0, alpha=0.55, label="Rater distribution"),
        Line2D([0], [0], color=COLORS["image"], lw=5.0, alpha=0.55, label="Image distribution"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["axis"], markeredgecolor=COLORS["axis"], markersize=6.5, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["axis"], markeredgewidth=1.2, markersize=6.5, label="Mean"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        handlelength=1.8,
        labelspacing=0.45,
        borderaxespad=0.0,
    )


def clear_non_tick_text(ax: plt.Axes) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")


def save_png_and_pdf(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")


def remove_tick_numbers(ax: plt.Axes) -> None:
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def draw_textless_caterpillar(
    output_path: Path,
    effects: pd.DataFrame,
    color: str,
    hide_tick_numbers: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=TEXTLESS_FIGURE_SIZE, dpi=400)
    draw_caterpillar(ax, effects, color, "")
    clear_non_tick_text(ax)
    if hide_tick_numbers:
        remove_tick_numbers(ax)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.985, bottom=0.12)
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def draw_textless_distribution(
    output_path: Path,
    rater: pd.DataFrame,
    image: pd.DataFrame,
    hide_tick_numbers: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=TEXTLESS_FIGURE_SIZE, dpi=400)
    x_grid = compute_distribution_grid(rater, image)
    draw_distribution_row(ax, rater, COLORS["rater"], 1.0, x_grid)
    draw_distribution_row(ax, image, COLORS["image"], 0.0, x_grid)

    ax.axvline(0, color=COLORS["axis"], lw=1.25, zorder=1)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(format_signed_tick))
    bold_tick_labels(ax)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.75)
    ax.set_axisbelow(True)
    clear_non_tick_text(ax)
    if hide_tick_numbers:
        remove_tick_numbers(ax)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.985, bottom=0.12)
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def draw_textless_split_figures(rater: pd.DataFrame, image: pd.DataFrame) -> None:
    draw_textless_caterpillar(TEXTLESS_RATER_OUTPUT, rater, COLORS["rater"])
    draw_textless_caterpillar(TEXTLESS_IMAGE_OUTPUT, image, COLORS["image"])
    draw_textless_distribution(TEXTLESS_DISTRIBUTION_OUTPUT, rater, image)
    draw_textless_caterpillar(NOTICK_RATER_OUTPUT, rater, COLORS["rater"], hide_tick_numbers=True)
    draw_textless_caterpillar(NOTICK_IMAGE_OUTPUT, image, COLORS["image"], hide_tick_numbers=True)
    draw_textless_distribution(NOTICK_DISTRIBUTION_OUTPUT, rater, image, hide_tick_numbers=True)


def draw_combined_figure() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rater = load_effects("1|rater[")
    image = load_effects("1|image[")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE,
        dpi=400,
        gridspec_kw={"width_ratios": PANEL_WIDTH_RATIOS},
    )
    ax_rt, ax_it, ax_dist = axes

    draw_caterpillar(ax_rt, rater, COLORS["rater"], "Rater index")
    draw_caterpillar(ax_it, image, COLORS["image"], "Image index")
    draw_combined_distribution(ax_dist, rater, image)

    ax_rt.set_title("Rater Caterpillar", fontsize=FONT_SIZES["title"], fontweight="bold", pad=16)
    ax_it.set_title("Image Caterpillar", fontsize=FONT_SIZES["title"], fontweight="bold", pad=16)
    ax_dist.set_title("Random Effect Distributions", fontsize=FONT_SIZES["title"], fontweight="bold", pad=16)

    fig.subplots_adjust(**SUBPLOT_ADJUST)
    fig.savefig(PNG_OUTPUT, facecolor="white")
    plt.close(fig)
    draw_textless_split_figures(rater, image)


def main() -> None:
    draw_combined_figure()
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved textless figure: {TEXTLESS_RATER_OUTPUT}")
    print(f"Saved textless figure: {TEXTLESS_IMAGE_OUTPUT}")
    print(f"Saved textless figure: {TEXTLESS_DISTRIBUTION_OUTPUT}")
    print(f"Saved no-tick-number figure: {NOTICK_RATER_OUTPUT}")
    print(f"Saved no-tick-number figure: {NOTICK_IMAGE_OUTPUT}")
    print(f"Saved no-tick-number figure: {NOTICK_DISTRIBUTION_OUTPUT}")


if __name__ == "__main__":
    main()
