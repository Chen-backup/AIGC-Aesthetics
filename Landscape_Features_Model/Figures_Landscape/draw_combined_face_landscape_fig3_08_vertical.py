from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parents[1]
FACE_SAMPLES = ROOT_DIR / "Picture_code" / "picture" / "image_variance_explained_3models_boxline_samples.csv"
LANDSCAPE_SAMPLES = OUTPUT_DIR / "landscape_image_variance_explained_3models_samples.csv"

PNG_OUTPUT = OUTPUT_DIR / "Fig3_08_face_landscape_image_variance_vertical.png"
TEXTLESS_PNG_OUTPUT = OUTPUT_DIR / "Fig3_08_face_landscape_image_variance_vertical_textless.png"
TEXTLESS_PDF_OUTPUT = OUTPUT_DIR / "Fig3_08_face_landscape_image_variance_vertical_textless.pdf"
TEXTLESS_SVG_OUTPUT = OUTPUT_DIR / "Fig3_08_face_landscape_image_variance_vertical_textless.svg"
CSV_OUTPUT = OUTPUT_DIR / "Fig3_08_face_landscape_image_variance_vertical_data.csv"

MODEL_ORDER = ["InsightFace", "StyleGAN", "DINOv2", "Places365"]
GROUP_COLORS = {
    "Face": "#88A0CB",
    "Landscape": "#ECA7C0",
}
MODEL_SPACING = 0.35


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.6,
            "xtick.major.width": 2.6,
            "ytick.major.width": 2.6,
            "xtick.major.size": 5.8,
            "ytick.major.size": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_samples() -> pd.DataFrame:
    face = pd.read_csv(FACE_SAMPLES).assign(ImageGroup="Face")
    landscape = pd.read_csv(LANDSCAPE_SAMPLES).assign(ImageGroup="Landscape")
    samples = pd.concat([face, landscape], ignore_index=True)
    samples = samples.loc[samples["Model"].isin(MODEL_ORDER)].copy()
    samples["Model"] = pd.Categorical(samples["Model"], categories=MODEL_ORDER, ordered=True)
    return samples.sort_values(["Model", "ImageGroup"]).reset_index(drop=True)


def draw_vertical_comparison(samples: pd.DataFrame, *, textless: bool = False) -> None:
    rng = np.random.default_rng(20260601)
    model_positions = {model: index * MODEL_SPACING for index, model in enumerate(MODEL_ORDER)}

    fig = plt.figure(figsize=(9.0, 12.0), dpi=400)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.055)
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1], sharex=ax_top)
    axes = [ax_top, ax_bottom]
    legend_handles = []
    group_means: dict[str, list[tuple[float, float]]] = {"Face": [], "Landscape": []}
    for group in ["Face", "Landscape"]:
        color = GROUP_COLORS[group]
        legend_handles.append(mpl.patches.Patch(facecolor=color, edgecolor="#262626", alpha=0.45, label=group))

        for model in MODEL_ORDER:
            values = samples.loc[
                (samples["ImageGroup"] == group) & (samples["Model"] == model),
                "ExplainedVariance",
            ].to_numpy(dtype=float)
            if len(values) == 0:
                continue

            mean = float(np.mean(values))
            position = model_positions[model]
            group_means[group].append((position, mean))
            display_values = rng.choice(values, size=min(115, len(values)), replace=False)
            jitter = rng.normal(0, 0.040, size=len(display_values))

            for ax in axes:
                box = ax.boxplot(
                    [values],
                    positions=[position],
                    vert=True,
                    widths=0.25,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="#262626", linewidth=1.8),
                    whiskerprops=dict(color="#454545", linewidth=1.45),
                    capprops=dict(color="#454545", linewidth=1.45),
                    boxprops=dict(linewidth=1.45, color="#262626"),
                )
                box["boxes"][0].set_facecolor(color)
                box["boxes"][0].set_alpha(0.45)
                ax.scatter(
                    np.full(len(display_values), position) + jitter,
                    display_values,
                    s=14,
                    color=color,
                    edgecolor="#1F1F1F",
                    linewidth=0.30,
                    alpha=0.72,
                    zorder=3,
                )
                ax.scatter(
                    [position],
                    [mean],
                    s=42,
                    color="#262626",
                    edgecolor="white",
                    linewidth=0.65,
                    zorder=6,
                )

    for group, points in group_means.items():
        points = sorted(points)
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        for ax in axes:
            ax.plot(x_values, y_values, color=GROUP_COLORS[group], linewidth=2.0, marker="o", markersize=5.2, zorder=5)

    ax_top.set_ylim(4.20, 6.00)
    # Keep the visible break conceptually at 3.1 while leaving enough room
    # for the Places365 upper whisker and posterior points to render fully.
    ax_bottom.set_ylim(2.65, 3.16)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.set_xticks([model_positions[model] for model in MODEL_ORDER])
    ax_bottom.set_xticklabels(MODEL_ORDER, fontsize=15, fontweight="bold")

    break_size = 0.012
    axis_linewidth = 3.6 if textless else 2.6
    kwargs = dict(color="#262626", clip_on=False, linewidth=axis_linewidth)
    ax_top.plot((-break_size, +break_size), (-break_size, +break_size), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), transform=ax_bottom.transAxes, **kwargs)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(axis="y", labelsize=15)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", color="#E6EAF0", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.20 if textless else -0.42, model_positions[MODEL_ORDER[-1]] + 0.42)

    if textless:
        for ax in axes:
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_yticklabels([])
        ax_bottom.set_xticklabels([])
    else:
        ax_top.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            fontsize=14,
        )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.14)
    fig.savefig(TEXTLESS_PNG_OUTPUT if textless else PNG_OUTPUT, facecolor="white")
    if textless:
        fig.savefig(TEXTLESS_PDF_OUTPUT, facecolor="white")
        fig.savefig(TEXTLESS_SVG_OUTPUT, facecolor="white")
    plt.close(fig)


def main() -> None:
    set_style()
    samples = load_samples()
    samples.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    draw_vertical_comparison(samples)
    draw_vertical_comparison(samples, textless=True)
    print(samples.groupby(["Model", "ImageGroup"], observed=True)["ExplainedVariance"].mean())
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved textless figure: {TEXTLESS_PNG_OUTPUT}")
    print(f"Saved textless PDF: {TEXTLESS_PDF_OUTPUT}")
    print(f"Saved textless SVG: {TEXTLESS_SVG_OUTPUT}")


if __name__ == "__main__":
    main()
