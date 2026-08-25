from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"
OUTPUT_DIR = ROOT_DIR / "BYS_Heterogeneity_Evidence"
PNG_OUTPUT = OUTPUT_DIR / "Plot1_Heterogeneity_Violin_ReferenceStyle.png"
PDF_OUTPUT = OUTPUT_DIR / "Plot1_Heterogeneity_Violin_ReferenceStyle.pdf"

EXCLUDE_COLUMNS = {"Cluster", "tSNE_1", "tSNE_2", "Final_Cluster"}
RNG_SEED = 20260506

DISPLAY_LABELS = {
    "upper_lower_ratio": "Upper/lower",
    "face_hw_ratio": "Face H/W",
    "mouth_nose_ratio": "Mouth/nose",
    "eye_face_w_ratio": "Eye/face W",
    "eye_y_ratio": "Eye Y",
    "edge_density": "Edge density",
    "mouth_face_w_ratio": "Mouth/face W",
    "face_clarity": "Clarity",
    "face_brightness": "Brightness",
    "le_nose_re_angle": "Nose angle",
    "face_contrast": "Contrast",
    "saturation": "Saturation",
    "total_symmetry": "Symmetry",
    "three_courts_balance": "Three-courts",
}


def load_preferences() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, index_col=0)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    return df[feature_cols].astype(float)


def clean_label(feature: str) -> str:
    return DISPLAY_LABELS.get(feature, feature.replace("_", " "))


def short_sd_label(sd: float) -> str:
    return f"SD={sd:.3f}"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sd_to_color(sd: float, sd_min: float, sd_max: float) -> tuple[float, float, float, float]:
    # Cool-to-warm scale, chosen to echo the blue/gold reference figure while
    # mapping color to disagreement magnitude rather than to artificial groups.
    cmap = mpl.colors.LinearSegmentedColormap.from_list("sd_scale", ["#2E86A6", "#E6B85C", "#B47A2B"])
    normed = (sd - sd_min) / max(sd_max - sd_min, 1e-12)
    return cmap(normed)


def draw_top_bracket(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str) -> None:
    left = x - width / 2
    right = x + width / 2
    ax.plot([left, left, right, right], [y, y + height, y + height, y], color="#262626", lw=0.8, clip_on=False)
    ax.text(x, y + height + 0.012, text, ha="center", va="bottom", fontsize=6.7, color="#222222")


def draw_violin_figure(df_raw: pd.DataFrame) -> None:
    set_style()
    stats = pd.DataFrame(
        {
            "feature": df_raw.columns,
            "mean": df_raw.mean(axis=0).to_numpy(),
            "sd": df_raw.std(axis=0).to_numpy(),
        }
    ).sort_values("sd", ascending=False)

    ordered_features = stats["feature"].tolist()
    n_features = len(ordered_features)
    all_values = df_raw[ordered_features].to_numpy(dtype=float).ravel()
    y_min = float(np.quantile(all_values, 0.003))
    y_max = float(np.quantile(all_values, 0.997))
    y_pad = (y_max - y_min) * 0.16
    ylim = (y_min - y_pad * 0.35, y_max + y_pad * 1.35)
    y_grid = np.linspace(ylim[0], ylim[1], 600)

    fig, ax = plt.subplots(figsize=(11.2, 7.0), dpi=450)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rng = np.random.default_rng(RNG_SEED)
    sd_min = float(stats["sd"].min())
    sd_max = float(stats["sd"].max())
    box_width = 0.23
    violin_max_width = 0.58

    for pos, feature in enumerate(ordered_features, start=1):
        values = df_raw[feature].dropna().to_numpy(dtype=float)
        sd = float(stats.loc[stats["feature"] == feature, "sd"].iloc[0])
        color = sd_to_color(sd, sd_min, sd_max)

        kde = gaussian_kde(values)
        density = kde(y_grid)
        density = density / max(float(density.max()), 1e-12) * violin_max_width / 2
        visible = density > violin_max_width * 0.004

        ax.fill_betweenx(
            y_grid[visible],
            pos - density[visible],
            pos + density[visible],
            facecolor=color,
            edgecolor="none",
            alpha=0.42,
            zorder=1,
        )
        ax.plot(pos - density[visible], y_grid[visible], color=color, lw=1.15, zorder=2)
        ax.plot(pos + density[visible], y_grid[visible], color=color, lw=1.15, zorder=2)

        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        lower = max(float(np.min(values)), q1 - 1.5 * iqr)
        upper = min(float(np.max(values)), q3 + 1.5 * iqr)

        rect = Rectangle(
            (pos - box_width / 2, q1),
            box_width,
            q3 - q1,
            facecolor=(*color[:3], 0.72),
            edgecolor="#303030",
            linewidth=0.8,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.plot([pos - box_width / 2, pos + box_width / 2], [median, median], color="#1F1F1F", lw=1.1, zorder=5)
        ax.plot([pos, pos], [lower, q1], color="#303030", lw=0.8, zorder=4)
        ax.plot([pos, pos], [q3, upper], color="#303030", lw=0.8, zorder=4)
        ax.plot([pos - box_width * 0.32, pos + box_width * 0.32], [lower, lower], color="#303030", lw=0.8, zorder=4)
        ax.plot([pos - box_width * 0.32, pos + box_width * 0.32], [upper, upper], color="#303030", lw=0.8, zorder=4)

        # Plot a deterministic subset of observations to keep the figure readable.
        sample_size = min(140, len(values))
        sample_idx = rng.choice(len(values), size=sample_size, replace=False)
        sample_values = values[sample_idx]
        jitter = rng.normal(0, 0.055, size=sample_size)
        ax.scatter(
            pos + jitter,
            sample_values,
            s=9,
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.74,
            zorder=3,
        )

        draw_top_bracket(ax, pos, y_max + y_pad * 0.22, 0.42, y_pad * 0.08, short_sd_label(sd))

    for boundary in np.arange(1.5, n_features + 0.5, 1.0):
        ax.axvline(boundary, color="#8A8A8A", lw=0.7, ls=(0, (4, 4)), alpha=0.75, zorder=0)

    ax.axhline(0, color="#C23B3B", lw=0.85, ls=(0, (4, 3)), alpha=0.85, zorder=0)
    ax.text(0.58, 0.01, "0", ha="right", va="bottom", fontsize=8.2, color="#C23B3B")

    ax.set_xlim(0.5, n_features + 0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(1, n_features + 1))
    ax.set_xticklabels([clean_label(feature) for feature in ordered_features], rotation=35, ha="right", fontsize=8.2)
    ax.set_ylabel("Individual preference slope (BLUP)", fontsize=10.0)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=8.6)
    ax.tick_params(axis="x", length=3)

    ax.set_title("Between-rater Heterogeneity in Aesthetic Preferences", fontsize=12.0, fontweight="bold", pad=10)
    ax.text(
        0.0,
        1.035,
        "a",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    high_patch = Patch(facecolor=sd_to_color(sd_max, sd_min, sd_max), edgecolor="#303030", label="Higher disagreement")
    low_patch = Patch(facecolor=sd_to_color(sd_min, sd_min, sd_max), edgecolor="#303030", label="Lower disagreement")
    ax.legend(
        handles=[high_patch, low_patch],
        loc="upper right",
        frameon=False,
        fontsize=7.8,
        handlelength=1.1,
        borderaxespad=0.4,
    )

    fig.tight_layout(pad=1.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUTPUT, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_raw = load_preferences()
    draw_violin_figure(df_raw)
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved figure: {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
