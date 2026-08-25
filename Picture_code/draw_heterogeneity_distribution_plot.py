from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"
OUTPUT_DIR = ROOT_DIR / "BYS_Heterogeneity_Evidence"
OUTPUT_PATH = OUTPUT_DIR / "Plot1_Heterogeneity_Distribution.png"
PDF_OUTPUT_PATH = OUTPUT_DIR / "Plot1_Heterogeneity_Distribution.pdf"

EXCLUDE_COLUMNS = {"Cluster", "tSNE_1", "tSNE_2", "Final_Cluster"}


def clean_label(feature: str) -> str:
    return feature.replace("_", " ")


def load_preferences() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, index_col=0)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    return df[feature_cols].astype(float)


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_distribution_plot(df_raw: pd.DataFrame) -> None:
    stats = pd.DataFrame(
        {
            "feature": df_raw.columns,
            "mean": df_raw.mean(axis=0).to_numpy(),
            "sd": df_raw.std(axis=0).to_numpy(),
        }
    ).sort_values("sd", ascending=False)

    ordered_features = stats["feature"].tolist()
    all_values = df_raw[ordered_features].to_numpy(dtype=float).ravel()
    x_min = float(np.quantile(all_values, 0.002))
    x_max = float(np.quantile(all_values, 0.998))
    pad = (x_max - x_min) * 0.08
    x_min -= pad
    x_max += pad
    x_grid = np.linspace(x_min, x_max, 700)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=400)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = LinearSegmentedColormap.from_list("heterogeneity", ["#4E79A7", "#F2C14E", "#D1495B"])
    norm = Normalize(vmin=float(stats["sd"].min()), vmax=float(stats["sd"].max()))

    row_height = 0.43
    n_features = len(ordered_features)

    for row_idx, feature in enumerate(ordered_features):
        y = n_features - 1 - row_idx
        values = df_raw[feature].dropna().to_numpy(dtype=float)
        kde = gaussian_kde(values)
        density = kde(x_grid)
        density = density / max(float(density.max()), 1e-12) * row_height
        visible = density > row_height * 0.015

        color = cmap(norm(float(stats.loc[stats["feature"] == feature, "sd"].iloc[0])))
        ax.fill_between(
            x_grid[visible],
            y - density[visible],
            y + density[visible],
            color=color,
            alpha=0.72,
            linewidth=0,
            zorder=2,
        )
        ax.plot(x_grid[visible], y + density[visible], color=color, linewidth=1.05, zorder=3)
        ax.plot(x_grid[visible], y - density[visible], color=color, linewidth=1.05, zorder=3)

        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        mean = float(np.mean(values))
        ax.hlines(y, q25, q75, color="#252525", linewidth=2.2, zorder=4)
        ax.scatter([median], [y], s=18, color="#252525", zorder=5)
        ax.scatter([mean], [y], s=24, facecolor="white", edgecolor="#252525", linewidth=0.8, zorder=5)

    ax.axvline(0, color="#7A7A7A", linestyle=(0, (4, 4)), linewidth=1.25, zorder=1)
    ax.grid(axis="x", color="#E5E8EC", linewidth=0.75)
    ax.set_axisbelow(True)

    y_positions = list(range(n_features - 1, -1, -1))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([clean_label(feature) for feature in ordered_features], fontsize=10.5)

    x_span = x_max - x_min
    label_x = x_max + x_span * 0.035
    for row_idx, row in stats.reset_index(drop=True).iterrows():
        y = n_features - 1 - row_idx
        ax.text(
            label_x,
            y,
            f"SD = {row['sd']:.3f}",
            va="center",
            ha="left",
            fontsize=9.2,
            color="#353535",
        )

    ax.set_xlim(x_min, x_max + x_span * 0.20)
    ax.set_ylim(-0.75, n_features - 0.25)
    ax.set_xlabel("Individual preference slope (BLUP)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Heterogeneity of Individual Aesthetic Preferences", fontsize=15, fontweight="bold", pad=30)
    ax.text(
        0.5,
        1.005,
        "Feature distributions are ordered by between-rater standard deviation",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.3,
        color="#4A4A4A",
    )

    legend_y = -0.095
    ax.scatter([], [], s=22, color="#252525", label="Median")
    ax.scatter([], [], s=28, facecolor="white", edgecolor="#252525", linewidth=0.8, label="Mean")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        frameon=False,
        fontsize=9.6,
        handletextpad=0.4,
        columnspacing=1.8,
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.032)
    cbar.set_label("Between-rater SD", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5, width=0.8, length=3)
    cbar.outline.set_linewidth(0.6)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_raw = load_preferences()
    draw_distribution_plot(df_raw)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
