from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


FEATURE_ORDER = [
    "warm_cool_balance",
    "horizon_y_norm",
    "depth_gradient_mean",
    "artificial_ratio",
    "saturation_mean",
    "thirds_brightness_mean",
    "line_strength",
    "depth_std",
    "semantic_diversity",
    "left_right_balance",
]

LABELS = {
    "warm_cool_balance": "Warm-cool balance",
    "horizon_y_norm": "Horizon position",
    "depth_gradient_mean": "Depth gradient",
    "artificial_ratio": "Artificial ratio",
    "saturation_mean": "Saturation",
    "thirds_brightness_mean": "Thirds brightness",
    "line_strength": "Line strength",
    "depth_std": "Depth variation",
    "semantic_diversity": "Semantic diversity",
    "left_right_balance": "Left-right balance",
}


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_half_violin(ax: plt.Axes, values: np.ndarray, center_y: float, color: str, x_grid: np.ndarray) -> None:
    kde = gaussian_kde(values)
    density = kde(x_grid)
    density = density / max(float(density.max()), 1e-12) * 0.34
    visible = density > 0.003
    ax.fill_between(x_grid[visible], center_y, center_y + density[visible], color=color, alpha=0.34, linewidth=0)
    ax.plot(x_grid[visible], center_y + density[visible], color=color, linewidth=1.3)


def draw_figures() -> None:
    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "Result"
    summary_path = result_dir / "Ultimate_Heterogeneity_Leaderboard.csv"
    preferences_path = result_dir / "Rater_10D_Preferences.csv"

    summary = pd.read_csv(summary_path).sort_values("heterogeneity_mean", ascending=False).reset_index(drop=True)
    preferences = pd.read_csv(preferences_path).set_index("rater")
    _set_style()

    print("================ Figure 1: heterogeneity forest plot ================")
    forest = summary.sort_values("heterogeneity_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(7.3, 5.5), dpi=400)
    ax.errorbar(
        forest["heterogeneity_mean"],
        y,
        xerr=[
            forest["heterogeneity_mean"] - forest["heterogeneity_hdi_3"],
            forest["heterogeneity_hdi_97"] - forest["heterogeneity_mean"],
        ],
        fmt="none",
        ecolor="#555555",
        elinewidth=1.4,
        capsize=3.2,
    )
    colors = mpl.colormaps["YlOrRd"](
        mpl.colors.Normalize(forest["heterogeneity_mean"].min(), forest["heterogeneity_mean"].max())(
            forest["heterogeneity_mean"]
        )
    )
    ax.scatter(forest["heterogeneity_mean"], y, s=46, c=colors, edgecolor="#222222", linewidth=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(f, f) for f in forest["feature"]], fontsize=8.8)
    ax.set_xlabel("Between-rater heterogeneity SD (posterior mean with 94% HDI)", fontsize=9.6)
    ax.set_title("Population Disagreement Across Landscape Features", fontsize=11.8, fontweight="bold")
    ax.grid(axis="x", color="#E7E9ED", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(result_dir / "Figure1_Landscape_Heterogeneity_Forest.png", bbox_inches="tight")
    plt.close(fig)

    print("================ Figure 2: top-5 rater preference spectra ================")
    top_features = summary.head(min(5, len(summary)))["feature"].tolist()
    all_values = preferences[top_features].to_numpy(dtype=float).reshape(-1)
    x_min, x_max = np.quantile(all_values, [0.005, 0.995])
    pad = (x_max - x_min) * 0.14
    x_grid = np.linspace(x_min - pad, x_max + pad, 640)
    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=400)
    rng = np.random.default_rng(20260601)
    palette = ["#C8553D", "#D9822B", "#E3B341", "#5E8C61", "#4E79A7"]
    for idx, feature in enumerate(top_features):
        y_pos = len(top_features) - 1 - idx
        values = preferences[feature].dropna().to_numpy(dtype=float)
        _draw_half_violin(ax, values, y_pos, palette[idx], x_grid)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.hlines(y_pos - 0.08, q1, q3, color="#222222", lw=2.0)
        ax.scatter([median], [y_pos - 0.08], s=23, color="#222222")
        sample = rng.choice(values, size=min(240, len(values)), replace=False)
        jitter = rng.uniform(-0.28, -0.02, size=len(sample))
        ax.scatter(sample, y_pos + jitter, s=8, color=palette[idx], edgecolor="#222222", linewidth=0.2, alpha=0.55)
    ax.axvline(0, color="#777777", linestyle=(0, (4, 4)), linewidth=1.1)
    ax.set_yticks(range(len(top_features) - 1, -1, -1))
    ax.set_yticklabels([LABELS.get(f, f) for f in top_features], fontsize=8.8)
    ax.set_xlabel("Rater-specific preference slope (posterior mean)", fontsize=9.6)
    ax.set_title("Individual Preference Spectra for Top Landscape Features", fontsize=11.8, fontweight="bold")
    ax.grid(axis="x", color="#E7E9ED", linewidth=0.8)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markersize=5, label="Median")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.1)
    fig.tight_layout()
    fig.savefig(result_dir / "Figure2_Top5_Landscape_Rater_Preference_Spectra.png", bbox_inches="tight")
    plt.close(fig)

    print("================ Figure 3: effect versus heterogeneity ================")
    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=400)
    ax.axvline(0, color="#777777", linestyle=(0, (4, 4)), linewidth=1.0)
    ax.axhline(summary["heterogeneity_mean"].median(), color="#777777", linestyle=(0, (4, 4)), linewidth=1.0)
    ax.errorbar(
        summary["fixed_mean"],
        summary["heterogeneity_mean"],
        xerr=[summary["fixed_mean"] - summary["fixed_hdi_3"], summary["fixed_hdi_97"] - summary["fixed_mean"]],
        yerr=[
            summary["heterogeneity_mean"] - summary["heterogeneity_hdi_3"],
            summary["heterogeneity_hdi_97"] - summary["heterogeneity_mean"],
        ],
        fmt="none",
        ecolor="#777777",
        elinewidth=0.95,
        capsize=2.2,
    )
    ax.scatter(summary["fixed_mean"], summary["heterogeneity_mean"], s=60, color="#D9822B", edgecolor="#222222")
    for _, row in summary.iterrows():
        ax.annotate(LABELS.get(row["feature"], row["feature"]), (row["fixed_mean"], row["heterogeneity_mean"]), xytext=(6, 5), textcoords="offset points", fontsize=7.2)
    ax.set_xlabel("Population-level effect on aesthetic rating", fontsize=9.6)
    ax.set_ylabel("Between-rater heterogeneity SD", fontsize=9.6)
    ax.set_title("Average Effect Versus Individual Disagreement", fontsize=11.8, fontweight="bold")
    ax.grid(color="#E7E9ED", linewidth=0.75)
    fig.tight_layout()
    fig.savefig(result_dir / "Figure3_Landscape_Effect_vs_Heterogeneity.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to: {result_dir}")


if __name__ == "__main__":
    draw_figures()
