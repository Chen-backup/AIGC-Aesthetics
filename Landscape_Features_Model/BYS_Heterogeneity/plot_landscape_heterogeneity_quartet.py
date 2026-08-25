from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_quartet() -> None:
    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "Result"
    preferences = pd.read_csv(result_dir / "Rater_10D_Preferences.csv").set_index("rater")
    preferences = preferences.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    values = preferences.to_numpy(dtype=float)

    print("================ Plot 1: parallel coordinates ================")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=400)
    x = np.arange(preferences.shape[1])
    for row in values:
        ax.plot(x, row, color="#365F7D", alpha=0.045, lw=0.75)
    ax.fill_between(x, preferences.quantile(0.05), preferences.quantile(0.95), color="#B9CEDD", alpha=0.3)
    ax.plot(x, preferences.median(), color="#202020", lw=1.7, marker="o", markersize=3.3)
    ax.axhline(0, color="#A23B3B", linestyle="--", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(preferences.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Rater-specific preference slope")
    ax.set_title("Parallel Coordinates of Landscape Aesthetic Preferences", fontweight="bold")
    fig.tight_layout()
    fig.savefig(result_dir / "Quartet1_Parallel_Coordinates.png", bbox_inches="tight")
    plt.close(fig)

    print("================ Plot 2: PCA-sorted heatmap ================")
    scaled = StandardScaler().fit_transform(values)
    pc1 = PCA(n_components=1).fit_transform(scaled).reshape(-1)
    row_order = np.argsort(pc1)[::-1]
    loading_model = PCA(n_components=1).fit(scaled)
    column_order = np.argsort(loading_model.components_[0])[::-1]
    ordered = pd.DataFrame(scaled[row_order][:, column_order], columns=preferences.columns[column_order])
    fig, ax = plt.subplots(figsize=(10, 9), dpi=350)
    sns.heatmap(ordered, cmap="RdBu_r", center=0, vmin=-3, vmax=3, yticklabels=False, ax=ax, cbar_kws={"label": "Z-scored preference strength"})
    ax.set_xlabel("Landscape features sorted by PC1 loading")
    ax.set_ylabel(f"{len(ordered)} raters sorted by PC1 score")
    ax.set_title("PCA-Sorted Preference Heatmap", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(result_dir / "Quartet2_PCA_Sorted_Heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print("================ Plot 3: PCA density map ================")
    pca_2d = PCA(n_components=2)
    xy = pca_2d.fit_transform(values)
    density = gaussian_kde(xy.T)(xy.T)
    order = np.argsort(density)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=350)
    scatter = ax.scatter(xy[order, 0], xy[order, 1], c=density[order], s=34, cmap="magma", edgecolor="none", alpha=0.82)
    fig.colorbar(scatter, ax=ax, label="Local population density")
    ax.set_xlabel(f"Principal component 1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"Principal component 2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Landscape Preference Density Map", fontweight="bold")
    fig.tight_layout()
    fig.savefig(result_dir / "Quartet3_PCA_Density_Map.png", bbox_inches="tight")
    plt.close(fig)

    print("================ Plot 4: random-slope forest plot ================")
    stats = pd.DataFrame({"mean": preferences.mean(), "sd": preferences.std()}).sort_values("sd")
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=350)
    y = np.arange(len(stats))
    ax.errorbar(stats["mean"], y, xerr=1.96 * stats["sd"], fmt="o", color="#2C3E50", ecolor="#E74C3C", capsize=4)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(stats.index, fontsize=8.5)
    ax.set_xlabel("Individual preference slope (mean +/- 1.96 SD)")
    ax.set_title("Landscape Rater Preference Heterogeneity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(result_dir / "Quartet4_Random_Slope_Forest.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Quartet figures saved to: {result_dir}")


if __name__ == "__main__":
    plot_quartet()
