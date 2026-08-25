from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"
OUTPUT_DIR = ROOT_DIR / "BYS_Heterogeneity_Evidence"
PNG_OUTPUT = OUTPUT_DIR / "Plot2_PCA_Sorted_Heatmap_Publication.png"
PDF_OUTPUT = OUTPUT_DIR / "Plot2_PCA_Sorted_Heatmap_Publication.pdf"

EXCLUDE_COLUMNS = {"Cluster", "tSNE_1", "tSNE_2", "Final_Cluster"}

DISPLAY_LABELS = {
    "upper_lower_ratio": "Upper/lower",
    "face_hw_ratio": "Face H/W",
    "mouth_nose_ratio": "Mouth/nose",
    "eye_face_w_ratio": "Eye/face W",
    "mouth_face_w_ratio": "Mouth/face W",
    "eye_y_ratio": "Eye Y",
    "edge_density": "Edge density",
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


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def short_label(feature: str) -> str:
    return DISPLAY_LABELS.get(feature, feature.replace("_", " "))


def build_sorted_pca_matrix(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray, float]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_raw)
    df_scaled = pd.DataFrame(scaled, index=df_raw.index, columns=df_raw.columns)

    pca = PCA(n_components=1, random_state=0)
    pc1_scores = pca.fit_transform(df_scaled).reshape(-1)
    loadings = pd.Series(pca.components_[0], index=df_raw.columns)

    # Fix the arbitrary PCA sign so that the right side of the heatmap tends to
    # represent larger positive loadings for interpretable facial proportions.
    if loadings[["upper_lower_ratio", "face_hw_ratio", "mouth_nose_ratio"]].mean() < 0:
        pc1_scores = -pc1_scores
        loadings = -loadings

    sorted_features = loadings.sort_values(ascending=False).index.tolist()
    df_scaled["PC1 score"] = pc1_scores
    df_sorted = df_scaled.sort_values("PC1 score", ascending=False).drop(columns="PC1 score")
    return df_sorted[sorted_features], loadings[sorted_features], pc1_scores, float(pca.explained_variance_ratio_[0])


def draw_heatmap() -> None:
    set_style()
    df_raw = load_preferences()
    df_sorted, loadings, pc1_scores, pc1_var = build_sorted_pca_matrix(df_raw)

    matrix = np.clip(df_sorted.to_numpy(dtype=float), -3.0, 3.0)
    sorted_scores = np.sort(pc1_scores)[::-1]
    score_strip = sorted_scores.reshape(-1, 1)

    fig = plt.figure(figsize=(8.4, 7.4), dpi=450)
    gs = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        width_ratios=[28, 0.75, 1.05],
        height_ratios=[2.0, 13.5, 0.35],
        wspace=0.07,
        hspace=0.05,
    )

    ax_load = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_score = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    cmap_heat = mpl.colormaps["RdBu_r"]
    cmap_score = mpl.colormaps["viridis"]
    norm_heat = mpl.colors.TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)

    im = ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap_heat, norm=norm_heat, rasterized=True)
    ax_heat.set_xticks(np.arange(df_sorted.shape[1]))
    ax_heat.set_xticklabels([short_label(feature) for feature in df_sorted.columns], rotation=42, ha="right", fontsize=7.7)
    ax_heat.set_yticks([])
    ax_heat.set_ylabel(f"{len(df_sorted)} raters ordered by PC1 score", fontsize=9.5)
    ax_heat.tick_params(axis="x", length=0, pad=2)
    for spine in ax_heat.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#333333")

    for boundary in np.arange(0.5, df_sorted.shape[1] - 0.5, 1.0):
        ax_heat.axvline(boundary, color="white", lw=0.35, alpha=0.75)

    loading_colors = np.where(loadings.to_numpy() >= 0, "#C55353", "#4C78A8")
    ax_load.bar(np.arange(len(loadings)), loadings.to_numpy(), color=loading_colors, width=0.74, edgecolor="none")
    ax_load.axhline(0, color="#333333", lw=0.75)
    ax_load.set_xlim(-0.5, len(loadings) - 0.5)
    y_abs = max(abs(float(loadings.min())), abs(float(loadings.max())))
    ax_load.set_ylim(-y_abs * 1.25, y_abs * 1.25)
    ax_load.set_xticks([])
    ax_load.set_ylabel("PC1\nloading", fontsize=8.5, labelpad=4)
    ax_load.tick_params(axis="y", labelsize=7.2, length=2)
    ax_load.spines["top"].set_visible(False)
    ax_load.spines["right"].set_visible(False)
    ax_load.spines["bottom"].set_visible(False)

    ax_score.imshow(score_strip, aspect="auto", interpolation="nearest", cmap=cmap_score, rasterized=True)
    ax_score.set_xticks([])
    ax_score.set_yticks([])
    ax_score.set_title("PC1\nscore", fontsize=7.6, pad=4)
    for spine in ax_score.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#333333")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Z-scored\npreference", fontsize=8.2)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.tick_params(labelsize=7.5, width=0.65, length=2.5)
    cbar.outline.set_linewidth(0.65)

    fig.suptitle("PCA-ordered continuum of individual aesthetic preferences", fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.946,
        f"Rows and columns are sorted by PC1; PC1 explains {pc1_var * 100:.1f}% of between-rater preference variation",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4A4A4A",
    )
    fig.text(0.015, 0.982, "b", fontsize=12.5, fontweight="bold", va="top", ha="left")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUTPUT, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_heatmap()
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved figure: {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
