from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Picture_code" / "picture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_PATH = ROOT / "PCA_14_dinov2.csv"
FEATURE_PATH = ROOT / "interpretable_face_features.csv"

OUTPUT_PNG = OUT_DIR / "DINOv2_PC_Interpretable_Bridge_Heatmap.png"
OUTPUT_DATA = OUT_DIR / "DINOv2_PC_Interpretable_Bridge_Heatmap_Data.csv"
OUTPUT_TOP_LINKS = OUT_DIR / "DINOv2_PC_Interpretable_Bridge_Heatmap_TopLinks.csv"

PCS = [f"PC{i}" for i in range(1, 15)]

FEATURES = [
    ("face_hw_ratio", "Face H/W ratio"),
    ("eye_face_w_ratio", "Eye-face width"),
    ("mouth_face_w_ratio", "Mouth-face width"),
    ("three_courts_balance", "Three-courts balance"),
    ("upper_lower_ratio", "Upper/lower ratio"),
    ("eye_y_ratio", "Eye vertical position"),
    ("total_symmetry", "Facial symmetry"),
    ("le_nose_re_angle", "Eye-nose angle"),
    ("mouth_nose_ratio", "Mouth-nose ratio"),
    ("face_brightness", "Brightness"),
    ("face_contrast", "Contrast"),
    ("face_clarity", "Clarity"),
    ("saturation", "Saturation"),
    ("edge_density", "Edge density"),
]


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.clip(adjusted, 0, 1)
    return q_values


def compute_correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, label in FEATURES:
        for pc in PCS:
            pair = data[[feature, pc]].dropna()
            if len(pair) < 4 or pair[feature].nunique() < 2 or pair[pc].nunique() < 2:
                r_value, p_value = np.nan, np.nan
            else:
                r_value, p_value = pearsonr(pair[feature], pair[pc])
            rows.append(
                {
                    "feature": feature,
                    "feature_label": label,
                    "PC": pc,
                    "r": r_value,
                    "p": p_value,
                    "n": len(pair),
                }
            )

    corr = pd.DataFrame(rows)
    valid = corr["p"].notna()
    corr.loc[valid, "q_fdr"] = benjamini_hochberg(corr.loc[valid, "p"].to_numpy())
    corr.loc[~valid, "q_fdr"] = np.nan
    corr["abs_r"] = corr["r"].abs()
    return corr


def load_data() -> pd.DataFrame:
    pca = pd.read_csv(PCA_PATH)
    features = pd.read_csv(FEATURE_PATH)

    missing_pcs = [pc for pc in PCS if pc not in pca.columns]
    missing_features = [feature for feature, _ in FEATURES if feature not in features.columns]
    if missing_pcs:
        raise ValueError(f"Missing PC columns in {PCA_PATH}: {missing_pcs}")
    if missing_features:
        raise ValueError(f"Missing feature columns in {FEATURE_PATH}: {missing_features}")

    data = pca[["image_name", *PCS]].merge(
        features[["image_name", *[feature for feature, _ in FEATURES]]],
        on="image_name",
        how="inner",
    )
    if data.empty:
        raise ValueError("No matched image_name records between DINOv2 PCA and interpretable features.")
    return data


def draw_heatmap(corr: pd.DataFrame) -> None:
    rcParams["font.family"] = "Times New Roman"
    rcParams["axes.unicode_minus"] = False

    max_abs = (
        corr.groupby(["feature", "feature_label"], as_index=False)["abs_r"]
        .max()
        .sort_values("abs_r", ascending=True)
    )
    ordered_labels = max_abs["feature_label"].tolist()

    heat = corr.pivot(index="feature_label", columns="PC", values="r").loc[ordered_labels, PCS]
    q = corr.pivot(index="feature_label", columns="PC", values="q_fdr").loc[ordered_labels, PCS]
    abs_heat = heat.abs()

    fig = plt.figure(figsize=(9.4, 6.2), dpi=450)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 0.18, 0.045],
        left=0.19,
        right=0.96,
        bottom=0.16,
        top=0.88,
        wspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    cax = fig.add_subplot(gs[0, 2])

    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(PCS)))
    ax.set_xticklabels(PCS, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(ordered_labels)))
    ax.set_yticklabels(ordered_labels, fontsize=9.2)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, len(PCS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Mark robust bridge links while keeping the heatmap clean enough for print.
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if q.iloc[i, j] < 0.05 and abs_heat.iloc[i, j] >= 0.30:
                color = "white" if abs_heat.iloc[i, j] >= 0.55 else "#1B1B1B"
                ax.scatter(j, i, s=10, c=color, edgecolors="none", zorder=3)

    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label("Pearson r", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8)

    bar_values = max_abs.set_index("feature_label").loc[ordered_labels, "abs_r"]
    y = np.arange(len(ordered_labels))
    ax_bar.barh(y, bar_values, height=0.58, color="#323B4B", alpha=0.88)
    ax_bar.set_xlim(0, max(0.35, bar_values.max() * 1.15))
    ax_bar.set_xlabel("max |r|", fontsize=8.8, labelpad=5)
    ax_bar.tick_params(axis="x", labelsize=8, length=3, width=0.6)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)

    for yi, value in zip(y, bar_values):
        ax_bar.text(value + 0.01, yi, f"{value:.2f}", va="center", ha="left", fontsize=7.6)

    fig.suptitle(
        "Bridge Between DINOv2 Principal Components and Interpretable Facial Features",
        x=0.19,
        y=0.965,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.19,
        0.915,
        "Dots mark FDR-adjusted significant correlations with |r| >= 0.30; the side bar summarizes the strongest bridge for each feature.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#4B5563",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")

    fig.savefig(OUTPUT_PNG, dpi=450)
    plt.close(fig)


def main() -> None:
    data = load_data()
    corr = compute_correlations(data)
    corr.to_csv(OUTPUT_DATA, index=False, encoding="utf-8-sig")
    (
        corr.sort_values("abs_r", ascending=False)
        .head(30)
        .to_csv(OUTPUT_TOP_LINKS, index=False, encoding="utf-8-sig")
    )
    draw_heatmap(corr)
    print(f"Saved figure: {OUTPUT_PNG}")
    print(f"Saved data: {OUTPUT_DATA}")
    print(f"Saved top links: {OUTPUT_TOP_LINKS}")


if __name__ == "__main__":
    main()
