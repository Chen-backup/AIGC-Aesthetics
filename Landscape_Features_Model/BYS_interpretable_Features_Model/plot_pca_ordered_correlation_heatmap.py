from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
FEATURES_CSV = BASE_DIR / "landscape_interpretable_features.csv"
RANKING_CSV = BASE_DIR / "PCA_landscape_feature_importance_ranking.csv"
OUTPUT_CORR_CSV = BASE_DIR / "PCA_ordered_feature_correlation_matrix.csv"
OUTPUT_HEATMAP_PNG = BASE_DIR / "PCA_ordered_feature_correlation_heatmap.png"


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    ranking_df = pd.read_csv(RANKING_CSV)

    ranking_col = "Unnamed: 0" if "Unnamed: 0" in ranking_df.columns else ranking_df.columns[0]
    ordered_features = ranking_df[ranking_col].astype(str).tolist()

    available_features = [col for col in ordered_features if col in df.columns]
    corr = df[available_features].corr(numeric_only=True)

    corr.to_csv(OUTPUT_CORR_CSV, encoding="utf-8-sig")

    plt.figure(figsize=(20, 16))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    sns.set_theme(style="white")

    ax = sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.15,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap Ordered by PCA Feature Importance", fontsize=18, weight="bold", pad=16)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ordered correlation matrix to: {OUTPUT_CORR_CSV}")
    print(f"Saved heatmap to: {OUTPUT_HEATMAP_PNG}")
    print("Top 15 ordered features:")
    for idx, name in enumerate(available_features[:15], start=1):
        print(f"{idx}. {name}")


if __name__ == "__main__":
    main()
