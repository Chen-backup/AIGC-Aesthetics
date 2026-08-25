from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT_CSV = Path(__file__).resolve().parent / "landscape_interpretable_features.csv"
OUTPUT_PCA_CSV = Path(__file__).resolve().parent / "PCA_landscape_interpretable_features.csv"
OUTPUT_LOADINGS_CSV = Path(__file__).resolve().parent / "PCA_landscape_feature_loadings.csv"
OUTPUT_RANKING_CSV = Path(__file__).resolve().parent / "PCA_landscape_feature_importance_ranking.csv"
OUTPUT_VARIANCE_PNG = Path(__file__).resolve().parent / "PCA_landscape_explained_variance.png"

# Features to drop before PCA because they are redundant or currently low-information.
MANUAL_DROP_COLUMNS = {
    "brightness_std",   # almost identical to contrast in the current feature set
    "depth_high_ratio", # nearly constant due to the current quantile split logic
}

# Correlation threshold for automatically removing redundant features.
AUTO_DROP_CORR_THRESHOLD = 0.98

# Fixed component count for a compact interpretable representation.
N_COMPONENTS = 10


def find_redundant_features(df: pd.DataFrame, threshold: float) -> list[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        if high_corr:
            to_drop.add(col)
    return sorted(to_drop)


def main() -> None:
    print("================ 1. Load landscape interpretable features ================")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} images from {INPUT_CSV.name}")

    feature_cols = [col for col in df.columns if col != "image_name"]
    X = df[feature_cols].copy()

    print("\n================ 2. Remove redundant features ================")
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    auto_drop_cols = find_redundant_features(X.drop(columns=constant_cols, errors="ignore"), AUTO_DROP_CORR_THRESHOLD)

    drop_cols = sorted(set(constant_cols) | MANUAL_DROP_COLUMNS | set(auto_drop_cols))
    keep_cols = [col for col in X.columns if col not in drop_cols]

    print(f"Constant columns removed: {constant_cols}")
    print(f"Manual columns removed: {sorted(MANUAL_DROP_COLUMNS)}")
    print(f"Auto high-correlation columns removed: {auto_drop_cols}")
    print(f"Remaining features for PCA: {len(keep_cols)}")

    X = X[keep_cols]

    print("\n================ 3. Standardize features ================")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    print("\n================ 4. Run PCA ================")
    n_components = min(N_COMPONENTS, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"PCA finished with {n_components} components.")
    print(f"Cumulative explained variance: {cumulative_variance[-1]:.2%}")

    print("\n================ 5. Save PCA features and loadings ================")
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
    df_pca["image_name"] = df["image_name"].values
    df_pca.to_csv(OUTPUT_PCA_CSV, index=False, encoding="utf-8-sig")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=keep_cols,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings.to_csv(OUTPUT_LOADINGS_CSV, encoding="utf-8-sig")

    importance = pd.DataFrame(index=keep_cols)
    importance["mean_abs_loading"] = np.mean(np.abs(pca.components_.T), axis=1)
    importance["max_abs_loading"] = np.max(np.abs(pca.components_.T), axis=1)
    importance["weighted_abs_loading"] = np.sum(
        np.abs(pca.components_.T) * explained_variance_ratio.reshape(1, -1),
        axis=1,
    )
    importance = importance.sort_values("weighted_abs_loading", ascending=False)
    importance.to_csv(OUTPUT_RANKING_CSV, encoding="utf-8-sig")

    print(f"Saved PCA features to: {OUTPUT_PCA_CSV.name}")
    print(f"Saved loadings to: {OUTPUT_LOADINGS_CSV.name}")
    print(f"Saved ranking to: {OUTPUT_RANKING_CSV.name}")

    print("\nTop 10 features by weighted absolute loading:")
    print(importance.head(10).to_string())

    print("\n================ 6. Save explained variance plot ================")
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.bar(
        range(1, n_components + 1),
        explained_variance_ratio,
        alpha=0.6,
        color="#1f77b4",
        label="Individual PC Variance",
    )
    plt.plot(
        range(1, n_components + 1),
        cumulative_variance,
        marker="o",
        linestyle="-",
        color="#d62728",
        label="Cumulative Variance",
    )
    plt.title("Landscape Interpretable Feature PCA", fontsize=16, weight="bold")
    plt.xlabel("Principal Component (PC)", fontsize=14)
    plt.ylabel("Explained Variance Ratio", fontsize=14)
    plt.xticks(range(1, n_components + 1))
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_VARIANCE_PNG, dpi=300)
    plt.close()

    print(f"Saved explained variance plot to: {OUTPUT_VARIANCE_PNG.name}")


if __name__ == "__main__":
    main()
