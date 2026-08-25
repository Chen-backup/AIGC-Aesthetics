from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def process_landscape_dinov2_pca_10() -> None:
    print("================ 1. Load 768D landscape DINOv2 features ================")
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "landscape_dinov2_features.csv"
    output_csv = base_dir / "PCA_10_dinov2.csv"
    output_plot = base_dir / "DINOv2_PCA_10_Variance.png"
    output_pkl = base_dir / "landscape_dinov2_pca_10d.pkl"

    df = pd.read_csv(input_csv)
    emb_cols = [f"emb_{i}" for i in range(768)]
    initial_len = len(df)
    df = df.dropna(subset=emb_cols)
    print(f"Loaded {len(df)} images (dropped {initial_len - len(df)} invalid rows).")

    print("\n================ 2. Build StandardScaler + PCA pipeline ================")
    X = df[emb_cols].values
    n_components = 10
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )

    X_pca = pipeline.fit_transform(X)
    pca_model = pipeline.named_steps["pca"]
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"PCA-10 finished. Cumulative explained variance: {cumulative_variance[-1]:.2%}")

    print("\n================ 3. Save PCA variance plot ================")
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.bar(
        range(1, n_components + 1),
        explained_variance_ratio,
        alpha=0.6,
        color="#ff7f0e",
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
    plt.title("Landscape DINOv2 Explained Variance: 768D to 10D", fontsize=16, weight="bold")
    plt.xlabel("Principal Component (PC)", fontsize=14)
    plt.ylabel("Explained Variance Ratio", fontsize=14)
    plt.xticks(range(1, n_components + 1))
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print("\n================ 4. Save PCA features and PCA tool ================")
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
    df_pca["image_name"] = df["image_name"].reset_index(drop=True)
    df_pca.to_csv(output_csv, index=False, encoding="utf-8-sig")
    joblib.dump(pipeline, output_pkl)

    print(f"Saved PCA features to: {output_csv}")
    print(f"Saved PCA pipeline to: {output_pkl}")
    print(f"Saved PCA variance plot to: {output_plot}")


if __name__ == "__main__":
    process_landscape_dinov2_pca_10()
