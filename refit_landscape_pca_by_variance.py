from argparse import ArgumentParser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
RAW_FEATURES_PATH = ROOT_DIR / "landscape_dinov2_features.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--target-variance", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.target_variance <= 1:
        raise ValueError("--target-variance must be in (0, 1].")
    if not RAW_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing raw feature file: {RAW_FEATURES_PATH}")

    df = pd.read_csv(RAW_FEATURES_PATH)
    emb_cols = [f"emb_{idx}" for idx in range(768)]
    df = df.dropna(subset=emb_cols).reset_index(drop=True)

    X = df[emb_cols].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    full_pca = PCA(random_state=42)
    full_pca.fit(X_scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= args.target_variance) + 1)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])
    X_pca = pipeline.fit_transform(X)

    csv_path = ROOT_DIR / f"landscape_PCA_{n_components}_dinov2.csv"
    pkl_path = ROOT_DIR / f"landscape_dinov2_pca_{n_components}d.pkl"
    plot_path = ROOT_DIR / f"landscape_DINOv2_PCA_{n_components}_Variance.png"

    df_pca = pd.DataFrame(X_pca, columns=[f"PC{idx}" for idx in range(1, n_components + 1)])
    df_pca["image_name"] = df["image_name"]
    df_pca.to_csv(csv_path, index=False, encoding="utf-8-sig")
    joblib.dump(pipeline, pkl_path)

    plt.figure(figsize=(12, 6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    ratios = full_pca.explained_variance_ratio_[:n_components]
    cum = cumulative_variance[:n_components]
    plt.bar(
        range(1, n_components + 1),
        ratios,
        alpha=0.6,
        color="#2a6f97",
        label="Individual PC Variance",
    )
    plt.plot(
        range(1, n_components + 1),
        cum,
        marker="o",
        linestyle="-",
        color="#bc4749",
        label="Cumulative Variance",
    )
    plt.axhline(args.target_variance, color="#6a994e", linestyle="--", label="Target Variance")
    plt.title(
        f"Landscape DINOv2 Explained Variance: 768D to {n_components}D",
        fontsize=16,
        weight="bold",
    )
    plt.xlabel("Principal Component (PC)", fontsize=14)
    plt.ylabel("Explained Variance Ratio", fontsize=14)
    plt.xticks(range(1, n_components + 1, max(1, n_components // 10)))
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    print(f"Target cumulative variance: {args.target_variance:.4f}")
    print(f"Selected n_components: {n_components}")
    print(f"Achieved cumulative variance: {cumulative_variance[n_components - 1]:.6f}")
    print(f"Saved PCA CSV to: {csv_path}")
    print(f"Saved PCA pipeline to: {pkl_path}")
    print(f"Saved variance plot to: {plot_path}")


if __name__ == "__main__":
    main()
