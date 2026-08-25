from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
RAW_FEATURES_PATH = ROOT_DIR / "landscape_dinov2_features.csv"
OUTPUT_CSV_PATH = ROOT_DIR / "landscape_PCA_14_dinov2.csv"
BACKUP_CSV_PATH = ROOT_DIR / "landscape_PCA_14_dinov2_face_pca_projection_backup.csv"
OUTPUT_PLOT_PATH = ROOT_DIR / "landscape_DINOv2_PCA_14_Variance.png"
OUTPUT_PKL_PATH = ROOT_DIR / "landscape_dinov2_pca_14d.pkl"
N_COMPONENTS = 14


def main() -> None:
    if not RAW_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing raw feature file: {RAW_FEATURES_PATH}")

    df = pd.read_csv(RAW_FEATURES_PATH)
    emb_cols = [f"emb_{idx}" for idx in range(768)]
    df = df.dropna(subset=emb_cols).reset_index(drop=True)

    if OUTPUT_CSV_PATH.exists() and not BACKUP_CSV_PATH.exists():
        shutil.copy2(OUTPUT_CSV_PATH, BACKUP_CSV_PATH)
        print(f"Backed up previous PCA CSV to: {BACKUP_CSV_PATH}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS, random_state=42)),
    ])

    X_pca = pipeline.fit_transform(df[emb_cols].to_numpy())
    pca = pipeline.named_steps["pca"]
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    df_pca = pd.DataFrame(X_pca, columns=[f"PC{idx}" for idx in range(1, N_COMPONENTS + 1)])
    df_pca["image_name"] = df["image_name"]
    df_pca.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    joblib.dump(pipeline, OUTPUT_PKL_PATH)

    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.bar(
        range(1, N_COMPONENTS + 1),
        explained_variance_ratio,
        alpha=0.6,
        color="#2a6f97",
        label="Individual PC Variance",
    )
    plt.plot(
        range(1, N_COMPONENTS + 1),
        cumulative_variance,
        marker="o",
        linestyle="-",
        color="#bc4749",
        label="Cumulative Variance",
    )
    plt.title("Landscape DINOv2 Explained Variance: 768D to 14D", fontsize=16, weight="bold")
    plt.xlabel("Principal Component (PC)", fontsize=14)
    plt.ylabel("Explained Variance Ratio", fontsize=14)
    plt.xticks(range(1, N_COMPONENTS + 1))
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)

    print(f"Saved refit landscape PCA CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved landscape PCA pipeline to: {OUTPUT_PKL_PATH}")
    print(f"Saved variance plot to: {OUTPUT_PLOT_PATH}")
    print(f"Cumulative explained variance (first 14 PCs): {cumulative_variance[-1]:.6f}")


if __name__ == "__main__":
    main()
