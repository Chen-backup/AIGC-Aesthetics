from __future__ import annotations

import os
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TEST_MODE = False


def run_landscape_stylegan_model_10() -> None:
    print("================ Stage 1: Load PCA-10 StyleGAN features and merge tables ================")
    base_dir = Path(__file__).resolve().parent
    ratings_path = base_dir.parent / "ratings_for_bayesian_model.csv"
    features_path = base_dir / "PCA_10_stylegan_w.csv"
    mapping_path = base_dir.parent / "landscape_number_mapping.csv"
    save_dir = base_dir / "Result"

    df_ratings = pd.read_csv(ratings_path)
    df_features = pd.read_csv(features_path)
    df_mapping = pd.read_csv(mapping_path)

    df_feat_mapped = pd.merge(df_features, df_mapping[["image_name", "image"]], on="image_name", how="inner")
    df = pd.merge(df_ratings, df_feat_mapped, on="image", how="inner")
    df = df.dropna().copy()
    print(f"Merged data ready: {len(df)} valid rating records.")

    if TEST_MODE:
        print("\nTest mode enabled: sampling 1000 rows for a quick validation run.")
        df = df.sample(n=1000, random_state=42)
        tune_steps, draw_steps = 500, 500
    else:
        print("\nFull mode enabled: using all landscape ratings.")
        tune_steps, draw_steps = 2000, 1000

    print("\n================ Stage 2: Re-standardize the 10 PCs ================")
    pc_features = [f"PC{i + 1}" for i in range(10)]

    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)

    for feat in pc_features:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ Stage 3: Run Bayesian cumulative model ================")
    formula = f"rating ~ 1 + {' + '.join(pc_features)} + (1|rater) + (1|image)"
    print(f"Model formula:\n{formula}\n")

    model = bmb.Model(formula, data=df, family="cumulative")
    results = model.fit(
        draws=draw_steps,
        tune=tune_steps,
        chains=1,
        cores=1,
        target_accept=0.95,
    )
    print("\nMCMC sampling finished.")

    print("\n================ Stage 4: Compute model metrics ================")
    summary = az.summary(results)

    sigma_image = summary.loc["1|image_sigma", "mean"]
    sigma_rater = summary.loc["1|rater_sigma", "mean"]
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    null_metrics_path = base_dir.parent / "BYS_Null_Model" / "BYS_landscape_null_result" / "Null_Model_Metrics_Report.txt"
    null_var_image = None
    if null_metrics_path.exists():
        text = null_metrics_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "Image Variance" in line or "sigma^2" in line:
                try:
                    null_var_image = float(line.split(":")[-1].strip().split()[0])
                    break
                except Exception:
                    continue

    marginal_r2_str = "Not computed"
    if null_var_image and null_var_image > 0:
        marginal_r2 = (null_var_image - var_image) / null_var_image
        marginal_r2_str = f"{marginal_r2:.2%}"

    metrics_text = (
        "========== Landscape StyleGAN PCA-10 Model Metrics ==========\n\n"
        "1. Variance Components:\n"
        f"   - Residual Image Variance (sigma^2): {var_image:.4f}\n"
        f"   - Rater Variance (sigma^2): {var_rater:.4f}\n\n"
        "2. Deep Feature Explanatory Power:\n"
        f"   - Landscape null-model image variance: {null_var_image if null_var_image is not None else 'Not found'}\n"
        f"   - StyleGAN PCA-10 marginal R^2: {marginal_r2_str}\n"
    )
    print("\n" + metrics_text)

    print("\n================ Stage 5: Save outputs and figures ================")
    save_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_dir / "StyleGAN_10D_Model_Summary.csv")
    with open(save_dir / "StyleGAN_10D_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)
    results.to_netcdf(save_dir / "StyleGAN_10d_model_trace.nc")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    sns.set_context("paper", font_scale=1.2)

    feat_summary = az.summary(results, var_names=pc_features)
    feat_summary_sorted = feat_summary.sort_values(by="mean", ascending=True)
    sorted_features = feat_summary_sorted.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    means = feat_summary_sorted["mean"]
    hdi_lower = feat_summary_sorted["hdi_3%"]
    hdi_upper = feat_summary_sorted["hdi_97%"]
    y_pos = np.arange(len(sorted_features))

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - hdi_lower, hdi_upper - means],
        fmt="o",
        color="#9b59b6",
        ecolor="#9b59b6",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(x=0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Standardized Effect Size (Posterior Mean with 95% HDI)", weight="bold")
    ax.set_title("Impact of 10 StyleGAN-W+ PCs on Landscape Aesthetic Ratings", weight="bold", pad=15)
    sns.despine()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "StyleGAN_10D_Fig1_Forest_Plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    trace_dir = save_dir / "TracePlots_Diagnostics"
    trace_dir.mkdir(parents=True, exist_ok=True)
    vars_to_plot = pc_features + ["1|rater_sigma", "1|image_sigma"]
    for var in vars_to_plot:
        axes_trace = az.plot_trace(results, var_names=[var])
        if isinstance(axes_trace, np.ndarray):
            for row in axes_trace:
                for ax in row:
                    ax.set_yticklabels([])
        else:
            for ax in axes_trace:
                ax.set_yticklabels([])
        plt.suptitle(f"Trace Plot: {var}", fontsize=16, weight="bold", y=1.05)
        plt.tight_layout()
        safe_var_name = var.replace("|", "_")
        plt.savefig(trace_dir / f"Trace_{safe_var_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nAll tasks finished. Result folder: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    run_landscape_stylegan_model_10()
