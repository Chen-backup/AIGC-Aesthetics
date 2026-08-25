import os
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TEST_MODE = False

SELECTED_FEATURES = [
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


def run_full_model() -> None:
    base_dir = Path(__file__).resolve().parent
    ratings_path = base_dir.parent / "ratings_for_bayesian_model.csv"
    features_path = base_dir / "landscape_interpretable_features.csv"
    mapping_path = base_dir.parent / "landscape_number_mapping.csv"
    save_dir = base_dir / "Result"

    print("================ Stage 1: Load and merge data ================")
    print("Reading ratings, landscape features, and image mapping...")
    df_ratings = pd.read_csv(ratings_path)
    df_features = pd.read_csv(features_path)
    df_mapping = pd.read_csv(mapping_path)

    print("Merging the three tables...")
    df_feat_mapped = pd.merge(
        df_features,
        df_mapping[["image_name", "image"]],
        on="image_name",
        how="inner",
    )
    df = pd.merge(df_ratings, df_feat_mapped, on="image", how="inner")

    initial_len = len(df)
    df = df.dropna(subset=SELECTED_FEATURES + ["rating", "rater", "image"]).copy()
    print(
        f"Merged data ready: {len(df)} valid rating records "
        f"(dropped {initial_len - len(df)} rows with missing values)."
    )

    if TEST_MODE:
        print("\nTest mode enabled: sampling 1000 rows for a quick validation run.")
        df = df.sample(n=1000, random_state=42)
        tune_steps = 500
        draw_steps = 500
    else:
        print("\nFull mode enabled: using all landscape rating records.")
        tune_steps = 2000
        draw_steps = 1000

    print("\n================ Stage 2: Clean and standardize features ================")
    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)

    print("Applying Z-score standardization to the selected 10 features...")
    for feat in SELECTED_FEATURES:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ Stage 3: Run Bayesian cumulative model ================")
    formula = f"rating ~ 1 + {' + '.join(SELECTED_FEATURES)} + (1|rater) + (1|image)"
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

    print("Trying to compute WAIC and LOO...")
    waic_str = "Not available for this cumulative model"
    loo_str = "Not available for this cumulative model"
    try:
        import pymc as pm

        if not hasattr(results, "log_likelihood"):
            pm.compute_log_likelihood(results, model=model.backend.model)
        waic_data = az.waic(results)
        loo_data = az.loo(results)
        waic_str = f"{waic_data.waic:.2f} (SE: {waic_data.waic_se:.2f})"
        loo_str = f"{loo_data.loo:.2f} (SE: {loo_data.loo_se:.2f})"
        print("WAIC and LOO computed successfully.")
    except Exception:
        print("Skipping WAIC/LOO because they are unstable for this cumulative model setup.")

    sigma_image = summary.loc["1|image_sigma", "mean"]
    sigma_rater = summary.loc["1|rater_sigma", "mean"]
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    metrics_text = (
        "========== Landscape Interpretable Feature Model Metrics ==========\n\n"
        "1. Variance Components (residual variance after adding features):\n"
        f"   - Residual Image Variance (sigma^2): {var_image:.4f}\n"
        f"   - Rater Variance (sigma^2): {var_rater:.4f}\n\n"
        "2. Model Comparison Metrics:\n"
        f"   - WAIC: {waic_str}\n"
        f"   - LOO:  {loo_str}\n"
    )
    print("\n" + metrics_text)

    print("\n================ Stage 5: Save outputs ================")
    save_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_dir / "Full_Model_Summary.csv")
    with open(save_dir / "Full_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)
    results.to_netcdf(save_dir / "full_model_trace.nc")
    print("Saved summary, metrics report, and trace file.")

    print("\n================ Stage 6: Generate figures ================")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks", {"font.family": "serif", "font.serif": ["Times New Roman"]})

    feat_summary = az.summary(results, var_names=SELECTED_FEATURES)
    feat_summary_sorted = feat_summary.sort_values(by="mean", ascending=True)
    sorted_features = feat_summary_sorted.index.tolist()

    print("Generating Fig 1: feature effect forest plot...")
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
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=2,
        capsize=4,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
    )
    ax.axvline(x=0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Standardized Effect Size (Posterior Mean with 95% HDI)", weight="bold")
    ax.set_title("Impact of Landscape Interpretable Features on Aesthetic Ratings", weight="bold", pad=15)
    sns.despine()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "Full_Fig1_Forest_Plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating Fig 2: posterior ridge plot...")
    az.plot_forest(
        results,
        var_names=SELECTED_FEATURES,
        kind="ridgeplot",
        combined=True,
        ridgeplot_alpha=0.6,
        ridgeplot_overlap=1.2,
        colors="#2ca02c",
        figsize=(10, 10),
    )
    plt.title("Posterior Density Distributions of Landscape Features", fontsize=18, weight="bold")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Parameter Value", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "Full_Fig2_Ridge_Plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating Fig 3: top 4 posterior distributions...")
    feat_summary["abs_mean"] = feat_summary["mean"].abs()
    top_4_features = feat_summary.sort_values(by="abs_mean", ascending=False).head(4).index.tolist()
    az.plot_posterior(results, var_names=top_4_features, hdi_prob=0.95, color="#8c564b", figsize=(12, 8), textsize=12)
    plt.suptitle("Detailed Posterior Distributions for Top 4 Influential Landscape Features", fontsize=18, weight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / "Full_Fig3_Top4_Posterior.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating trace plots for convergence diagnostics...")
    trace_dir = save_dir / "TracePlots_Diagnostics"
    trace_dir.mkdir(parents=True, exist_ok=True)

    vars_to_plot = SELECTED_FEATURES + ["1|rater_sigma", "1|image_sigma"]
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

    print(f"\nAll tasks finished. Output folder: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    run_full_model()
