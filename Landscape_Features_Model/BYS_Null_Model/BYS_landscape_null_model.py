import os
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# True: use a 1000-row sample for a quick test run.
# False: use the full landscape ratings dataset.
TEST_MODE = False


def run_landscape_null_model() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "ratings_for_bayesian_model.csv"
    save_dir = base_dir / "BYS_landscape_null_result"

    print("================ Stage 1: Load data ================")
    df = pd.read_csv(data_path)

    if TEST_MODE:
        print("\nTest mode enabled: sampling 1000 rows for a faster run.")
        df = df.sample(n=1000, random_state=42)
        tune_steps = 500
        draw_steps = 500
    else:
        print("\nFull mode enabled: using the entire landscape ratings dataset.")
        tune_steps = 2000
        draw_steps = 1000

    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)

    print(f"Loaded {len(df)} rating records.\n")

    print("================ Stage 2: Fit null model ================")
    model = bmb.Model("rating ~ 1 + (1|rater) + (1|image)", data=df, family="cumulative")
    results = model.fit(
        draws=draw_steps,
        tune=tune_steps,
        chains=1,
        cores=1,
        target_accept=0.95,
    )
    print("\nMCMC sampling finished.")

    print("\n================ Stage 3: Compute metrics ================")
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
        print("Skipping WAIC/LOO and keeping ICC as the main baseline metric.")

    sigma_image = summary.loc["1|image_sigma", "mean"]
    sigma_rater = summary.loc["1|rater_sigma", "mean"]
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2
    icc_image = var_image / (var_image + var_rater)

    metrics_text = (
        "========== Landscape Null Model Baseline Metrics ==========\n\n"
        "1. Variance Components:\n"
        f"   - Image Variance (sigma^2): {var_image:.4f}\n"
        f"   - Rater Variance (sigma^2): {var_rater:.4f}\n\n"
        "2. Explanatory Baseline:\n"
        f"   - ICC_image: {icc_image:.2%}\n\n"
        "3. Model Comparison Metrics:\n"
        f"   - WAIC: {waic_str}\n"
        f"   - LOO:  {loo_str}\n"
    )
    print("\n" + metrics_text)

    print("\n================ Stage 4: Save outputs ================")
    save_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_dir / "Null_Model_Summary.csv")
    with open(save_dir / "Null_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)
    results.to_netcdf(save_dir / "null_model_trace.nc")
    print("Saved summary, metrics report, and trace file.")

    print("\n================ Stage 5: Make figures ================")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks", {"font.family": "serif", "font.serif": ["Times New Roman"]})

    print("Generating Fig 1: trace plot...")
    axes_trace = az.plot_trace(results, var_names=["1|rater_sigma", "1|image_sigma"])
    if isinstance(axes_trace, np.ndarray):
        for row in axes_trace:
            for ax in row:
                ax.set_yticklabels([])
    else:
        for ax in axes_trace:
            ax.set_yticklabels([])
    plt.suptitle("Trace Plots for Global Variances (Landscape Null Model)", fontsize=16, weight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / "Null_Fig1_TracePlot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating Fig 2: image effects forest plot...")
    axes_img = az.plot_forest(results, var_names=["1|image"], combined=True, hdi_prob=0.95, figsize=(8, 12))
    if isinstance(axes_img, np.ndarray):
        for ax in axes_img.flatten():
            ax.set_yticklabels([])
            ax.set_ylabel("")
    else:
        axes_img.set_yticklabels([])
        axes_img.set_ylabel("")
    plt.title("Intrinsic Landscape Aesthetic Scores Distribution", fontsize=16, weight="bold")
    plt.axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(save_dir / "Null_Fig2_ImageScores_Clean.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating Fig 3: rater effects forest plot...")
    axes_rater = az.plot_forest(results, var_names=["1|rater"], combined=True, hdi_prob=0.95, figsize=(8, 15))
    if isinstance(axes_rater, np.ndarray):
        for ax in axes_rater.flatten():
            ax.set_yticklabels([])
            ax.set_ylabel("")
    else:
        axes_rater.set_yticklabels([])
        axes_rater.set_ylabel("")
    plt.title("Landscape Rater Strictness/Leniency Distribution", fontsize=16, weight="bold")
    plt.axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(save_dir / "Null_Fig3_RaterScores_Clean.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nAll tasks finished. Output folder: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    run_landscape_null_model()
