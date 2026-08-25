from __future__ import annotations

from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd


TEST_MODE = False

NONLINEAR_FEATURES = [
    "horizon_y_norm",
    "saturation_mean",
    "artificial_ratio",
    "semantic_diversity",
]


def _coerce_summary_value(value: object, name: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not parse numeric summary value for {name}: {value!r}")
    return float(numeric)


def _save_inference_data(results: az.InferenceData, output_path: Path) -> Path:
    try:
        results.to_netcdf(output_path)
        return output_path
    except Exception as exc:
        fallback_path = output_path.with_suffix(".pkl")
        print(
            f"Could not save NetCDF trace because {exc}. "
            f"Falling back to pickle: {fallback_path}"
        )
        pd.to_pickle(results, fallback_path)
        return fallback_path


def run_landscape_gamm_nonlinear_model() -> None:
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    ratings_path = project_dir / "ratings_for_bayesian_model.csv"
    mapping_path = project_dir / "landscape_number_mapping.csv"
    features_path = project_dir / "BYS_interpretable_Features_Model" / "landscape_interpretable_features.csv"
    save_dir = base_dir / "Result"

    print("================ Stage 1: Load and merge landscape data ================")
    df_ratings = pd.read_csv(ratings_path)
    df_mapping = pd.read_csv(mapping_path)
    df_features = pd.read_csv(features_path)

    df_feat_mapped = pd.merge(df_features, df_mapping[["image_name", "image"]], on="image_name", how="inner")
    df = pd.merge(df_ratings, df_feat_mapped, on="image", how="inner")

    initial_len = len(df)
    df = df.dropna(subset=NONLINEAR_FEATURES + ["rating", "rater", "image"]).copy()
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
        print("\nFull mode enabled: fitting the nonlinear landscape GAMM with all ratings.")
        tune_steps = 2000
        draw_steps = 1000

    print("\n================ Stage 2: Clean and standardize 4 nonlinear features ================")
    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)

    feature_stats_rows = []
    for feat in NONLINEAR_FEATURES:
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        z_min = ((df[feat] - mean_val) / std_val).min()
        z_max = ((df[feat] - mean_val) / std_val).max()
        df[feat] = (df[feat] - mean_val) / std_val
        feature_stats_rows.append(
            {
                "feature": feat,
                "mean": mean_val,
                "std": std_val,
                "z_min": z_min,
                "z_max": z_max,
            }
        )

    print("\n================ Stage 3: Fit Bayesian GAMM nonlinear model ================")
    spline_terms = [f"bs({feat}, df=4)" for feat in NONLINEAR_FEATURES]
    formula = f"rating ~ 1 + {' + '.join(spline_terms)} + (1|rater) + (1|image)"
    print(f"Model formula:\n{formula}\n")

    model = bmb.Model(formula, data=df, family="cumulative")
    results = model.fit(
        draws=draw_steps,
        tune=tune_steps,
        chains=1,
        cores=1,
        target_accept=0.95,
        init="adapt_diag",
    )
    print("\nMCMC sampling finished.")

    print("\n================ Stage 4: Save summary and metrics ================")
    summary = az.summary(results)
    sigma_image = _coerce_summary_value(summary.loc["1|image_sigma", "mean"], "1|image_sigma")
    sigma_rater = _coerce_summary_value(summary.loc["1|rater_sigma", "mean"], "1|rater_sigma")
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    null_metrics_path = project_dir / "BYS_Null_Model" / "BYS_landscape_null_result" / "Null_Model_Metrics_Report.txt"
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
        "========== Landscape GAMM Nonlinear Model Metrics ==========\n\n"
        f"Nonlinear features: {', '.join(NONLINEAR_FEATURES)}\n\n"
        "1. Variance Components:\n"
        f"   - Residual Image Variance (sigma^2): {var_image:.4f}\n"
        f"   - Rater Variance (sigma^2): {var_rater:.4f}\n\n"
        "2. Nonlinear Explanatory Power:\n"
        f"   - Landscape null-model image variance: {null_var_image if null_var_image is not None else 'Not found'}\n"
        f"   - GAMM nonlinear marginal R^2: {marginal_r2_str}\n"
    )
    print("\n" + metrics_text)

    save_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_dir / "GAMM_Model_Summary.csv")
    with open(save_dir / "GAMM_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)
    trace_path = _save_inference_data(results, save_dir / "GAMM_model_trace_4features.nc")
    pd.DataFrame(feature_stats_rows).to_csv(save_dir / "GAMM_feature_standardization_stats.csv", index=False)

    print(f"Saved model trace to: {trace_path}")
    print(f"\nAll training outputs saved to: {save_dir}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    run_landscape_gamm_nonlinear_model()
