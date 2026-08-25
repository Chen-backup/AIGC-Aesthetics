from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


TEST_MODE = False
BLACK_BOX_MODEL = "Places365"

# Set these values when running multiple devices without command-line arguments.
# Use None to fit every feature and "Result" to write the final combined output.
DEFAULT_SELECTED_FEATURES = [
    "warm_cool_balance",
    "horizon_y_norm",
    "depth_gradient_mean",
    "artificial_ratio",
]
DEFAULT_OUTPUT_DIR = "Result_device1"

INTERPRETABLE_FEATURES = [
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

PC_FEATURES = [f"PC{i}" for i in range(1, 11)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit landscape rater-heterogeneity models.")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=INTERPRETABLE_FEATURES,
        help="Subset of interpretable features to fit. Default: all features.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory relative to this script, or an absolute path. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def _save_inference_data(results: az.InferenceData, output_path: Path) -> Path:
    try:
        results.to_netcdf(output_path)
        return output_path
    except Exception as exc:
        fallback_path = output_path.with_suffix(".pkl")
        print(f"Could not save NetCDF trace because {exc}. Falling back to pickle: {fallback_path}")
        pd.to_pickle(results, fallback_path)
        return fallback_path


def _posterior_summary(results: az.InferenceData, variable: str) -> tuple[float, float, float]:
    if variable not in results.posterior:
        raise KeyError(f"Posterior variable not found: {variable}")
    draws = np.asarray(results.posterior[variable].values, dtype=float).reshape(-1)
    low, high = np.quantile(draws, [0.03, 0.97])
    return float(draws.mean()), float(low), float(high)


def _extract_rater_slope_means(results: az.InferenceData, feature: str) -> pd.DataFrame:
    variable = f"{feature}|rater"
    if variable not in results.posterior:
        raise KeyError(f"Posterior random-slope variable not found: {variable}")

    posterior = results.posterior[variable]
    slope_means = np.asarray(posterior.mean(dim=("chain", "draw")).values, dtype=float).reshape(-1)
    factor_dims = [dim for dim in posterior.dims if dim not in {"chain", "draw"}]
    if not factor_dims:
        raise ValueError(f"Could not find rater coordinate dimension for {variable}")

    rater_ids = np.asarray(posterior.coords[factor_dims[-1]].values).reshape(-1)
    if len(rater_ids) != len(slope_means):
        raise ValueError(f"Rater coordinate mismatch for {variable}: {len(rater_ids)} vs {len(slope_means)}")

    return pd.DataFrame({"rater": rater_ids.astype(str), feature: slope_means})


def _load_data(project_dir: Path) -> pd.DataFrame:
    ratings = pd.read_csv(project_dir / "ratings_for_bayesian_model.csv")
    mapping = pd.read_csv(project_dir / "landscape_number_mapping.csv")
    interpretable = pd.read_csv(
        project_dir / "BYS_interpretable_Features_Model" / "landscape_interpretable_features.csv"
    )
    places365 = pd.read_csv(project_dir / "BYS_Places365_Features_Model" / "PCA_10_places365.csv")

    mapped_interpretable = pd.merge(interpretable, mapping[["image_name", "image"]], on="image_name", how="inner")
    image_features = pd.merge(mapped_interpretable, places365, on="image_name", how="inner")
    df = pd.merge(ratings, image_features, on="image", how="inner")

    required = INTERPRETABLE_FEATURES + PC_FEATURES + ["rating", "rater", "image"]
    return df.dropna(subset=required).copy()


def run_landscape_heterogeneity(
    selected_features: list[str] | None = None,
    output_dir: str = "Result",
) -> None:
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    result_dir = Path(output_dir)
    if not result_dir.is_absolute():
        result_dir = base_dir / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    requested_features = selected_features or DEFAULT_SELECTED_FEATURES or INTERPRETABLE_FEATURES

    print(f"================ Stage 1: Load landscape ratings and {BLACK_BOX_MODEL} features ================")
    df = _load_data(project_dir)
    print(f"Merged data ready: {len(df)} ratings, {df['rater'].nunique()} raters, {df['image'].nunique()} images.")

    if TEST_MODE:
        print("\nTest mode enabled: fitting one feature on a 1000-row sample.")
        df = df.sample(n=1000, random_state=42)
        target_features = requested_features[:1]
        tune_steps, draw_steps = 500, 500
    else:
        print("\nFull mode enabled: fitting one random-slope model per interpretable feature.")
        target_features = requested_features
        tune_steps, draw_steps = 1500, 1000
    print(f"Selected features: {', '.join(target_features)}")
    print(f"Output directory: {result_dir}")

    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)

    print(f"\n================ Stage 2: Standardize 10 interpretable + 10 {BLACK_BOX_MODEL} PC features ================")
    for feature in INTERPRETABLE_FEATURES + PC_FEATURES:
        std = df[feature].std()
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Feature cannot be standardized because its standard deviation is invalid: {feature}")
        df[feature] = (df[feature] - df[feature].mean()) / std

    fixed_effects = " + ".join(INTERPRETABLE_FEATURES + PC_FEATURES)
    leaderboard_rows: list[dict[str, float | str]] = []
    preference_df: pd.DataFrame | None = None

    print("\n================ Stage 3: Fit random-slope models ================")
    for index, target_feature in enumerate(target_features, start=1):
        loop_start = time.time()
        print(f"\n[{index}/{len(target_features)}] Fitting rater heterogeneity for: {target_feature}")
        formula = (
            f"rating ~ 1 + {fixed_effects} + (1|rater) + (1|image) "
            f"+ (0+{target_feature}|rater)"
        )
        print(f"Formula:\n{formula}\n")

        model = bmb.Model(formula, data=df, family="cumulative")
        results = model.fit(
            draws=draw_steps,
            tune=tune_steps,
            chains=1,
            cores=1,
            target_accept=0.95,
            init="adapt_diag",
        )

        trace_path = _save_inference_data(
            results,
            result_dir / f"Ultimate_Heterogeneity_{target_feature}.nc",
        )
        print(f"Saved trace to: {trace_path}")

        sigma_name = f"{target_feature}|rater_sigma"
        heterogeneity_mean, heterogeneity_low, heterogeneity_high = _posterior_summary(results, sigma_name)
        fixed_mean, fixed_low, fixed_high = _posterior_summary(results, target_feature)
        leaderboard_rows.append(
            {
                "feature": target_feature,
                "heterogeneity_mean": heterogeneity_mean,
                "heterogeneity_hdi_3": heterogeneity_low,
                "heterogeneity_hdi_97": heterogeneity_high,
                "fixed_mean": fixed_mean,
                "fixed_hdi_3": fixed_low,
                "fixed_hdi_97": fixed_high,
            }
        )

        slope_df = _extract_rater_slope_means(results, target_feature)
        preference_df = slope_df if preference_df is None else pd.merge(preference_df, slope_df, on="rater", how="outer")

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values("heterogeneity_mean", ascending=False)
        leaderboard.to_csv(result_dir / "Ultimate_Heterogeneity_Leaderboard.csv", index=False)
        if preference_df is not None:
            preference_df.to_csv(result_dir / "Rater_10D_Preferences.csv", index=False)

        print(
            f"Heterogeneity SD: {heterogeneity_mean:.4f} "
            f"[94% HDI: {heterogeneity_low:.4f}, {heterogeneity_high:.4f}]"
        )
        print(f"Elapsed for this feature: {(time.time() - loop_start) / 60:.1f} minutes")

        del model
        del results
        gc.collect()

    print(f"\nAll models finished in {(time.time() - start_time) / 60:.1f} minutes.")
    print(f"Outputs are in: {result_dir}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    args = _parse_args()
    run_landscape_heterogeneity(selected_features=args.features, output_dir=args.output_dir)
