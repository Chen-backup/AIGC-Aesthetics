from __future__ import annotations

from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


NONLINEAR_FEATURES = [
    "horizon_y_norm",
    "saturation_mean",
    "artificial_ratio",
    "semantic_diversity",
]


def plot_from_saved_landscape_gamm_model() -> None:
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    result_dir = base_dir / "Result"
    trace_path = result_dir / "GAMM_model_trace_4features.nc"
    fallback_trace_path = result_dir / "GAMM_model_trace_4features.pkl"
    stats_path = result_dir / "GAMM_feature_standardization_stats.csv"

    print("================ 1. Load saved GAMM trace ================")
    if not stats_path.exists():
        raise FileNotFoundError(f"Standardization stats not found: {stats_path}")

    if trace_path.exists():
        trace = az.from_netcdf(trace_path)
    elif fallback_trace_path.exists():
        trace = pd.read_pickle(fallback_trace_path)
    else:
        raise FileNotFoundError(f"Trace file not found: {trace_path} or {fallback_trace_path}")
    stats_df = pd.read_csv(stats_path).set_index("feature")
    print("Saved model trace and feature stats loaded successfully.")

    print("\n================ 2. Rebuild model shell ================")
    ratings_path = project_dir / "ratings_for_bayesian_model.csv"
    mapping_path = project_dir / "landscape_number_mapping.csv"
    features_path = project_dir / "BYS_interpretable_Features_Model" / "landscape_interpretable_features.csv"

    df_ratings = pd.read_csv(ratings_path)
    df_mapping = pd.read_csv(mapping_path)
    df_features = pd.read_csv(features_path)

    df_feat_mapped = pd.merge(df_features, df_mapping[["image_name", "image"]], on="image_name", how="inner")
    df = pd.merge(df_ratings, df_feat_mapped, on="image", how="inner")
    df = df.dropna(subset=NONLINEAR_FEATURES + ["rating", "rater", "image"]).copy()

    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)
    rating_cats_numeric = np.array(categories).astype(float)

    feature_stats: dict[str, dict[str, float]] = {}
    for feat in NONLINEAR_FEATURES:
        mean_val = float(stats_df.loc[feat, "mean"])
        std_val = float(stats_df.loc[feat, "std"])
        z_min = float(stats_df.loc[feat, "z_min"])
        z_max = float(stats_df.loc[feat, "z_max"])
        df[feat] = (df[feat] - mean_val) / std_val
        feature_stats[feat] = {"mean": mean_val, "std": std_val, "z_min": z_min, "z_max": z_max}

    spline_terms = [f"bs({feat}, df=4)" for feat in NONLINEAR_FEATURES]
    formula = f"rating ~ 1 + {' + '.join(spline_terms)} + (1|rater) + (1|image)"
    model = bmb.Model(formula, data=df, family="cumulative")
    print("Model shell rebuilt.")

    print("\n================ 3. Predict and draw expected nonlinear curves ================")
    plt.figure(figsize=(15, 10))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif"]
    sns.set_context("paper", font_scale=1.2)

    base_rater = df["rater"].iloc[0]
    base_image = df["image"].iloc[0]
    curve_rows = []

    for i, feat in enumerate(NONLINEAR_FEATURES):
        print(f"Rendering nonlinear curve for {feat} ...")
        z_min = feature_stats[feat]["z_min"]
        z_max = feature_stats[feat]["z_max"]
        margin = (z_max - z_min) * 0.01
        x_zscores = np.linspace(z_min + margin, z_max - margin, 100)

        dummy_data = {f: np.zeros(100) for f in NONLINEAR_FEATURES}
        dummy_data[feat] = x_zscores
        dummy_data["rater"] = base_rater
        dummy_data["image"] = base_image
        dummy_df = pd.DataFrame(dummy_data)

        pred = model.predict(trace, data=dummy_df, kind="response_params", include_group_specific=False, inplace=False)

        pred_values = None
        available_vars = list(pred.posterior.data_vars.keys())
        for name in ["p", "rating_response_params", "rating_probs", "rating_mean", "rating"]:
            if name in available_vars:
                val = pred.posterior[name].values
                if len(val.shape) >= 3:
                    pred_values = val
                    break

        if pred_values is None:
            raise ValueError(f"Prediction probabilities not found. Available vars: {available_vars}")

        expected_scores = np.sum(pred_values * rating_cats_numeric, axis=-1)
        mean_expected_score = expected_scores.mean(axis=(0, 1))
        lower_bound = np.percentile(expected_scores, 2.5, axis=(0, 1))
        upper_bound = np.percentile(expected_scores, 97.5, axis=(0, 1))

        x_real = x_zscores * feature_stats[feat]["std"] + feature_stats[feat]["mean"]
        valid_mask = ~np.isnan(mean_expected_score)

        for x_val, mean_val, low_val, up_val in zip(
            x_real[valid_mask],
            mean_expected_score[valid_mask],
            lower_bound[valid_mask],
            upper_bound[valid_mask],
        ):
            curve_rows.append(
                {
                    "feature": feat,
                    "x_real": x_val,
                    "expected_score": mean_val,
                    "ci_lower": low_val,
                    "ci_upper": up_val,
                }
            )

        plt.subplot(2, 2, i + 1)
        plt.plot(x_real[valid_mask], mean_expected_score[valid_mask], color="#d62728", linewidth=3, label="Expected Score")
        plt.fill_between(
            x_real[valid_mask],
            lower_bound[valid_mask],
            upper_bound[valid_mask],
            color="#d62728",
            alpha=0.2,
            label="95% Credible Interval",
        )
        plt.title(f"Nonlinear Aesthetic Effect of {feat}", weight="bold", pad=10)
        plt.xlabel(f"Real values of {feat}", weight="bold")
        plt.ylabel("Expected Aesthetic Score", weight="bold")
        plt.legend(loc="best")
        plt.grid(axis="both", linestyle="--", alpha=0.3)
        sns.despine()

    plt.tight_layout()
    figure_path = result_dir / "GAMM_NonLinear_Expected_Curves_4Features.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    curve_data_path = result_dir / "GAMM_NonLinear_Expected_Curves_4Features_data.csv"
    pd.DataFrame(curve_rows).to_csv(curve_data_path, index=False, encoding="utf-8-sig")

    print(f"\nExpected nonlinear curves saved to: {figure_path}")
    print(f"Curve data saved to: {curve_data_path}")


if __name__ == "__main__":
    plot_from_saved_landscape_gamm_model()
