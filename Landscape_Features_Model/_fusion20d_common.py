from __future__ import annotations

import os
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


TEST_MODE = False

SELECTED_INTERPRETABLE_FEATURES = [
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


def _get_ai_color(ai_model_type: str) -> str:
    if ai_model_type == "DINOv2":
        return "#ff7f0e"
    if ai_model_type == "Places365":
        return "#2ca02c"
    if ai_model_type == "StyleGAN":
        return "#9b59b6"
    return "#1f77b4"


def _read_null_image_variance(base_dir: Path) -> float | None:
    null_metrics_path = base_dir / "BYS_Null_Model" / "BYS_landscape_null_result" / "Null_Model_Metrics_Report.txt"
    if not null_metrics_path.exists():
        return None

    text = null_metrics_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "Image Variance" in line or "sigma^2" in line:
            try:
                return float(line.split(":")[-1].strip().split()[0])
            except Exception:
                continue
    return None


def _coerce_summary_value(value: object, name: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not parse numeric summary value for {name}: {value!r}")
    return float(numeric)


def _coerce_summary_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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


def _get_hdi_column_names(df: pd.DataFrame) -> tuple[str, str]:
    hdi_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("hdi_")]
    if len(hdi_cols) >= 2:
        def _extract_percent(col_name: str) -> float:
            tail = col_name.replace("hdi_", "").replace("%", "")
            try:
                return float(tail)
            except Exception:
                return float("nan")

        hdi_cols_sorted = sorted(hdi_cols, key=_extract_percent)
        return hdi_cols_sorted[0], hdi_cols_sorted[-1]

    eti_lower_candidates = [col for col in df.columns if isinstance(col, str) and ("eti" in col.lower()) and col.lower().endswith("_lb")]
    eti_upper_candidates = [col for col in df.columns if isinstance(col, str) and ("eti" in col.lower()) and col.lower().endswith("_ub")]
    if eti_lower_candidates and eti_upper_candidates:
        return eti_lower_candidates[0], eti_upper_candidates[0]

    ci_lower_candidates = [col for col in df.columns if isinstance(col, str) and col.lower().endswith("_lb")]
    ci_upper_candidates = [col for col in df.columns if isinstance(col, str) and col.lower().endswith("_ub")]
    if ci_lower_candidates and ci_upper_candidates:
        return ci_lower_candidates[0], ci_upper_candidates[0]

    raise KeyError(
        "Could not find interval columns in summary output. "
        f"Available columns: {list(df.columns)}"
    )


def _iter_axes_from_trace_plot(plot_obj: object) -> list[object]:
    if isinstance(plot_obj, np.ndarray):
        return plot_obj.ravel().tolist()
    if isinstance(plot_obj, (list, tuple)):
        axes = []
        for item in plot_obj:
            if isinstance(item, np.ndarray):
                axes.extend(item.ravel().tolist())
            elif isinstance(item, (list, tuple)):
                axes.extend(list(item))
            else:
                axes.append(item)
        return axes
    if hasattr(plot_obj, "axes"):
        try:
            axes_attr = getattr(plot_obj, "axes")
            if isinstance(axes_attr, np.ndarray):
                return axes_attr.ravel().tolist()
            if isinstance(axes_attr, (list, tuple)):
                return list(axes_attr)
        except Exception:
            pass
    return []


def run_fusion_20d_model(
    *,
    ai_model_type: str,
    ai_features_path: Path,
    output_dir: Path,
) -> None:
    base_dir = Path(__file__).resolve().parent
    ratings_path = base_dir / "ratings_for_bayesian_model.csv"
    mapping_path = base_dir / "landscape_number_mapping.csv"
    interpretable_path = base_dir / "BYS_interpretable_Features_Model" / "landscape_interpretable_features.csv"

    print(f"================ Stage 1: Load 20D fusion inputs ({ai_model_type}) ================")
    df_ratings = pd.read_csv(ratings_path)
    df_mapping = pd.read_csv(mapping_path)
    df_ai_features = pd.read_csv(ai_features_path)
    df_interpretable = pd.read_csv(interpretable_path)

    pc_features = [f"PC{i + 1}" for i in range(10)]
    all_20_features = SELECTED_INTERPRETABLE_FEATURES + pc_features

    df_ai_mapped = pd.merge(df_ai_features, df_mapping[["image_name", "image"]], on="image_name", how="inner")
    df_combined_features = pd.merge(df_ai_mapped, df_interpretable, on="image_name", how="inner")
    df = pd.merge(df_ratings, df_combined_features, on="image", how="inner")

    initial_len = len(df)
    df = df.dropna(subset=all_20_features + ["rating", "rater", "image"]).copy()
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
        print(f"\nFull mode enabled: running the 20D fusion model with all landscape ratings ({ai_model_type}).")
        tune_steps = 2000
        draw_steps = 1000

    print("\n================ Stage 2: Standardize 20 features ================")
    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)
    categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=categories, ordered=True)

    for feat in all_20_features:
        df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()

    print("\n================ Stage 3: Run Bayesian cumulative model ================")
    formula = f"rating ~ 1 + {' + '.join(all_20_features)} + (1|rater) + (1|image)"
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

    print("\n================ Stage 4: Compute fusion metrics ================")
    summary = az.summary(results)

    sigma_image = _coerce_summary_value(summary.loc["1|image_sigma", "mean"], "1|image_sigma")
    sigma_rater = _coerce_summary_value(summary.loc["1|rater_sigma", "mean"], "1|rater_sigma")
    var_image = sigma_image ** 2
    var_rater = sigma_rater ** 2

    null_var_image = _read_null_image_variance(base_dir)
    marginal_r2_str = "Not computed"
    if null_var_image and null_var_image > 0:
        marginal_r2 = (null_var_image - var_image) / null_var_image
        marginal_r2_str = f"{marginal_r2:.2%}"

    metrics_text = (
        f"========== Landscape 20D Fusion Model Metrics ({ai_model_type} + Interpretable Features) ==========\n\n"
        "1. Variance Components:\n"
        f"   - Residual Image Variance (sigma^2): {var_image:.4f}\n"
        f"   - Rater Variance (sigma^2): {var_rater:.4f}\n\n"
        "2. Fusion Explanatory Power:\n"
        f"   - Landscape null-model image variance: {null_var_image if null_var_image is not None else 'Not found'}\n"
        f"   - 20D fusion marginal R^2: {marginal_r2_str}\n"
    )
    print("\n" + metrics_text)

    print("\n================ Stage 5: Save outputs and figures ================")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "Fusion_20D_Model_Summary.csv")
    with open(output_dir / "Fusion_20D_Model_Metrics_Report.txt", "w", encoding="utf-8") as f:
        f.write(metrics_text)
    trace_path = _save_inference_data(results, output_dir / f"Fusion_20D_{ai_model_type}_model_trace.nc")
    print(f"Saved model trace to: {trace_path}")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    sns.set_context("paper", font_scale=1.0)

    feat_summary = az.summary(results, var_names=all_20_features)
    hdi_lower_col, hdi_upper_col = _get_hdi_column_names(feat_summary)
    feat_summary = _coerce_summary_columns(feat_summary, ["mean", hdi_lower_col, hdi_upper_col])
    feat_summary_sorted = feat_summary.sort_values(by="mean", ascending=True)
    sorted_features = feat_summary_sorted.index.tolist()

    human_color = "#1f77b4"
    ai_color = _get_ai_color(ai_model_type)
    colors = [ai_color if feat.startswith("PC") else human_color for feat in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 10))
    means = feat_summary_sorted["mean"]
    hdi_lower = feat_summary_sorted[hdi_lower_col]
    hdi_upper = feat_summary_sorted[hdi_upper_col]
    y_pos = np.arange(len(sorted_features))

    for i in range(len(sorted_features)):
        ax.errorbar(
            means.iloc[i],
            y_pos[i],
            xerr=[[means.iloc[i] - hdi_lower.iloc[i]], [hdi_upper.iloc[i] - means.iloc[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2,
            capsize=4,
        )

    ax.axvline(x=0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Standardized Effect Size (Posterior Mean with 95% HDI)", weight="bold")
    ax.set_title(f"Landscape Fusion: 10 Interpretable + 10 {ai_model_type} PCs", weight="bold", pad=15)

    custom_lines = [
        Line2D([0], [0], color=human_color, marker="o", lw=2),
        Line2D([0], [0], color=ai_color, marker="o", lw=2),
    ]
    ax.legend(custom_lines, ["Interpretable Features", f"{ai_model_type} Deep PCs"], loc="lower right")

    sns.despine()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "Fusion_20D_Fig1_Forest_Plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    trace_dir = output_dir / "TracePlots_Diagnostics"
    trace_dir.mkdir(parents=True, exist_ok=True)

    vars_to_plot = all_20_features + ["1|rater_sigma", "1|image_sigma"]
    for var in vars_to_plot:
        try:
            axes_trace = az.plot_trace(results, var_names=[var])
            for ax in _iter_axes_from_trace_plot(axes_trace):
                try:
                    ax.set_yticklabels([])
                except Exception:
                    continue

            plt.suptitle(f"Trace Plot: {var}", fontsize=16, weight="bold", y=1.05)
            plt.tight_layout()
            safe_var_name = var.replace("|", "_")
            plt.savefig(trace_dir / f"Trace_{safe_var_name}.png", dpi=300, bbox_inches="tight")
        except Exception as exc:
            print(f"Skipping trace plot cleanup for {var} because of plotting backend differences: {exc}")
            safe_var_name = var.replace("|", "_")
            try:
                plt.savefig(trace_dir / f"Trace_{safe_var_name}.png", dpi=300, bbox_inches="tight")
            except Exception:
                pass
        finally:
            plt.close("all")

    print(f"\nAll tasks finished. Output folder: {os.path.abspath(output_dir)}")
