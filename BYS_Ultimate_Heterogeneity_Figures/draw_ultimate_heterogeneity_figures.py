from __future__ import annotations

import gc
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "BYS_Ultimate_Heterogeneity_Results"
OUTPUT_DIR = Path(__file__).resolve().parent

FEATURE_ORDER = [
    "face_hw_ratio",
    "eye_face_w_ratio",
    "mouth_face_w_ratio",
    "three_courts_balance",
    "upper_lower_ratio",
    "eye_y_ratio",
    "total_symmetry",
    "le_nose_re_angle",
    "mouth_nose_ratio",
    "face_brightness",
    "face_contrast",
    "face_clarity",
    "saturation",
    "edge_density",
]

DISPLAY_LABELS = {
    "face_hw_ratio": "Face H/W",
    "eye_face_w_ratio": "Eye/face W",
    "mouth_face_w_ratio": "Mouth/face W",
    "three_courts_balance": "Three-courts",
    "upper_lower_ratio": "Upper/lower",
    "eye_y_ratio": "Eye Y",
    "total_symmetry": "Symmetry",
    "le_nose_re_angle": "Nose angle",
    "mouth_nose_ratio": "Mouth/nose",
    "face_brightness": "Brightness",
    "face_contrast": "Contrast",
    "face_clarity": "Clarity",
    "saturation": "Saturation",
    "edge_density": "Edge density",
}

FULL_LABELS = {
    "face_hw_ratio": "Face height-width ratio",
    "eye_face_w_ratio": "Eye-to-face width ratio",
    "mouth_face_w_ratio": "Mouth-to-face width ratio",
    "three_courts_balance": "Three-courts balance",
    "upper_lower_ratio": "Upper-lower face ratio",
    "eye_y_ratio": "Eye vertical position",
    "total_symmetry": "Total symmetry",
    "le_nose_re_angle": "Nose angle",
    "mouth_nose_ratio": "Mouth-nose ratio",
    "face_brightness": "Face brightness",
    "face_contrast": "Face contrast",
    "face_clarity": "Face clarity",
    "saturation": "Saturation",
    "edge_density": "Edge density",
}

COLORS = {
    "text": "#222222",
    "axis": "#333333",
    "grid": "#E7E9ED",
    "zero": "#7B7B7B",
    "blue": "#4E79A7",
    "orange": "#D99532",
    "red": "#C84C4C",
    "teal": "#4F9A94",
    "purple": "#7B6BB3",
}


def feature_label(feature: str, full: bool = False) -> str:
    labels = FULL_LABELS if full else DISPLAY_LABELS
    return labels.get(feature, feature.replace("_", " "))


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def flatten_draws(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def hdi(draws: np.ndarray, low: float = 0.03, high: float = 0.97) -> tuple[float, float]:
    flat = flatten_draws(draws)
    return float(np.quantile(flat, low)), float(np.quantile(flat, high))


def load_model(feature: str) -> az.InferenceData:
    path = MODEL_DIR / f"Ultimate_Heterogeneity_{feature}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return az.from_netcdf(path)


def extract_summary_and_slopes(top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    slope_frames: list[pd.DataFrame] = []

    print("Extracting posterior summaries from .nc models...")
    for idx, feature in enumerate(FEATURE_ORDER, start=1):
        print(f"  [{idx:02d}/{len(FEATURE_ORDER)}] {feature}")
        idata = load_model(feature)
        posterior = idata.posterior

        sigma_name = f"{feature}|rater_sigma"
        slope_name = f"{feature}|rater"
        if sigma_name not in posterior:
            raise KeyError(f"Cannot find posterior variable: {sigma_name}")
        if feature not in posterior:
            raise KeyError(f"Cannot find fixed-effect posterior variable: {feature}")

        sigma_draws = flatten_draws(posterior[sigma_name].values)
        fixed_draws = flatten_draws(posterior[feature].values)
        sigma_hdi_low, sigma_hdi_high = hdi(sigma_draws)
        fixed_hdi_low, fixed_hdi_high = hdi(fixed_draws)

        rows.append(
            {
                "feature": feature,
                "label": feature_label(feature),
                "full_label": feature_label(feature, full=True),
                "heterogeneity_mean": float(np.mean(sigma_draws)),
                "heterogeneity_hdi_3": sigma_hdi_low,
                "heterogeneity_hdi_97": sigma_hdi_high,
                "fixed_mean": float(np.mean(fixed_draws)),
                "fixed_hdi_3": fixed_hdi_low,
                "fixed_hdi_97": fixed_hdi_high,
            }
        )

        # Store all slope means temporarily; the final top-N filter happens after ranking.
        if slope_name in posterior:
            slopes = np.asarray(posterior[slope_name].values, dtype=float)
            slope_mean = slopes.mean(axis=(0, 1))
            rater_coord = posterior[slope_name].coords.get("rater__factor_dim")
            if rater_coord is None:
                rater_ids = np.arange(len(slope_mean))
            else:
                rater_ids = np.asarray(rater_coord.values)
            slope_frames.append(
                pd.DataFrame(
                    {
                        "feature": feature,
                        "label": feature_label(feature),
                        "rater": rater_ids,
                        "random_slope_mean": slope_mean,
                    }
                )
            )

        del idata
        gc.collect()

    summary = pd.DataFrame(rows).sort_values("heterogeneity_mean", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary_path = OUTPUT_DIR / "Ultimate_Heterogeneity_14Feature_Summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    slope_df = pd.concat(slope_frames, ignore_index=True)
    top_features = summary.head(top_n)["feature"].tolist()
    slope_df = slope_df[slope_df["feature"].isin(top_features)].copy()
    slope_path = OUTPUT_DIR / f"Top{top_n}_Rater_Random_Slope_Means.csv"
    slope_df.to_csv(slope_path, index=False)
    print(f"Saved slope means: {slope_path}")

    return summary, slope_df


def draw_figure1_forest(summary: pd.DataFrame) -> None:
    df = summary.sort_values("heterogeneity_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 6.7), dpi=450)
    fig.patch.set_facecolor("white")

    xerr_low = df["heterogeneity_mean"] - df["heterogeneity_hdi_3"]
    xerr_high = df["heterogeneity_hdi_97"] - df["heterogeneity_mean"]
    colors = mpl.colormaps["YlOrRd"](
        mpl.colors.Normalize(df["heterogeneity_mean"].min(), df["heterogeneity_mean"].max())(df["heterogeneity_mean"])
    )

    ax.errorbar(
        df["heterogeneity_mean"],
        y,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="#555555",
        elinewidth=1.45,
        capsize=3.5,
        capthick=1.15,
        zorder=2,
    )
    ax.scatter(df["heterogeneity_mean"], y, s=46, c=colors, edgecolor="#222222", linewidth=0.55, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["full_label"], fontsize=8.9)
    ax.set_xlabel("Between-rater heterogeneity SD (posterior mean with 94% HDI)", fontsize=9.8)
    ax.set_title("Population Disagreement Across Aesthetic Features", fontsize=12.0, fontweight="bold", pad=12)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.7)

    for row_idx, row in df.iterrows():
        ax.text(
            row["heterogeneity_hdi_97"] + 0.006,
            row_idx,
            f"{row['heterogeneity_mean']:.3f}",
            va="center",
            ha="left",
            fontsize=7.7,
            color=COLORS["text"],
        )

    ax.text(-0.14, 1.03, "a", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    ax.set_xlim(left=0, right=max(df["heterogeneity_hdi_97"]) * 1.18)
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "Figure1_Ultimate_Heterogeneity_Forest.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure1_Ultimate_Heterogeneity_Forest.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_half_violin(ax: plt.Axes, values: np.ndarray, center_y: float, color: str, x_grid: np.ndarray, width: float = 0.34) -> None:
    kde = gaussian_kde(values)
    density = kde(x_grid)
    density = density / max(float(density.max()), 1e-12) * width
    visible = density > width * 0.008
    ax.fill_between(
        x_grid[visible],
        center_y,
        center_y + density[visible],
        color=color,
        alpha=0.34,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x_grid[visible], center_y + density[visible], color=color, linewidth=1.4, zorder=2)


def draw_figure2_top5_slopes(summary: pd.DataFrame, slope_df: pd.DataFrame) -> None:
    top_features = summary.head(5)["feature"].tolist()
    df = slope_df[slope_df["feature"].isin(top_features)].copy()
    all_values = df["random_slope_mean"].to_numpy(dtype=float)
    x_min = float(np.quantile(all_values, 0.005))
    x_max = float(np.quantile(all_values, 0.995))
    pad = (x_max - x_min) * 0.14
    x_grid = np.linspace(x_min - pad, x_max + pad, 640)

    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=450)
    fig.patch.set_facecolor("white")

    rng = np.random.default_rng(20260506)
    palette = ["#C8553D", "#D9822B", "#E3B341", "#5E8C61", "#4E79A7"]

    for idx, feature in enumerate(top_features):
        y = len(top_features) - 1 - idx
        values = df.loc[df["feature"] == feature, "random_slope_mean"].to_numpy(dtype=float)
        color = palette[idx]
        draw_half_violin(ax, values, y, color, x_grid)

        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        ax.hlines(y - 0.08, q1, q3, color="#222222", lw=2.1, zorder=4)
        ax.scatter([med], [y - 0.08], s=24, color="#222222", zorder=5)
        ax.scatter([mean], [y - 0.08], s=28, facecolor="white", edgecolor="#222222", linewidth=0.75, zorder=5)

        sample_size = min(240, len(values))
        sample = rng.choice(values, size=sample_size, replace=False)
        jitter = rng.uniform(-0.28, -0.02, size=sample_size)
        ax.scatter(
            sample,
            y + jitter,
            s=8.5,
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.25,
            alpha=0.58,
            zorder=3,
        )

    ax.axvline(0, color="#777777", linestyle=(0, (4, 4)), linewidth=1.1)
    ax.set_yticks(range(len(top_features) - 1, -1, -1))
    ax.set_yticklabels([feature_label(f, full=True) for f in top_features], fontsize=8.9)
    ax.set_xlabel("Rater-specific preference slope (posterior mean)", fontsize=9.8)
    ax.set_title("Individual Preference Spectra for the Top Heterogeneous Features", fontsize=11.8, fontweight="bold", pad=11)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.6)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markeredgecolor="#222222", markersize=5, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#222222", markersize=5, label="Mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.2, ncol=2, handletextpad=0.4, columnspacing=1.2)
    ax.text(-0.14, 1.03, "b", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    ax.set_xlim(x_grid.min(), x_grid.max())
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "Figure2_Top5_Rater_Preference_Spectra.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure2_Top5_Rater_Preference_Spectra.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_figure3_effect_vs_heterogeneity(summary: pd.DataFrame) -> None:
    df = summary.copy()
    x_threshold = 0.0
    y_threshold = float(df["heterogeneity_mean"].median())

    fig, ax = plt.subplots(figsize=(7.7, 5.9), dpi=450)
    fig.patch.set_facecolor("white")

    ax.axvline(x_threshold, color="#777777", linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)
    ax.axhline(y_threshold, color="#777777", linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)

    xerr = [df["fixed_mean"] - df["fixed_hdi_3"], df["fixed_hdi_97"] - df["fixed_mean"]]
    yerr = [
        df["heterogeneity_mean"] - df["heterogeneity_hdi_3"],
        df["heterogeneity_hdi_97"] - df["heterogeneity_mean"],
    ]

    norm = mpl.colors.Normalize(vmin=df["heterogeneity_mean"].min(), vmax=df["heterogeneity_mean"].max())
    colors = mpl.colormaps["YlOrRd"](norm(df["heterogeneity_mean"]))

    ax.errorbar(
        df["fixed_mean"],
        df["heterogeneity_mean"],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor="#7A7A7A",
        elinewidth=0.95,
        capsize=2.2,
        capthick=0.8,
        alpha=0.85,
        zorder=2,
    )
    ax.scatter(
        df["fixed_mean"],
        df["heterogeneity_mean"],
        s=58,
        c=colors,
        edgecolor="#222222",
        linewidth=0.55,
        zorder=3,
    )

    label_offsets = {
        "upper_lower_ratio": (8, 5),
        "face_hw_ratio": (-10, 5),
        "mouth_nose_ratio": (9, 6),
        "eye_y_ratio": (-32, 11),
        "eye_face_w_ratio": (-40, -3),
        "edge_density": (-12, 7),
        "mouth_face_w_ratio": (9, 6),
        "face_clarity": (-14, 8),
        "le_nose_re_angle": (-18, 8),
        "face_brightness": (10, 5),
        "face_contrast": (11, 3),
        "saturation": (-10, 9),
        "total_symmetry": (10, 8),
        "three_courts_balance": (9, 7),
    }
    for _, row in df.iterrows():
        offset = label_offsets.get(str(row["feature"]), (8, 5))
        ax.annotate(
            row["label"],
            xy=(row["fixed_mean"], row["heterogeneity_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            color=COLORS["text"],
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.45, shrinkA=0, shrinkB=5),
            zorder=4,
        )

    ax.text(
        0.02,
        0.96,
        "weak average effect\nhigh disagreement",
        transform=ax.transAxes,
        fontsize=7.8,
        ha="left",
        va="top",
        color="#5E5E5E",
    )
    ax.text(
        0.98,
        0.96,
        "strong average effect\nhigh disagreement",
        transform=ax.transAxes,
        fontsize=7.8,
        ha="right",
        va="top",
        color="#5E5E5E",
    )

    ax.set_xlabel("Population-level effect on aesthetic rating (posterior mean)", fontsize=9.7)
    ax.set_ylabel("Between-rater heterogeneity SD", fontsize=9.7)
    ax.set_title("Average Aesthetic Effect Versus Individual Disagreement", fontsize=11.8, fontweight="bold", pad=11)
    ax.grid(color=COLORS["grid"], linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.4)
    ax.text(-0.14, 1.03, "c", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps["YlOrRd"])
    cbar = fig.colorbar(sm, ax=ax, pad=0.018, fraction=0.045)
    cbar.set_label("Heterogeneity SD", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.65)
    cbar.outline.set_linewidth(0.65)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figure3_Effect_vs_Heterogeneity_Quadrant.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure3_Effect_vs_Heterogeneity_Quadrant.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    summary, slope_df = extract_summary_and_slopes(top_n=5)
    draw_figure1_forest(summary)
    draw_figure2_top5_slopes(summary, slope_df)
    draw_figure3_effect_vs_heterogeneity(summary)
    print(f"Done. Figures and source data are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
