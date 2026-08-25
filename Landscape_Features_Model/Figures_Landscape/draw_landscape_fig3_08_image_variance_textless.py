from __future__ import annotations

import re
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


LANDSCAPE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
NULL_MODEL_REPORT = LANDSCAPE_DIR / "BYS_Null_Model" / "BYS_landscape_null_result" / "Null_Model_Metrics_Report.txt"

MODEL_ORDER = ["StyleGAN", "DINOv2", "Places365"]
MODEL_SPECS = [
    {
        "name": "StyleGAN",
        "trace": LANDSCAPE_DIR / "BYS_Fusion_20D_StyleGAN" / "Result" / "Fusion_20D_StyleGAN_model_trace.nc",
        "color": "#CC8FB0",
    },
    {
        "name": "DINOv2",
        "trace": LANDSCAPE_DIR / "BYS_Fusion_20D_DINOv2" / "Result" / "Fusion_20D_DINOv2_model_trace.nc",
        "color": "#B96F9A",
    },
    {
        "name": "Places365",
        "trace": LANDSCAPE_DIR / "BYS_Fusion_20D_Places365" / "Result" / "Fusion_20D_Places365_model_trace.nc",
        "color": "#A84F7E",
    },
]

MODEL_COLORS = {spec["name"]: spec["color"] for spec in MODEL_SPECS}
PNG_OUTPUT = OUTPUT_DIR / "Fig3_08_image_variance_textless.png"
SUMMARY_OUTPUT = OUTPUT_DIR / "landscape_image_variance_explained_3models_summary.csv"
SAMPLES_OUTPUT = OUTPUT_DIR / "landscape_image_variance_explained_3models_samples.csv"
SIGNIFICANCE_OUTPUT = OUTPUT_DIR / "landscape_image_variance_explained_3models_significance.csv"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.8,
            "xtick.major.width": 2.8,
            "ytick.major.width": 2.8,
            "xtick.major.size": 5.8,
            "ytick.major.size": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clear_text(ax: plt.Axes) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([])
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")


def load_null_image_variance() -> float:
    text = NULL_MODEL_REPORT.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Image Variance .*?:\s*([0-9.]+)", text)
    if match is None:
        raise ValueError(f"Could not parse image variance from: {NULL_MODEL_REPORT}")
    return float(match.group(1))


def hdi(values: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    interval = az.hdi(values, hdi_prob=prob)
    return float(interval[0]), float(interval[1])


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    null_image_variance = load_null_image_variance()
    sample_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for spec in MODEL_SPECS:
        trace_path = Path(spec["trace"])
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing fusion trace: {trace_path}")

        trace = az.from_netcdf(trace_path)
        image_sigma = np.asarray(trace.posterior["1|image_sigma"].values, dtype=float).reshape(-1)
        explained_variance = null_image_variance - image_sigma**2
        low, high = hdi(explained_variance)

        for value in explained_variance:
            sample_rows.append({"Model": spec["name"], "ExplainedVariance": float(value)})

        summary_rows.append(
            {
                "Model": spec["name"],
                "Mean": float(explained_variance.mean()),
                "Median": float(np.median(explained_variance)),
                "Q1": float(np.quantile(explained_variance, 0.25)),
                "Q3": float(np.quantile(explained_variance, 0.75)),
                "HDI_3%": low,
                "HDI_97%": high,
                "MarginalR2Mean": float(explained_variance.mean() / null_image_variance),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def significance_label(probability: float) -> str:
    if probability < 0.001:
        return "***"
    if probability < 0.01:
        return "**"
    if probability < 0.05:
        return "*"
    return "ns"


def calculate_pairwise_significance(samples: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("StyleGAN", "DINOv2"), ("DINOv2", "Places365"), ("StyleGAN", "Places365")]
    rows: list[dict[str, float | str]] = []
    for model_a, model_b in comparisons:
        values_a = samples.loc[samples["Model"] == model_a, "ExplainedVariance"].to_numpy(dtype=float)
        values_b = samples.loc[samples["Model"] == model_b, "ExplainedVariance"].to_numpy(dtype=float)
        if len(values_a) != len(values_b):
            raise ValueError(f"Posterior sample counts differ: {model_a}={len(values_a)}, {model_b}={len(values_b)}")
        difference = values_b - values_a
        posterior_probability = float(np.mean(difference > 0))
        two_sided_tail_probability = float(2 * min(posterior_probability, 1 - posterior_probability))
        rows.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "Mean_Difference_B_minus_A": float(difference.mean()),
                "Posterior_Probability_B_Greater_A": posterior_probability,
                "Two_Sided_Posterior_Tail_Probability": two_sided_tail_probability,
                "Significance": significance_label(two_sided_tail_probability),
            }
        )
    return pd.DataFrame(rows)


def draw_model_boxline(summary: pd.DataFrame, samples: pd.DataFrame, significance: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260601)
    df = summary.set_index("Model").loc[MODEL_ORDER].reset_index()
    positions = np.arange(len(MODEL_ORDER))
    grouped = [
        samples.loc[samples["Model"] == model, "ExplainedVariance"].to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]

    fig, ax = plt.subplots(figsize=(15, 6), dpi=400)
    box = ax.boxplot(
        grouped,
        positions=positions,
        vert=False,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#262626", linewidth=1.65),
        whiskerprops=dict(color="#454545", linewidth=1.35),
        capprops=dict(color="#454545", linewidth=1.35),
        boxprops=dict(linewidth=1.35, color="#262626"),
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.42)

    for pos, model, values in zip(positions, MODEL_ORDER, grouped):
        display_values = rng.choice(values, size=min(115, len(values)), replace=False)
        jitter = rng.normal(0, 0.075, size=len(display_values))
        ax.scatter(
            display_values,
            np.full(len(display_values), pos) + jitter,
            s=13,
            color=MODEL_COLORS[model],
            edgecolor="#1F1F1F",
            linewidth=0.32,
            alpha=0.76,
            zorder=3,
        )

    means = df["Mean"].to_numpy(dtype=float)
    ax.plot(means, positions, color="#262626", lw=2.15, marker="o", markersize=5.8, zorder=5)

    for pos, (_, row) in zip(positions, df.iterrows()):
        mean = float(row["Mean"])
        low = float(row["HDI_3%"])
        high = float(row["HDI_97%"])
        ax.errorbar(
            mean,
            pos,
            xerr=np.array([[mean - low], [high - mean]], dtype=float),
            fmt="none",
            ecolor="#575757",
            elinewidth=1.35,
            capsize=3.8,
            capthick=1.15,
            zorder=4,
        )

    x_min = float(min(samples["ExplainedVariance"].quantile(0.002), df["HDI_3%"].min()))
    x_max = float(max(samples["ExplainedVariance"].quantile(0.998), df["HDI_97%"].max()))
    x_pad = max((x_max - x_min) * 0.10, 0.05)
    bracket_start = x_max + x_pad * 0.45
    bracket_step = x_pad * 0.68
    bracket_tick = x_pad * 0.20
    for index, row in significance.iterrows():
        pos_a = MODEL_ORDER.index(str(row["Model_A"]))
        pos_b = MODEL_ORDER.index(str(row["Model_B"]))
        bracket_x = bracket_start + index * bracket_step
        ax.plot(
            [bracket_x - bracket_tick, bracket_x, bracket_x, bracket_x - bracket_tick],
            [pos_a, pos_a, pos_b, pos_b],
            color="#3B2632",
            linewidth=1.55,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            bracket_x + bracket_tick * 0.55,
            (pos_a + pos_b) / 2,
            str(row["Significance"]),
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#3B2632",
        )

    ax.set_xlim(x_min - x_pad, bracket_start + (len(significance) - 1) * bracket_step + x_pad * 1.20)
    ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.95)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="x", color="#E6EAF0", lw=0.8)
    ax.set_axisbelow(True)
    clear_text(ax)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.985, bottom=0.16)
    fig.savefig(PNG_OUTPUT, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    summary, samples = load_plot_data()
    significance = calculate_pairwise_significance(samples)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    samples.to_csv(SAMPLES_OUTPUT, index=False)
    significance.to_csv(SIGNIFICANCE_OUTPUT, index=False)
    draw_model_boxline(summary, samples, significance)
    print(summary[["Model", "Mean", "HDI_3%", "HDI_97%", "MarginalR2Mean"]].to_string(index=False))
    print(significance.to_string(index=False))
    print(f"Saved: {PNG_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")
    print(f"Saved: {SAMPLES_OUTPUT}")
    print(f"Saved: {SIGNIFICANCE_OUTPUT}")


if __name__ == "__main__":
    main()
