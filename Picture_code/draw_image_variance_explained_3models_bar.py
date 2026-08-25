from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

NULL_MODEL_REPORT = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Metrics_Report.txt"
MODEL_SPECS = [
    {
        "name": "StyleGAN",
        "trace": ROOT_DIR / "BYS_StyleGAN_model_14D_result" / "StyleGAN_14d_model_trace.nc",
        "color": "#6BAED6",
    },
    {
        "name": "InsightFace",
        "trace": ROOT_DIR / "BYS_Insightface_model_14D_result" / "deep_14d_model_trace.nc",
        "color": "#D9A441",
    },
    {
        "name": "DINOv2",
        "trace": ROOT_DIR / "BYS_DINOv2_model_14D_result" / "DINOv2_14d_model_trace.nc",
        "color": "#7A9A01",
    },
]

PNG_OUTPUT = OUTPUT_DIR / "image_variance_explained_3models_bar.png"
CSV_OUTPUT = OUTPUT_DIR / "image_variance_explained_3models_bar_data.csv"
SAMPLES_OUTPUT = OUTPUT_DIR / "image_variance_explained_3models_boxline_samples.csv"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def extract_value(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find {label} with pattern: {pattern}")
    return float(match.group(1))


def load_null_image_variance() -> float:
    null_text = NULL_MODEL_REPORT.read_text(encoding="utf-8", errors="ignore")
    return extract_value(r"Image Variance .*?:\s*([0-9.]+)", null_text, "null image variance")


def hdi(values: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    interval = az.hdi(values, hdi_prob=prob)
    return float(interval[0]), float(interval[1])


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    null_image_variance = load_null_image_variance()

    samples_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    for spec in MODEL_SPECS:
        trace = az.from_netcdf(spec["trace"])
        image_sigma = np.asarray(trace.posterior["1|image_sigma"].values, dtype=float).reshape(-1)
        explained = null_image_variance - image_sigma**2
        low, high = hdi(explained)

        for value in explained:
            samples_rows.append({"Model": spec["name"], "ExplainedVariance": float(value)})

        summary_rows.append(
            {
                "Model": spec["name"],
                "Mean": float(explained.mean()),
                "Median": float(np.median(explained)),
                "Q1": float(np.quantile(explained, 0.25)),
                "Q3": float(np.quantile(explained, 0.75)),
                "HDI_3%": low,
                "HDI_97%": high,
            }
        )

    return pd.DataFrame(samples_rows), pd.DataFrame(summary_rows)


def posterior_probability_greater(samples: pd.DataFrame, left_model: str, right_model: str) -> float:
    left = samples.loc[samples["Model"] == left_model, "ExplainedVariance"].to_numpy(dtype=float)
    right = samples.loc[samples["Model"] == right_model, "ExplainedVariance"].to_numpy(dtype=float)
    n = min(len(left), len(right))
    return float(np.mean(right[:n] > left[:n]))


def stars_from_probability(prob: float) -> str:
    if prob >= 0.995:
        return "***"
    if prob >= 0.975:
        return "**"
    if prob >= 0.95:
        return "*"
    return "n.s."


def draw_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    tick = 0.045
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#222222", lw=0.75, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.025, text, ha="center", va="bottom", fontsize=8.2, fontweight="bold")


def draw_figure(samples: pd.DataFrame, summary: pd.DataFrame) -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_OUTPUT, index=False)
    samples.to_csv(SAMPLES_OUTPUT, index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=450)
    rng = np.random.default_rng(20260506)
    model_order = [spec["name"] for spec in MODEL_SPECS]
    colors = {spec["name"]: spec["color"] for spec in MODEL_SPECS}
    positions = np.arange(1, len(model_order) + 1)

    box_data = [samples.loc[samples["Model"] == model, "ExplainedVariance"].to_numpy(dtype=float) for model in model_order]
    box = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.35),
        whiskerprops=dict(color="#555555", linewidth=0.95),
        capprops=dict(color="#555555", linewidth=0.95),
        boxprops=dict(linewidth=0.95, color="#333333"),
    )

    for patch, model in zip(box["boxes"], model_order):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.38)

    for pos, model, values in zip(positions, model_order, box_data):
        display_values = rng.choice(values, size=min(90, len(values)), replace=False)
        jitter = rng.normal(0, 0.055, size=len(display_values))
        ax.scatter(
            np.full(len(display_values), pos) + jitter,
            display_values,
            s=12,
            color=colors[model],
            edgecolor="#222222",
            linewidth=0.25,
            alpha=0.62,
            zorder=3,
        )

    means = summary.set_index("Model").loc[model_order, "Mean"].to_numpy(dtype=float)
    ax.plot(positions, means, color="#222222", lw=1.25, marker="o", markersize=4.0, zorder=5)

    for pos, model, mean in zip(positions, model_order, means):
        row = summary.set_index("Model").loc[model]
        ax.errorbar(
            pos,
            mean,
            yerr=np.array([[mean - row["HDI_3%"]], [row["HDI_97%"] - mean]], dtype=float),
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3.0,
            capthick=0.8,
            zorder=4,
        )
        ax.text(pos, row["HDI_97%"] + 0.055, f"{mean:.2f}", ha="center", va="bottom", fontsize=7.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_order, fontsize=8.8)
    ax.set_ylabel("Image variance explained (sigma^2)", fontsize=9.2)
    ax.set_title("Image Variance Explained by Pretrained Models", fontsize=11.0, fontweight="bold", pad=8)
    ax.grid(axis="y", color="#E7E9ED", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8.2)

    y_min = float(samples["ExplainedVariance"].quantile(0.001)) - 0.18
    y_max = float(samples["ExplainedVariance"].quantile(0.999)) + 0.85
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0.45, len(model_order) + 0.55)

    bracket_base = float(summary["HDI_97%"].max()) + 0.22
    for level, (left_idx, right_idx) in enumerate(combinations(range(len(model_order)), 2)):
        left_model = model_order[left_idx]
        right_model = model_order[right_idx]
        prob = posterior_probability_greater(samples, left_model, right_model)
        stars = stars_from_probability(prob)
        label = f"+{(means[right_idx] - means[left_idx]):.2f}\n{stars}"
        draw_significance_bracket(ax, positions[left_idx], positions[right_idx], bracket_base + level * 0.25, label)

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="#222222", markersize=4, lw=1.25, label="Posterior mean"),
        mpl.lines.Line2D([0], [0], color="#222222", lw=0.9, marker="_", markersize=7, label="94% HDI"),
        mpl.patches.Patch(facecolor="#999999", edgecolor="#333333", alpha=0.38, label="IQR"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=7.6)

    fig.savefig(PNG_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    samples, summary = load_plot_data()
    draw_figure(samples, summary)
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved samples: {SAMPLES_OUTPUT}")


if __name__ == "__main__":
    main()
