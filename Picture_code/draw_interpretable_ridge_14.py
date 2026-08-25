from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"
TRACE_PATH = ROOT_DIR / "BYS_interpretable_model_result" / "full_model_trace.nc"

PNG_OUTPUT = OUTPUT_DIR / "Interpretable_14D_Ridge_1x2.png"
CSV_OUTPUT = OUTPUT_DIR / "Interpretable_14D_Boxline_Data.csv"
SAMPLES_OUTPUT = OUTPUT_DIR / "Interpretable_14D_Boxline_Samples.csv"

FEATURES = [
    ("face_hw_ratio", "Facial height-to-width ratio"),
    ("eye_face_w_ratio", "Eye-to-face width ratio"),
    ("mouth_face_w_ratio", "Mouth-to-face width ratio"),
    ("three_courts_balance", "Three-courts facial balance"),
    ("upper_lower_ratio", "Upper-to-lower face ratio"),
    ("eye_y_ratio", "Vertical eye position"),
    ("total_symmetry", "Overall facial symmetry"),
    ("le_nose_re_angle", "Left eye-nose-right eye angle"),
    ("mouth_nose_ratio", "Mouth-to-nose distance ratio"),
    ("face_brightness", "Facial brightness"),
    ("face_contrast", "Facial contrast"),
    ("face_clarity", "Facial clarity"),
    ("saturation", "Color saturation"),
    ("edge_density", "Edge density"),
]

COLORS = {
    "positive": "#2C7FB8",
    "negative": "#D95F59",
    "neutral": "#F2C94C",
    "mean": "#222222",
    "zero": "#333333",
    "grid": "#E7E9ED",
    "text": "#222222",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def hdi(values: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    interval = az.hdi(values, hdi_prob=prob)
    return float(interval[0]), float(interval[1])


def significance_label(values: np.ndarray) -> str:
    prob_positive = float(np.mean(values > 0))
    prob_direction = max(prob_positive, 1.0 - prob_positive)
    if prob_direction >= 0.995:
        return "***"
    if prob_direction >= 0.975:
        return "**"
    if prob_direction >= 0.95:
        return "*"
    return ""


def effect_color(low: float, high: float, mean: float) -> str:
    if low > 0:
        return COLORS["positive"]
    if high < 0:
        return COLORS["negative"]
    if mean >= 0:
        return COLORS["positive"]
    return COLORS["negative"]


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = az.from_netcdf(TRACE_PATH)
    sample_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for feature, label in FEATURES:
        values = np.asarray(trace.posterior[feature].values, dtype=float).reshape(-1)
        low, high = hdi(values)
        q1, q3 = np.quantile(values, [0.25, 0.75])
        mean = float(np.mean(values))
        median = float(np.median(values))
        stars = significance_label(values)

        for value in values:
            sample_rows.append({"feature": feature, "label": label, "effect": float(value)})

        summary_rows.append(
            {
                "feature": feature,
                "label": label,
                "mean": mean,
                "median": median,
                "q1": float(q1),
                "q3": float(q3),
                "hdi_3": low,
                "hdi_97": high,
                "p_direction": max(float(np.mean(values > 0)), float(np.mean(values < 0))),
                "stars": stars,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean", ascending=False).reset_index(drop=True)
    samples = pd.DataFrame(sample_rows)
    return samples, summary


def draw_figure(samples: pd.DataFrame, summary: pd.DataFrame) -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    samples.to_csv(SAMPLES_OUTPUT, index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(20260508)
    order = summary["feature"].tolist()
    labels = summary["label"].tolist()
    positions = np.arange(1, len(order) + 1)
    data = [samples.loc[samples["feature"] == feature, "effect"].to_numpy(dtype=float) for feature in order]

    fig, ax = plt.subplots(figsize=(12.0, 5.05), dpi=600)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLORS["mean"], linewidth=1.2),
        whiskerprops=dict(color="#555555", linewidth=0.85),
        capprops=dict(color="#555555", linewidth=0.85),
        boxprops=dict(linewidth=0.85, color="#333333"),
    )

    color_map: dict[str, str] = {}
    for _, row in summary.iterrows():
        color = effect_color(float(row["hdi_3"]), float(row["hdi_97"]), float(row["mean"]))
        if float(row["hdi_3"]) <= 0 <= float(row["hdi_97"]):
            color = COLORS["neutral"]
        color_map[str(row["feature"])] = color

    for patch, feature in zip(box["boxes"], order):
        patch.set_facecolor(color_map[feature])
        patch.set_alpha(0.34)

    displayed_min = []
    displayed_max = []
    for pos, feature, values in zip(positions, order, data):
        display_values = rng.choice(values, size=min(70, len(values)), replace=False)
        displayed_min.append(float(np.min(display_values)))
        displayed_max.append(float(np.max(display_values)))
        jitter = rng.normal(0, 0.055, size=len(display_values))
        ax.scatter(
            np.full(len(display_values), pos) + jitter,
            display_values,
            s=7.5,
            color=color_map[feature],
            edgecolor="#222222",
            linewidth=0.18,
            alpha=0.42,
            zorder=3,
        )

    means = summary["mean"].to_numpy(dtype=float)
    ax.plot(positions, means, color=COLORS["mean"], lw=1.05, marker="o", markersize=3.2, zorder=5)

    hdi_low = float(summary["hdi_3"].min())
    hdi_high = float(summary["hdi_97"].max())
    y_min = min(hdi_low, min(displayed_min)) - 0.14
    y_max = max(hdi_high, max(displayed_max)) + 0.46
    y_min = min(y_min, -0.12)
    y_max = max(y_max, 0.12)
    ax.set_ylim(y_min, y_max)
    y_range = y_max - y_min

    for pos, (_, row) in zip(positions, summary.iterrows()):
        mean = float(row["mean"])
        low = float(row["hdi_3"])
        high = float(row["hdi_97"])
        ax.errorbar(
            pos,
            mean,
            yerr=np.array([[mean - low], [high - mean]], dtype=float),
            fmt="none",
            ecolor=COLORS["mean"],
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.75,
            zorder=4,
        )
        label = str(row["stars"])
        if label:
            star_y = high + y_range * 0.075
            ax.text(
                pos,
                star_y,
                label,
                ha="center",
                va="bottom",
                fontsize=15.0,
                fontweight="bold",
                color=COLORS["text"],
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                },
                zorder=6,
            )

    ax.axhline(0, color=COLORS["zero"], lw=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=10.8, fontweight="bold")
    ax.set_ylabel("Standardized posterior effect", fontsize=12.2, fontweight="bold")
    ax.set_title("Posterior Effects of 14 Interpretable Facial Features", fontsize=15.5, fontweight="bold", pad=44)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=10.8)

    ax.set_xlim(0.35, len(order) + 0.65)

    legend_handles = [
        mpl.patches.Patch(facecolor=COLORS["positive"], edgecolor="#333333", alpha=0.34, label="Positive 94% HDI"),
        mpl.patches.Patch(facecolor=COLORS["negative"], edgecolor="#333333", alpha=0.34, label="Negative 94% HDI"),
        mpl.patches.Patch(facecolor=COLORS["neutral"], edgecolor="#333333", alpha=0.44, label="HDI crosses 0"),
        mpl.lines.Line2D([0], [0], color=COLORS["mean"], lw=1.05, marker="o", markersize=3.2, label="Posterior mean"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=10.0,
        ncol=4,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0,
    )

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
