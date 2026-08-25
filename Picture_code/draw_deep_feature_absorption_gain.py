from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

NULL_REPORT = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Metrics_Report.txt"
INTERPRETABLE_REPORT = ROOT_DIR / "BYS_interpretable_model_result" / "Full_Model_Metrics_Report.txt"

MODEL_REPORTS = [
    {
        "Model": "StyleGAN",
        "DeepOnly": ROOT_DIR / "BYS_StyleGAN_model_14D_result" / "StyleGAN_14D_Model_Metrics_Report.txt",
        "Fusion": ROOT_DIR / "BYS_Fusion_28D_StyleGAN_result" / "Fusion_28D_Model_Metrics.txt",
    },
    {
        "Model": "InsightFace",
        "DeepOnly": ROOT_DIR / "BYS_Insightface_model_14D_result" / "Deep_14D_Model_Metrics_Report.txt",
        "Fusion": ROOT_DIR / "BYS_Fusion_28D_InsightFace_result" / "Fusion_28D_Model_Metrics.txt",
    },
    {
        "Model": "DINOv2",
        "DeepOnly": ROOT_DIR / "BYS_DINOv2_model_14D_result" / "DINOv2_14D_Model_Metrics_Report.txt",
        "Fusion": ROOT_DIR / "BYS_Fusion_28D_DINOv2_result" / "Fusion_28D_Model_Metrics.txt",
    },
]

PNG_OUTPUT = OUTPUT_DIR / "Deep_Feature_Absorption_Gain.png"
CSV_OUTPUT = OUTPUT_DIR / "Deep_Feature_Absorption_Gain_Data.csv"

COLORS = {
    "deep": "#D55E00",
    "fusion": "#0072B2",
    "line": "#8A9099",
    "interpretable": "#2F5D90",
    "grid": "#E7E9ED",
    "text": "#222222",
    "gain": "#3F4854",
}


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find {label}")
    return float(match.group(1))


def residual_image_variance(report_path: Path) -> float:
    return extract_float(r"Residual Image Variance .*?:\s*([0-9.]+)", read_text(report_path), f"residual variance in {report_path}")


def load_plot_data() -> pd.DataFrame:
    null_image_variance = extract_float(r"Image Variance .*?:\s*([0-9.]+)", read_text(NULL_REPORT), "null image variance")
    interpretable_residual = residual_image_variance(INTERPRETABLE_REPORT)
    interpretable_r2 = (1.0 - interpretable_residual / null_image_variance) * 100.0

    rows: list[dict[str, float | str]] = []
    for spec in MODEL_REPORTS:
        deep_residual = residual_image_variance(spec["DeepOnly"])
        fusion_residual = residual_image_variance(spec["Fusion"])
        deep_r2 = (1.0 - deep_residual / null_image_variance) * 100.0
        fusion_r2 = (1.0 - fusion_residual / null_image_variance) * 100.0
        rows.append(
            {
                "Model": spec["Model"],
                "DeepOnly_R2": deep_r2,
                "Fusion_R2": fusion_r2,
                "FusionGain": fusion_r2 - deep_r2,
                "DeepOnly_ResidualImageVariance": deep_residual,
                "Fusion_ResidualImageVariance": fusion_residual,
                "InterpretableOnly_R2": interpretable_r2,
            }
        )
    return pd.DataFrame(rows)


def draw_figure(df: pd.DataFrame) -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)

    ordered = df.sort_values("DeepOnly_R2", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))

    fig = plt.figure(figsize=(6.9, 2.95), dpi=600)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[3.7, 1.0],
        left=0.16,
        right=0.98,
        bottom=0.25,
        top=0.78,
        wspace=0.10,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1], sharey=ax)

    for idx, row in ordered.iterrows():
        ax.plot(
            [row["DeepOnly_R2"], row["Fusion_R2"]],
            [idx, idx],
            color=COLORS["line"],
            lw=2.2,
            solid_capstyle="round",
            zorder=1,
        )

    ax.scatter(ordered["DeepOnly_R2"], y, s=44, color=COLORS["deep"], edgecolor="#222222", linewidth=0.45, label="Deep only", zorder=3)
    ax.scatter(
        ordered["Fusion_R2"],
        y,
        s=48,
        color=COLORS["fusion"],
        edgecolor="#222222",
        linewidth=0.45,
        label="Deep + interpretable",
        zorder=4,
    )

    interpretable_r2 = float(ordered["InterpretableOnly_R2"].iloc[0])
    fig.suptitle("Deep Models Absorb Most Interpretable Facial Information", x=0.16, y=0.965, ha="left", fontsize=10.6, fontweight="bold")
    fig.text(
        0.16,
        0.875,
        f"Interpretable-only benchmark = {interpretable_r2:.2f}% explained variance",
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["interpretable"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Model"], fontsize=8.4)
    ax.set_xlabel("Explained image-level variance (%)", fontsize=8.4)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.65)
    ax.tick_params(axis="x", labelsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(74, 90.2)
    ax.set_ylim(-0.55, len(ordered) - 0.45)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.78, 0.18),
        ncol=2,
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0,
    )
    ax.set_axisbelow(True)

    ax_gain.barh(y, ordered["FusionGain"], height=0.48, color=COLORS["gain"], alpha=0.92)
    for idx, gain in enumerate(ordered["FusionGain"]):
        ax_gain.text(gain + 0.08, idx, f"+{gain:.2f}", va="center", ha="left", fontsize=7.7, color=COLORS["text"])
    ax_gain.axvline(0, color="#222222", lw=0.8)
    ax_gain.set_xlim(0, 4.65)
    ax_gain.set_xlabel("Gain (pp)", fontsize=8.4)
    ax_gain.tick_params(axis="x", labelsize=7.8)
    ax_gain.tick_params(axis="y", left=False, labelleft=False)
    ax_gain.grid(axis="x", color=COLORS["grid"], lw=0.65)
    ax_gain.set_axisbelow(True)

    fig.savefig(PNG_OUTPUT)
    plt.close(fig)


def main() -> None:
    data = load_plot_data()
    draw_figure(data)
    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved data: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
