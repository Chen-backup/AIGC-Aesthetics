from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT_DIR = Path(__file__).resolve().parent

FACE_CURVE_DATA = ROOT_DIR / "Picture_code" / "picture" / "GAMM_NonLinear_Expected_Curves_Top6_data.csv"
FACE_OUTPUT_DIR = ROOT_DIR / "Picture_fig3"

LANDSCAPE_CURVE_DATA = (
    ROOT_DIR
    / "Landscape_Features_Model"
    / "BYS_GAMM_NonLinear"
    / "Result"
    / "GAMM_NonLinear_Expected_Curves_4Features_data.csv"
)
LANDSCAPE_OUTPUT_DIR = ROOT_DIR / "Landscape_Features_Model" / "BYS_GAMM_NonLinear" / "Result"

PANEL_SIZE = (5.2, 4.4)
DPI = 300

FACE_FEATURES = [
    (4, "mouth_face_w_ratio"),
    (5, "total_symmetry"),
    (6, "edge_density"),
]

LANDSCAPE_FEATURES = [
    (5, "semantic_diversity"),
    (3, "saturation_mean"),
    (2, "horizon_y_norm"),
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def strip_all_text(ax: plt.Axes) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)


def draw_panel(
    *,
    x: pd.Series,
    y: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    output_path: Path,
    line_color: str,
    band_color: str,
    band_alpha: float,
) -> None:
    y_min = float(lower.min())
    y_max = float(upper.max())
    y_pad = max((y_max - y_min) * 0.08, 0.02)

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=DPI)
    ax.fill_between(
        x.to_numpy(dtype=float),
        lower.to_numpy(dtype=float),
        upper.to_numpy(dtype=float),
        color=band_color,
        alpha=band_alpha,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        color=line_color,
        linewidth=2.8,
        zorder=2,
    )

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=3.5, length=7.0)
    for spine in ax.spines.values():
        spine.set_linewidth(3.5)
    strip_all_text(ax)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.985, bottom=0.14)
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def draw_face_panels() -> None:
    curve_df = pd.read_csv(FACE_CURVE_DATA)
    for figure_index, feature in FACE_FEATURES:
        panel = curve_df[
            (curve_df["feature_name"] == feature) & (curve_df["curve_index"] >= 0)
        ].sort_values("curve_index")
        if panel.empty:
            raise ValueError(f"Face GAMM curve not found: {feature}")

        output = FACE_OUTPUT_DIR / f"Fig3_{figure_index:02d}_{feature}_textless_no_rug_no_text.png"
        draw_panel(
            x=panel["x"],
            y=panel["mean_expected_score"],
            lower=panel["lower_95"],
            upper=panel["upper_95"],
            output_path=output,
            line_color="#1F6EAA",
            band_color="#8DB9E2",
            band_alpha=0.42,
        )


def draw_landscape_panels() -> None:
    curve_df = pd.read_csv(LANDSCAPE_CURVE_DATA)
    for figure_index, feature in LANDSCAPE_FEATURES:
        panel = curve_df[curve_df["feature"] == feature].sort_values("x_real")
        if panel.empty:
            raise ValueError(f"Landscape GAMM curve not found: {feature}")

        output = LANDSCAPE_OUTPUT_DIR / f"Fig3_{figure_index:02d}_{feature}_textless_no_rug_no_text.png"
        draw_panel(
            x=panel["x_real"],
            y=panel["expected_score"],
            lower=panel["ci_lower"],
            upper=panel["ci_upper"],
            output_path=output,
            line_color="#DC1F1E",
            band_color="#DCC7D9",
            band_alpha=0.68,
        )


def main() -> None:
    if not FACE_CURVE_DATA.exists():
        raise FileNotFoundError(f"Face curve data not found: {FACE_CURVE_DATA}")
    if not LANDSCAPE_CURVE_DATA.exists():
        raise FileNotFoundError(f"Landscape curve data not found: {LANDSCAPE_CURVE_DATA}")

    set_style()
    draw_face_panels()
    draw_landscape_panels()


if __name__ == "__main__":
    main()
