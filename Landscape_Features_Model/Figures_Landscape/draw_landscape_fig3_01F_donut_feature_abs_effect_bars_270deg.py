from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
LANDSCAPE_DIR = SCRIPT_DIR.parent
ROOT_DIR = LANDSCAPE_DIR.parent
FACE_FIGURE_DIR = ROOT_DIR / "Picture_fig3"
sys.path.insert(0, str(FACE_FIGURE_DIR))

import draw_fig3_01D_donut_feature_means_sd_effect_values as base  # noqa: E402
import draw_fig3_01F_donut_feature_abs_effect_bars_270deg as face_donut  # noqa: E402


FEATURES = [
    {"code": "warm_cool_balance", "label": "Warm-cool color balance", "short_label": "Warm-cool"},
    {"code": "horizon_y_norm", "label": "Normalized horizon position", "short_label": "Horizon"},
    {"code": "depth_gradient_mean", "label": "Mean depth gradient", "short_label": "Depth gradient"},
    {"code": "artificial_ratio", "label": "Artificial element ratio", "short_label": "Artificial"},
    {"code": "saturation_mean", "label": "Mean color saturation", "short_label": "Saturation"},
    {"code": "thirds_brightness_mean", "label": "Rule-of-thirds brightness", "short_label": "Thirds brightness"},
    {"code": "line_strength", "label": "Compositional line strength", "short_label": "Line strength"},
    {"code": "depth_std", "label": "Depth variation", "short_label": "Depth variation"},
    {"code": "semantic_diversity", "label": "Semantic diversity", "short_label": "Semantic diversity"},
    {"code": "left_right_balance", "label": "Left-right visual balance", "short_label": "Left-right balance"},
]

PNG_OUTPUT = SCRIPT_DIR / "Fig3_01F_landscape_donut_feature_abs_effect_bars_270deg.png"
CSV_OUTPUT = SCRIPT_DIR / "Fig3_01F_landscape_donut_feature_abs_effect_bars_270deg_data.csv"


def configure_landscape_inputs() -> None:
    base.INTERPRETABLE_FEATURES_PATH = (
        LANDSCAPE_DIR / "BYS_interpretable_Features_Model" / "landscape_interpretable_features.csv"
    )
    base.TRACE_PATH = LANDSCAPE_DIR / "BYS_interpretable_Features_Model" / "Result" / "full_model_trace.nc"
    base.FEATURES = FEATURES
    base.FEATURE_ORDER_BY_IMPORTANCE_RANK = list(range(1, len(FEATURES) + 1))
    face_donut.PNG_OUTPUT = PNG_OUTPUT
    face_donut.CSV_OUTPUT = CSV_OUTPUT


def main() -> None:
    configure_landscape_inputs()
    values_by_feature = base.load_posterior_values()
    plot_df = base.build_plot_table(values_by_feature)
    plot_df = (
        plot_df.assign(abs_effect_value=plot_df["effect_value"].abs())
        .sort_values("abs_effect_value", ascending=False)
        .reset_index(drop=True)
    )
    plot_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    face_donut.draw_chart_abs_effect_bars(plot_df, values_by_feature)
    print(plot_df[["source_feature", "effect_value", "significance", "importance_rank"]].to_string(index=False))
    print(f"Saved data: {CSV_OUTPUT}")
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
