from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from draw_combined_face_landscape_parallel_coordinates import (
    BASELINE_COLOR,
    CENTER_GAP,
    DATA_OUTPUT,
    FACE_COLOR,
    FACE_LINE_COLOR,
    GRID_COLOR,
    LANDSCAPE_COLOR,
    LANDSCAPE_LINE_COLOR,
    MEDIAN_COLOR,
    RESULT_DIR,
    X_SPACING,
    Y_MAX,
    Y_MIN,
    compute_summary,
    draw_group,
    load_face_preferences,
    load_landscape_preferences,
    set_style,
)


PNG_OUTPUT_TEXTLESS = RESULT_DIR / "Combined_Face_Landscape_Consensus_Centered_Parallel_Coordinates_textless.png"

HIGH_SD_SHADE_COLOR = "#FAA200"
HIGH_SD_SHADE_ALPHA = 0.40
LOW_SD_SHADE_COLOR = "#FFEE00"
LOW_SD_SHADE_ALPHA = 0.25
SHADE_HALF_WIDTH = X_SPACING / 2
SHOW_SD_SHADES = False


def draw_connected_shades(ax, x_positions, feature_order, sd_values, low_threshold, high_threshold):
    shade_groups = []
    current_kind = None
    start_x = None
    last_x = None

    for x, feature in zip(x_positions, feature_order):
        sd = sd_values[feature]
        if sd >= high_threshold:
            kind = "high"
        elif sd <= low_threshold:
            kind = "low"
        else:
            kind = None

        if kind != current_kind:
            if current_kind is not None:
                shade_groups.append((current_kind, start_x, last_x))
            current_kind = kind
            start_x = x if kind is not None else None

        if kind is not None:
            last_x = x

    if current_kind is not None:
        shade_groups.append((current_kind, start_x, last_x))

    for kind, start, end in shade_groups:
        color = HIGH_SD_SHADE_COLOR if kind == "high" else LOW_SD_SHADE_COLOR
        alpha = HIGH_SD_SHADE_ALPHA if kind == "high" else LOW_SD_SHADE_ALPHA
        ax.axvspan(
            start - SHADE_HALF_WIDTH,
            end + SHADE_HALF_WIDTH,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    face_df = load_face_preferences()
    landscape_df = load_landscape_preferences()

    face_sd = face_df.std()
    landscape_sd = landscape_df.std()

    landscape_order = landscape_sd.sort_values(ascending=True).index.tolist()
    face_order = face_sd.sort_values(ascending=False).index.tolist()

    landscape_sorted = landscape_df[landscape_order]
    face_sorted = face_df[face_order]

    landscape_x = -CENTER_GAP - np.arange(len(landscape_order) - 1, -1, -1) * X_SPACING
    face_x = CENTER_GAP + np.arange(len(face_order)) * X_SPACING
    all_x = np.concatenate([landscape_x, face_x])

    fig, ax = plt.subplots(figsize=(16.0, 4.5), dpi=450)

    landscape_low_sd_threshold = landscape_sd.quantile(0.20)
    landscape_high_sd_threshold = landscape_sd.quantile(0.80)
    face_low_sd_threshold = face_sd.quantile(0.20)
    face_high_sd_threshold = face_sd.quantile(0.80)

    if SHOW_SD_SHADES:
        draw_connected_shades(
            ax,
            landscape_x,
            landscape_order,
            landscape_sd,
            landscape_low_sd_threshold,
            landscape_high_sd_threshold,
        )
        draw_connected_shades(ax, face_x, face_order, face_sd, face_low_sd_threshold, face_high_sd_threshold)

    draw_group(ax, landscape_sorted, landscape_x, LANDSCAPE_COLOR, LANDSCAPE_LINE_COLOR, "Landscape")
    draw_group(ax, face_sorted, face_x, FACE_COLOR, FACE_LINE_COLOR, "Face")
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=2.4, alpha=0.95, zorder=4)

    ax.set_xticks(all_x)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(landscape_x.min(), face_x.max())
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3.2)
    ax.spines["bottom"].set_linewidth(3.2)
    ax.tick_params(axis="both", which="major", length=8, width=2.8, labelbottom=False, labelleft=False)

    for xi in all_x:
        ax.axvline(xi, color="#D8DEE8", linestyle=":", lw=0.72, alpha=0.72, zorder=0)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.72, alpha=0.95)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.012, right=0.998, bottom=0.045, top=0.985)
    fig.savefig(PNG_OUTPUT_TEXTLESS, dpi=450, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)

    landscape_summary = compute_summary(landscape_sorted).reset_index(names="feature")
    landscape_summary.insert(0, "image_type", "Landscape")
    landscape_summary.insert(1, "x_position", landscape_x)

    face_summary = compute_summary(face_sorted).reset_index(names="feature")
    face_summary.insert(0, "image_type", "Face")
    face_summary.insert(1, "x_position", face_x)

    pd.concat([landscape_summary, face_summary], ignore_index=True).to_csv(
        DATA_OUTPUT, index=False, encoding="utf-8-sig"
    )

    print(f"Saved textless figure: {PNG_OUTPUT_TEXTLESS}")


if __name__ == "__main__":
    main()
