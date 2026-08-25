import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_DIR = "BYS_Heterogeneity_Evidence"
CSV_PATH = os.path.join("BYS_Clustering_Results_Advanced", "Rater_14D_Preferences.csv")
OUTPUT_PATH = os.path.join(SAVE_DIR, "Plot1B_Parallel_Coordinates_SD_Sorted_Enhanced.png")

FACE_MAIN_COLOR = "#88A0CB"
TOP_HIGHLIGHT_COLOR = "#F0D6E0"
MEDIAN_COLOR = "#1E2430"
BASELINE_COLOR = "#8C3D3D"

TOP_HETEROGENEITY_N = 3


def center_low_sd_and_push_high_sd_to_edges(sd_series):
    """Place low-disagreement features near the center and high-disagreement features at both edges."""
    low_to_high = sd_series.sort_values(ascending=True).index.tolist()
    arranged = []
    for i, feature in enumerate(low_to_high):
        if i % 2 == 0:
            arranged.insert(0, feature)
        else:
            arranged.append(feature)
    return arranged[::-1]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df_preferences = pd.read_csv(CSV_PATH, index_col=0)
    excluded_columns = ["Cluster", "tSNE_1", "tSNE_2", "Final_Cluster"]
    feature_columns = [col for col in df_preferences.columns if col not in excluded_columns]
    df_raw = df_preferences[feature_columns].apply(pd.to_numeric, errors="coerce")

    feature_sd = df_raw.std()
    sd_sorted_features = center_low_sd_and_push_high_sd_to_edges(feature_sd)
    top_heterogeneity_features = set(feature_sd.sort_values(ascending=False).head(TOP_HETEROGENEITY_N).index)
    df_sorted = df_raw[sd_sorted_features]

    x = np.arange(df_sorted.shape[1])
    values = df_sorted.to_numpy(dtype=float)
    q05 = df_sorted.quantile(0.05).to_numpy(dtype=float)
    q25 = df_sorted.quantile(0.25).to_numpy(dtype=float)
    q50 = df_sorted.quantile(0.50).to_numpy(dtype=float)
    q75 = df_sorted.quantile(0.75).to_numpy(dtype=float)
    q95 = df_sorted.quantile(0.95).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=450)

    for xi, feature in enumerate(sd_sorted_features):
        if feature in top_heterogeneity_features:
            ax.axvspan(xi - 0.42, xi + 0.42, color=TOP_HIGHLIGHT_COLOR, alpha=0.34, linewidth=0, zorder=0)

    for row in values:
        ax.plot(x, row, color=FACE_MAIN_COLOR, alpha=0.030, lw=0.64, zorder=1)

    ax.fill_between(x, q05, q95, color=FACE_MAIN_COLOR, alpha=0.16, linewidth=0, zorder=2)
    ax.fill_between(x, q25, q75, color=FACE_MAIN_COLOR, alpha=0.32, linewidth=0, zorder=3)
    ax.plot(
        x,
        q50,
        color=MEDIAN_COLOR,
        lw=2.45,
        marker="o",
        markersize=4.2,
        markerfacecolor="#FFFFFF",
        markeredgecolor=MEDIAN_COLOR,
        markeredgewidth=1.05,
        zorder=5,
    )
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.7, alpha=0.9, zorder=4)

    y_min = -0.4
    y_max = 0.4
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(sd_sorted_features, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_xlim(0, len(sd_sorted_features) - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Consensus-Centered Parallel Coordinates of Individual Aesthetic Preferences",
        fontsize=16,
        fontweight="bold",
        pad=13,
    )
    ax.set_ylabel("Preference Slope (BLUPs)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for xi in x:
        ax.axvline(xi, color="#D8DEE8", linestyle=":", lw=0.75, alpha=0.78, zorder=0)
    ax.grid(axis="y", color="#E8ECF2", lw=0.7, alpha=0.95)
    ax.set_axisbelow(True)

    for xi, feature in enumerate(sd_sorted_features):
        if feature in top_heterogeneity_features:
            ax.text(
                xi,
                y_max - 0.02,
                "High\nSD",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#9B5270",
                fontweight="bold",
                linespacing=0.82,
            )

    legend_handles = [
        mpl.lines.Line2D([0], [0], color=FACE_MAIN_COLOR, lw=1.0, alpha=0.30, label="Individual raters"),
        mpl.patches.Patch(facecolor=FACE_MAIN_COLOR, edgecolor="none", alpha=0.16, label="5-95% envelope"),
        mpl.patches.Patch(facecolor=FACE_MAIN_COLOR, edgecolor="none", alpha=0.32, label="25-75% envelope"),
        mpl.lines.Line2D(
            [0],
            [0],
            color=MEDIAN_COLOR,
            lw=2.45,
            marker="o",
            markersize=4.2,
            markerfacecolor="#FFFFFF",
            markeredgecolor=MEDIAN_COLOR,
            label="Median",
        ),
        mpl.patches.Patch(facecolor=TOP_HIGHLIGHT_COLOR, edgecolor="none", alpha=0.34, label="Top heterogeneity"),
        mpl.lines.Line2D([0], [0], color=BASELINE_COLOR, lw=1.7, linestyle="--", label="Neutral baseline"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        fontsize=8.2,
        ncol=3,
        columnspacing=0.75,
        handlelength=1.55,
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.29, top=0.89)
    fig.savefig(OUTPUT_PATH, dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    data_path = os.path.join(SAVE_DIR, "Plot1B_Parallel_Coordinates_SD_Sorted_Enhanced_Data.csv")
    summary = pd.DataFrame(
        {
            "feature": sd_sorted_features,
            "heterogeneity_sd": feature_sd[sd_sorted_features].to_numpy(dtype=float),
            "q05": q05,
            "q25": q25,
            "median": q50,
            "q75": q75,
            "q95": q95,
        }
    )
    summary.to_csv(data_path, index=False, encoding="utf-8-sig")

    print(f"Saved enhanced parallel coordinates figure: {OUTPUT_PATH}")
    print(f"Saved summary data: {data_path}")


if __name__ == "__main__":
    main()
