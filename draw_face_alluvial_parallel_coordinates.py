from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RESULT_DIR = Path("BYS_Heterogeneity_Evidence")
PREFERENCES_PATH = Path("BYS_Clustering_Results_Advanced") / "Rater_14D_Preferences.csv"

QUANTILE_OUTPUT = RESULT_DIR / "Face_Alluvial_Quantile_Bands_14Features.png"
CLUSTER_OUTPUT = RESULT_DIR / "Face_Alluvial_Cluster_Bands_14Features.png"
QUANTILE_DATA_OUTPUT = RESULT_DIR / "Face_Alluvial_Quantile_Bands_14Features_Data.csv"
CLUSTER_DATA_OUTPUT = RESULT_DIR / "Face_Alluvial_Cluster_Bands_14Features_Data.csv"
CLUSTER_ASSIGNMENT_OUTPUT = RESULT_DIR / "Face_Alluvial_Cluster_Assignments.csv"

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

FEATURE_LABELS = {
    "face_hw_ratio": "face\nH/W",
    "eye_face_w_ratio": "eye/face\nwidth",
    "mouth_face_w_ratio": "mouth/face\nwidth",
    "three_courts_balance": "three-courts\nbalance",
    "upper_lower_ratio": "upper/lower\nratio",
    "eye_y_ratio": "eye\ny-position",
    "total_symmetry": "total\nsymmetry",
    "le_nose_re_angle": "eye-nose\nangle",
    "mouth_nose_ratio": "mouth/nose\nratio",
    "face_brightness": "face\nbrightness",
    "face_contrast": "face\ncontrast",
    "face_clarity": "face\nclarity",
    "saturation": "saturation",
    "edge_density": "edge\ndensity",
}

RANDOM_SEED = 20260606
SCATTER_N_PER_AXIS = 120
CLUSTER_N = 5
Y_MIN = -0.45
Y_MAX = 0.45

AXIS_COLOR = "#30333A"
GRID_COLOR = "#E8ECF2"
SCATTER_COLOR = "#2A2D33"

QUANTILE_BOUNDS = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
QUANTILE_COLORS = ["#3C7FB8", "#86B8CF", "#E9D6A7", "#F0A66A", "#D24A43"]
QUANTILE_ALPHA = 0.48

CLUSTER_COLORS = ["#C84B4B", "#D99B42", "#62AD79", "#54A6B6", "#6389C8"]
CLUSTER_ALPHA = 0.36
CLUSTER_LINE_WIDTH = 3.5


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_preferences() -> pd.DataFrame:
    if not PREFERENCES_PATH.exists():
        raise FileNotFoundError(f"Preference matrix not found: {PREFERENCES_PATH}")
    df = pd.read_csv(PREFERENCES_PATH, index_col=0)
    missing = [feature for feature in FEATURE_ORDER if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")


def smooth_fill_between(
    ax: plt.Axes,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    alpha: float,
    zorder: int = 2,
) -> None:
    dense_x = np.linspace(float(x.min()), float(x.max()), 900)
    lower_dense = PchipInterpolator(x, lower)(dense_x)
    upper_dense = PchipInterpolator(x, upper)(dense_x)
    low = np.minimum(lower_dense, upper_dense)
    high = np.maximum(lower_dense, upper_dense)
    ax.fill_between(dense_x, low, high, color=color, alpha=alpha, linewidth=0, zorder=zorder)


def smooth_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, lw: float, alpha: float = 1.0) -> None:
    dense_x = np.linspace(float(x.min()), float(x.max()), 900)
    dense_y = PchipInterpolator(x, y)(dense_x)
    ax.plot(dense_x, dense_y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=5)


def draw_axes_and_sampled_points(ax: plt.Axes, df: pd.DataFrame, x: np.ndarray, rng: np.random.Generator) -> None:
    for xi, feature in zip(x, FEATURE_ORDER):
        ax.vlines(xi, Y_MIN, Y_MAX, color=AXIS_COLOR, lw=1.65, alpha=0.72, zorder=3)
        values = df[feature].dropna().to_numpy(dtype=float)
        if len(values) > SCATTER_N_PER_AXIS:
            values = rng.choice(values, size=SCATTER_N_PER_AXIS, replace=False)
        jitter = rng.normal(0, 0.022, size=len(values))
        ax.scatter(
            np.full_like(values, xi, dtype=float) + jitter,
            values,
            s=10,
            color=SCATTER_COLOR,
            alpha=0.22,
            linewidths=0,
            zorder=4,
        )


def format_axis(ax: plt.Axes, title: str, x: np.ndarray) -> None:
    ax.axhline(0, color="#7D3E3E", linestyle="--", lw=1.35, alpha=0.82, zorder=1)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(float(x.min()) - 0.45, float(x.max()) + 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[feature] for feature in FEATURE_ORDER], fontsize=9.1, fontweight="bold")
    ax.set_ylabel("Rater-specific preference slope", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.8, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_axisbelow(True)


def draw_quantile_band_figure(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    x = np.arange(len(FEATURE_ORDER), dtype=float)
    records = []

    fig, ax = plt.subplots(figsize=(13.8, 6.4), dpi=450)

    for band_idx, (q_low, q_high) in enumerate(zip(QUANTILE_BOUNDS[:-1], QUANTILE_BOUNDS[1:])):
        lower = df.quantile(q_low).loc[FEATURE_ORDER].to_numpy(dtype=float)
        upper = df.quantile(q_high).loc[FEATURE_ORDER].to_numpy(dtype=float)
        mid = df.quantile((q_low + q_high) / 2).loc[FEATURE_ORDER].to_numpy(dtype=float)
        smooth_fill_between(ax, x, lower, upper, QUANTILE_COLORS[band_idx], QUANTILE_ALPHA, zorder=2)
        smooth_line(ax, x, mid, QUANTILE_COLORS[band_idx], lw=1.9, alpha=0.78)

        for feature, xi, lo, md, hi in zip(FEATURE_ORDER, x, lower, mid, upper):
            records.append(
                {
                    "band": band_idx + 1,
                    "q_low": q_low,
                    "q_high": q_high,
                    "feature": feature,
                    "x": xi,
                    "lower": lo,
                    "median_within_band": md,
                    "upper": hi,
                }
            )

    draw_axes_and_sampled_points(ax, df, x, rng)
    format_axis(ax, "Quantile-band alluvial parallel coordinates of face preferences", x)

    legend_handles = [
        mpl.patches.Patch(color=color, alpha=QUANTILE_ALPHA, label=f"{int(lo*100)}-{int(hi*100)}%")
        for color, lo, hi in zip(QUANTILE_COLORS, QUANTILE_BOUNDS[:-1], QUANTILE_BOUNDS[1:])
    ]
    ax.legend(
        handles=legend_handles,
        title="Preference quantile band",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        fontsize=8.8,
        title_fontsize=9.6,
    )

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.84)
    fig.savefig(QUANTILE_OUTPUT, dpi=450, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return pd.DataFrame.from_records(records)


def ordered_kmeans_labels(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    scaled = StandardScaler().fit_transform(df)
    raw_labels = KMeans(n_clusters=CLUSTER_N, random_state=RANDOM_SEED, n_init=50).fit_predict(scaled)
    pc1 = PCA(n_components=1, random_state=RANDOM_SEED).fit_transform(scaled).ravel()
    cluster_pc1 = pd.Series(pc1).groupby(raw_labels).mean().sort_values()
    label_map = {old_label: new_label for new_label, old_label in enumerate(cluster_pc1.index)}
    labels = np.array([label_map[label] for label in raw_labels], dtype=int)
    assignment = pd.DataFrame({"rater_index": np.arange(len(df)), "cluster": labels, "pc1_score": pc1})
    return labels, assignment


def draw_cluster_band_figure(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + 17)
    x = np.arange(len(FEATURE_ORDER), dtype=float)
    labels, assignment = ordered_kmeans_labels(df)
    records = []

    fig, ax = plt.subplots(figsize=(13.8, 6.4), dpi=450)

    for cluster_id in range(CLUSTER_N):
        cluster_df = df.loc[labels == cluster_id, FEATURE_ORDER]
        q25 = cluster_df.quantile(0.25).to_numpy(dtype=float)
        q50 = cluster_df.quantile(0.50).to_numpy(dtype=float)
        q75 = cluster_df.quantile(0.75).to_numpy(dtype=float)
        color = CLUSTER_COLORS[cluster_id]

        smooth_fill_between(ax, x, q25, q75, color, CLUSTER_ALPHA, zorder=2 + cluster_id)
        smooth_line(ax, x, q50, color, lw=CLUSTER_LINE_WIDTH, alpha=0.95)

        for feature, xi, lo, md, hi in zip(FEATURE_ORDER, x, q25, q50, q75):
            records.append(
                {
                    "cluster": cluster_id + 1,
                    "n_raters": int((labels == cluster_id).sum()),
                    "feature": feature,
                    "x": xi,
                    "q25": lo,
                    "median": md,
                    "q75": hi,
                }
            )

    for xi, feature in zip(x, FEATURE_ORDER):
        for cluster_id in range(CLUSTER_N):
            values = df.loc[labels == cluster_id, feature].dropna().to_numpy(dtype=float)
            sample_n = min(35, len(values))
            if len(values) > sample_n:
                values = rng.choice(values, size=sample_n, replace=False)
            jitter = rng.normal(0, 0.020, size=len(values))
            ax.scatter(
                np.full_like(values, xi, dtype=float) + jitter,
                values,
                s=10,
                color=CLUSTER_COLORS[cluster_id],
                alpha=0.28,
                linewidths=0,
                zorder=4,
            )

    for xi in x:
        ax.vlines(xi, Y_MIN, Y_MAX, color=AXIS_COLOR, lw=1.65, alpha=0.72, zorder=3)
    format_axis(ax, "Cluster-band alluvial parallel coordinates of face preferences", x)

    legend_handles = [
        mpl.patches.Patch(
            color=CLUSTER_COLORS[cluster_id],
            alpha=CLUSTER_ALPHA,
            label=f"Cluster {cluster_id + 1} (n={(labels == cluster_id).sum()})",
        )
        for cluster_id in range(CLUSTER_N)
    ]
    ax.legend(
        handles=legend_handles,
        title="Rater preference cluster",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=CLUSTER_N,
        fontsize=9,
        title_fontsize=9.8,
    )

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.84)
    fig.savefig(CLUSTER_OUTPUT, dpi=450, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return pd.DataFrame.from_records(records), assignment


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    df = load_preferences()

    quantile_data = draw_quantile_band_figure(df)
    cluster_data, assignment = draw_cluster_band_figure(df)

    quantile_data.to_csv(QUANTILE_DATA_OUTPUT, index=False, encoding="utf-8-sig")
    cluster_data.to_csv(CLUSTER_DATA_OUTPUT, index=False, encoding="utf-8-sig")
    assignment.to_csv(CLUSTER_ASSIGNMENT_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved quantile-band figure: {QUANTILE_OUTPUT}")
    print(f"Saved cluster-band figure: {CLUSTER_OUTPUT}")
    print(f"Saved quantile data: {QUANTILE_DATA_OUTPUT}")
    print(f"Saved cluster data: {CLUSTER_DATA_OUTPUT}")
    print(f"Saved cluster assignments: {CLUSTER_ASSIGNMENT_OUTPUT}")


if __name__ == "__main__":
    main()
