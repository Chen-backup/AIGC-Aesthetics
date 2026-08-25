from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent / "Result"
PREFERENCES_PATH = RESULT_DIR / "Rater_10D_Preferences.csv"
PNG_OUTPUT = RESULT_DIR / "Landscape_Alluvial_State_Transitions_10Features.png"
PNG_OUTPUT_TEXTLESS = RESULT_DIR / "Landscape_Alluvial_State_Transitions_10Features_textless.png"
PNG_OUTPUT_FULL_RANGE = RESULT_DIR / "Landscape_Alluvial_State_Transitions_10Features_0_100.png"
PNG_OUTPUT_FULL_RANGE_TEXTLESS = RESULT_DIR / "Landscape_Alluvial_State_Transitions_10Features_0_100_textless.png"
DATA_OUTPUT = RESULT_DIR / "Landscape_Alluvial_State_Transitions_10Features_Data.csv"

FEATURE_ORDER = [
    "warm_cool_balance",
    "horizon_y_norm",
    "depth_gradient_mean",
    "artificial_ratio",
    "saturation_mean",
    "thirds_brightness_mean",
    "line_strength",
    "depth_std",
    "semantic_diversity",
    "left_right_balance",
]

FEATURE_LABELS = {
    "warm_cool_balance": "warm/cool\nbalance",
    "horizon_y_norm": "horizon\ny-position",
    "depth_gradient_mean": "depth\ngradient",
    "artificial_ratio": "artificial\nratio",
    "saturation_mean": "saturation",
    "thirds_brightness_mean": "thirds\nbrightness",
    "line_strength": "line\nstrength",
    "depth_std": "depth\nvariation",
    "semantic_diversity": "semantic\ndiversity",
    "left_right_balance": "left/right\nbalance",
}

STATE_ORDER = ["High", "Medium", "Consensus", "Low"]
STATE_COLORS = {
    "High": "#E982B2",
    "Medium": "#E9D081",
    "Consensus": "#86B676",
    "Low": "#7E9DD3",
}
STATE_LABELS = {
    "High": "High",
    "Medium": "Medium",
    "Consensus": "Consensus",
    "Low": "Low",
}

RANDOM_SEED = 20260606
FIGSIZE = (16.0, 7.2)
X_SPACING = 0.82
NODE_WIDTH = 0.18
COLUMN_GAP = 0.06
NODE_GAP = 0.045
FLOW_ALPHA = 0.34
FLOW_EDGE_ALPHA = 0.05
MIN_FLOW_WIDTH = 0.001
CONSENSUS_SD_QUANTILE = 0.20
SLOPE_THRESHOLD = 0.10
MIN_STATE_PROPORTION = 0.03
MIN_COLUMN_HEIGHT = 0.22
MAX_COLUMN_HEIGHT = 1.00
CONSENSUS_COLUMN_HEIGHT_SCALE = 0.68
NODE_LABEL_MIN_FONT_SIZE = 7.0
NODE_LABEL_MAX_FONT_SIZE = 15.0


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_preferences() -> pd.DataFrame:
    if not PREFERENCES_PATH.exists():
        raise FileNotFoundError(f"Preference matrix not found: {PREFERENCES_PATH}")
    df = pd.read_csv(PREFERENCES_PATH)
    missing = [feature for feature in FEATURE_ORDER if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")


def assign_state_by_mixed_strategy(values: pd.Series, is_consensus_feature: bool) -> pd.Series:
    states = pd.Series(index=values.index, dtype="object")
    if is_consensus_feature:
        states.loc[:] = "Consensus"
        return states

    states.loc[values < -SLOPE_THRESHOLD] = "Low"
    states.loc[(values >= -SLOPE_THRESHOLD) & (values <= SLOPE_THRESHOLD)] = "Medium"
    states.loc[values > SLOPE_THRESHOLD] = "High"

    min_count = int(np.ceil(len(values) * MIN_STATE_PROPORTION))
    counts = states.value_counts()
    for state in ("Low", "High"):
        if counts.get(state, 0) < min_count:
            states.loc[states == state] = "Medium"
    return states


def build_state_table(df: pd.DataFrame) -> pd.DataFrame:
    state_df = pd.DataFrame(index=df.index)
    feature_sd = df.std()
    consensus_sd_threshold = feature_sd.quantile(CONSENSUS_SD_QUANTILE)
    for feature in FEATURE_ORDER:
        state_df[feature] = assign_state_by_mixed_strategy(
            df[feature],
            is_consensus_feature=feature_sd[feature] <= consensus_sd_threshold,
        )
    return state_df


def compute_column_heights(df: pd.DataFrame, range_mode: str = "q05_q95") -> pd.Series:
    if range_mode == "full":
        envelope_width = df.max() - df.min()
    elif range_mode == "q05_q95":
        envelope_width = df.quantile(0.95) - df.quantile(0.05)
    else:
        raise ValueError(f"Unsupported range_mode: {range_mode}")
    max_width = float(envelope_width.max())
    if max_width <= 0:
        return pd.Series(MAX_COLUMN_HEIGHT, index=df.columns)
    scaled = envelope_width / max_width
    return MIN_COLUMN_HEIGHT + (MAX_COLUMN_HEIGHT - MIN_COLUMN_HEIGHT) * scaled


def compute_node_layout(state_df: pd.DataFrame, column_heights: pd.Series) -> tuple[dict, pd.DataFrame]:
    n_raters = len(state_df)
    node_rects = {}
    records = []

    for col_idx, feature in enumerate(FEATURE_ORDER):
        x_center = float(col_idx) * X_SPACING
        counts = state_df[feature].value_counts().to_dict()
        present_states = [state for state in STATE_ORDER if int(counts.get(state, 0)) > 0]
        column_height = float(column_heights[feature])
        if present_states == ["Consensus"]:
            column_height *= CONSENSUS_COLUMN_HEIGHT_SCALE
        y_top = 0.5 + column_height / 2
        total_gap = NODE_GAP * (len(present_states) - 1)
        usable_height = max(column_height - total_gap, column_height * 0.55)

        for state in present_states:
            count = int(counts.get(state, 0))
            height = usable_height * count / n_raters
            y1 = y_top
            y0 = y1 - height
            node_rects[(feature, state)] = {
                "x0": x_center - NODE_WIDTH / 2,
                "x1": x_center + NODE_WIDTH / 2,
                "y0": y0,
                "y1": y1,
                "count": count,
                "column_height": column_height,
            }
            records.append(
                {
                    "feature": feature,
                    "state": state,
                    "count": count,
                    "x": x_center,
                    "y0": y0,
                    "y1": y1,
                    "column_height": column_height,
                }
            )
            y_top = y0 - NODE_GAP

        for state in STATE_ORDER:
            if state not in present_states:
                node_rects[(feature, state)] = {
                    "x0": x_center - NODE_WIDTH / 2,
                    "x1": x_center + NODE_WIDTH / 2,
                    "y0": 0.0,
                    "y1": 0.0,
                    "count": 0,
                    "column_height": column_height,
                }

    return node_rects, pd.DataFrame(records)


def node_label_font_size(rect: dict, state: str) -> float:
    node_height = max(rect["y1"] - rect["y0"], 0.0)
    label_length = len(STATE_LABELS[state])
    size = node_height * 98.0 / max(label_length, 4)
    return float(np.clip(size, NODE_LABEL_MIN_FONT_SIZE, NODE_LABEL_MAX_FONT_SIZE))


def compute_flows(state_df: pd.DataFrame, node_rects: dict) -> tuple[list, pd.DataFrame]:
    n_raters = len(state_df)
    source_offsets = defaultdict(float)
    target_offsets = defaultdict(float)
    flows = []
    records = []

    for i in range(len(FEATURE_ORDER) - 1):
        source_feature = FEATURE_ORDER[i]
        target_feature = FEATURE_ORDER[i + 1]
        pair_counts = (
            state_df.groupby([source_feature, target_feature], observed=False)
            .size()
            .reset_index(name="count")
        )
        pair_counts["_source_order"] = pair_counts[source_feature].map({state: idx for idx, state in enumerate(STATE_ORDER)})
        pair_counts["_target_order"] = pair_counts[target_feature].map({state: idx for idx, state in enumerate(STATE_ORDER)})
        pair_counts = pair_counts.sort_values(["_source_order", "_target_order"])

        for _, row in pair_counts.iterrows():
            source_state = str(row[source_feature])
            target_state = str(row[target_feature])
            count = int(row["count"])
            if count <= 0:
                continue

            source_rect = node_rects[(source_feature, source_state)]
            target_rect = node_rects[(target_feature, target_state)]
            flow_height = max((source_rect["y1"] - source_rect["y0"]) * count / max(source_rect["count"], 1), MIN_FLOW_WIDTH)
            target_height = max((target_rect["y1"] - target_rect["y0"]) * count / max(target_rect["count"], 1), MIN_FLOW_WIDTH)

            s_y1 = source_rect["y1"] - source_offsets[(source_feature, source_state)]
            s_y0 = s_y1 - flow_height
            source_offsets[(source_feature, source_state)] += flow_height

            t_y1 = target_rect["y1"] - target_offsets[(target_feature, target_state)]
            t_y0 = t_y1 - target_height
            target_offsets[(target_feature, target_state)] += target_height

            flow = {
                "source_feature": source_feature,
                "target_feature": target_feature,
                "source_state": source_state,
                "target_state": target_state,
                "count": count,
                "proportion": count / n_raters,
                "x0": source_rect["x1"],
                "x1": target_rect["x0"],
                "s_y0": s_y0,
                "s_y1": s_y1,
                "t_y0": t_y0,
                "t_y1": t_y1,
            }
            flows.append(flow)
            records.append(flow.copy())

    return flows, pd.DataFrame(records)


def flow_color(flow: dict) -> str:
    if flow["source_state"] == "Consensus" and flow["target_state"] != "Consensus":
        return STATE_COLORS[flow["target_state"]]
    return STATE_COLORS[flow["source_state"]]


def draw_flow(ax: plt.Axes, flow: dict) -> None:
    Path = mpath.Path
    x0 = flow["x0"]
    x1 = flow["x1"]
    dx = x1 - x0
    c0 = x0 + dx * 0.45
    c1 = x1 - dx * 0.45

    verts = [
        (x0, flow["s_y1"]),
        (c0, flow["s_y1"]),
        (c1, flow["t_y1"]),
        (x1, flow["t_y1"]),
        (x1, flow["t_y0"]),
        (c1, flow["t_y0"]),
        (c0, flow["s_y0"]),
        (x0, flow["s_y0"]),
        (x0, flow["s_y1"]),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    color = flow_color(flow)
    patch = mpatches.PathPatch(
        Path(verts, codes),
        facecolor=color,
        edgecolor=color,
        lw=0.35,
        alpha=FLOW_ALPHA,
        zorder=1,
    )
    patch.set_edgecolor((*mpl.colors.to_rgba(color)[:3], FLOW_EDGE_ALPHA))
    ax.add_patch(patch)


def draw_nodes(ax: plt.Axes, node_rects: dict) -> None:
    for feature in FEATURE_ORDER:
        for state in STATE_ORDER:
            rect = node_rects[(feature, state)]
            if rect["count"] <= 0:
                continue
            patch = mpatches.FancyBboxPatch(
                (rect["x0"], rect["y0"]),
                NODE_WIDTH,
                rect["y1"] - rect["y0"],
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=STATE_COLORS[state],
                edgecolor="#FFFFFF",
                linewidth=1.0,
                alpha=0.92,
                zorder=3,
            )
            ax.add_patch(patch)

            if rect["count"] >= 60:
                ax.text(
                    (rect["x0"] + rect["x1"]) / 2,
                    (rect["y0"] + rect["y1"]) / 2,
                    STATE_LABELS[state],
                    ha="center",
                    va="center",
                    fontsize=node_label_font_size(rect, state),
                    color="#202020",
                    fontweight="bold",
                    rotation=90,
                    zorder=4,
                )


def draw_feature_labels(ax: plt.Axes) -> None:
    for i, feature in enumerate(FEATURE_ORDER):
        ax.text(
            i * X_SPACING,
            -0.075,
            FEATURE_LABELS[feature],
            ha="center",
            va="top",
            fontsize=16,
            fontweight="normal",
        )


def draw_column_height_guides(ax: plt.Axes, column_heights: pd.Series) -> None:
    for i, feature in enumerate(FEATURE_ORDER):
        height = float(column_heights[feature])
        y0 = 0.5 - height / 2
        y1 = 0.5 + height / 2
        ax.vlines(i * X_SPACING, y0, y1, color="#20242A", lw=0.65, alpha=0.18, zorder=0)


def draw_sankey(
    png_output: Path = PNG_OUTPUT,
    range_mode: str = "q05_q95",
    save_data: bool = True,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    df = load_preferences()
    state_df = build_state_table(df)
    column_heights = compute_column_heights(df, range_mode=range_mode)
    node_rects, node_data = compute_node_layout(state_df, column_heights)
    flows, flow_data = compute_flows(state_df, node_rects)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=450)

    draw_column_height_guides(ax, column_heights)
    large_flows = sorted(flows, key=lambda item: item["count"], reverse=True)
    for flow in reversed(large_flows):
        draw_flow(ax, flow)
    draw_nodes(ax, node_rects)
    draw_feature_labels(ax)

    legend_handles = [
        mpatches.Patch(facecolor=STATE_COLORS[state], edgecolor="none", label=STATE_LABELS[state], alpha=0.92)
        for state in STATE_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        title="Preference state",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=9.8,
        title_fontsize=10.2,
    )
    ax.set_title(
        "Alluvial transitions of rater preference states across landscape features",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlim(-0.45, (len(FEATURE_ORDER) - 1) * X_SPACING + 0.45)
    ax.set_ylim(-0.13, 1.05)
    ax.axis("off")

    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.13, top=0.86)
    fig.savefig(png_output, dpi=450, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    if save_data:
        node_data.to_csv(DATA_OUTPUT.with_name(DATA_OUTPUT.stem + "_Nodes.csv"), index=False, encoding="utf-8-sig")
        flow_data.to_csv(DATA_OUTPUT, index=False, encoding="utf-8-sig")
        state_df.to_csv(DATA_OUTPUT.with_name(DATA_OUTPUT.stem + "_RaterStates.csv"), index=False, encoding="utf-8-sig")

    print(f"Saved figure: {png_output}")
    if save_data:
        print(f"Saved flow data: {DATA_OUTPUT}")
        print(f"Saved node data: {DATA_OUTPUT.with_name(DATA_OUTPUT.stem + '_Nodes.csv')}")
        print(f"Saved rater states: {DATA_OUTPUT.with_name(DATA_OUTPUT.stem + '_RaterStates.csv')}")


def draw_sankey_textless(
    png_output: Path = PNG_OUTPUT_TEXTLESS,
    range_mode: str = "q05_q95",
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    df = load_preferences()
    state_df = build_state_table(df)
    column_heights = compute_column_heights(df, range_mode=range_mode)
    node_rects, _ = compute_node_layout(state_df, column_heights)
    flows, _ = compute_flows(state_df, node_rects)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=450)
    draw_column_height_guides(ax, column_heights)
    large_flows = sorted(flows, key=lambda item: item["count"], reverse=True)
    for flow in reversed(large_flows):
        draw_flow(ax, flow)

    for feature in FEATURE_ORDER:
        for state in STATE_ORDER:
            rect = node_rects[(feature, state)]
            if rect["count"] <= 0:
                continue
            patch = mpatches.FancyBboxPatch(
                (rect["x0"], rect["y0"]),
                NODE_WIDTH,
                rect["y1"] - rect["y0"],
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=STATE_COLORS[state],
                edgecolor="#FFFFFF",
                linewidth=1.0,
                alpha=0.92,
                zorder=3,
            )
            ax.add_patch(patch)

    ax.set_xlim(-0.45, (len(FEATURE_ORDER) - 1) * X_SPACING + 0.45)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.035, top=0.985)
    fig.savefig(png_output, dpi=450, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"Saved textless figure: {png_output}")


if __name__ == "__main__":
    draw_sankey()
    draw_sankey_textless()
    draw_sankey(PNG_OUTPUT_FULL_RANGE, range_mode="full", save_data=False)
    draw_sankey_textless(PNG_OUTPUT_FULL_RANGE_TEXTLESS, range_mode="full")
