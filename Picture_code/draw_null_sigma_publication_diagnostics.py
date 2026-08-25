from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT_DIR / "BYS_kong_2_result" / "null_model_trace.nc"
OUTPUT_DIR = ROOT_DIR / "Picture_code" / "picture"

PNG_OUTPUT = OUTPUT_DIR / "Null_Sigma_Posterior_Diagnostics_Publication.png"
CSV_OUTPUT = OUTPUT_DIR / "Null_Sigma_Posterior_Diagnostics_Summary.csv"

PARAMS = [
    {
        "var": "1|rater_sigma",
        "label": "Rater random-effect SD",
        "short": "Rater SD",
        "color": "#C84C61",
        "fill": "#EFC1CA",
    },
    {
        "var": "1|image_sigma",
        "label": "Image random-effect SD",
        "short": "Image SD",
        "color": "#2F78B7",
        "fill": "#BDD7EE",
    },
]

CHAIN_COLORS = ["#3B528B", "#21918C", "#5DC863", "#FDE725"]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def flatten_draws(trace: az.InferenceData, var: str) -> np.ndarray:
    values = np.asarray(trace.posterior[var].values, dtype=float)
    return values.reshape(values.shape[0], -1)


def posterior_summary(values_by_chain: np.ndarray, hdi_prob: float = 0.94) -> dict[str, float]:
    values = values_by_chain.reshape(-1)
    hdi = az.hdi(values, hdi_prob=hdi_prob)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)),
        "hdi_low": float(hdi[0]),
        "hdi_high": float(hdi[1]),
    }


def draw_ridge(
    ax: plt.Axes,
    values_by_chain: np.ndarray,
    y: float,
    color: str,
    fill: str,
    x_grid: np.ndarray,
    width: float = 0.30,
) -> None:
    chain_densities = np.vstack([gaussian_kde(chain_values)(x_grid) for chain_values in values_by_chain])
    mean_density = chain_densities.mean(axis=0)
    scale = max(float(chain_densities.max()), float(mean_density.max()), 1e-12)
    chain_densities = chain_densities / scale * width
    mean_density = mean_density / scale * width

    ax.fill_between(x_grid, y, y + mean_density, color=fill, alpha=0.78, linewidth=0, zorder=1)
    for density in chain_densities:
        ax.plot(x_grid, y + density, color=color, lw=0.78, alpha=0.58, linestyle=(0, (3, 2)), zorder=2)
    ax.plot(x_grid, y + mean_density, color=color, lw=1.7, zorder=3)


def draw_posterior_panel(
    ax: plt.Axes,
    trace: az.InferenceData,
    param: dict[str, str],
    summaries: pd.DataFrame,
    show_xlabel: bool = False,
) -> None:
    chain_values = flatten_draws(trace, param["var"])
    values = chain_values.reshape(-1)
    row = summaries.loc[param["var"]]

    x_min = float(np.quantile(values, 0.001))
    x_max = float(np.quantile(values, 0.999))
    pad = (x_max - x_min) * 0.18
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)
    y = 0.0

    draw_ridge(ax, chain_values, y, param["color"], param["fill"], x_grid, width=0.55)

    ax.hlines(y, row["hdi_low"], row["hdi_high"], color=param["color"], lw=4.0, alpha=0.46, zorder=3)
    ax.hlines(y, np.quantile(values, 0.25), np.quantile(values, 0.75), color="#222222", lw=2.0, zorder=4)
    ax.scatter(row["mean"], y, s=46, facecolor="white", edgecolor="#222222", linewidth=0.9, zorder=5)
    ax.scatter(row["median"], y, s=34, color="#222222", zorder=5)

    label = f"Mean {row['mean']:.2f}; 94% HDI [{row['hdi_low']:.2f}, {row['hdi_high']:.2f}]"
    ax.text(
        0.48,
        0.42,
        label,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#222222",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
    )

    ax.text(
        0.035,
        0.50,
        param["short"],
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#222222",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        zorder=8,
    )

    ax.set_yticks([])
    ax.set_ylim(-0.12, 0.68)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.grid(axis="x", color="#E7E9ED", lw=0.7)
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xlabel("Posterior standard deviation", fontsize=9.0)
    else:
        ax.set_xlabel("")


def add_posterior_legend(ax: plt.Axes) -> None:
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markeredgecolor="#222222", markersize=4.8, label="Median"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#222222", markersize=5.2, label="Mean"),
        mpl.lines.Line2D([0], [0], color="#222222", lw=2.0, label="IQR"),
        mpl.lines.Line2D([0], [0], color="#666666", lw=0.9, linestyle=(0, (3, 2)), label="Chain KDE"),
        mpl.lines.Line2D([0], [0], color="#666666", lw=1.7, label="Mean KDE"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7.0, ncol=3, handlelength=1.5, columnspacing=0.9)


def draw_trace_panel(ax: plt.Axes, trace: az.InferenceData, param: dict[str, str], summaries: pd.DataFrame) -> None:
    values_by_chain = flatten_draws(trace, param["var"])
    draws = np.arange(values_by_chain.shape[1])
    for i, chain_values in enumerate(values_by_chain):
        ax.plot(draws, chain_values, color=CHAIN_COLORS[i % len(CHAIN_COLORS)], lw=0.52, alpha=0.62)

    row = summaries.loc[param["var"]]
    ax.axhline(row["mean"], color="#222222", lw=0.9, linestyle=(0, (4, 3)))
    ax.fill_between(draws, row["hdi_low"], row["hdi_high"], color=param["color"], alpha=0.10, linewidth=0)

    ax.set_title(
        f"{param['short']} trace: R-hat={row['r_hat']:.2f}, ESS={row['ess_bulk']:.0f}",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    ax.set_ylabel(param["short"], fontsize=8.0)
    ax.grid(axis="y", color="#E7E9ED", lw=0.6)
    ax.tick_params(axis="both", labelsize=7.4)
    ax.set_xlim(draws.min(), draws.max())
    ax.set_axisbelow(True)


def build_summary(trace: az.InferenceData) -> pd.DataFrame:
    rows = []
    diagnostics = az.summary(trace, var_names=[param["var"] for param in PARAMS], hdi_prob=0.94)
    for param in PARAMS:
        values_by_chain = flatten_draws(trace, param["var"])
        summary = posterior_summary(values_by_chain)
        summary.update(
            {
                "parameter": param["var"],
                "label": param["label"],
                "r_hat": float(diagnostics.loc[param["var"], "r_hat"]),
                "ess_bulk": float(diagnostics.loc[param["var"], "ess_bulk"]),
                "ess_tail": float(diagnostics.loc[param["var"], "ess_tail"]),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows).set_index("parameter")


def draw_figure() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trace = az.from_netcdf(TRACE_PATH)
    summaries = build_summary(trace)
    summaries.reset_index().to_csv(CSV_OUTPUT, index=False)

    fig = plt.figure(figsize=(8.8, 5.1), dpi=450)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.25, hspace=0.36)

    ax_rater_posterior = fig.add_subplot(gs[0, 0])
    ax_image_posterior = fig.add_subplot(gs[1, 0])
    ax_rater = fig.add_subplot(gs[0, 1])
    ax_image = fig.add_subplot(gs[1, 1])

    draw_posterior_panel(ax_rater_posterior, trace, PARAMS[0], summaries, show_xlabel=False)
    draw_posterior_panel(ax_image_posterior, trace, PARAMS[1], summaries, show_xlabel=True)
    ax_rater_posterior.set_title("Posterior uncertainty of variance components", fontsize=10.0, fontweight="bold", pad=8)
    add_posterior_legend(ax_rater_posterior)
    draw_trace_panel(ax_rater, trace, PARAMS[0], summaries)
    draw_trace_panel(ax_image, trace, PARAMS[1], summaries)
    ax_image.set_xlabel("Post-warmup draw", fontsize=8.0)

    fig.suptitle("Null Model Sigma Parameters", fontsize=12.0, fontweight="bold", y=0.985)
    fig.savefig(PNG_OUTPUT, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {PNG_OUTPUT}")
    print(f"Saved summary: {CSV_OUTPUT}")


if __name__ == "__main__":
    draw_figure()
