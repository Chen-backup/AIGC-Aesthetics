from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
FACE_FIGURE_DIR = ROOT_DIR / "Picture_fig3"
sys.path.insert(0, str(FACE_FIGURE_DIR))

import draw_fig3_01D_donut_feature_means_sd_effect_values as base  # noqa: E402


PNG_OUTPUT = SCRIPT_DIR / "Posterior_effect_value_colorbar.png"

FIG_SIZE = (3.6, 6.0)
COLORBAR_RECT = [0.14, 0.055, 0.18, 0.90]
OUTLINE_WIDTH = 2.6
TICK_WIDTH = 2.4
TICK_LENGTH = 8.0
TICK_SIZE = 32
LABEL_SIZE = 36


def draw_colorbar() -> None:
    base.set_style()
    norm = TwoSlopeNorm(vmin=-base.EFFECT_COLOR_LIMIT, vcenter=0.0, vmax=base.EFFECT_COLOR_LIMIT)
    cmap = base.make_effect_cmap()

    fig = plt.figure(figsize=FIG_SIZE, dpi=400, facecolor="white")
    cax = fig.add_axes(COLORBAR_RECT)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label(
        "Posterior effect value",
        fontsize=LABEL_SIZE,
        labelpad=10,
    )
    colorbar.ax.tick_params(
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    fig.savefig(PNG_OUTPUT, dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {PNG_OUTPUT}")


if __name__ == "__main__":
    draw_colorbar()
