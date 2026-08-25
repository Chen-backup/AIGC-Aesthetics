from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path("..")
OUTPUT_DIR = Path("picture")
SUMMARY_PATH = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Summary.csv"

COLORS = {
    "background": "#FFFFFF",
    "axis": "#333333",
    "grid": "#E7EBF0",
    "text": "#202020",
    "rater": "#D1495B",
    "image": "#2F78B7",
    "zero": "#202020",
}

IMAGE_SIZE = (2000, 2000)  # 1:1 layout
MARGINS = {
    "left": 280,
    "right": 90,
    "top": 140,
    "bottom": 190,
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "timesbd.ttf" if bold else "times.ttf",
        "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf",
        "timesbi.ttf" if bold else "timesi.ttf",
    ]
    font_dirs = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts")]
    for font_dir in font_dirs:
        for candidate in candidates:
            font_path = font_dir / candidate
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def load_effects(prefix: str) -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH, index_col=0)
    mask = df.index.to_series().str.startswith(prefix)
    effects = df.loc[mask, ["mean", "hdi_3%", "hdi_97%"]].copy()
    effects = effects.sort_values("mean", ascending=True).reset_index(names="term")
    return effects


def compute_limits(df: pd.DataFrame) -> tuple[float, float]:
    min_val = float(df["hdi_3%"].min())
    max_val = float(df["hdi_97%"].max())
    pad = (max_val - min_val) * 0.08
    if pad == 0:
        pad = 0.3
    return min_val - pad, max_val + pad


def value_to_x(value: float, vmin: float, vmax: float, x_left: int, x_right: int) -> float:
    if vmax == vmin:
        return (x_left + x_right) / 2
    return x_left + (value - vmin) / (vmax - vmin) * (x_right - x_left)


def x_to_pixel(x: float, xmin: float, xmax: float, x0: int, x1: int) -> float:
    if xmax == xmin:
        return (x0 + x1) / 2
    return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)


def value_to_y(value: float, vmin: float, vmax: float, y_top: int, y_bottom: int) -> float:
    if vmax == vmin:
        return (y_top + y_bottom) / 2
    return y_bottom - (value - vmin) / (vmax - vmin) * (y_bottom - y_top)


def format_signed_tick(value: float) -> str:
    if abs(value) < 1e-10:
        return "0"
    abs_text = f"{abs(value):.1f}".rstrip("0").rstrip(".")
    return f"+{abs_text}" if value > 0 else f"-{abs_text}"


def compute_rank_ticks(n: int, num_ticks: int = 6) -> list[int]:
    if n <= 1:
        return [1]
    candidates = [int(round(v)) for v in np.linspace(1, n, num_ticks)]
    ticks = sorted(set(candidates))
    if ticks[0] != 1:
        ticks.insert(0, 1)
    if ticks[-1] != n:
        ticks.append(n)
    return ticks


def compute_effect_ticks(vmin: float, vmax: float, num_ticks: int = 7) -> list[float]:
    ticks = [float(tick) for tick in np.linspace(vmin, vmax, num_ticks)]
    if vmin <= 0 <= vmax:
        ticks.append(0.0)
        min_gap = (vmax - vmin) / 16.0
        if min_gap > 0:
            filtered = []
            for tick in ticks:
                if abs(tick) < 1e-10:
                    filtered.append(0.0)
                elif abs(tick) < min_gap:
                    continue
                else:
                    filtered.append(tick)
            ticks = filtered
    return sorted(set(ticks))


def parse_entity_id(term: str) -> int:
    match = re.search(r"\[(\d+)\]$", term)
    if match:
        return int(match.group(1))
    return -1


def draw_caterpillar(df: pd.DataFrame, title: str, y_axis_label: str, color: str, out_path: Path) -> None:
    canvas = Image.new("RGB", IMAGE_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(88, bold=True)
    tick_font = load_font(60, bold=True)
    axis_label_font = load_font(80, bold=True)

    plot_left = MARGINS["left"]
    plot_right = IMAGE_SIZE[0] - MARGINS["right"]
    plot_top = MARGINS["top"]
    plot_bottom = IMAGE_SIZE[1] - MARGINS["bottom"]
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    vmin, vmax = compute_limits(df)
    n = len(df)
    y_step = plot_height / max(n, 1)

    draw_centered_text(draw, IMAGE_SIZE[0] / 2, 50, title, title_font, COLORS["text"])

    # X ticks + light vertical grid
    xticks = compute_effect_ticks(vmin, vmax, num_ticks=7)
    for tick in xticks:
        px = value_to_x(float(tick), vmin, vmax, plot_left, plot_right)
        draw.line((px, plot_top, px, plot_bottom), fill=COLORS["grid"], width=2)
        label = format_signed_tick(float(tick))
        tb = draw.textbbox((0, 0), label, font=tick_font)
        tw = tb[2] - tb[0]
        draw.text((px - tw / 2, plot_bottom + 14), label, font=tick_font, fill=COLORS["axis"])

    # Y ticks as index range (1..N) + horizontal grid
    rank_ticks = compute_rank_ticks(n, num_ticks=6)
    max_y_tick_label_w = 0
    for rank in rank_ticks:
        py = plot_top + ((rank - 0.5) / n) * plot_height
        draw.line((plot_left, py, plot_right, py), fill="#F2F4F7", width=1)
        label = str(rank)
        tb = draw.textbbox((0, 0), label, font=tick_font)
        tw = tb[2] - tb[0]
        if tw > max_y_tick_label_w:
            max_y_tick_label_w = tw
        th = tb[3] - tb[1]
        draw.text((plot_left - tw - 14, py - th / 2), label, font=tick_font, fill=COLORS["axis"])

    # Zero reference line
    if vmin <= 0 <= vmax:
        zx = value_to_x(0.0, vmin, vmax, plot_left, plot_right)
        draw.line((zx, plot_top, zx, plot_bottom), fill=COLORS["zero"], width=3)

    # Main caterpillar
    dot_radius = 2 if n > 800 else 3
    line_width = 1 if n > 800 else 2
    ordered = df.copy()
    ordered["id"] = ordered["term"].map(parse_entity_id)
    ordered = ordered.sort_values("mean", ascending=False, kind="mergesort").reset_index(drop=True)

    for i, row in ordered.iterrows():
        y = int(plot_top + (i + 0.5) * y_step)
        x_low = int(value_to_x(float(row["hdi_3%"]), vmin, vmax, plot_left, plot_right))
        x_mid = int(value_to_x(float(row["mean"]), vmin, vmax, plot_left, plot_right))
        x_high = int(value_to_x(float(row["hdi_97%"]), vmin, vmax, plot_left, plot_right))

        draw.line((x_low, y, x_high, y), fill=color, width=line_width)
        draw.ellipse(
            (x_mid - dot_radius, y - dot_radius, x_mid + dot_radius, y + dot_radius),
            fill=color,
            outline=color,
        )

    # Axis frame (left + bottom only)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["axis"], width=15)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["axis"], width=15)

    # Axis annotations (no x tick labels by design)
    draw_centered_text(
        draw,
        (plot_left + plot_right) / 2,
        IMAGE_SIZE[1] - 54,
        "Posterior Random Effect (Mean with 95% HDI)",
        axis_label_font,
        COLORS["axis"],
    )

    y_label = y_axis_label
    y_bbox = draw.textbbox((0, 0), y_label, font=axis_label_font)
    y_w = y_bbox[2] - y_bbox[0]
    y_h = y_bbox[3] - y_bbox[1]
    temp = Image.new("RGBA", (y_w + 24, y_h + 24), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((12, 12), y_label, font=axis_label_font, fill=COLORS["axis"])
    rotated = temp.rotate(90, expand=True)
    tick_left_x = plot_left - max_y_tick_label_w - 14
    y_label_gap = 34
    y_label_x = max(0, int(tick_left_x - y_label_gap - rotated.size[0]))
    canvas.paste(rotated, (y_label_x, int((plot_top + plot_bottom - rotated.size[1]) / 2)), rotated)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(300, 300))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rater_df = load_effects("1|rater[")
    image_df = load_effects("1|image[")

    draw_caterpillar(
        rater_df,
        title="Null Model Caterpillar Plot: Rater Random Effects",
        y_axis_label="Rater index",
        color=COLORS["rater"],
        out_path=OUTPUT_DIR / "Null_Caterpillar_Rater_Compact_2x3.png",
    )
    draw_caterpillar(
        image_df,
        title="Null Model Caterpillar Plot: Image Random Effects",
        y_axis_label="Image index",
        color=COLORS["image"],
        out_path=OUTPUT_DIR / "Null_Caterpillar_Image_Compact_2x3.png",
    )

    print(f"Saved figure: {OUTPUT_DIR / 'Null_Caterpillar_Rater_Compact_2x3.png'}")
    print(f"Saved figure: {OUTPUT_DIR / 'Null_Caterpillar_Image_Compact_2x3.png'}")


if __name__ == "__main__":
    main()
