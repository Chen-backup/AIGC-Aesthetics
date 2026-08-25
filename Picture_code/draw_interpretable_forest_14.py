from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path("..")
OUTPUT_DIR = Path("picture")
SUMMARY_PATH = ROOT_DIR / "BYS_interpretable_model_result" / "Full_Model_Summary.csv"

FEATURES = [
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

COLORS = {
    "background": "#FFFFFF",
    "axis": "#333333",
    "grid": "#E7EBF0",
    "text": "#202020",
    "point": "#F28E2B",
    "zero": "#D1495B",
}

IMAGE_SIZE = (2400, 3000)  # 4:5 ratio
MARGINS = {
    "left": 560,
    "right": 120,
    "top": 220,
    "bottom": 210,
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


def draw_right_text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w, y - h / 2), text, font=font, fill=fill)


def x_to_pixel(x: float, xmin: float, xmax: float, x0: int, x1: int) -> float:
    if xmax == xmin:
        return (x0 + x1) / 2
    return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)


def format_signed_tick(value: float) -> str:
    if abs(value) < 1e-10:
        return "0"
    abs_text = f"{abs(value):.1f}".rstrip("0").rstrip(".")
    return f"+{abs_text}" if value > 0 else f"-{abs_text}"


def load_feature_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH, index_col=0)
    feature_df = df.loc[FEATURES, ["mean", "hdi_3%", "hdi_97%"]].copy()
    feature_df = feature_df.sort_values("mean", ascending=False).reset_index(names="feature")
    return feature_df


def draw_forest_plot(df: pd.DataFrame, output_path: Path) -> None:
    canvas = Image.new("RGB", IMAGE_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(100, bold=True)
    feature_font = load_font(55, bold=True)
    tick_font = load_font(70, bold=True)
    xlabel_font = load_font(75, bold=True)

    plot_left = MARGINS["left"]
    plot_right = IMAGE_SIZE[0] - MARGINS["right"]
    plot_top = MARGINS["top"]
    plot_bottom = IMAGE_SIZE[1] - MARGINS["bottom"]
    plot_height = plot_bottom - plot_top

    x_min = float(df["hdi_3%"].min())
    x_max = float(df["hdi_97%"].max())
    pad = (x_max - x_min) * 0.10
    x_min -= pad
    x_max += pad

    draw_centered_text(
        draw,
        IMAGE_SIZE[0] / 2,
        90,
        "Impact of Facial Features on Aesthetic Ratings",
        title_font,
        COLORS["text"],
    )

    xticks = [float(tick) for tick in np.linspace(x_min, x_max, 7)]
    if x_min <= 0.0 <= x_max:
        xticks.append(0.0)
    xticks = sorted(set(xticks))
    for tick in xticks:
        x = x_to_pixel(float(tick), x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill=COLORS["grid"], width=2)
        tick_text = format_signed_tick(float(tick))
        bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw / 2, plot_bottom + 16), tick_text, font=tick_font, fill=COLORS["axis"])

    if x_min <= 0.0 <= x_max:
        zero_x = x_to_pixel(0.0, x_min, x_max, plot_left, plot_right)
        dash = 16
        gap = 10
        y = plot_top
        while y < plot_bottom:
            y2 = min(y + dash, plot_bottom)
            draw.line((zero_x, y, zero_x, y2), fill=COLORS["zero"], width=8)
            y += dash + gap

    n = len(df)
    row_step = plot_height / max(n, 1)
    for i, row in df.iterrows():
        y = int(plot_top + (i + 0.5) * row_step)
        feature_name = str(row["feature"])
        draw_right_text(draw, plot_left - 24, y, feature_name, feature_font, COLORS["text"])

        x_low = int(x_to_pixel(float(row["hdi_3%"]), x_min, x_max, plot_left, plot_right))
        x_mid = int(x_to_pixel(float(row["mean"]), x_min, x_max, plot_left, plot_right))
        x_high = int(x_to_pixel(float(row["hdi_97%"]), x_min, x_max, plot_left, plot_right))

        draw.line((x_low, y, x_high, y), fill=COLORS["point"], width=10)
        draw.line((x_low, y - 9, x_low, y + 9), fill=COLORS["point"], width=8)
        draw.line((x_high, y - 9, x_high, y + 9), fill=COLORS["point"], width=8)
        draw.ellipse((x_mid - 8, y - 8, x_mid + 8, y + 8), fill=COLORS["point"], outline=COLORS["point"])

    # Keep only left and bottom axes (remove top/right frame lines).
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["axis"], width=15)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["axis"], width=15)
    draw_centered_text(
        draw,
        (plot_left + plot_right) / 2,
        IMAGE_SIZE[1] - 64,
        "Standardized Effect Size (Posterior Mean with 95% HDI)",
        xlabel_font,
        COLORS["axis"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(300, 300))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "Interpretable_14D_Forest_1x2.png"
    summary_df = load_feature_summary()
    draw_forest_plot(summary_df, out_path)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
