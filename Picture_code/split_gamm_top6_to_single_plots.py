from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parent / "picture"
SOURCE_DATA = OUTPUT_DIR / "GAMM_NonLinear_Expected_Curves_Top6_data.csv"

COLORS = {
    "background": "#FFFFFF",
    "axis": "#2F343A",
    "grid": "#E3EAF2",
    "title": "#181818",
    "tick": "#4A4A4A",
    "rug": "#61708A",
}
LINE_COLOR = "#155A9C"
BAND_COLOR = "#9CC2E5"
BAND_EDGE_COLOR = "#5F93C8"
LINE_HALO_COLOR = "#FFFFFF"

IMAGE_SIZE = (1800, 1800)  # 1:1 ratio
MARGINS = {
    "left": 230,
    "right": 85,
    "top": 180,
    "bottom": 210,
}
# Extra separation near the bottom-left axis intersection to avoid overlap
# between the first x tick label and the lowest y tick label.
ORIGIN_X_LABEL_SHIFT = 28
ORIGIN_Y_LABEL_SHIFT = 14

Y_AXIS_SPECS = {
    "le_nose_re_angle": (2.5, 5.5, 0.5),
    "upper_lower_ratio": (3.5, 7.0, 0.5),
    "mouth_face_w_ratio": (2.0, 5.5, 0.5),
    "total_symmetry": (3.2, 5.8, 0.5),
    "edge_density": (3.5, 6.5, 0.5),
    "eye_y_ratio": (2.0, 5.5, 0.5),
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


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_bbox(draw, text, font)
    draw.text((center_x - width / 2, center_y - height / 2), text, font=font, fill=fill)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int = 2,
    dash_length: int = 12,
    gap_length: int = 10,
) -> None:
    x1, y1 = start
    x2, y2 = end
    distance = float(np.hypot(x2 - x1, y2 - y1))
    if distance == 0:
        return

    dx = (x2 - x1) / distance
    dy = (y2 - y1) / distance
    progress = 0.0
    while progress < distance:
        dash_end = min(progress + dash_length, distance)
        draw.line(
            (
                x1 + dx * progress,
                y1 + dy * progress,
                x1 + dx * dash_end,
                y1 + dy * dash_end,
            ),
            fill=fill,
            width=width,
        )
        progress += dash_length + gap_length


def map_value(value: float, min_value: float, max_value: float, pixel_min: float, pixel_max: float) -> float:
    if max_value == min_value:
        return pixel_min
    ratio = (value - min_value) / (max_value - min_value)
    return pixel_min + ratio * (pixel_max - pixel_min)


def trim_number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_x_tick(value: float, span: float) -> str:
    if span < 0.05:
        precision = 3
    elif span < 0.5:
        precision = 2
    elif span < 5:
        precision = 2
    else:
        precision = 1
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def render_single_panel(
    feature_name: str,
    feature_label: str,
    feature_curve: pd.DataFrame,
    rug_data: pd.DataFrame,
    order_idx: int,
) -> Path:
    canvas = Image.new("RGBA", IMAGE_SIZE, ImageColor.getrgb(COLORS["background"]) + (255,))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(100, bold=True)
    axis_title_font = load_font(80, bold=True)
    tick_font = load_font(80, bold=True)

    axis_left = MARGINS["left"]
    axis_right = IMAGE_SIZE[0] - MARGINS["right"]
    axis_top = MARGINS["top"]
    axis_bottom = IMAGE_SIZE[1] - MARGINS["bottom"]

    draw_centered_text(draw, IMAGE_SIZE[0] / 2, 85, feature_label, title_font, COLORS["title"])

    x_values = feature_curve["x"].to_numpy(dtype=float)
    mean_values = feature_curve["mean_expected_score"].to_numpy(dtype=float)
    lower_values = feature_curve["lower_95"].to_numpy(dtype=float)
    upper_values = feature_curve["upper_95"].to_numpy(dtype=float)

    x_min = float(x_values.min())
    x_max = float(x_values.max())
    x_span = x_max - x_min

    y_min, y_max, y_step = Y_AXIS_SPECS[feature_name]
    y_ticks = np.arange(y_min, y_max + y_step * 0.5, y_step)
    x_ticks = np.linspace(x_min, x_max, 4)

    # Place y-axis label dynamically so it stays clear of y tick labels
    # even if font sizes are increased later.
    max_y_tick_w = max(text_bbox(draw, trim_number(float(tick)), tick_font)[0] for tick in y_ticks)
    y_label = "Expected Aesthetic Score"
    y_w, y_h = text_bbox(draw, y_label, axis_title_font)
    temp = Image.new("RGBA", (y_w + 30, y_h + 30), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((15, 15), y_label, font=axis_title_font, fill=COLORS["axis"])
    rotated = temp.rotate(90, expand=True)
    tick_left_x = axis_left - max_y_tick_w - 14
    y_label_gap = 34
    y_label_x = max(8, int(tick_left_x - y_label_gap - rotated.size[0]))
    canvas.alpha_composite(rotated, (y_label_x, int((axis_top + axis_bottom - rotated.size[1]) / 2)))

    for y_idx, y_tick in enumerate(y_ticks):
        y_px = map_value(float(y_tick), y_min, y_max, axis_bottom, axis_top)
        draw_dashed_line(draw, (axis_left, y_px), (axis_right, y_px), COLORS["grid"], width=2, dash_length=18, gap_length=16)
        tick_text = trim_number(float(y_tick))
        tw, th = text_bbox(draw, tick_text, tick_font)
        y_text_y = y_px - th / 2
        if y_idx == 0:
            y_text_y -= ORIGIN_Y_LABEL_SHIFT
        draw.text((axis_left - tw - 14, y_text_y), tick_text, font=tick_font, fill=COLORS["tick"])

    for x_idx, x_tick in enumerate(x_ticks):
        x_px = map_value(float(x_tick), x_min, x_max, axis_left, axis_right)
        tick_text = format_x_tick(float(x_tick), x_span)
        tw, _ = text_bbox(draw, tick_text, tick_font)
        x_text_x = x_px - tw / 2
        if x_idx == 0:
            x_text_x += ORIGIN_X_LABEL_SHIFT
        draw.text((x_text_x, axis_bottom + 16), tick_text, font=tick_font, fill=COLORS["tick"])

    band_points = []
    for x_val, up in zip(x_values, upper_values):
        band_points.append(
            (
                map_value(float(x_val), x_min, x_max, axis_left, axis_right),
                map_value(float(up), y_min, y_max, axis_bottom, axis_top),
            )
        )
    for x_val, lo in zip(x_values[::-1], lower_values[::-1]):
        band_points.append(
            (
                map_value(float(x_val), x_min, x_max, axis_left, axis_right),
                map_value(float(lo), y_min, y_max, axis_bottom, axis_top),
            )
        )
    clamped_band = [(x, min(max(y, axis_top), axis_bottom)) for x, y in band_points]
    draw.polygon(clamped_band, fill=ImageColor.getrgb(BAND_COLOR) + (104,))

    upper_points = [
        (
            map_value(float(x_val), x_min, x_max, axis_left, axis_right),
            min(max(map_value(float(up), y_min, y_max, axis_bottom, axis_top), axis_top), axis_bottom),
        )
        for x_val, up in zip(x_values, upper_values)
    ]
    lower_points = [
        (
            map_value(float(x_val), x_min, x_max, axis_left, axis_right),
            min(max(map_value(float(lo), y_min, y_max, axis_bottom, axis_top), axis_top), axis_bottom),
        )
        for x_val, lo in zip(x_values, lower_values)
    ]
    draw.line(upper_points, fill=ImageColor.getrgb(BAND_EDGE_COLOR) + (70,), width=3, joint="curve")
    draw.line(lower_points, fill=ImageColor.getrgb(BAND_EDGE_COLOR) + (70,), width=3, joint="curve")

    line_points = [
        (
            map_value(float(x_val), x_min, x_max, axis_left, axis_right),
            map_value(float(mean), y_min, y_max, axis_bottom, axis_top),
        )
        for x_val, mean in zip(x_values, mean_values)
    ]
    draw.line(line_points, fill=ImageColor.getrgb(LINE_HALO_COLOR) + (235,), width=15, joint="curve")
    draw.line(line_points, fill=LINE_COLOR, width=8, joint="curve")

    rug_y_top = axis_bottom - 23
    rug_y_bottom = axis_bottom - 5
    for rug_val in rug_data["x"].to_numpy(dtype=float):
        rug_x = map_value(float(rug_val), x_min, x_max, axis_left, axis_right)
        draw.line((rug_x, rug_y_top, rug_x, rug_y_bottom), fill=ImageColor.getrgb(COLORS["rug"]) + (125,), width=3)

    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=COLORS["axis"], width=10)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=COLORS["axis"], width=10)

    out_name = f"GAMM_NonLinear_Expected_Curve_{order_idx + 1:02d}_{feature_name}.png"
    out_path = OUTPUT_DIR / out_name
    canvas.convert("RGB").save(out_path, format="PNG", dpi=(300, 300))
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_df = pd.read_csv(SOURCE_DATA)

    order_df = (
        curve_df[["feature_order", "feature_name", "feature_label"]]
        .drop_duplicates()
        .sort_values("feature_order")
    )

    saved_paths: list[Path] = []
    for _, row in order_df.iterrows():
        feature_name = str(row["feature_name"])
        feature_label = str(row["feature_label"])
        feature_order = int(row["feature_order"])

        feature_curve = curve_df[(curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] >= 0)].sort_values(
            "curve_index"
        )
        rug_data = curve_df[(curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] < 0)].sort_values(
            "rug_index"
        )

        saved_paths.append(render_single_panel(feature_name, feature_label, feature_curve, rug_data, feature_order))

    print("Saved single-panel figures:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
