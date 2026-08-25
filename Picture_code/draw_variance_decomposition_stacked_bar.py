from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

NULL_MODEL_REPORT = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Metrics_Report.txt"
MODEL_REPORTS = {
    "Interpretable": ROOT_DIR / "BYS_interpretable_model_result" / "Full_Model_Metrics_Report.txt",
    "StyleGAN": ROOT_DIR / "BYS_StyleGAN_model_14D_result" / "StyleGAN_14D_Model_Metrics_Report.txt",
    "InsightFace": ROOT_DIR / "BYS_Insightface_model_14D_result" / "Deep_14D_Model_Metrics_Report.txt",
    "DINOv2": ROOT_DIR / "BYS_DINOv2_model_14D_result" / "DINOv2_14D_Model_Metrics_Report.txt",
}

COLORS = {
    "background": "#FFFFFF",
    "axis": "#454545",
    "grid": "#E9EDF2",
    "Rater Variance": "#2F78B7",
    "Image Variance Explained by Model": "#F28E2B",
    "Unexplained Image Variance": "#B9BDC6",
    "white_text": "#FFFFFF",
    "dark_text": "#181818",
    "subtle_text": "#666666",
}

IMAGE_SIZE = (2400, 1200)
MARGINS = {
    "left": 190,
    "right": 90,
    "top": 185,
    "bottom": 180,
}
Y_MAX = 8.5
Y_TICKS = [float(tick) for tick in range(0, 9)]


def extract_value(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find {label} using pattern: {pattern}")
    return float(match.group(1))


def load_variance_components() -> pd.DataFrame:
    null_text = NULL_MODEL_REPORT.read_text(encoding="utf-8")
    null_image_variance = extract_value(
        r"Image Variance .*?:\s*([0-9.]+)",
        null_text,
        "null image variance",
    )

    rows: list[dict[str, float | str]] = []
    for model_name, report_path in MODEL_REPORTS.items():
        report_text = report_path.read_text(encoding="utf-8")
        residual_image_variance = extract_value(
            r"Residual Image Variance .*?:\s*([0-9.]+)",
            report_text,
            f"{model_name} residual image variance",
        )
        rater_variance = extract_value(
            r"Rater Variance .*?:\s*([0-9.]+)",
            report_text,
            f"{model_name} rater variance",
        )
        explained_image_variance = null_image_variance - residual_image_variance

        rows.append(
            {
                "Model": model_name,
                "Rater Variance": rater_variance,
                "Image Variance Explained by Model": explained_image_variance,
                "Unexplained Image Variance": residual_image_variance,
                "Explained Share": explained_image_variance / null_image_variance,
            }
        )

    return pd.DataFrame(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "timesbd.ttf" if bold else "times.ttf",
        "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf",
        "timesbi.ttf" if bold else "timesi.ttf",
    ]
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
    ]

    for font_dir in font_dirs:
        for candidate in candidates:
            font_path = font_dir / candidate
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size=size)

    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int = 4) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    return right - left, bottom - top


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 4,
) -> None:
    width, height = text_bbox(draw, text, font=font, spacing=spacing)
    draw.multiline_text(
        (center_x - width / 2, center_y - height / 2),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def draw_left_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    draw.text((x, y), text, font=font, fill=fill)


def format_segment_label(value: float, share: float | None = None) -> str:
    if share is None:
        return f"{value:.2f}"
    return f"{value:.2f}\n({share * 100:.1f}%)"


def map_y(value: float, plot_top: int, plot_bottom: int) -> float:
    plot_height = plot_bottom - plot_top
    return plot_bottom - (value / Y_MAX) * plot_height


def draw_plot(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", IMAGE_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(54, bold=True)
    legend_font = load_font(34, bold=False)
    axis_font = load_font(34, bold=True)
    tick_font = load_font(28, bold=True)
    bar_label_font = load_font(34, bold=True)
    segment_font = load_font(34, bold=True)

    plot_left = MARGINS["left"]
    plot_right = IMAGE_SIZE[0] - MARGINS["right"]
    plot_top = MARGINS["top"] + 100
    plot_bottom = IMAGE_SIZE[1] - MARGINS["bottom"]

    title = "Variance Decomposition Across Four Image-Based Models"
    draw_centered_text(draw, IMAGE_SIZE[0] / 2, 100, title, title_font, COLORS["dark_text"])

    legend_items = [
        "Rater Variance",
        "Image Variance Explained by Model",
        "Unexplained Image Variance",
    ]
    legend_y = 180
    legend_box = 34
    cursor_x = 200
    for item in legend_items:
        draw.rectangle(
            (cursor_x, legend_y, cursor_x + legend_box, legend_y + legend_box),
            fill=COLORS[item],
            outline=COLORS[item],
        )
        draw.text(
            (cursor_x + legend_box + 14, legend_y - 6),
            item,
            font=legend_font,
            fill=COLORS["dark_text"],
        )
        text_width, _ = text_bbox(draw, item, legend_font)
        cursor_x += legend_box + 14 + text_width + 52

    for tick in Y_TICKS:
        y = map_y(tick, plot_top, plot_bottom)
        for dash_start in range(plot_left, plot_right, 24):
            dash_end = min(dash_start + 12, plot_right)
            draw.line((dash_start, y, dash_end, y), fill=COLORS["grid"], width=2)
        tick_label = f"{tick:.1f}"
        tick_width, tick_height = text_bbox(draw, tick_label, tick_font)
        draw.text(
            (plot_left - 18 - tick_width, y - tick_height / 2),
            tick_label,
            font=tick_font,
            fill=COLORS["axis"],
        )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["axis"], width=5)

    total_variance = float(
        df["Rater Variance"].iloc[0]
        + df["Image Variance Explained by Model"].iloc[0]
        + df["Unexplained Image Variance"].iloc[0]
    )
    y_label = f"Variance Component (σ²; Total = {total_variance:.2f})"
    y_label_width, y_label_height = text_bbox(draw, y_label, axis_font)
    temp = Image.new("RGBA", (y_label_width + 20, y_label_height + 20), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((10, 10), y_label, font=axis_font, fill=COLORS["axis"])
    rotated = temp.rotate(90, expand=True)
    canvas.paste(rotated, (52, int((plot_top + plot_bottom - rotated.size[1]) / 2)), rotated)

    bar_area_width = plot_right - plot_left
    slot_width = bar_area_width / len(df)
    bar_width = slot_width * 0.72

    for idx, row in df.iterrows():
        x_center = plot_left + slot_width * (idx + 0.5)
        bar_left = x_center - bar_width / 2
        bar_right = x_center + bar_width / 2

        rater = float(row["Rater Variance"])
        explained = float(row["Image Variance Explained by Model"])
        residual = float(row["Unexplained Image Variance"])
        explained_share = float(row["Explained Share"])
        residual_share = 1.0 - explained_share

        y_rater_top = map_y(rater, plot_top, plot_bottom)
        y_explained_top = map_y(rater + explained, plot_top, plot_bottom)
        y_total_top = map_y(rater + explained + residual, plot_top, plot_bottom)

        draw.rectangle((bar_left, y_rater_top, bar_right, plot_bottom), fill=COLORS["Rater Variance"])
        draw.rectangle(
            (bar_left, y_explained_top, bar_right, y_rater_top),
            fill=COLORS["Image Variance Explained by Model"],
        )
        draw.rectangle((bar_left, y_total_top, bar_right, y_explained_top), fill=COLORS["Unexplained Image Variance"])

        draw.line((bar_left, y_rater_top, bar_right, y_rater_top), fill=COLORS["background"], width=4)
        draw.line((bar_left, y_explained_top, bar_right, y_explained_top), fill=COLORS["background"], width=4)

        draw_centered_text(
            draw,
            x_center,
            (plot_bottom + y_rater_top) / 2,
            format_segment_label(rater, rater / total_variance),
            segment_font,
            COLORS["dark_text"],
        )
        draw_centered_text(
            draw,
            x_center,
            (y_rater_top + y_explained_top) / 2,
            format_segment_label(explained, explained_share),
            segment_font,
            COLORS["dark_text"],
        )
        draw_centered_text(
            draw,
            x_center,
            (y_explained_top + y_total_top) / 2,
            format_segment_label(residual, residual_share),
            segment_font,
            COLORS["dark_text"],
        )

        draw_centered_text(
            draw,
            x_center,
            plot_bottom + 52,
            str(row["Model"]),
            bar_label_font,
            COLORS["dark_text"],
        )

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["axis"], width=5)

    png_path = OUTPUT_DIR / "variance_decomposition_stacked_bar.png"
    csv_path = OUTPUT_DIR / "variance_decomposition_stacked_bar_data.csv"
    df.to_csv(csv_path, index=False)
    canvas.save(png_path, format="PNG")

    print(f"Saved figure: {png_path}")
    print(f"Saved data: {csv_path}")


def main() -> None:
    df = load_variance_components()
    draw_plot(df)


if __name__ == "__main__":
    main()
