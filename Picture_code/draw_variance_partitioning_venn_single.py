from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

NULL_REPORT = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Metrics_Report.txt"
INTERPRETABLE_REPORT = ROOT_DIR / "BYS_interpretable_model_result" / "Full_Model_Metrics_Report.txt"

DEEP_SINGLE_REPORTS = [
    ROOT_DIR / "BYS_StyleGAN_model_14D_result" / "StyleGAN_14D_Model_Metrics_Report.txt",
    ROOT_DIR / "BYS_Insightface_model_14D_result" / "Deep_14D_Model_Metrics_Report.txt",
    ROOT_DIR / "BYS_DINOv2_model_14D_result" / "DINOv2_14D_Model_Metrics_Report.txt",
]
DEEP_FUSION_REPORTS = [
    ROOT_DIR / "BYS_Fusion_28D_StyleGAN_result" / "Fusion_28D_Model_Metrics.txt",
    ROOT_DIR / "BYS_Fusion_28D_InsightFace_result" / "Fusion_28D_Model_Metrics.txt",
    ROOT_DIR / "BYS_Fusion_28D_DINOv2_result" / "Fusion_28D_Model_Metrics.txt",
]

COLORS = {
    "background": "#FFFFFF",
    "title": "#1A1A1A",
    "text": "#222222",
    "subtle_text": "#5F6670",
    "inter_fill": "#4E79A7",
    "inter_outline": "#2F5D90",
    "deep_fill": "#E15759",
    "deep_outline": "#B9383A",
    "overlap_fill": "#B8BEC8",
    "summary_fill": "#E5E8ED",
    "summary_outline": "#B9C0CA",
}

IMAGE_SIZE = (2400, 3000)  # 4:5 ratio


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


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    red, green, blue = ImageColor.getrgb(hex_color)
    return red, green, blue, alpha


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find {label}")
    return float(match.group(1))


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 4,
) -> None:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    width = right - left
    height = bottom - top
    draw.multiline_text(
        (center_x - width / 2, center_y - height / 2),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def radius_from_r2(r2_value: float, base_r2: float, base_radius: float) -> float:
    return base_radius * ((r2_value / base_r2) ** 0.5)


def load_partition_data() -> pd.DataFrame:
    null_text = read_text(NULL_REPORT)
    interpretable_text = read_text(INTERPRETABLE_REPORT)

    null_image_variance = extract_float(r"Image Variance .*?:\s*([0-9.]+)", null_text, "null image variance")
    interpretable_residual = extract_float(
        r"Residual Image Variance .*?:\s*([0-9.]+)",
        interpretable_text,
        "interpretable residual image variance",
    )
    interpretable_r2 = (1.0 - interpretable_residual / null_image_variance) * 100.0

    deep_r2_values = []
    for report_path in DEEP_SINGLE_REPORTS:
        deep_r2_values.append(extract_float(r"([0-9]+\.[0-9]+)%", read_text(report_path), f"deep r2 from {report_path.name}"))

    fusion_r2_values = []
    for report_path in DEEP_FUSION_REPORTS:
        fusion_r2_values.append(
            extract_float(r"([0-9]+\.[0-9]+)%", read_text(report_path), f"fusion r2 from {report_path.name}")
        )

    deep_r2 = float(np.mean(deep_r2_values))
    fusion_r2 = float(np.mean(fusion_r2_values))

    shared = interpretable_r2 + deep_r2 - fusion_r2
    unique_interpretable = fusion_r2 - deep_r2
    unique_deep = fusion_r2 - interpretable_r2
    unexplained = 100.0 - fusion_r2

    return pd.DataFrame(
        [
            {
                "interpretable_r2": round(interpretable_r2, 2),
                "deep_r2_avg": round(deep_r2, 2),
                "fusion_r2_avg": round(fusion_r2, 2),
                "shared": round(shared, 2),
                "unique_interpretable": round(unique_interpretable, 2),
                "unique_deep": round(unique_deep, 2),
                "unexplained": round(unexplained, 2),
            }
        ]
    )


def render_venn(df: pd.DataFrame) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "variance_partitioning_venn_single_data.csv"
    png_path = OUTPUT_DIR / "variance_partitioning_venn_single.png"
    df.to_csv(csv_path, index=False)

    row = df.iloc[0]
    canvas = Image.new("RGBA", IMAGE_SIZE, rgba(COLORS["background"], 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(100, bold=True)
    label_font = load_font(90, bold=True)
    region_font = load_font(100, bold=True)

    draw_centered_text(
        draw,
        IMAGE_SIZE[0] / 2,
        110,
        "Variance Partitioning: Interpretable vs Deep Features",
        title_font,
        COLORS["title"],
    )

    inter_r2 = float(row["interpretable_r2"])
    deep_r2 = float(row["deep_r2_avg"])

    center_x = IMAGE_SIZE[0] * 0.50
    top_radius = radius_from_r2(inter_r2, base_r2=42.60, base_radius=600.0)
    bottom_radius = radius_from_r2(deep_r2, base_r2=42.60, base_radius=600.0)

    overlap_depth = 600.0
    top_center_y = 980.0
    bottom_center_y = top_center_y + (top_radius + bottom_radius - overlap_depth)

    top_box = (
        center_x - top_radius,
        top_center_y - top_radius,
        center_x + top_radius,
        top_center_y + top_radius,
    )
    bottom_box = (
        center_x - bottom_radius,
        bottom_center_y - bottom_radius,
        center_x + bottom_radius,
        bottom_center_y + bottom_radius,
    )

    fill_layer = Image.new("RGBA", IMAGE_SIZE, (255, 255, 255, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    fill_draw.ellipse(bottom_box, fill=rgba(COLORS["deep_fill"], 165))
    fill_draw.ellipse(top_box, fill=rgba(COLORS["inter_fill"], 165))

    top_mask = Image.new("L", IMAGE_SIZE, 0)
    bottom_mask = Image.new("L", IMAGE_SIZE, 0)
    ImageDraw.Draw(top_mask).ellipse(top_box, fill=255)
    ImageDraw.Draw(bottom_mask).ellipse(bottom_box, fill=255)
    overlap_mask = ImageChops.multiply(top_mask, bottom_mask)

    overlap_layer = Image.new("RGBA", IMAGE_SIZE, rgba(COLORS["overlap_fill"], 230))
    overlap_layer.putalpha(overlap_mask)

    overlap_outer = overlap_mask.filter(ImageFilter.MaxFilter(7))
    overlap_inner = overlap_mask.filter(ImageFilter.MinFilter(7))
    overlap_edge = ImageChops.subtract(overlap_outer, overlap_inner)
    overlap_edge_layer = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 255))
    overlap_edge_layer.putalpha(overlap_edge)

    canvas.alpha_composite(fill_layer)
    canvas.alpha_composite(overlap_layer)
    draw = ImageDraw.Draw(canvas)

    outline_width = 7
    draw.ellipse(top_box, outline=COLORS["inter_outline"], width=outline_width)
    draw.ellipse(bottom_box, outline=COLORS["deep_outline"], width=outline_width)
    canvas.alpha_composite(overlap_edge_layer)
    draw = ImageDraw.Draw(canvas)

    draw_centered_text(
        draw,
        center_x,
        top_center_y - top_radius - 100,
        f"Interpretable Features\n{format_percent(inter_r2)}",
        label_font,
        COLORS["inter_outline"],
    )
    draw_centered_text(
        draw,
        center_x,
        bottom_center_y + bottom_radius + 85,
        f"Deep Features (Average of 3 Models)\n{format_percent(deep_r2)}",
        label_font,
        COLORS["deep_outline"],
    )

    draw_centered_text(
        draw,
        center_x,
        top_center_y - top_radius * 0.40,
        f"Interpretable Only\n{format_percent(float(row['unique_interpretable']))}",
        region_font,
        COLORS["text"],
    )
    draw_centered_text(
        draw,
        center_x,
        (top_center_y + bottom_center_y) / 2 - 150,
        f"Shared\n{format_percent(float(row['shared']))}",
        region_font,
        COLORS["text"],
    )
    draw_centered_text(
        draw,
        center_x,
        bottom_center_y + bottom_radius * 0.42,
        f"Deep Only\n{format_percent(float(row['unique_deep']))}",
        region_font,
        COLORS["text"],
    )

    canvas.convert("RGB").save(png_path, format="PNG", dpi=(300, 300))
    return png_path, csv_path


def main() -> None:
    df = load_partition_data()
    png_path, csv_path = render_venn(df)
    print(f"Saved figure: {png_path}")
    print(f"Saved data: {csv_path}")


if __name__ == "__main__":
    main()
