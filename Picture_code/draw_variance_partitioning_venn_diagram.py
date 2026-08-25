from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

NULL_REPORT = ROOT_DIR / "BYS_kong_2_result" / "Null_Model_Metrics_Report.txt"
INTERPRETABLE_REPORT = ROOT_DIR / "BYS_interpretable_model_result" / "Full_Model_Metrics_Report.txt"

MODEL_SPECS = [
    {
        "name": "StyleGAN",
        "single_report": ROOT_DIR / "BYS_StyleGAN_model_14D_result" / "StyleGAN_14D_Model_Metrics_Report.txt",
        "fusion_report": ROOT_DIR / "BYS_Fusion_28D_StyleGAN_result" / "Fusion_28D_Model_Metrics.txt",
        "model_color": "#F28E2B",
        "model_outline": "#CC7518",
    },
    {
        "name": "InsightFace",
        "single_report": ROOT_DIR / "BYS_Insightface_model_14D_result" / "Deep_14D_Model_Metrics_Report.txt",
        "fusion_report": ROOT_DIR / "BYS_Fusion_28D_InsightFace_result" / "Fusion_28D_Model_Metrics.txt",
        "model_color": "#59A14F",
        "model_outline": "#3F7E39",
    },
    {
        "name": "DINOv2",
        "single_report": ROOT_DIR / "BYS_DINOv2_model_14D_result" / "DINOv2_14D_Model_Metrics_Report.txt",
        "fusion_report": ROOT_DIR / "BYS_Fusion_28D_DINOv2_result" / "Fusion_28D_Model_Metrics.txt",
        "model_color": "#E15759",
        "model_outline": "#C43F43",
    },
]

COLORS = {
    "background": "#FFFFFF",
    "title": "#181818",
    "text": "#2B2B2B",
    "subtle_text": "#646A73",
    "inter_fill": "#4E79A7",
    "inter_outline": "#2F5D90",
    "overlap_fill": "#B8BEC8",
    "overlap_outline": "#7D858F",
    "summary_fill": "#E0E3E8",
    "summary_outline": "#B7BDC6",
}

IMAGE_SIZE = (3600, 1800)
MARGINS = {
    "left": 55,
    "right": 55,
    "top": 140,
    "bottom": 90,
}
PANEL_GAP = 8


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
    spacing: int = 4,
) -> None:
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="left")


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


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def radius_from_r2(r2_value: float, base_r2: float = 42.60, base_radius: float = 220.0) -> float:
    return base_radius * ((r2_value / base_r2) ** 0.5)


def load_partition_table() -> pd.DataFrame:
    null_text = read_text(NULL_REPORT)
    interpretable_text = read_text(INTERPRETABLE_REPORT)

    null_image_variance = extract_float(r"Image Variance .*?:\s*([0-9.]+)", null_text, "null image variance")
    interpretable_residual = extract_float(
        r"Residual Image Variance .*?:\s*([0-9.]+)",
        interpretable_text,
        "interpretable residual image variance",
    )
    interpretable_r2 = (1.0 - interpretable_residual / null_image_variance) * 100.0

    rows = []
    for spec in MODEL_SPECS:
        single_text = read_text(spec["single_report"])
        fusion_text = read_text(spec["fusion_report"])

        single_r2 = extract_float(r"([0-9]+\.[0-9]+)%", single_text, f"{spec['name']} single r2")
        fusion_r2 = extract_float(r"([0-9]+\.[0-9]+)%", fusion_text, f"{spec['name']} fusion r2")

        rows.append(
            {
                "model": spec["name"],
                "interpretable_r2": round(interpretable_r2, 2),
                "model_r2": round(single_r2, 2),
                "fusion_r2": round(fusion_r2, 2),
                "shared": round(interpretable_r2 + single_r2 - fusion_r2, 2),
                "unique_interpretable": round(fusion_r2 - single_r2, 2),
                "unique_model": round(fusion_r2 - interpretable_r2, 2),
                "unexplained": round(100.0 - fusion_r2, 2),
                "model_color": spec["model_color"],
                "model_outline": spec["model_outline"],
            }
        )

    return pd.DataFrame(rows)


def render_diagram(df: pd.DataFrame) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "variance_partitioning_venn_diagram_data.csv"
    png_path = OUTPUT_DIR / "variance_partitioning_venn_diagram.png"
    df.to_csv(csv_path, index=False)

    canvas = Image.new("RGBA", IMAGE_SIZE, rgba(COLORS["background"], 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(80, bold=True)
    panel_title_font = load_font(60, bold=True)
    set_label_font = load_font(50, bold=True)
    region_font = load_font(50, bold=True)
    callout_font = load_font(50, bold=True)

    draw_centered_text(
        draw,
        IMAGE_SIZE[0] / 2,
        70,
        "Variance Partitioning Between Interpretable and Deep Features",
        title_font,
        COLORS["title"],
    )

    panel_width = (IMAGE_SIZE[0] - MARGINS["left"] - MARGINS["right"] - PANEL_GAP * 2) / 3
    panel_height = IMAGE_SIZE[1] - MARGINS["top"] - MARGINS["bottom"]
    outline_width = 5

    for index, row in df.iterrows():
        panel_left = MARGINS["left"] + index * (panel_width + PANEL_GAP)
        panel_right = panel_left + panel_width
        panel_top = MARGINS["top"]
        panel_bottom = panel_top + panel_height

        draw_centered_text(
            draw,
            (panel_left + panel_right) / 2,
            panel_top + 60,
            f"Interpretable + {row['model']}",
            panel_title_font,
            COLORS["title"],
        )

        venn_center_x = panel_left + panel_width * 0.50
        top_radius = radius_from_r2(float(row["interpretable_r2"]), base_radius=275.0)
        bottom_radius = radius_from_r2(float(row["model_r2"]), base_radius=275.0)
        top_center_y = panel_top + 520
        bottom_center_y = top_center_y + 345

        top_box = (
            venn_center_x - top_radius,
            top_center_y - top_radius,
            venn_center_x + top_radius,
            top_center_y + top_radius,
        )
        bottom_box = (
            venn_center_x - bottom_radius,
            bottom_center_y - bottom_radius,
            venn_center_x + bottom_radius,
            bottom_center_y + bottom_radius,
        )

        fill_layer = Image.new("RGBA", IMAGE_SIZE, (255, 255, 255, 0))
        fill_draw = ImageDraw.Draw(fill_layer)
        fill_draw.ellipse(bottom_box, fill=rgba(str(row["model_color"]), 175))
        fill_draw.ellipse(top_box, fill=rgba(COLORS["inter_fill"], 175))

        top_mask = Image.new("L", IMAGE_SIZE, 0)
        bottom_mask = Image.new("L", IMAGE_SIZE, 0)
        ImageDraw.Draw(top_mask).ellipse(top_box, fill=255)
        ImageDraw.Draw(bottom_mask).ellipse(bottom_box, fill=255)
        overlap_mask = ImageChops.multiply(top_mask, bottom_mask)

        overlap_layer = Image.new("RGBA", IMAGE_SIZE, rgba(COLORS["overlap_fill"], 220))
        overlap_layer.putalpha(overlap_mask)

        overlap_outer = overlap_mask.filter(ImageFilter.MaxFilter(7))
        overlap_inner = overlap_mask.filter(ImageFilter.MinFilter(7))
        overlap_edge = ImageChops.subtract(overlap_outer, overlap_inner)
        overlap_edge_layer = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 255))
        overlap_edge_layer.putalpha(overlap_edge)

        canvas.alpha_composite(fill_layer)
        canvas.alpha_composite(overlap_layer)
        draw = ImageDraw.Draw(canvas)

        draw.ellipse(bottom_box, outline=str(row["model_outline"]), width=outline_width)
        draw.ellipse(top_box, outline=COLORS["inter_outline"], width=outline_width)
        canvas.alpha_composite(overlap_edge_layer)
        draw = ImageDraw.Draw(canvas)

        draw_centered_text(
            draw,
            venn_center_x,
            top_center_y - top_radius - 60,
            f"Interpretable\n{format_percent(float(row['interpretable_r2']))}",
            set_label_font,
            COLORS["inter_outline"],
        )
        draw_centered_text(
            draw,
            venn_center_x,
            bottom_center_y + bottom_radius + 60,
            f"{row['model']}\n{format_percent(float(row['model_r2']))}",
            set_label_font,
            str(row["model_outline"]),
        )

        draw_centered_text(
            draw,
            venn_center_x,
            top_center_y - top_radius * 0.50,
            f"Interpretable only\n{format_percent(float(row['unique_interpretable']))}",
            callout_font,
            "#111111",
        )
        draw_centered_text(
            draw,
            venn_center_x,
            (top_center_y + bottom_center_y) / 2 - 55,
            f"Shared\n{format_percent(float(row['shared']))}",
            region_font,
            "#111111",
        )
        draw_centered_text(
            draw,
            venn_center_x,
            bottom_center_y + bottom_radius * 0.46,
            f"{row['model']} only\n{format_percent(float(row['unique_model']))}",
            callout_font,
            "#111111",
        )

        summary_box = (
            panel_left + 95,
            panel_bottom - 130,
            panel_right - 95,
            panel_bottom - 18,
        )
        draw.rounded_rectangle(
            summary_box,
            radius=22,
            fill=COLORS["summary_fill"],
            outline=COLORS["summary_outline"],
            width=3,
        )
        draw_centered_text(
            draw,
            (summary_box[0] + summary_box[2]) / 2,
            (summary_box[1] + summary_box[3]) / 2 - 2,
            f"Fusion explained = {format_percent(float(row['fusion_r2']))}\nUnexplained = {format_percent(float(row['unexplained']))}",
            set_label_font,
            COLORS["text"],
        )

    canvas.convert("RGB").save(png_path, format="PNG", dpi=(300, 300))
    return png_path, csv_path


def main() -> None:
    partition_df = load_partition_table()
    png_path, csv_path = render_diagram(partition_df)
    print(f"Saved figure: {png_path}")
    print(f"Saved data: {csv_path}")


if __name__ == "__main__":
    main()
