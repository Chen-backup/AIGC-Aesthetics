from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


OUTPUT_DIR = Path("picture")

OBJECTIVE_SHARE = 80.97
SUBJECTIVE_SHARE = 19.03

COLORS = {
    "background": "#FFFFFF",
    "title": "#181818",
    "text": "#2B2B2B",
    "subtle_text": "#666666",
    "objective": "#4E79A7",
    "subjective": "#F28E2B",
    "outline": "#FFFFFF",
}

IMAGE_SIZE = (1600, 1600)


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "icc_variance_pie_chart.png"

    canvas = Image.new("RGB", IMAGE_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    center_font = load_font(70, bold=True)
    label_font = load_font(52, bold=True)

    pie_size = 1180
    pie_left = (IMAGE_SIZE[0] - pie_size) // 2
    pie_top = (IMAGE_SIZE[1] - pie_size) // 2
    pie_box = (pie_left, pie_top, pie_left + pie_size, pie_top + pie_size)

    start_angle = -90
    objective_end = start_angle + 360 * OBJECTIVE_SHARE / 100.0
    draw.pieslice(
        pie_box,
        start=start_angle,
        end=objective_end,
        fill=COLORS["objective"],
        outline=COLORS["outline"],
        width=6,
    )
    draw.pieslice(
        pie_box,
        start=objective_end,
        end=start_angle + 360,
        fill=COLORS["subjective"],
        outline=COLORS["outline"],
        width=6,
    )

    inner_margin = 330
    inner_box = (
        pie_left + inner_margin,
        pie_top + inner_margin,
        pie_left + pie_size - inner_margin,
        pie_top + pie_size - inner_margin,
    )
    draw.ellipse(inner_box, fill=COLORS["background"])
    draw_centered_text(
        draw,
        pie_left + pie_size / 2,
        pie_top + pie_size / 2,
        "Null Model\nScore\nFluctuation",
        center_font,
        COLORS["title"],
    )

    draw_centered_text(
        draw,
        pie_left + pie_size * 0.50,
        pie_top + pie_size * 0.83,
        "Objective characteristics\nof images\n\n80.97%",
        label_font,
        "#FFFFFF",
    )
    draw_centered_text(
        draw,
        pie_left + pie_size * 0.30,
        pie_top + pie_size * 0.20,
        "Subjective\naesthetic\nheterogeneity\n\n19.03%",
        label_font,
        COLORS["title"],
    )

    canvas.save(output_path, format="PNG", dpi=(300, 300))
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
