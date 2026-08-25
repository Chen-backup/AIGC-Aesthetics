from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT_DIR = Path("..")
OUTPUT_DIR = Path("picture")
CSV_PATH = ROOT_DIR / "BYS_Clustering_Results_Advanced" / "Rater_14D_Preferences.csv"

COLORS = {
    "background": "#FFFFFF",
    "title": "#181818",
    "axis": "#3F3F3F",
    "tick": "#4F4F4F",
    "border": "#D5DCE6",
}

IMAGE_SIZE = (2200, 1700)
MARGINS = {
    "left": 190,
    "right": 230,
    "top": 150,
    "bottom": 290,
}


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


def interpolate_color(value: float, anchors: list[tuple[float, str]]) -> tuple[int, int, int]:
    if value <= anchors[0][0]:
        return ImageColor.getrgb(anchors[0][1])
    if value >= anchors[-1][0]:
        return ImageColor.getrgb(anchors[-1][1])

    for idx in range(len(anchors) - 1):
        left_val, left_color = anchors[idx]
        right_val, right_color = anchors[idx + 1]
        if left_val <= value <= right_val:
            ratio = (value - left_val) / (right_val - left_val)
            left_rgb = np.array(ImageColor.getrgb(left_color), dtype=float)
            right_rgb = np.array(ImageColor.getrgb(right_color), dtype=float)
            rgb = left_rgb + (right_rgb - left_rgb) * ratio
            return tuple(int(round(channel)) for channel in rgb)
    return ImageColor.getrgb(anchors[-1][1])


def build_sorted_matrix() -> pd.DataFrame:
    df_preferences = pd.read_csv(CSV_PATH, index_col=0)
    features_only = [
        col for col in df_preferences.columns if col not in ["Cluster", "tSNE_1", "tSNE_2", "Final_Cluster"]
    ]
    df_raw = df_preferences[features_only].copy()

    values = df_raw.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    scaled_values = (values - mean) / std
    df_scaled = pd.DataFrame(scaled_values, index=df_raw.index, columns=df_raw.columns)

    num_rows, num_cols = scaled_values.shape
    covariance = np.zeros((num_cols, num_cols), dtype=float)
    for i in range(num_cols):
        for j in range(num_cols):
            covariance[i, j] = float(np.sum(scaled_values[:, i] * scaled_values[:, j]) / max(num_rows - 1, 1))

    pc1_loadings = np.ones(num_cols, dtype=float)
    pc1_loadings = pc1_loadings / np.sqrt(np.sum(pc1_loadings**2))
    for _ in range(60):
        next_vec = np.zeros(num_cols, dtype=float)
        for i in range(num_cols):
            total = 0.0
            for j in range(num_cols):
                total += covariance[i, j] * pc1_loadings[j]
            next_vec[i] = total
        norm = float(np.sqrt(np.sum(next_vec**2)))
        if norm == 0:
            break
        next_vec = next_vec / norm
        if float(np.max(np.abs(next_vec - pc1_loadings))) < 1e-10:
            pc1_loadings = next_vec
            break
        pc1_loadings = next_vec

    pc1_scores = np.sum(scaled_values * pc1_loadings, axis=1)
    df_scaled["PC1_Score"] = pc1_scores
    df_sorted = df_scaled.sort_values(by="PC1_Score", ascending=False).drop(columns=["PC1_Score"])

    loadings = pd.Series(pc1_loadings, index=df_raw.columns)
    sorted_features = loadings.sort_values(ascending=False).index
    return df_sorted[sorted_features]


def matrix_to_heatmap_image(df_sorted: pd.DataFrame, heatmap_width: int, heatmap_height: int) -> Image.Image:
    anchors = [
        (-3.0, "#2166AC"),
        (-2.0, "#4393C3"),
        (-1.0, "#92C5DE"),
        (0.0, "#F7F7F7"),
        (1.0, "#F4A582"),
        (2.0, "#D6604D"),
        (3.0, "#B2182B"),
    ]

    clipped = np.clip(df_sorted.to_numpy(dtype=float), -3.0, 3.0)
    rows, cols = clipped.shape
    rgb_array = np.zeros((rows, cols, 3), dtype=np.uint8)
    for row_idx in range(rows):
        for col_idx in range(cols):
            rgb_array[row_idx, col_idx] = interpolate_color(float(clipped[row_idx, col_idx]), anchors)

    base_image = Image.fromarray(rgb_array, mode="RGB")
    return base_image.resize((heatmap_width, heatmap_height), resample=Image.Resampling.BILINEAR)


def build_colorbar(height: int, width: int) -> Image.Image:
    anchors = [
        (-3.0, "#2166AC"),
        (-2.0, "#4393C3"),
        (-1.0, "#92C5DE"),
        (0.0, "#F7F7F7"),
        (1.0, "#F4A582"),
        (2.0, "#D6604D"),
        (3.0, "#B2182B"),
    ]
    values = np.linspace(3.0, -3.0, height)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for row_idx, value in enumerate(values):
        arr[row_idx, :, :] = interpolate_color(float(value), anchors)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "Plot2_PCA_Sorted_Heatmap.png"

    df_sorted = build_sorted_matrix()

    canvas = Image.new("RGB", IMAGE_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(46, bold=True)
    axis_font = load_font(34, bold=True)
    tick_font = load_font(28, bold=True)
    cbar_font = load_font(30, bold=True)

    heatmap_left = MARGINS["left"]
    heatmap_top = MARGINS["top"] + 80
    heatmap_right = IMAGE_SIZE[0] - MARGINS["right"] - 70
    heatmap_bottom = IMAGE_SIZE[1] - MARGINS["bottom"]
    heatmap_width = heatmap_right - heatmap_left
    heatmap_height = heatmap_bottom - heatmap_top

    title = "PCA-Sorted Continuous Gradient Heatmap of Aesthetic Preferences"
    draw_centered_text(draw, IMAGE_SIZE[0] / 2, 72, title, title_font, COLORS["title"])

    heatmap_img = matrix_to_heatmap_image(df_sorted, heatmap_width, heatmap_height)
    canvas.paste(heatmap_img, (heatmap_left, heatmap_top))
    draw.rectangle((heatmap_left, heatmap_top, heatmap_right, heatmap_bottom), outline=COLORS["border"], width=2)

    xtick_y = heatmap_bottom + 26
    feature_slot = heatmap_width / len(df_sorted.columns)
    for idx, feature_name in enumerate(df_sorted.columns):
        tick_x = heatmap_left + feature_slot * (idx + 0.5)
        temp = Image.new("RGBA", (260, 90), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp)
        feature_text = feature_name.replace("_", " ")
        temp_draw.text((0, 0), feature_text, font=tick_font, fill=COLORS["tick"])
        rotated = temp.rotate(45, expand=True)
        canvas.paste(rotated, (int(tick_x - rotated.size[0] / 2), int(xtick_y)), rotated)

    y_label = f"{len(df_sorted)} Raters (Sorted Top-to-Bottom by PC1 Score)"
    y_label_width, y_label_height = text_bbox(draw, y_label, axis_font)
    temp = Image.new("RGBA", (y_label_width + 30, y_label_height + 30), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((15, 15), y_label, font=axis_font, fill=COLORS["axis"])
    rotated_y = temp.rotate(90, expand=True)
    canvas.paste(rotated_y, (36, int((heatmap_top + heatmap_bottom - rotated_y.size[1]) / 2)), rotated_y)

    cbar_x = heatmap_right + 46
    cbar_width = 40
    cbar_img = build_colorbar(height=heatmap_height, width=cbar_width)
    canvas.paste(cbar_img, (cbar_x, heatmap_top))
    draw.rectangle((cbar_x, heatmap_top, cbar_x + cbar_width, heatmap_bottom), outline=COLORS["border"], width=2)

    for tick_val in [-3, -2, -1, 0, 1, 2, 3]:
        ratio = (3 - tick_val) / 6
        y = heatmap_top + ratio * heatmap_height
        draw.line((cbar_x + cbar_width, y, cbar_x + cbar_width + 12, y), fill=COLORS["axis"], width=2)
        draw.text((cbar_x + cbar_width + 18, y - 14), f"{tick_val}", font=tick_font, fill=COLORS["tick"])

    cbar_label = "Z-scored Preference Strength"
    label_w, label_h = text_bbox(draw, cbar_label, cbar_font)
    temp = Image.new("RGBA", (label_w + 30, label_h + 30), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((15, 15), cbar_label, font=cbar_font, fill=COLORS["axis"])
    rotated_cbar = temp.rotate(90, expand=True)
    canvas.paste(
        rotated_cbar,
        (cbar_x + 82, int((heatmap_top + heatmap_bottom - rotated_cbar.size[1]) / 2)),
        rotated_cbar,
    )

    canvas.save(output_path, format="PNG", dpi=(300, 300))
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
