from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

MODEL_CONFIGS = [
    {
        "panel_title": "Interpretable Features",
        "trace_path": ROOT_DIR / "BYS_interpretable_model_result" / "full_model_trace.nc",
        "variables": [
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
        ],
        "labels": [
            "HW Ratio",
            "Eye-Face Ratio",
            "Mouth-Face Ratio",
            "Three Courts",
            "Upper-Lower",
            "Eye Y Ratio",
            "Total Symmetry",
            "Eye-Nose Angle",
            "Mouth-Nose",
            "Brightness",
            "Contrast",
            "Clarity",
            "Saturation",
            "Edge Density",
        ],
        "fill": "#4E79A7",
        "line": "#2F5D90",
    },
    {
        "panel_title": "StyleGAN Features",
        "trace_path": ROOT_DIR / "BYS_StyleGAN_model_14D_result" / "StyleGAN_14d_model_trace.nc",
        "variables": [f"PC{i}" for i in range(1, 15)],
        "labels": [f"PC{i}" for i in range(1, 15)],
        "fill": "#F28E2B",
        "line": "#C96F12",
    },
    {
        "panel_title": "InsightFace Features",
        "trace_path": ROOT_DIR / "BYS_Insightface_model_14D_result" / "deep_14d_model_trace.nc",
        "variables": [f"PC{i}" for i in range(1, 15)],
        "labels": [f"PC{i}" for i in range(1, 15)],
        "fill": "#59A14F",
        "line": "#3E7F38",
    },
    {
        "panel_title": "DINOv2 Features",
        "trace_path": ROOT_DIR / "BYS_DINOv2_model_14D_result" / "DINOv2_14d_model_trace.nc",
        "variables": [f"PC{i}" for i in range(1, 15)],
        "labels": [f"PC{i}" for i in range(1, 15)],
        "fill": "#E15759",
        "line": "#BF3F43",
    },
]

COLORS = {
    "background": "#FFFFFF",
    "title": "#181818",
    "text": "#2F2F2F",
    "subtle_text": "#666666",
    "zero": "#D1495B",
    "axis": "#4A4A4A",
    "grid": "#E7EBF0",
}

IMAGE_SIZE = (3600, 1800)
MARGINS = {
    "left": 72,
    "right": 48,
    "top": 145,
    "bottom": 125,
}
PANEL_GAP = 10
GRID_STEPS = 6


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


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    red, green, blue = ImageColor.getrgb(hex_color)
    return red, green, blue, alpha


def kernel_density_estimate(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.zeros_like(x_grid)
    std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.1
    bandwidth = 1.06 * std * (samples.size ** (-1 / 5)) if std > 0 else 0.08
    bandwidth = max(bandwidth, (x_grid[-1] - x_grid[0]) / 100.0, 0.03)
    z = (x_grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return density


def collect_posterior_data() -> list[dict[str, object]]:
    panel_data: list[dict[str, object]] = []

    for config in MODEL_CONFIGS:
        trace = az.from_netcdf(config["trace_path"])
        feature_rows = []
        panel_samples: list[np.ndarray] = []
        for var_name, label in zip(config["variables"], config["labels"]):
            samples = np.asarray(trace.posterior[var_name].values, dtype=float).reshape(-1)
            feature_rows.append(
                {
                    "var_name": var_name,
                    "label": label,
                    "samples": samples,
                    "mean": float(np.mean(samples)),
                    "hdi_3": float(np.percentile(samples, 3)),
                    "hdi_97": float(np.percentile(samples, 97)),
                }
            )
            panel_samples.append(samples)

        panel_min = min(float(np.percentile(samples, 0.5)) for samples in panel_samples)
        panel_max = max(float(np.percentile(samples, 99.5)) for samples in panel_samples)
        pad = (panel_max - panel_min) * 0.06
        if pad == 0:
            pad = 0.1

        panel_data.append(
            {
                "panel_title": config["panel_title"],
                "fill": config["fill"],
                "line": config["line"],
                "x_min": panel_min - pad,
                "x_max": panel_max + pad,
                "features": feature_rows,
            }
        )

    return panel_data


def format_tick(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def render_ridgeplot(panel_data: list[dict[str, object]]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "posterior_ridgeplot_4models.png"
    csv_path = OUTPUT_DIR / "posterior_ridgeplot_4models_data.csv"

    csv_rows: list[dict[str, object]] = []
    for panel in panel_data:
        x_min = float(panel["x_min"])
        x_max = float(panel["x_max"])
        x_grid = np.linspace(x_min, x_max, 260)
        for feature in panel["features"]:
            density = kernel_density_estimate(feature["samples"], x_grid)
            feature["density"] = density
            feature["density_max"] = float(density.max())
            feature["x_grid"] = x_grid
            for x_value, density_value in zip(x_grid, density):
                csv_rows.append(
                {
                        "panel": panel["panel_title"],
                        "feature": feature["label"],
                        "x": float(x_value),
                        "density": float(density_value),
                        "mean": feature["mean"],
                        "hdi_3": feature["hdi_3"],
                        "hdi_97": feature["hdi_97"],
                    }
                )
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    canvas = Image.new("RGBA", IMAGE_SIZE, rgba(COLORS["background"], 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(70, bold=True)
    panel_title_font = load_font(48, bold=True)
    tick_font = load_font(30, bold=True)
    axis_font = load_font(38, bold=True)

    draw_centered_text(
        draw,
        IMAGE_SIZE[0] / 2,
        68,
        "Posterior Ridge Distributions Across Interpretable and Deep Features",
        title_font,
        COLORS["title"],
    )

    plot_top = MARGINS["top"] + 30
    plot_bottom = IMAGE_SIZE[1] - MARGINS["bottom"] - 20
    plot_height = plot_bottom - plot_top
    panel_width = (IMAGE_SIZE[0] - MARGINS["left"] - MARGINS["right"] - PANEL_GAP * 3) / 4
    tick_values = np.linspace(x_min, x_max, GRID_STEPS)

    for panel_index, panel in enumerate(panel_data):
        left = MARGINS["left"] + panel_index * (panel_width + PANEL_GAP)
        right = left + panel_width
        axis_left = left + 18
        axis_right = right - 10
        x_min = float(panel["x_min"])
        x_max = float(panel["x_max"])
        title_y = MARGINS["top"] - 6
        draw_centered_text(draw, (left + right) / 2, title_y, str(panel["panel_title"]), panel_title_font, COLORS["title"])

        feature_rows = panel["features"]
        baseline_space = plot_height / len(feature_rows)
        ridge_height = baseline_space * 0.82
        tick_values = np.linspace(x_min, x_max, GRID_STEPS)

        for tick_value in tick_values:
            x = axis_left + (tick_value - x_min) / (x_max - x_min) * (axis_right - axis_left)
            draw.line((x, plot_top - 8, x, plot_bottom + 8), fill=COLORS["grid"], width=2)
            tick_text = format_tick(float(tick_value))
            tick_w, _ = text_bbox(draw, tick_text, tick_font)
            draw.text((x - tick_w / 2, plot_bottom + 18), tick_text, font=tick_font, fill=COLORS["subtle_text"])

        if x_min <= 0.0 <= x_max:
            zero_x = axis_left + (0.0 - x_min) / (x_max - x_min) * (axis_right - axis_left)
            draw.line((zero_x, plot_top - 14, zero_x, plot_bottom + 6), fill=COLORS["zero"], width=3)

        for ridge_index, feature in enumerate(feature_rows):
            baseline_y = plot_top + baseline_space * (ridge_index + 0.76)

            density = feature["density"]
            x_grid = feature["x_grid"]
            density_scale = ridge_height / max(float(feature["density_max"]), 1e-6)
            polygon = [(axis_left, baseline_y)]
            for x_value, density_value in zip(x_grid, density):
                px = axis_left + (x_value - x_min) / (x_max - x_min) * (axis_right - axis_left)
                py = baseline_y - density_value * density_scale
                polygon.append((px, py))
            polygon.append((axis_right, baseline_y))

            draw.polygon(polygon, fill=rgba(str(panel["fill"]), 185))
            draw.line(polygon[1:-1], fill=str(panel["line"]), width=4)
            draw.line((axis_left, baseline_y, axis_right, baseline_y), fill=rgba(str(panel["line"]), 80), width=2)

        draw.line((axis_left, plot_bottom + 6, axis_right, plot_bottom + 6), fill=COLORS["axis"], width=3)
        draw_centered_text(draw, (axis_left + axis_right) / 2, IMAGE_SIZE[1] - 36, "Parameter Value", axis_font, COLORS["title"])

    png_rgb = canvas.convert("RGB")
    png_rgb.save(png_path, format="PNG", dpi=(300, 300))
    return png_path, csv_path


def main() -> None:
    panel_data = collect_posterior_data()
    png_path, csv_path = render_ridgeplot(panel_data)
    print(f"Saved figure: {png_path}")
    print(f"Saved data: {csv_path}")


if __name__ == "__main__":
    main()
