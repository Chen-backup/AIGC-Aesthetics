from __future__ import annotations

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"

TRACE_PATH = ROOT_DIR / "BYS_GAMM_NonLinear_Result" / "GAMM_model_trace_Top6.nc"
RATINGS_PATH = ROOT_DIR / "ratings_for_bayesian_model.xlsx"
MAPPING_PATH = ROOT_DIR / "renumber&gender.xlsx"
FEATURES_PATH = ROOT_DIR / "interpretable_face_features.csv"

FEATURES = [
    ("le_nose_re_angle", "Left Eye-Nose-Right Eye Angle"),
    ("upper_lower_ratio", "Upper-Lower Face Ratio"),
    ("mouth_face_w_ratio", "Mouth-Face Width Ratio"),
    ("total_symmetry", "Total Symmetry"),
    ("edge_density", "Edge Density"),
    ("eye_y_ratio", "Eye Vertical Position Ratio"),
]

COLORS = {
    "background": "#FFFFFF",
    "axis": "#454545",
    "grid": "#E9EDF2",
    "title": "#181818",
    "tick": "#4A4A4A",
    "rug": "#8A94A6",
}

LINE_COLOR = "#2067A9"
BAND_COLOR = "#A8C6E6"
RUG_QUANTILE_COUNT = 20

IMAGE_SIZE = (3200, 1850)
MARGINS = {
    "left": 200,
    "right": 80,
    "top": 210,
    "bottom": 155,
}
GRID = {
    "cols": 3,
    "rows": 2,
    "x_gap": 55,
    "y_gap": 70,
}

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


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    r, g, b = ImageColor.getrgb(hex_color)
    return r, g, b, alpha


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int = 1,
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


def trim_number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def build_dataset() -> tuple[pd.DataFrame, list[float], dict[str, dict[str, float]]]:
    df_ratings = pd.read_excel(RATINGS_PATH)
    df_mapping = pd.read_excel(MAPPING_PATH)
    df_features = pd.read_csv(FEATURES_PATH)

    geom_merge_key = "face_id" if "face_id" in df_features.columns else "image_name"
    df_combined = pd.merge(
        df_mapping[["face_id", "Number"]],
        df_features,
        left_on="face_id",
        right_on=geom_merge_key,
        how="inner",
    )
    df = pd.merge(df_ratings, df_combined, left_on="image", right_on="Number", how="inner")

    feature_names = [name for name, _ in FEATURES]
    df = df.dropna(subset=feature_names + ["rating", "rater", "image"]).copy()
    df["rater"] = df["rater"].astype(str)
    df["image"] = df["image"].astype(str)

    rating_categories = sorted(df["rating"].unique())
    df["rating"] = pd.Categorical(df["rating"], categories=rating_categories, ordered=True)

    feature_stats: dict[str, dict[str, float]] = {}
    for feature_name, _ in FEATURES:
        mean_value = float(df[feature_name].mean())
        std_value = float(df[feature_name].std())
        df[feature_name] = (df[feature_name] - mean_value) / std_value
        feature_stats[feature_name] = {
            "mean": mean_value,
            "std": std_value,
            "z_min": float(df[feature_name].min()),
            "z_max": float(df[feature_name].max()),
        }

    return df, [float(value) for value in rating_categories], feature_stats


def compute_curve_data() -> pd.DataFrame:
    df, rating_categories, feature_stats = build_dataset()
    trace = az.from_netcdf(TRACE_PATH)

    spline_terms = [f"bs({feature_name}, df=4)" for feature_name, _ in FEATURES]
    formula = f"rating ~ 1 + {' + '.join(spline_terms)} + (1|rater) + (1|image)"
    model = bmb.Model(formula, data=df, family="cumulative")

    rating_cats_numeric = np.array(rating_categories, dtype=float)
    base_rater = df["rater"].iloc[0]
    base_image = df["image"].iloc[0]

    rows: list[dict[str, float | int | str]] = []

    for feature_index, (feature_name, feature_label) in enumerate(FEATURES):
        z_min = feature_stats[feature_name]["z_min"]
        z_max = feature_stats[feature_name]["z_max"]
        margin = (z_max - z_min) * 0.01
        x_zscores = np.linspace(z_min + margin, z_max - margin, 120)

        dummy_df = pd.DataFrame({name: np.zeros(len(x_zscores)) for name, _ in FEATURES})
        dummy_df[feature_name] = x_zscores
        dummy_df["rater"] = base_rater
        dummy_df["image"] = base_image

        pred = model.predict(
            trace,
            data=dummy_df,
            kind="response_params",
            include_group_specific=False,
            inplace=False,
        )

        probabilities = pred.posterior["p"].values
        expected_scores = np.sum(probabilities * rating_cats_numeric, axis=-1)

        mean_expected_score = expected_scores.mean(axis=(0, 1))
        lower_bound = np.percentile(expected_scores, 2.5, axis=(0, 1))
        upper_bound = np.percentile(expected_scores, 97.5, axis=(0, 1))

        x_real = x_zscores * feature_stats[feature_name]["std"] + feature_stats[feature_name]["mean"]
        observed_real = df[feature_name] * feature_stats[feature_name]["std"] + feature_stats[feature_name]["mean"]

        rug_values = np.quantile(
            observed_real.to_numpy(dtype=float),
            np.linspace(0.05, 0.95, RUG_QUANTILE_COUNT),
        )

        for idx, (x_value, mean_value, lower_value, upper_value) in enumerate(
            zip(x_real, mean_expected_score, lower_bound, upper_bound)
        ):
            rows.append(
                {
                    "feature_order": feature_index,
                    "feature_name": feature_name,
                    "feature_label": feature_label,
                    "curve_index": idx,
                    "x": float(x_value),
                    "mean_expected_score": float(mean_value),
                    "lower_95": float(lower_value),
                    "upper_95": float(upper_value),
                }
            )

        for rug_index, rug_value in enumerate(rug_values):
            rows.append(
                {
                    "feature_order": feature_index,
                    "feature_name": feature_name,
                    "feature_label": feature_label,
                    "curve_index": -1,
                    "x": float(rug_value),
                    "mean_expected_score": np.nan,
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "rug_index": rug_index,
                }
            )

    curve_df = pd.DataFrame(rows)
    if "rug_index" not in curve_df.columns:
        curve_df["rug_index"] = np.nan
    return curve_df


def get_panel_rectangles() -> list[tuple[int, int, int, int]]:
    usable_width = IMAGE_SIZE[0] - MARGINS["left"] - MARGINS["right"] - GRID["x_gap"] * (GRID["cols"] - 1)
    usable_height = IMAGE_SIZE[1] - MARGINS["top"] - MARGINS["bottom"] - GRID["y_gap"] * (GRID["rows"] - 1)
    panel_width = usable_width // GRID["cols"]
    panel_height = usable_height // GRID["rows"]

    rectangles = []
    for row in range(GRID["rows"]):
        for col in range(GRID["cols"]):
            left = MARGINS["left"] + col * (panel_width + GRID["x_gap"])
            top = MARGINS["top"] + row * (panel_height + GRID["y_gap"])
            rectangles.append((left, top, left + panel_width, top + panel_height))
    return rectangles


def map_value(value: float, min_value: float, max_value: float, pixel_min: float, pixel_max: float) -> float:
    if max_value == min_value:
        return pixel_min
    ratio = (value - min_value) / (max_value - min_value)
    return pixel_min + ratio * (pixel_max - pixel_min)


def render_curves(curve_df: pd.DataFrame) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "GAMM_NonLinear_Expected_Curves_Top6_data.csv"
    curve_df.to_csv(csv_path, index=False)

    canvas = Image.new("RGBA", IMAGE_SIZE, rgba(COLORS["background"], 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(80, bold=True)
    axis_font = load_font(68, bold=True)
    tick_font = load_font(38, bold=True)
    panel_title_font = load_font(46, bold=True)

    title = "Nonlinear Expected Aesthetic Curves for Six Core Features"
    draw_centered_text(draw, IMAGE_SIZE[0] / 2, 95, title, title_font, COLORS["title"])

    y_label = "Expected Aesthetic Score"
    y_label_width, y_label_height = text_bbox(draw, y_label, axis_font)
    temp = Image.new("RGBA", (y_label_width + 20, y_label_height + 20), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((10, 10), y_label, font=axis_font, fill=COLORS["axis"])
    rotated = temp.rotate(90, expand=True)
    canvas.alpha_composite(rotated, (48, int((MARGINS["top"] + IMAGE_SIZE[1] - MARGINS["bottom"] - rotated.size[1]) / 2)))

    panel_rectangles = get_panel_rectangles()

    for feature_index, ((feature_name, feature_label), panel_rect) in enumerate(zip(FEATURES, panel_rectangles)):
        left, top, right, bottom = panel_rect

        panel_title_y = top + 30
        draw_centered_text(draw, (left + right) / 2, panel_title_y, feature_label, panel_title_font, COLORS["title"])

        axis_left = left + 80
        axis_right = right - 18
        axis_top = top + 64
        axis_bottom = bottom - 62

        feature_curve = curve_df[
            (curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] >= 0)
        ].sort_values("curve_index")
        rug_data = curve_df[
            (curve_df["feature_name"] == feature_name) & (curve_df["curve_index"] < 0)
        ].sort_values("rug_index")

        x_values = feature_curve["x"].to_numpy(dtype=float)
        mean_values = feature_curve["mean_expected_score"].to_numpy(dtype=float)
        lower_values = feature_curve["lower_95"].to_numpy(dtype=float)
        upper_values = feature_curve["upper_95"].to_numpy(dtype=float)

        x_min = float(x_values.min())
        x_max = float(x_values.max())
        x_span = x_max - x_min
        axis_spec = Y_AXIS_SPECS[feature_name]
        y_min, y_max, y_step = axis_spec
        y_ticks = np.arange(y_min, y_max + y_step * 0.5, y_step)

        x_ticks = np.linspace(x_min, x_max, 4)
        for y_tick in y_ticks:
            y = map_value(y_tick, y_min, y_max, axis_bottom, axis_top)
            draw_dashed_line(draw, (axis_left, y), (axis_right, y), fill=COLORS["grid"], width=2)
            tick_text = trim_number(float(y_tick))
            tick_width, tick_height = text_bbox(draw, tick_text, tick_font)
            draw_left_text(draw, axis_left - tick_width - 16, y - tick_height / 2, tick_text, tick_font, COLORS["tick"])

        for x_tick in x_ticks:
            x = map_value(float(x_tick), x_min, x_max, axis_left, axis_right)
            tick_text = format_x_tick(float(x_tick), x_span)
            tick_width, _ = text_bbox(draw, tick_text, tick_font)
            draw_left_text(draw, x - tick_width / 2, axis_bottom + 14, tick_text, tick_font, COLORS["tick"])

        band_points = []
        for x_value, upper_value in zip(x_values, upper_values):
            band_points.append(
                (
                    map_value(float(x_value), x_min, x_max, axis_left, axis_right),
                    map_value(float(upper_value), y_min, y_max, axis_bottom, axis_top),
                )
            )
        for x_value, lower_value in zip(x_values[::-1], lower_values[::-1]):
            band_points.append(
                (
                    map_value(float(x_value), x_min, x_max, axis_left, axis_right),
                    map_value(float(lower_value), y_min, y_max, axis_bottom, axis_top),
                )
            )

        clamped_band_points = []
        for px, py in band_points:
            clamped_band_points.append((px, min(max(py, axis_top), axis_bottom)))

        draw.polygon(clamped_band_points, fill=rgba(BAND_COLOR, 135))

        line_points = [
            (
                map_value(float(x_value), x_min, x_max, axis_left, axis_right),
                map_value(float(mean_value), y_min, y_max, axis_bottom, axis_top),
            )
            for x_value, mean_value in zip(x_values, mean_values)
        ]
        draw.line(line_points, fill=LINE_COLOR, width=7, joint="curve")

        rug_y_top = axis_bottom - 14
        rug_y_bottom = axis_bottom - 2
        for rug_value in rug_data["x"].to_numpy(dtype=float):
            rug_x = map_value(float(rug_value), x_min, x_max, axis_left, axis_right)
            draw.line((rug_x, rug_y_top, rug_x, rug_y_bottom), fill=COLORS["rug"], width=2)

        draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=COLORS["axis"], width=4)
        draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=COLORS["axis"], width=4)

    png_path = OUTPUT_DIR / "GAMM_NonLinear_Expected_Curves_Top6.png"
    canvas.convert("RGB").save(png_path, format="PNG", dpi=(300, 300))
    return png_path, csv_path


def main() -> None:
    curve_df = compute_curve_data()
    png_path, csv_path = render_curves(curve_df)
    print(f"Saved figure: {png_path}")
    print(f"Saved data: {csv_path}")


if __name__ == "__main__":
    main()
