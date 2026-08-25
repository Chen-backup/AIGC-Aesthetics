from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "picture"
TRACE_PATH = ROOT_DIR / "BYS_kong_2_result" / "null_model_trace.nc"

IMAGE_SIZE = (2000, 2000)  # 1:1
# Reserve more left whitespace so axis labels can be enlarged later without crowding.
MARGINS = {"left": 300, "right": 80, "top": 160, "bottom": 210}

COLORS = {
    "background": "#FFFFFF",
    "axis": "#333333",
    "grid": "#DDE3EA",
    "text": "#202020",
}

PLOTS = [
    {
        "var": "1|rater_sigma",
        "title": "Posterior Density: 1|rater_sigma",
        "x_label": "Value of 1|rater_sigma",
        "y_label": "Posterior Density",
        "chain_color": "#D66076",
        "mean_color": "#B63A4D",
        "fill_color": "#F4D2D8",
        "output_name": "Null_Trace_1_rater_sigma_1x1.png",
    },
    {
        "var": "1|image_sigma",
        "title": "Posterior Density: 1|image_sigma",
        "x_label": "Value of 1|image_sigma",
        "y_label": "Posterior Density",
        "chain_color": "#5A97D0",
        "mean_color": "#1F6FB8",
        "fill_color": "#D4E5F5",
        "output_name": "Null_Trace_2_image_sigma_1x1.png",
    },
]


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


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    draw.text((center_x - width / 2, center_y - height / 2), text, font=font, fill=fill)


def map_x(value: float, xmin: float, xmax: float, x0: int, x1: int) -> float:
    if xmax == xmin:
        return (x0 + x1) / 2
    return x0 + ((value - xmin) / (xmax - xmin)) * (x1 - x0)


def map_y(value: float, vmin: float, vmax: float, y0: int, y1: int) -> float:
    if vmax == vmin:
        return (y0 + y1) / 2
    return y1 - ((value - vmin) / (vmax - vmin)) * (y1 - y0)


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int,
    dash: int = 10,
    gap: int = 8,
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
        dash_end = min(progress + dash, distance)
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
        progress += dash + gap


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int,
    dash: int = 10,
    gap: int = 8,
) -> None:
    for i in range(len(points) - 1):
        draw_dashed_segment(draw, points[i], points[i + 1], fill=fill, width=width, dash=dash, gap=gap)


def format_tick(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def draw_single_density(trace_values: np.ndarray, spec: dict[str, str]) -> Path:
    # trace_values shape: (n_chain, n_draw)
    n_chain, _ = trace_values.shape

    canvas = Image.new("RGB", IMAGE_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(100, bold=True)
    axis_label_font = load_font(90, bold=True)
    tick_font = load_font(58, bold=True)
    legend_font = load_font(65, bold=True)

    plot_left = MARGINS["left"]
    plot_right = IMAGE_SIZE[0] - MARGINS["right"]
    plot_top = MARGINS["top"]
    plot_bottom = IMAGE_SIZE[1] - MARGINS["bottom"]

    xmin = float(np.min(trace_values))
    xmax = float(np.max(trace_values))
    xpad = (xmax - xmin) * 0.10
    if xpad <= 0:
        xpad = 0.05
    xmin -= xpad
    xmax += xpad

    x_grid = np.linspace(xmin, xmax, 400)
    chain_densities = []
    for c in range(n_chain):
        kde = gaussian_kde(trace_values[c])
        chain_densities.append(kde(x_grid))
    density_arr = np.vstack(chain_densities)
    mean_density = density_arr.mean(axis=0)
    ymin = 0.0
    ymax = float(max(np.max(density_arr), np.max(mean_density))) * 1.08

    draw_centered_text(draw, IMAGE_SIZE[0] / 2, 70, spec["title"], title_font, COLORS["text"])

    x_ticks = np.linspace(xmin, xmax, 6)
    y_ticks = np.linspace(ymin, ymax, 6)

    # Grid + tick labels
    for xt in x_ticks:
        x = map_x(float(xt), xmin, xmax, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill=COLORS["grid"], width=2)
        t = format_tick(float(xt))
        tb = draw.textbbox((0, 0), t, font=tick_font)
        tw = tb[2] - tb[0]
        draw.text((x - tw / 2, plot_bottom + 16), t, font=tick_font, fill=COLORS["axis"])

    max_y_tick_label_w = 0
    for yt in y_ticks:
        y = map_y(float(yt), ymin, ymax, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill=COLORS["grid"], width=2)
        t = format_tick(float(yt))
        tb = draw.textbbox((0, 0), t, font=tick_font)
        tw = tb[2] - tb[0]
        if tw > max_y_tick_label_w:
            max_y_tick_label_w = tw
        th = tb[3] - tb[1]
        draw.text((plot_left - tw - 14, y - th / 2), t, font=tick_font, fill=COLORS["axis"])

    # Fill area under the mean KDE curve for stronger visual presence.
    fill_points = [
        (
            map_x(float(x_grid[i]), xmin, xmax, plot_left, plot_right),
            map_y(float(mean_density[i]), ymin, ymax, plot_top, plot_bottom),
        )
        for i in range(len(x_grid))
    ]
    fill_points.append((map_x(float(x_grid[-1]), xmin, xmax, plot_left, plot_right), plot_bottom))
    fill_points.append((map_x(float(x_grid[0]), xmin, xmax, plot_left, plot_right), plot_bottom))
    draw.polygon(fill_points, fill=spec["fill_color"])

    # 4 dashed chain density curves
    for c in range(n_chain):
        points = [
            (
                map_x(float(x_grid[i]), xmin, xmax, plot_left, plot_right),
                map_y(float(density_arr[c, i]), ymin, ymax, plot_top, plot_bottom),
            )
            for i in range(len(x_grid))
        ]
        draw_dashed_polyline(draw, points, fill=spec["chain_color"], width=4, dash=12, gap=8)

    # Mean solid density curve
    mean_points = [
        (
            map_x(float(x_grid[i]), xmin, xmax, plot_left, plot_right),
            map_y(float(mean_density[i]), ymin, ymax, plot_top, plot_bottom),
        )
        for i in range(len(x_grid))
    ]
    draw.line(mean_points, fill=spec["mean_color"], width=8, joint="curve")

    # Axes: keep left + bottom only
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["axis"], width=15)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["axis"], width=15)

    # Axis labels
    draw_centered_text(
        draw,
        (plot_left + plot_right) / 2,
        IMAGE_SIZE[1] - 92,
        spec["x_label"],
        axis_label_font,
        COLORS["axis"],
    )

    y_label = spec["y_label"]
    yl_box = draw.textbbox((0, 0), y_label, font=axis_label_font)
    yl_w = yl_box[2] - yl_box[0]
    yl_h = yl_box[3] - yl_box[1]
    temp = Image.new("RGBA", (yl_w + 24, yl_h + 24), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((12, 12), y_label, font=axis_label_font, fill=COLORS["axis"])
    rotated = temp.rotate(90, expand=True)
    tick_left_x = plot_left - max_y_tick_label_w - 14
    y_label_gap = 34
    y_label_x = max(0, int(tick_left_x - y_label_gap - rotated.size[0]))
    canvas.paste(rotated, (y_label_x, int((plot_top + plot_bottom - rotated.size[1]) / 2)), rotated)

    # Legend
    legend_line_label_1 = "4 chains KDE (dashed)"
    legend_line_label_2 = "Mean KDE across chains"
    sample_len = 100
    gap = 14
    text_gap = 16
    l1_w = draw.textbbox((0, 0), legend_line_label_1, font=legend_font)[2]
    l2_w = draw.textbbox((0, 0), legend_line_label_2, font=legend_font)[2]
    legend_width = sample_len + text_gap + max(l1_w, l2_w) + gap * 2
    legend_x = int(plot_right - legend_width - 18)
    legend_y = plot_top + 20
    # Align legend sample lines to the vertical center of each text row.
    text_x = legend_x + sample_len + text_gap
    row1_text_y = legend_y - 8
    row2_text_y = legend_y + 66

    row1_box = draw.textbbox((0, 0), legend_line_label_1, font=legend_font)
    row1_h = row1_box[3] - row1_box[1]
    row1_line_y = row1_text_y + row1_h / 2

    row2_box = draw.textbbox((0, 0), legend_line_label_2, font=legend_font)
    row2_h = row2_box[3] - row2_box[1]
    row2_line_y = row2_text_y + row2_h / 2

    draw_dashed_segment(
        draw,
        (legend_x, row1_line_y),
        (legend_x + sample_len, row1_line_y),
        fill=spec["chain_color"],
        width=4,
        dash=12,
        gap=8,
    )
    draw.text((text_x, row1_text_y), legend_line_label_1, font=legend_font, fill=COLORS["text"])

    draw.line((legend_x, row2_line_y, legend_x + sample_len, row2_line_y), fill=spec["mean_color"], width=8)
    draw.text((text_x, row2_text_y), legend_line_label_2, font=legend_font, fill=COLORS["text"])

    output_path = OUTPUT_DIR / spec["output_name"]
    canvas.save(output_path, format="PNG", dpi=(300, 300))
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trace = az.from_netcdf(TRACE_PATH)
    saved_paths: list[Path] = []

    for spec in PLOTS:
        values = np.asarray(trace.posterior[spec["var"]].values, dtype=float)
        if values.ndim != 2:
            values = values.reshape(values.shape[0], -1)
        saved_paths.append(draw_single_density(values, spec))

    print("Saved figures:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
