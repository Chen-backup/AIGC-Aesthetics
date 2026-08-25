import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from landscape_feature_config import HANDCRAFTED_FEATURES_CSV
from landscape_feature_utils import iter_landscape_images, read_image_bgr, safe_ratio, shannon_entropy


def colorfulness_metric(image_bgr: np.ndarray) -> float:
    image = image_bgr.astype(np.float32)
    rg = np.abs(image[:, :, 2] - image[:, :, 1])
    yb = np.abs(0.5 * (image[:, :, 2] + image[:, :, 1]) - image[:, :, 0])
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def hue_entropy(hsv_image: np.ndarray) -> float:
    hist, _ = np.histogram(hsv_image[:, :, 0], bins=36, range=(0, 180), density=True)
    return shannon_entropy(hist / (hist.sum() + 1e-8))


def compute_visual_center(edge_strength: np.ndarray) -> tuple[float, float, float]:
    total = float(edge_strength.sum())
    if total <= 1e-8:
        return 0.5, 0.5, 0.0

    ys, xs = np.indices(edge_strength.shape)
    center_x = float((xs * edge_strength).sum() / total / max(1, edge_strength.shape[1] - 1))
    center_y = float((ys * edge_strength).sum() / total / max(1, edge_strength.shape[0] - 1))
    offset = float(np.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2))
    return center_x, center_y, offset


def compute_line_strength(gray_image: np.ndarray) -> float:
    edges = cv2.Canny(gray_image, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None:
        return 0.0
    total_length = 0.0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        total_length += np.hypot(x2 - x1, y2 - y1)
    return float(total_length / (gray_image.shape[0] * gray_image.shape[1]))


def main() -> None:
    rows = []

    for image_path in tqdm(iter_landscape_images(), desc="Handcrafted features"):
        image_bgr = read_image_bgr(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        brightness = gray.astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        hue = hsv[:, :, 0].astype(np.float32)

        edges = cv2.Canny(gray, 80, 160)
        edge_strength = edges.astype(np.float32) / 255.0
        center_x, center_y, center_offset = compute_visual_center(edge_strength)

        h, w = gray.shape
        left_mean = float(brightness[:, : w // 2].mean())
        right_mean = float(brightness[:, w // 2 :].mean())
        top_mean = float(brightness[: h // 2, :].mean())
        bottom_mean = float(brightness[h // 2 :, :].mean())

        thirds_points = [
            (h // 3, w // 3),
            (h // 3, 2 * w // 3),
            (2 * h // 3, w // 3),
            (2 * h // 3, 2 * w // 3),
        ]
        thirds_values = [brightness[y, x] for y, x in thirds_points]

        warm_mask = ((hue >= 0) & (hue <= 30)) | ((hue >= 150) & (hue <= 179))
        cool_mask = (hue >= 60) & (hue <= 120)

        row = {
            "image_name": image_path.name,
            "brightness_mean": float(brightness.mean()),
            "brightness_std": float(brightness.std()),
            "contrast": float(gray.std()),
            "saturation_mean": float(saturation.mean()),
            "saturation_std": float(saturation.std()),
            "hue_entropy": float(hue_entropy(hsv)),
            "warm_ratio": float(warm_mask.mean()),
            "cool_ratio": float(cool_mask.mean()),
            "warm_cool_balance": float(warm_mask.mean() - cool_mask.mean()),
            "colorfulness": float(colorfulness_metric(image_bgr)),
            "edge_density": float(edge_strength.mean()),
            "left_right_balance": float(abs(left_mean - right_mean) / 255.0),
            "top_bottom_balance": float(abs(top_mean - bottom_mean) / 255.0),
            "thirds_brightness_mean": float(np.mean(thirds_values) / 255.0),
            "visual_center_x": center_x,
            "visual_center_y": center_y,
            "visual_center_offset": center_offset,
            "line_strength": float(compute_line_strength(gray)),
            "clarity": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(HANDCRAFTED_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved handcrafted features to: {HANDCRAFTED_FEATURES_CSV}")


if __name__ == "__main__":
    main()

