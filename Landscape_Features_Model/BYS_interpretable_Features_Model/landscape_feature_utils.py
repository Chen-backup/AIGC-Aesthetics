from pathlib import Path

import cv2
import numpy as np

from landscape_feature_config import DATASET_DIR, IMAGE_EXTENSIONS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_landscape_images() -> list[Path]:
    return sorted(
        path for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image_bgr(image_path: Path) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def read_image_rgb(image_path: Path) -> np.ndarray:
    image_bgr = read_image_bgr(image_path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def shannon_entropy(probabilities: np.ndarray) -> float:
    probs = probabilities[probabilities > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

