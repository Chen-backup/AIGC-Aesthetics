from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from landscape_feature_config import (
    DEPTH_ANYTHING_MODEL_DIR,
    DEPTH_FEATURES_CSV,
    DEPTH_MAP_DIR,
    OUTPUT_DIR,
)
from landscape_feature_utils import ensure_dir, iter_landscape_images, read_image_rgb


SAVE_DEPTH_MAPS = False


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - depth_min) / (depth_max - depth_min)


def compute_depth_features(depth_norm: np.ndarray) -> dict[str, float]:
    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    q1, q2 = np.quantile(depth_norm, [1 / 3, 2 / 3])
    low_mask = depth_norm <= q1
    mid_mask = (depth_norm > q1) & (depth_norm <= q2)
    high_mask = depth_norm > q2

    center_patch = depth_norm[
        depth_norm.shape[0] // 4: depth_norm.shape[0] * 3 // 4,
        depth_norm.shape[1] // 4: depth_norm.shape[1] * 3 // 4,
    ]

    hist, _ = np.histogram(depth_norm, bins=5, range=(0.0, 1.0))
    layer_count = int((hist > (0.05 * depth_norm.size)).sum())

    return {
        "depth_mean": float(depth_norm.mean()),
        "depth_std": float(depth_norm.std()),
        "depth_low_ratio": float(low_mask.mean()),
        "depth_mid_ratio": float(mid_mask.mean()),
        "depth_high_ratio": float(high_mask.mean()),
        "depth_gradient_mean": float(grad.mean()),
        "depth_gradient_std": float(grad.std()),
        "depth_center_bias": float(center_patch.mean() - depth_norm.mean()),
        "depth_top_bottom_diff": float(depth_norm[: depth_norm.shape[0] // 2].mean() - depth_norm[depth_norm.shape[0] // 2 :].mean()),
        "depth_layer_count": float(layer_count),
    }


def main() -> None:
    if not DEPTH_ANYTHING_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Depth Anything V2 model directory not found: {DEPTH_ANYTHING_MODEL_DIR}"
        )

    ensure_dir(OUTPUT_DIR)
    if SAVE_DEPTH_MAPS:
        ensure_dir(DEPTH_MAP_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(str(DEPTH_ANYTHING_MODEL_DIR), local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        str(DEPTH_ANYTHING_MODEL_DIR), local_files_only=True
    ).to(device)
    model.eval()

    rows = []
    image_paths = iter_landscape_images()
    for image_path in tqdm(image_paths, desc="Depth features"):
        image_rgb = read_image_rgb(image_path)
        pil_image = Image.fromarray(image_rgb)
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = predicted_depth.cpu().numpy()
        depth_norm = normalize_depth(depth)

        feature_row = {"image_name": image_path.name}
        feature_row.update(compute_depth_features(depth_norm))
        rows.append(feature_row)

        if SAVE_DEPTH_MAPS:
            depth_path = DEPTH_MAP_DIR / f"{image_path.stem}_depth.png"
            cv2.imwrite(str(depth_path), (depth_norm * 255).astype(np.uint8))

    df = pd.DataFrame(rows)
    df.to_csv(DEPTH_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved depth features to: {DEPTH_FEATURES_CSV}")


if __name__ == "__main__":
    main()

