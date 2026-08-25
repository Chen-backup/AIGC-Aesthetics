from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from landscape_feature_config import (
    OUTPUT_DIR,
    SEGFORMER_FEATURES_CSV,
    SEGFORMER_MASK_DIR,
    SEGFORMER_MODEL_DIR,
)
from landscape_feature_utils import ensure_dir, iter_landscape_images, read_image_bgr, shannon_entropy


SAVE_MASKS = False


def _find_label_ids(id2label: dict[int, str], keywords: tuple[str, ...]) -> set[int]:
    matched = set()
    for label_id, label_name in id2label.items():
        name = label_name.lower()
        if any(keyword in name for keyword in keywords):
            matched.add(int(label_id))
    return matched


def build_label_groups(id2label: dict[int, str]) -> dict[str, set[int]]:
    groups = {
        "sky": _find_label_ids(id2label, ("sky",)),
        "vegetation": _find_label_ids(id2label, ("tree", "grass", "plant", "palm", "flower", "field")),
        "water": _find_label_ids(id2label, ("water", "river", "sea", "lake", "waterfall")),
        "mountain": _find_label_ids(id2label, ("mountain", "hill", "rock")),
        "building": _find_label_ids(id2label, ("building", "house", "skyscraper", "tower")),
        "road": _find_label_ids(id2label, ("road", "path", "sidewalk", "street", "bridge")),
        "person_vehicle": _find_label_ids(id2label, ("person", "car", "bus", "truck", "van", "boat", "ship", "airplane", "bicycle")),
        "ground": _find_label_ids(id2label, ("earth", "sand", "dirt", "soil")),
    }
    groups["natural"] = (
        groups["sky"] | groups["vegetation"] | groups["water"] | groups["mountain"] | groups["ground"]
    )
    groups["artificial"] = groups["building"] | groups["road"] | groups["person_vehicle"]
    return groups


def compute_horizon_y(sky_mask: np.ndarray) -> float:
    if not np.any(sky_mask):
        return 0.0
    bottoms = []
    for col in range(sky_mask.shape[1]):
        rows = np.where(sky_mask[:, col])[0]
        if rows.size > 0:
            bottoms.append(rows.max())
    if not bottoms:
        return 0.0
    return float(np.mean(bottoms) / max(1, sky_mask.shape[0] - 1))


def extract_features_from_mask(mask: np.ndarray, label_groups: dict[str, set[int]]) -> dict[str, float]:
    total_pixels = mask.size
    label_ids, counts = np.unique(mask, return_counts=True)
    ratios = counts / total_pixels
    ratio_lookup = {int(label_id): float(ratio) for label_id, ratio in zip(label_ids, ratios)}

    def group_ratio(group_name: str) -> float:
        return float(sum(ratio_lookup.get(label_id, 0.0) for label_id in label_groups[group_name]))

    sky_mask = np.isin(mask, list(label_groups["sky"]))

    features = {
        "sky_ratio": group_ratio("sky"),
        "vegetation_ratio": group_ratio("vegetation"),
        "water_ratio": group_ratio("water"),
        "mountain_ratio": group_ratio("mountain"),
        "ground_ratio": group_ratio("ground"),
        "building_ratio": group_ratio("building"),
        "road_ratio": group_ratio("road"),
        "person_vehicle_ratio": group_ratio("person_vehicle"),
        "natural_ratio": group_ratio("natural"),
        "artificial_ratio": group_ratio("artificial"),
        "semantic_diversity": shannon_entropy(ratios),
        "largest_region_ratio": float(ratios.max()),
        "sky_centroid_y": float(np.where(sky_mask)[0].mean() / max(1, mask.shape[0] - 1)) if np.any(sky_mask) else 0.0,
        "horizon_y_norm": compute_horizon_y(sky_mask),
    }
    return features


def main() -> None:
    if not SEGFORMER_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"SegFormer model directory not found: {SEGFORMER_MODEL_DIR}"
        )

    ensure_dir(OUTPUT_DIR)
    if SAVE_MASKS:
        ensure_dir(SEGFORMER_MASK_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(str(SEGFORMER_MODEL_DIR), local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(SEGFORMER_MODEL_DIR), local_files_only=True
    ).to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_groups = build_label_groups(id2label)

    rows = []
    image_paths = iter_landscape_images()
    for image_path in tqdm(image_paths, desc="SegFormer features"):
        image_rgb = Image.fromarray(cv2.cvtColor(read_image_bgr(image_path), cv2.COLOR_BGR2RGB))
        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        segmentation = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_rgb.size[::-1]]
        )[0].cpu().numpy()

        feature_row = {"image_name": image_path.name}
        feature_row.update(extract_features_from_mask(segmentation, label_groups))
        rows.append(feature_row)

        if SAVE_MASKS:
            mask_path = SEGFORMER_MASK_DIR / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), segmentation.astype(np.uint8))

    df = pd.DataFrame(rows)
    df.to_csv(SEGFORMER_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved SegFormer features to: {SEGFORMER_FEATURES_CSV}")


if __name__ == "__main__":
    main()

