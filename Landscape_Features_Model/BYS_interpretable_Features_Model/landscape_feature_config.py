from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parents[1]
DATASET_DIR = ROOT_DIR.parents[2] / "Dataset" / "scape_dataset"

MODELS_DIR = ROOT_DIR / "models"
SEGFORMER_MODEL_DIR = MODELS_DIR / "segformer_ade20k"
DEPTH_ANYTHING_MODEL_DIR = MODELS_DIR / "depth_anything_v2_small_hf"

OUTPUT_DIR = ROOT_DIR / "feature_outputs"
SEGFORMER_MASK_DIR = OUTPUT_DIR / "segformer_masks"
DEPTH_MAP_DIR = OUTPUT_DIR / "depth_maps"

SEGFORMER_FEATURES_CSV = ROOT_DIR / "landscape_segformer_features.csv"
DEPTH_FEATURES_CSV = ROOT_DIR / "landscape_depth_features.csv"
HANDCRAFTED_FEATURES_CSV = ROOT_DIR / "landscape_handcrafted_features.csv"
MERGED_FEATURES_CSV = ROOT_DIR / "landscape_interpretable_features.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
