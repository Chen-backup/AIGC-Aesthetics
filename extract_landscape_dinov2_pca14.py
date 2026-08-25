from pathlib import Path
import warnings

import joblib
import pandas as pd
import torch
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(r"G:\E\CJH-SJTU\课题组\图像美学\Dataset\scape_dataset")
MODEL_DIR = ROOT_DIR / "dinov2-base-local"
PCA_MODEL_PATH = ROOT_DIR / "BYS_Fusion_28D_DINOv2_result" / "dinov2_pca_14d.pkl"
RAW_OUTPUT_PATH = ROOT_DIR / "landscape_dinov2_features.csv"
PCA_OUTPUT_PATH = ROOT_DIR / "landscape_PCA_14_dinov2.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BATCH_SIZE = 8


def collect_image_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_batch_images(batch_paths: list[Path], dataset_dir: Path) -> tuple[list[str], list[Image.Image], list[str]]:
    image_names: list[str] = []
    images: list[Image.Image] = []
    failed: list[str] = []

    for path in batch_paths:
        image_name = path.relative_to(dataset_dir).as_posix()
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
            image_names.append(image_name)
        except Exception as exc:
            failed.append(f"{image_name}: {exc}")

    return image_names, images, failed


def extract_embeddings(
    image_paths: list[Path],
    dataset_dir: Path,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[list[float]] = []
    failed: list[str] = []

    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting DINOv2 features"):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        image_names, images, batch_failed = load_batch_images(batch_paths, dataset_dir)
        failed.extend(batch_failed)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for image_name, embedding in zip(image_names, batch_embeddings, strict=True):
            rows.append([image_name, *embedding.tolist()])

    columns = ["image_name", *[f"emb_{idx}" for idx in range(768)]]
    return pd.DataFrame(rows, columns=columns), failed


def project_with_existing_pca(raw_df: pd.DataFrame, pca_model_path: Path) -> pd.DataFrame:
    emb_cols = [f"emb_{idx}" for idx in range(768)]

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    pipeline = joblib.load(pca_model_path)

    projected = pipeline.transform(raw_df[emb_cols].to_numpy())
    pca_df = pd.DataFrame(projected, columns=[f"PC{idx}" for idx in range(1, 15)])
    pca_df["image_name"] = raw_df["image_name"].reset_index(drop=True)
    return pca_df


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"DINOv2 model directory not found: {MODEL_DIR}")
    if not PCA_MODEL_PATH.exists():
        raise FileNotFoundError(f"PCA model not found: {PCA_MODEL_PATH}")

    image_paths = collect_image_paths(DATASET_DIR)
    if not image_paths:
        raise RuntimeError(f"No images found in: {DATASET_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} landscape images.")

    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    raw_df, failed = extract_embeddings(image_paths, DATASET_DIR, processor, model, device)
    raw_df.to_csv(RAW_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    pca_df = project_with_existing_pca(raw_df, PCA_MODEL_PATH)
    pca_df.to_csv(PCA_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved raw 768D features to: {RAW_OUTPUT_PATH}")
    print(f"Saved 14D PCA features to: {PCA_OUTPUT_PATH}")
    print(f"Processed images: {len(raw_df)}")
    print(f"Failed images: {len(failed)}")

    if failed:
        failed_log_path = ROOT_DIR / "landscape_dinov2_failed.txt"
        failed_log_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"Saved failed image log to: {failed_log_path}")


if __name__ == "__main__":
    main()
