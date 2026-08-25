from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def extract_landscape_dinov2_features() -> None:
    print("================ 1. Initialize local DINOv2 model ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "dinov2-base-local"
    image_dir = base_dir.parents[2] / "Dataset" / "scape_dataset"
    output_csv = base_dir / "landscape_dinov2_features.csv"

    print(f"Loading model from local directory: {model_dir}")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    model.eval()
    print("Local DINOv2 model loaded successfully.")

    print("\n================ 2. Read landscape image list ================")
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    print(f"Found {len(image_paths)} landscape images.")

    print("\n================ 3. Extract 768D black-box features ================")
    rows = []
    for image_path in tqdm(image_paths, desc="DINOv2 feature extraction"):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        row = [image_path.name] + cls_embedding.tolist()
        rows.append(row)

    col_names = ["image_name"] + [f"emb_{i}" for i in range(768)]
    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n================ 4. Save feature bank ================")
    print(f"Saved {len(df)} image embeddings to: {output_csv}")


if __name__ == "__main__":
    extract_landscape_dinov2_features()
