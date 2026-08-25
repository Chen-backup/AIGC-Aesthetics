from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class Places365FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


def _load_places365_model(weights_path: Path, device: torch.device) -> tuple[nn.Module, Places365FeatureExtractor]:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 365)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval().to(device)

    extractor = Places365FeatureExtractor(model).eval().to(device)
    return model, extractor


def extract_landscape_places365_features() -> None:
    print("================ 1. Initialize local Places365 model ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "places365-resnet18-local"
    weights_path = model_dir / "resnet18_places365.pth.tar"
    image_dir = base_dir.parents[2] / "Dataset" / "scape_dataset"
    output_features = base_dir / "landscape_places365_features.csv"
    output_logits = base_dir / "landscape_places365_logits.csv"
    output_top5 = base_dir / "landscape_places365_top5_predictions.csv"

    if not weights_path.exists():
        raise FileNotFoundError(f"Places365 weights not found: {weights_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Landscape image directory not found: {image_dir}")

    category_path = model_dir / "categories_places365.txt"
    categories = []
    has_categories = category_path.exists()
    if has_categories:
        with open(category_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_name = parts[0][3:] if parts[0].startswith("/a/") or parts[0].startswith("/b/") else parts[0].lstrip("/")
                categories.append(cls_name)
    else:
        print(f"Category file not found, top-5 scene labels will be skipped: {category_path}")

    model, extractor = _load_places365_model(weights_path, device)
    print("Local Places365 model loaded successfully.")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("\n================ 2. Read landscape image list ================")
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    print(f"Found {len(image_paths)} landscape images.")

    print("\n================ 3. Extract 512D deep features and scene logits ================")
    feature_rows: list[list[float | str]] = []
    logit_rows: list[list[float | str]] = []
    top5_rows: list[dict[str, str | float]] = []

    for image_path in tqdm(image_paths, desc="Places365 feature extraction"):
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = extractor(input_tensor)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        feature_rows.append([image_path.name] + features.cpu().numpy().reshape(-1).tolist())
        logit_rows.append([image_path.name] + logits.cpu().numpy().reshape(-1).tolist())

        if has_categories:
            top_probs, top_indices = torch.topk(probs, k=5, dim=1)
            top_probs = top_probs[0].cpu().tolist()
            top_indices = top_indices[0].cpu().tolist()

            row: dict[str, str | float] = {"image_name": image_path.name}
            for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
                row[f"top{rank}_class"] = categories[idx] if idx < len(categories) else str(idx)
                row[f"top{rank}_prob"] = prob
            top5_rows.append(row)

    print("\n================ 4. Save outputs ================")
    feature_columns = ["image_name"] + [f"feat_{i}" for i in range(512)]
    pd.DataFrame(feature_rows, columns=feature_columns).to_csv(output_features, index=False, encoding="utf-8-sig")

    logit_columns = ["image_name"] + [f"logit_{i}" for i in range(365)]
    pd.DataFrame(logit_rows, columns=logit_columns).to_csv(output_logits, index=False, encoding="utf-8-sig")

    print(f"Saved 512D features to: {output_features}")
    print(f"Saved 365D logits to: {output_logits}")
    if has_categories:
        pd.DataFrame(top5_rows).to_csv(output_top5, index=False, encoding="utf-8-sig")
        print(f"Saved top-5 scene predictions to: {output_top5}")
    else:
        print("Skipped top-5 scene prediction export because categories_places365.txt was not found.")


if __name__ == "__main__":
    extract_landscape_places365_features()
