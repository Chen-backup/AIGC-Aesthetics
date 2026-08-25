from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def extract_landscape_stylegan_features() -> None:
    print("================ 1. Initialize local StyleGAN/e4e model ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parents[1]
    e4e_root = project_dir / "encoder4editing-main"
    model_path = e4e_root / "e4e_ffhq_encode.pt"
    image_dir = project_dir.parent / "Dataset" / "scape_dataset"
    mapping_path = base_dir.parent / "landscape_number_mapping.csv"
    output_csv = base_dir / "landscape_stylegan_w_features.csv"

    if not e4e_root.exists():
        raise FileNotFoundError(f"encoder4editing-main not found: {e4e_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"e4e checkpoint not found: {model_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Landscape image directory not found: {image_dir}")

    sys.path.insert(0, str(e4e_root))
    from models.psp import pSp  # pylint: disable=import-error,import-outside-toplevel

    print(f"Loading checkpoint from: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = str(model_path)
    opts["device"] = device.type
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.to(device)
    print("Local StyleGAN/e4e model loaded successfully.")

    print("\n================ 2. Read landscape image list ================")
    if mapping_path.exists():
        df_mapping = pd.read_csv(mapping_path)
        image_names = df_mapping["image_name"].astype(str).tolist()
        print(f"Loaded {len(image_names)} image names from mapping table.")
    else:
        image_names = sorted(path.name for path in image_dir.iterdir() if path.is_file())
        print(f"Mapping table not found. Falling back to directory scan: {len(image_names)} images.")

    print("\n================ 3. Extract 9216D W+ black-box features ================")
    img_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    rows: list[list[float | str]] = []
    missing_images: list[str] = []
    failed_images: list[str] = []

    for image_name in tqdm(image_names, desc="StyleGAN W+ extraction"):
        image_path = image_dir / image_name
        if not image_path.exists():
            missing_images.append(image_name)
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
            input_tensor = img_transforms(raw_image).unsqueeze(0).to(device)

            with torch.no_grad():
                _, latents = net(input_tensor, randomize_noise=False, return_latents=True)

            latent_flat = latents.cpu().numpy().reshape(-1)
            rows.append([image_name] + latent_flat.tolist())
        except Exception as exc:  # pragma: no cover - runtime path
            print(f"\nFailed on image {image_name}: {exc}")
            failed_images.append(image_name)

    if missing_images:
        print(f"\nWarning: {len(missing_images)} images were missing from the dataset folder.")
    if failed_images:
        print(f"Warning: {len(failed_images)} images failed during extraction.")

    if not rows:
        raise RuntimeError("No StyleGAN W+ features were extracted.")

    print("\n================ 4. Save feature bank ================")
    col_names = ["image_name"] + [f"w_{i}" for i in range(9216)]
    df_stylegan = pd.DataFrame(rows, columns=col_names)
    df_stylegan.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df_stylegan)} image embeddings to: {output_csv}")


if __name__ == "__main__":
    extract_landscape_stylegan_features()
