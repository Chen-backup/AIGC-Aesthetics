from __future__ import annotations

from pathlib import Path

import pandas as pd

import draw_landscape_gamm_textless_panels as panels


BASE_DIR = Path(__file__).resolve().parent
LANDSCAPE_DIR = BASE_DIR.parent
RATINGS_PATH = LANDSCAPE_DIR / "ratings_for_bayesian_model.csv"
MAPPING_PATH = LANDSCAPE_DIR / "landscape_number_mapping.csv"
SCATTER_DATA_OUTPUT = panels.RESULT_DIR / "GAMM_NonLinear_Image_Mean_Scatter_Data.csv"


def load_image_points(raw_features: pd.DataFrame) -> pd.DataFrame:
    ratings = pd.read_csv(RATINGS_PATH)
    mapping = pd.read_csv(MAPPING_PATH)
    rating_means = ratings.groupby("image", as_index=False)["rating"].mean().rename(columns={"rating": "rating_mean"})
    return raw_features.merge(mapping, on="image_name", how="inner").merge(rating_means, on="image", how="inner")


def main() -> None:
    panels.set_style()
    panels.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    curve_df = pd.read_csv(panels.CURVE_DATA)
    raw_features = pd.read_csv(panels.RAW_FEATURE_DATA)
    image_points = load_image_points(raw_features)
    image_points.to_csv(SCATTER_DATA_OUTPUT, index=False, encoding="utf-8-sig")

    for index, feature in enumerate(panels.FEATURES, start=2):
        output = panels.RESULT_DIR / f"Fig3_{index:02d}_{feature}_textless_scatter.png"
        panels.draw_curve_panel(curve_df, raw_features, feature, output, image_points=image_points)
        print(f"Saved: {output}")
    print(f"Saved data: {SCATTER_DATA_OUTPUT}")


if __name__ == "__main__":
    main()
