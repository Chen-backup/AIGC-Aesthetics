from __future__ import annotations

from pathlib import Path

import pandas as pd

import draw_fig3_textless_panels as panels


ROOT_DIR = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT_DIR / "ratings_for_bayesian_model.xlsx"
MAPPING_PATH = ROOT_DIR / "renumber&gender.xlsx"
FEATURES_PATH = ROOT_DIR / "interpretable_face_features.csv"
SCATTER_DATA_OUTPUT = ROOT_DIR / "Picture_fig3" / "GAMM_NonLinear_Image_Mean_Scatter_Data.csv"


def load_image_points() -> pd.DataFrame:
    ratings = pd.read_excel(RATINGS_PATH)
    mapping = pd.read_excel(MAPPING_PATH)
    features = pd.read_csv(FEATURES_PATH)

    rating_means = ratings.groupby("image", as_index=False)["rating"].mean().rename(columns={"rating": "rating_mean"})
    points = mapping[["face_id", "Number"]].merge(
        features,
        left_on="face_id",
        right_on="image_name",
        how="inner",
    )
    return points.merge(rating_means, left_on="Number", right_on="image", how="inner")


def main() -> None:
    panels.set_style()
    panels.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    curve_df = pd.read_csv(panels.GAMM_DATA)
    image_points = load_image_points()
    image_points.to_csv(SCATTER_DATA_OUTPUT, index=False, encoding="utf-8-sig")

    curve_features = (
        curve_df.loc[curve_df["curve_index"] >= 0, ["feature_order", "feature_name"]]
        .drop_duplicates()
        .sort_values("feature_order")
        .reset_index(drop=True)
    )
    for idx, row in curve_features.iterrows():
        feature = str(row["feature_name"])
        output = panels.OUTPUT_DIR / f"Fig3_{idx + 2:02d}_{feature}_textless_scatter.png"
        panels.draw_gamm_curve(curve_df, feature, output, image_points=image_points)
        print(f"Saved: {output}")
    print(f"Saved data: {SCATTER_DATA_OUTPUT}")


if __name__ == "__main__":
    main()
