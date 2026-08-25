import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "score_data"
SCAPE_DIR = ROOT.parent / "Dataset" / "scape_dataset"
RATER_MAP_PATH = ROOT / "rater_number_mapping.xlsx"
OUTPUT_PATH = Path(__file__).resolve().parent / "ratings_for_bayesian_model.csv"
OUTPUT_MAP_PATH = Path(__file__).resolve().parent / "landscape_number_mapping.csv"


def build_landscape_mapping() -> pd.DataFrame:
    image_names = sorted(
        path.name for path in SCAPE_DIR.iterdir() if path.is_file()
    )
    return pd.DataFrame(
        {
            "image_name": image_names,
            "image": range(1, len(image_names) + 1),
        }
    )


def build_rater_mapping() -> dict[str, int]:
    df = pd.read_excel(RATER_MAP_PATH)
    return {
        str(row["Original_ID"]): int(row["Rater_Number"])
        for _, row in df.iterrows()
    }


def extract_landscape_ratings(
    landscape_map: pd.DataFrame, rater_map: dict[str, int]
) -> pd.DataFrame:
    image_lookup = dict(
        zip(landscape_map["image_name"].astype(str), landscape_map["image"].astype(int))
    )
    rows: list[dict[str, int | float]] = []

    for csv_path in sorted(SCORE_DIR.glob("*.csv")):
        rater_key = csv_path.stem.replace("Aesthetic_", "", 1)
        if rater_key not in rater_map:
            raise KeyError(f"Rater {rater_key} not found in {RATER_MAP_PATH.name}")

        df = pd.read_csv(csv_path, usecols=["block", "image_url", "score"])
        df["block"] = df["block"].fillna("").astype(str)
        df["image_url"] = df["image_url"].fillna("").astype(str)
        df["image_name"] = df["image_url"].str.split("/").str[-1].astype(str)

        mask = df["block"].eq("landscapes") | df["image_url"].str.contains(
            "/images/landscapes/", regex=False
        )
        df = df.loc[mask].copy()
        df = df[df["image_name"].ne("") & df["image_name"].ne("nan")].copy()
        df["image"] = df["image_name"].map(image_lookup)

        missing_images = sorted(
            str(name) for name in df.loc[df["image"].isna(), "image_name"].unique()
        )
        if missing_images:
            raise ValueError(
                "Found landscape images missing from scape_dataset: "
                + ", ".join(missing_images[:10])
            )

        df = df.dropna(subset=["score"]).copy()
        df["rater"] = rater_map[rater_key]
        df["image"] = df["image"].astype(int)
        df["rating"] = pd.to_numeric(df["score"], errors="raise").astype(int)

        rows.extend(df[["rater", "image", "rating"]].to_dict("records"))

    return pd.DataFrame(rows).sort_values(["rater", "image"], kind="stable").reset_index(
        drop=True
    )


def main() -> None:
    landscape_map = build_landscape_mapping()
    rater_map = build_rater_mapping()
    ratings_df = extract_landscape_ratings(landscape_map, rater_map)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    landscape_map.to_csv(OUTPUT_MAP_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved ratings: {OUTPUT_PATH}")
    print(f"Saved mapping: {OUTPUT_MAP_PATH}")
    print(f"Rows: {len(ratings_df)}")
    print(f"Unique raters: {ratings_df['rater'].nunique()}")
    print(f"Unique images: {ratings_df['image'].nunique()}")


if __name__ == "__main__":
    main()
