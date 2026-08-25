import pandas as pd

from landscape_feature_config import (
    DEPTH_FEATURES_CSV,
    HANDCRAFTED_FEATURES_CSV,
    MERGED_FEATURES_CSV,
    SEGFORMER_FEATURES_CSV,
)


def main() -> None:
    df_seg = pd.read_csv(SEGFORMER_FEATURES_CSV)
    df_depth = pd.read_csv(DEPTH_FEATURES_CSV)
    df_hand = pd.read_csv(HANDCRAFTED_FEATURES_CSV)

    df = df_seg.merge(df_depth, on="image_name", how="inner")
    df = df.merge(df_hand, on="image_name", how="inner")
    df.to_csv(MERGED_FEATURES_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved merged features to: {MERGED_FEATURES_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()

