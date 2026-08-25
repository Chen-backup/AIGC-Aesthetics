from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0

SCRIPT_DIR = Path(__file__).resolve().parent
LANDSCAPE_ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR

RATINGS_PATH = LANDSCAPE_ROOT / "ratings_for_bayesian_model.csv"
MAPPING_PATH = LANDSCAPE_ROOT / "landscape_number_mapping.csv"

FEATURE_SETS = {
    "DINOv2": LANDSCAPE_ROOT / "BYS_DINOv2_Features_Model" / "PCA_10_dinov2.csv",
    "Places365": LANDSCAPE_ROOT / "BYS_Places365_Features_Model" / "PCA_10_places365.csv",
    "StyleGAN": LANDSCAPE_ROOT / "BYS_StyleGAN_Features_Model" / "PCA_10_stylegan_w.csv",
}

PC_COLS = [f"PC{i}" for i in range(1, 11)]


def infer_subject_id(image_name: str) -> str:
    """Use the stable landscape series prefix as the subject/group ID."""
    stem = Path(str(image_name).strip()).stem
    match = re.match(r"^([A-Za-z]+_\d+)", stem)
    if match:
        return match.group(1)
    return stem


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
        ]
    )


def make_subject_folds(subject_ids: pd.Series, n_splits: int, seed: int):
    unique_subjects = np.array(sorted(subject_ids.astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    subject_blocks = np.array_split(unique_subjects, n_splits)
    for fold_idx, test_subjects in enumerate(subject_blocks, start=1):
        test_mask = subject_ids.astype(str).isin(test_subjects).to_numpy()
        train_mask = ~test_mask
        yield fold_idx, train_mask, test_mask


def load_image_level_dataset(model_name: str, feature_path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(RATINGS_PATH)
    mapping = pd.read_csv(MAPPING_PATH)
    features = pd.read_csv(feature_path)

    missing_pc = [col for col in PC_COLS if col not in features.columns]
    if missing_pc:
        raise ValueError(f"{feature_path.name} is missing PC columns: {missing_pc}")

    features["image_name"] = features["image_name"].astype(str).str.strip()
    mapping["image_name"] = mapping["image_name"].astype(str).str.strip()

    rating_summary = (
        ratings.groupby("image", as_index=False)
        .agg(
            rating_mean=("rating", "mean"),
            rating_sd=("rating", "std"),
            n_ratings=("rating", "size"),
        )
    )

    mapped_features = features.merge(
        mapping[["image_name", "image"]],
        on="image_name",
        how="inner",
        validate="one_to_one",
    )
    df = mapped_features.merge(rating_summary, on="image", how="inner", validate="one_to_one")
    df["subject_id"] = df["image_name"].map(infer_subject_id)
    df["model_name"] = model_name
    df = df.dropna(subset=PC_COLS + ["rating_mean", "subject_id"])

    cols = [
        "model_name",
        "image",
        "image_name",
        "subject_id",
        "rating_mean",
        "rating_sd",
        "n_ratings",
    ] + PC_COLS
    return df[cols].sort_values("image").reset_index(drop=True)


def run_cv(df: pd.DataFrame, model_name: str, cv_type: str):
    x = df[PC_COLS].to_numpy(dtype=float)
    y = df["rating_mean"].to_numpy(dtype=float)

    if cv_type == "image-level":
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = (
            (fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x), start=1)
        )
    elif cv_type == "subject-level":
        folds = []
        for fold_idx, train_mask, test_mask in make_subject_folds(df["subject_id"], N_SPLITS, SEED):
            folds.append((fold_idx, np.where(train_mask)[0], np.where(test_mask)[0]))
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    fold_rows = []
    pred_rows = []

    for fold_idx, train_idx, test_idx in folds:
        model = make_model()
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        fold_r2 = r2_score(y[test_idx], y_pred)

        train_subjects = df.iloc[train_idx]["subject_id"].nunique()
        test_subjects = df.iloc[test_idx]["subject_id"].nunique()

        fold_rows.append(
            {
                "model_name": model_name,
                "cv_type": cv_type,
                "fold": fold_idx,
                "r2": fold_r2,
                "n_train_images": len(train_idx),
                "n_test_images": len(test_idx),
                "n_train_subjects": train_subjects,
                "n_test_subjects": test_subjects,
                "ridge_alpha": RIDGE_ALPHA,
                "random_seed": SEED,
            }
        )

        test_df = df.iloc[test_idx][["image", "image_name", "subject_id", "rating_mean"]].copy()
        test_df["model_name"] = model_name
        test_df["cv_type"] = cv_type
        test_df["fold"] = fold_idx
        test_df["y_true"] = y[test_idx]
        test_df["y_pred"] = y_pred
        test_df["residual"] = test_df["y_true"] - test_df["y_pred"]
        pred_rows.append(test_df)

    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True)


def summarize_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_results.groupby(["model_name", "cv_type"], as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            min_r2=("r2", "min"),
            max_r2=("r2", "max"),
            n_folds=("fold", "count"),
            mean_test_images=("n_test_images", "mean"),
            mean_test_subjects=("n_test_subjects", "mean"),
        )
        .sort_values(["model_name", "cv_type"])
        .reset_index(drop=True)
    )
    summary["mean_r2_pm_sd"] = summary.apply(
        lambda row: f"{row['mean_r2']:.4f} +/- {row['sd_r2']:.4f}", axis=1
    )
    summary["ridge_alpha"] = RIDGE_ALPHA
    summary["random_seed"] = SEED
    return summary


def write_report(summary: pd.DataFrame, fold_results: pd.DataFrame, dataset_info: pd.DataFrame):
    lines = [
        "Landscape black-box feature Ridge 5-fold CV",
        "=" * 52,
        "",
        "Task:",
        "  Predict image-level mean aesthetic rating from 10 PCA features.",
        "",
        "Model:",
        f"  Pipeline: StandardScaler + Ridge(alpha={RIDGE_ALPHA})",
        f"  Random seed: {SEED}",
        "",
        "CV paradigms:",
        "  image-level: shuffled KFold over individual landscape images.",
        "  subject-level: held-out image-series groups inferred from filename prefixes, e.g. average_001.",
        "",
        "Dataset summary:",
    ]

    for _, row in dataset_info.iterrows():
        lines.append(
            f"  {row['model_name']}: {int(row['n_images'])} images, "
            f"{int(row['n_subjects'])} subject/groups, "
            f"mean ratings per image = {row['mean_n_ratings_per_image']:.2f}"
        )

    lines.extend(["", "R2 summary:"])
    for _, row in summary.iterrows():
        lines.append(
            f"  {row['model_name']} | {row['cv_type']}: "
            f"R2 = {row['mean_r2']:.4f} +/- {row['sd_r2']:.4f} "
            f"(fold range {row['min_r2']:.4f} to {row['max_r2']:.4f})"
        )

    lines.extend(["", "Fold-level R2:"])
    for _, row in fold_results.sort_values(["model_name", "cv_type", "fold"]).iterrows():
        lines.append(
            f"  {row['model_name']} | {row['cv_type']} | fold {int(row['fold'])}: "
            f"R2={row['r2']:.4f}, test images={int(row['n_test_images'])}, "
            f"test subject/groups={int(row['n_test_subjects'])}"
        )

    (OUT_DIR / "Landscape_BlackBox_Ridge_CV_Report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    all_predictions = []
    all_image_datasets = []
    dataset_info = []

    for model_name, feature_path in FEATURE_SETS.items():
        print(f"\n=== Loading {model_name}: {feature_path.name} ===")
        df = load_image_level_dataset(model_name, feature_path)
        all_image_datasets.append(df)
        dataset_info.append(
            {
                "model_name": model_name,
                "feature_file": feature_path.name,
                "n_images": len(df),
                "n_subjects": df["subject_id"].nunique(),
                "mean_n_ratings_per_image": df["n_ratings"].mean(),
                "min_n_ratings_per_image": df["n_ratings"].min(),
                "max_n_ratings_per_image": df["n_ratings"].max(),
            }
        )
        print(
            f"Data: {len(df)} images, {df['subject_id'].nunique()} subject/groups, "
            f"rating mean range {df['rating_mean'].min():.3f}-{df['rating_mean'].max():.3f}"
        )

        for cv_type in ["image-level", "subject-level"]:
            fold_df, pred_df = run_cv(df, model_name, cv_type)
            print(
                f"{model_name} {cv_type}: "
                f"R2 = {fold_df['r2'].mean():.4f} +/- {fold_df['r2'].std():.4f}"
            )
            all_fold_results.append(fold_df)
            all_predictions.append(pred_df)

    fold_results = pd.concat(all_fold_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    image_dataset = pd.concat(all_image_datasets, ignore_index=True)
    dataset_info_df = pd.DataFrame(dataset_info)
    summary = summarize_folds(fold_results)

    fold_results.to_csv(OUT_DIR / "Landscape_BlackBox_Ridge_CV_Fold_Results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "Landscape_BlackBox_Ridge_CV_Summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "Landscape_BlackBox_Ridge_CV_Predictions.csv", index=False, encoding="utf-8-sig")
    image_dataset.to_csv(OUT_DIR / "Landscape_BlackBox_Ridge_CV_Image_Dataset.csv", index=False, encoding="utf-8-sig")
    dataset_info_df.to_csv(OUT_DIR / "Landscape_BlackBox_Ridge_CV_Dataset_Info.csv", index=False, encoding="utf-8-sig")
    write_report(summary, fold_results, dataset_info_df)

    print("\n=== Summary ===")
    print(summary[["model_name", "cv_type", "mean_r2", "sd_r2", "mean_r2_pm_sd"]].to_string(index=False))
    print(f"\nSaved all outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
