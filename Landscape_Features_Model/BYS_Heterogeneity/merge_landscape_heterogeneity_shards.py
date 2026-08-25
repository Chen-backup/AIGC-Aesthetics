from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


EXPECTED_FEATURES = [
    "warm_cool_balance",
    "horizon_y_norm",
    "depth_gradient_mean",
    "artificial_ratio",
    "saturation_mean",
    "thirds_brightness_mean",
    "line_strength",
    "depth_std",
    "semantic_diversity",
    "left_right_balance",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge distributed landscape heterogeneity outputs.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Shard result directories, relative to this script or absolute paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="Result",
        help="Merged output directory relative to this script, or an absolute path. Default: Result.",
    )
    return parser.parse_args()


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def merge_shards(input_dirs: list[str], output_dir: str) -> None:
    base_dir = Path(__file__).resolve().parent
    shard_dirs = [_resolve(base_dir, raw_path) for raw_path in input_dirs]
    merged_dir = _resolve(base_dir, output_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_frames = []
    preference_frames = []
    trace_files: dict[str, Path] = {}
    for shard_dir in shard_dirs:
        leaderboard_path = shard_dir / "Ultimate_Heterogeneity_Leaderboard.csv"
        preferences_path = shard_dir / "Rater_10D_Preferences.csv"
        if not leaderboard_path.exists() or not preferences_path.exists():
            raise FileNotFoundError(f"Missing shard CSV outputs in: {shard_dir}")

        leaderboard_frames.append(pd.read_csv(leaderboard_path))
        preference_frames.append(pd.read_csv(preferences_path))
        for trace_path in shard_dir.glob("Ultimate_Heterogeneity_*"):
            if trace_path.suffix not in {".nc", ".pkl"}:
                continue
            feature = trace_path.stem.replace("Ultimate_Heterogeneity_", "")
            trace_files[feature] = trace_path

    leaderboard = pd.concat(leaderboard_frames, ignore_index=True)
    leaderboard = leaderboard.drop_duplicates(subset=["feature"], keep="last")
    leaderboard = leaderboard.sort_values("heterogeneity_mean", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(merged_dir / "Ultimate_Heterogeneity_Leaderboard.csv", index=False)

    preferences = preference_frames[0]
    for frame in preference_frames[1:]:
        overlapping_features = [col for col in frame.columns if col != "rater" and col in preferences.columns]
        frame = frame.drop(columns=overlapping_features)
        preferences = pd.merge(preferences, frame, on="rater", how="outer")
    preferences.to_csv(merged_dir / "Rater_10D_Preferences.csv", index=False)

    leaderboard_features = set(leaderboard["feature"])
    preference_features = set(preferences.columns) - {"rater"}
    trace_features = set(trace_files)
    expected_features = set(EXPECTED_FEATURES)
    missing_leaderboard = expected_features - leaderboard_features
    missing_preferences = expected_features - preference_features
    missing_traces = expected_features - trace_features
    if missing_leaderboard or missing_preferences or missing_traces:
        raise ValueError(
            "Incomplete shard outputs. "
            f"Missing leaderboard features: {sorted(missing_leaderboard)}; "
            f"missing preference columns: {sorted(missing_preferences)}; "
            f"missing traces: {sorted(missing_traces)}"
        )

    for feature in EXPECTED_FEATURES:
        trace_path = trace_files[feature]
        shutil.copy2(trace_path, merged_dir / trace_path.name)

    print(f"Merged {len(shard_dirs)} shards.")
    print(f"Features in leaderboard: {len(leaderboard)}")
    print(f"Raters in preference matrix: {len(preferences)}")
    print(f"Copied trace files: {len(EXPECTED_FEATURES)}")
    print(f"Merged outputs saved to: {merged_dir}")


if __name__ == "__main__":
    args = _parse_args()
    merge_shards(args.input_dirs, args.output_dir)
