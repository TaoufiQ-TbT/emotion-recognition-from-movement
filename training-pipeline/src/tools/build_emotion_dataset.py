"""List the extracted clips in a CSV and split them into train, validation and test.

The emotion label is the first word of each file name. The split is stratified by emotion and
made per clip, so two views of one performance can land in different splits (see "How to read
these numbers" in the main README).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QC_DIR = PROJECT_ROOT / "outputs" / "run_blended_qc"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "emotion_dataset.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "emotion_dataset_train.csv"
DEFAULT_VAL_PATH = PROJECT_ROOT / "data" / "emotion_dataset_val.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "emotion_dataset_test.csv"
# Labels the model is trained on by default (see --target-emotions in train_gesture.py).
EMOTIONS = ("angry", "disgusted", "fear", "happiness", "sadness", "surprised")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan *_poses.json and *_features.json pairs in the QC folder and emit "
            "a CSV referencing each clip plus its inferred emotion label."
        )
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=DEFAULT_QC_DIR,
        help="Folder containing final pose + feature JSON files (default: outputs/run_blended_qc).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Destination CSV path (default: data/emotion_dataset.csv).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Train CSV output path (default: data/emotion_dataset_train.csv).",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help="Validation CSV output path (default: data/emotion_dataset_val.csv).",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Test CSV output path (default: data/emotion_dataset_test.csv).",
    )
    return parser.parse_args()


def infer_clip_name(pose_file: Path) -> str:
    """Remove the "_poses" suffix from a pose file name."""
    stem = pose_file.stem
    return stem[: -len("_poses")] if stem.endswith("_poses") else stem


def infer_emotion(clip_name: str) -> str:
    """Return the first word of the clip name, which is its emotion label."""
    return clip_name.split("_", 1)[0]


def normalise_video_path(raw: str) -> str:
    """Return a portable path: relative to the project, starting at assets/ if possible."""
    if not raw:
        return ""

    candidate = raw
    try:
        candidate_path = Path(raw).expanduser().resolve()
        if candidate_path.is_absolute():
            try:
                candidate = str(candidate_path.relative_to(PROJECT_ROOT))
            except ValueError:
                candidate = str(candidate_path)
    except OSError:
        candidate = raw

    normalized = candidate.replace("\\", "/")
    marker = "assets/"
    if marker in normalized:
        idx = normalized.index(marker)
        normalized = normalized[idx:]
    return normalized


def project_relative(path: Path) -> str:
    """Return the path relative to the project, or as it is when it lies outside."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_entries(qc_dir: Path) -> List[dict]:
    """Return one row per clip that has both a pose file and a feature file."""
    entries: List[dict] = []
    for pose_file in sorted(qc_dir.glob("*_poses.json")):
        clip_name = infer_clip_name(pose_file)
        features_file = qc_dir / f"{clip_name}_features.json"
        if not features_file.exists():
            print(f"Skipping {clip_name}: features file missing ({features_file}).")
            continue

        with pose_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_video_path = payload.get("video_path", "")
        video_path = normalise_video_path(raw_video_path)
        if not raw_video_path:
            print(f"Warning: {pose_file} missing video_path metadata; storing empty string.")

        entries.append(
            {
                "video_path": video_path,
                "pose_json": project_relative(pose_file),
                "features_json": project_relative(features_file),
                "emotion": infer_emotion(clip_name),
            }
        )

    if not entries:
        raise SystemExit(f"No pose/feature pairs found in {qc_dir}.")
    return entries


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    qc_dir = args.qc_dir.expanduser().resolve()
    if not qc_dir.is_dir():
        raise SystemExit(f"QC directory not found: {qc_dir}")

    entries = build_entries(qc_dir)
    df_full = pd.DataFrame(entries)

    counts = df_full["emotion"].value_counts()
    unknown = sorted(set(counts.index) - set(EMOTIONS))
    if unknown:
        print(f"Warning: unknown emotion in the file names, ignored by default in training: {', '.join(unknown)}.")

    output_csv = args.output_csv.expanduser().resolve()
    write_df(df_full, output_csv)
    print(f"Wrote full dataset ({len(df_full)} clips) to {output_csv}.")

    # Split train/val/test at the clip level
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise SystemExit("Invalid val/test ratios; ensure they are >=0 and sum to < 1.")

    # scikit-learn splits in two: first set aside val + test, then divide that part.
    remaining = val_ratio + test_ratio
    try:
        train_df, temp_df = train_test_split(
            df_full,
            test_size=remaining,
            stratify=df_full["emotion"],
            random_state=args.seed,
        )
        if remaining > 0:
            # split temp into val/test
            val_size = val_ratio / remaining if remaining > 0 else 0
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size,
                stratify=temp_df["emotion"],
                random_state=args.seed,
            )
        else:
            val_df = pd.DataFrame(columns=df_full.columns)
            test_df = pd.DataFrame(columns=df_full.columns)
    except ValueError as exc:
        found = ", ".join(f"{emotion} {count}" for emotion, count in counts.items())
        raise SystemExit(
            f"Not enough clips to split the dataset ({found}). "
            "With the default ratios, each emotion needs at least 7 clips."
        ) from exc

    write_df(train_df, args.train_out.expanduser().resolve())
    write_df(val_df, args.val_out.expanduser().resolve())
    write_df(test_df, args.test_out.expanduser().resolve())
    print(
        f"Splits -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} "
        f"(ratios ~ {1-remaining:.2f}/{val_ratio:.2f}/{test_ratio:.2f})"
    )


if __name__ == "__main__":
    main()