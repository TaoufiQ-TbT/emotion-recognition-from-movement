"""Command-line entry point of stage 3: see features/lma.py for the logic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from features import compute_lma_features
from utils.io import load_pose_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LMA-inspired features from pose JSON output."
    )
    parser.add_argument(
        "pose_json",
        type=Path,
        help="Pose JSON file (e.g., outputs/run_blended_qc/poses.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Destination for the feature JSON (default: features_<name>.json).",
    )
    parser.add_argument(
        "--min-visibility-world",
        type=float,
        default=0.0,
        help="Minimum visibility required when consuming 3D landmarks (default: 0.0).",
    )
    return parser.parse_args()


def default_output_path(pose_path: Path) -> Path:
    """Return <pose file name>_features.json, next to the pose file."""
    return pose_path.with_name(f"{pose_path.stem}_features.json")


def main() -> None:
    args = parse_args()

    if not args.pose_json.exists():
        raise SystemExit(f"Pose JSON not found: {args.pose_json}")

    metadata, frames, _diagnostics = load_pose_run(args.pose_json)

    # Apply visibility threshold before feature extraction
    filtered_frames = []
    for frame in frames:
        if frame.landmarks_world:
            frame.landmarks_world = [
                lm
                for lm in frame.landmarks_world
                if lm.visibility >= args.min_visibility_world
            ]
        filtered_frames.append(frame)

    features = compute_lma_features(filtered_frames)
    output_path = args.output_json or default_output_path(args.pose_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_path": metadata.get("video_path", ""),
        "fps": metadata.get("fps", 0.0),
        "frame_stride": metadata.get("frame_stride", 1),
        "num_frames": len(features),
        "features": [feat.as_dict() for feat in features],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Computed {len(features)} feature frames. Output saved to {output_path}."
    )


if __name__ == "__main__":
    main()