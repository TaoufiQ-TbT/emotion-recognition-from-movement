"""Command-line entry point of stage 2: see processing/quality.py for the logic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.quality import (
    QualityControlConfig,
    apply_quality_control,
)
from utils.io import load_pose_run, serialise_pose_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter pose landmarks based on visibility thresholds."
    )
    parser.add_argument(
        "pose_json",
        type=Path,
        help="Path to the pose JSON file produced by pose_estimation.py.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Destination for the quality-controlled pose JSON. "
        "Defaults to adding '_qc' before the extension.",
    )
    parser.add_argument(
        "--min-visibility-world",
        type=float,
        default=0.5,
        help="Minimum visibility threshold for 3D landmarks (default: 0.5).",
    )
    parser.add_argument(
        "--min-visibility-image",
        type=float,
        default=0.5,
        help="Minimum visibility threshold for 2D landmarks (default: 0.5).",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.6,
        help="Drop frames whose surviving landmarks fall below this ratio (default: 0.6).",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=0.4,
        help="Maximum allowed joint displacement between frames in metres (default: 0.4).",
    )
    parser.add_argument(
        "--gap-fill-frames",
        type=int,
        default=2,
        help="Number of consecutive frames to propagate missing landmarks (default: 2).",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    """Add "_qc" before the file extension."""
    return input_path.with_name(f"{input_path.stem}_qc{input_path.suffix}")


def main() -> None:
    args = parse_args()

    pose_json = args.pose_json
    if not pose_json.exists():
        raise SystemExit(f"Pose JSON not found: {pose_json}")

    metadata, frames, _diagnostics = load_pose_run(pose_json)

    config = QualityControlConfig(
        min_visibility_world=args.min_visibility_world,
        min_visibility_image=args.min_visibility_image,
        min_valid_landmarks_ratio=args.min_valid_ratio,
        max_joint_speed=args.max_joint_speed,
        gap_fill_frames=args.gap_fill_frames,
    )

    processed_frames, report = apply_quality_control(frames, config)

    payload = serialise_pose_run(
        video_path=Path(metadata.get("video_path", pose_json)),
        stride=int(metadata.get("frame_stride", 1)),
        fps=float(metadata.get("fps", 30.0)),
        frames=processed_frames,
        visibility_threshold=float(
            metadata.get("visibility_threshold", args.min_visibility_world)
        ),
    )

    output_path = args.output_json or default_output_path(pose_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        "Quality control complete: "
        f"{report.kept_frames}/{report.total_frames} frames retained "
        f"({report.dropped_frames} dropped)."
    )
    if report.mean_valid_landmarks_world is not None:
        print(
            f"  Mean valid 3D landmarks/frame: {report.mean_valid_landmarks_world:.1f}"
        )
    if report.mean_valid_landmarks_image is not None:
        print(
            f"  Mean valid 2D landmarks/frame: {report.mean_valid_landmarks_image:.1f}"
        )
    print(
        "  Corrections: "
        f"velocity_clamps={report.velocity_clamps}, "
        f"gap_fills={report.gap_fill_count}"
    )
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()