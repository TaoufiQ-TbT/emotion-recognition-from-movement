"""Extract the 3D pose of one video with MediaPipe Pose.

Samples the frames, applies one preprocessing pipeline (--preprocess-mode), runs MediaPipe, and
writes poses/poses.json with a CSV of per-frame visibility metrics. run_full_pipeline.py calls
this script once per preprocessing variant.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    PoseEstimationConfig,
    ensure_output_layout,
)
from preprocessing import get_pipeline, list_pipelines
from utils.io import serialise_pose_run
from utils.pose_utils import estimate_pose_landmarks
from utils.types import FrameDiagnostics
from utils.video_utils import sample_video_frames, save_frame_images

AVAILABLE_PIPELINES = list_pipelines()


def export_metrics_csv(
    diagnostics: list[FrameDiagnostics], target_path: Path
) -> None:
    """Write one row of visibility metrics per sampled frame."""
    if not diagnostics:
        return

    fieldnames = [
        "frame_index",
        "timestamp_s",
        "preprocess_applied",
        "mask_coverage",
        "mask_confidence",
        "crop_applied",
        "crop_area_ratio",
        "background_blur_applied",
        "fallback_reason",
        "world_landmarks_total",
        "world_landmarks_visible",
        "world_visible_ratio",
        "world_visibility_mean",
        "image_landmarks_total",
        "image_landmarks_visible",
        "image_visible_ratio",
        "image_visibility_mean",
    ]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in diagnostics:
            row = asdict(entry)
            world_total = entry.world_landmarks_total or 0
            image_total = entry.image_landmarks_total or 0
            row["world_visible_ratio"] = (
                entry.world_landmarks_visible / world_total if world_total else 0.0
            )
            row["image_visible_ratio"] = (
                entry.image_landmarks_visible / image_total if image_total else 0.0
            )
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarise_diagnostics(
    diagnostics: list[FrameDiagnostics],
) -> tuple[float, float, float]:
    """Return the mean mask coverage and the mean share of visible 3D and 2D landmarks."""
    if not diagnostics:
        return 0.0, 0.0, 0.0

    mean_mask = sum(d.mask_coverage for d in diagnostics) / len(diagnostics)
    world_ratios = [
        d.world_landmarks_visible / d.world_landmarks_total
        for d in diagnostics
        if d.world_landmarks_total
    ]
    image_ratios = [
        d.image_landmarks_visible / d.image_landmarks_total
        for d in diagnostics
        if d.image_landmarks_total
    ]
    mean_world = sum(world_ratios) / len(world_ratios) if world_ratios else 0.0
    mean_image = sum(image_ratios) / len(image_ratios) if image_ratios else 0.0
    return mean_mask, mean_world, mean_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sparse 3D pose landmarks from a dance video."
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=DEFAULT_VIDEO,
        help="Path to the input video file (default: bundled sample clip).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extracted frames and pose JSON.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Process every Nth frame (default: 4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=25,
        help="Stop after this many sampled frames (default: 25). Use 0 for unlimited.",
    )
    parser.add_argument(
        "--skip-frame-export",
        action="store_true",
        help="Do not save the sampled RGB frames alongside the pose data.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="MediaPipe Pose model complexity (higher = more accurate, slower).",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.6,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.6,
        help="Minimum tracking confidence threshold.",
    )
    parser.add_argument(
        "--include-2d",
        "--2d",
        action="store_true",
        help="Store image-space (2D) landmarks in addition to 3D world coordinates.",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Disable storage of 3D world landmarks (2D landmarks will be kept if available).",
    )
    parser.add_argument(
        "--no-person-crop",
        action="store_true",
        help="Disable segmentation-guided person cropping.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.15,
        help="Padding (0-0.5) added around the detected person crop (default: 0.15).",
    )
    parser.add_argument(
        "--min-crop-area",
        type=float,
        default=0.05,
        help="Minimum area ratio required to keep the segmentation crop (default: 0.05).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.1,
        help="Segmentation mask threshold applied before cropping/background blur (default: 0.1).",
    )
    parser.add_argument(
        "--no-background-blur",
        action="store_true",
        help="Skip blurring the background when segmentation succeeds and cropping is disabled.",
    )
    parser.add_argument(
        "--background-blur-kernel",
        type=int,
        default=21,
        help="Odd kernel size used for Gaussian background blur (default: 21).",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Visibility threshold used when tallying landmark coverage (default: 0.5).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Optional path to write per-frame visibility diagnostics (CSV).",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=AVAILABLE_PIPELINES,
        default="segmentation",
        help="Preprocessing pipeline to run before MediaPipe Pose (default: segmentation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Without 3D landmarks, the 2D ones are the only output left.
    include_2d = args.include_2d or args.no_3d

    settings = PoseEstimationConfig(
        stride=args.stride,
        max_frames=args.max_frames,
        save_frames=not args.skip_frame_export,
        model_complexity=args.model_complexity,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        capture_world=not args.no_3d,
        capture_image=include_2d,
        enable_person_crop=not args.no_person_crop,
        crop_padding=args.crop_padding,
        min_crop_area=args.min_crop_area,
        segmentation_confidence=args.mask_threshold,
        background_blur=not args.no_background_blur,
        background_blur_kernel=args.background_blur_kernel,
        visibility_threshold=args.visibility_threshold,
        preprocess_mode=args.preprocess_mode,
    )
    try:
        settings.validate()
    except ValueError as exc:
        raise SystemExit(str(exc))

    try:
        get_pipeline(settings.preprocess_mode)
    except KeyError as exc:
        raise SystemExit(str(exc))

    try:
        samples, fps = sample_video_frames(
            args.video, settings.stride, settings.resolved_max_frames()
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    if not samples:
        raise SystemExit(
            "No frames were sampled. Try lowering the stride or verifying the video file."
        )

    limit = settings.resolved_max_frames()
    if limit and len(samples) >= limit:
        print(f"Note: stopped at {limit} sampled frames, so the rest of the video is ignored. Raise --max-frames to keep it.")

    frames_dir, poses_dir = ensure_output_layout(
        args.output_dir, create_frames_dir=settings.save_frames
    )

    frame_paths = None
    if settings.save_frames:
        frame_paths = save_frame_images(samples, frames_dir)

    frame_poses, diagnostics = estimate_pose_landmarks(
        samples, settings, frame_paths=frame_paths
    )

    payload = serialise_pose_run(
        video_path=args.video,
        stride=settings.stride,
        fps=fps,
        frames=frame_poses,
        frame_metrics=diagnostics,
        visibility_threshold=settings.visibility_threshold,
    )

    output_path = poses_dir / "poses.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metrics_path = args.metrics_csv or (poses_dir / "visibility_metrics.csv")
    if diagnostics:
        export_metrics_csv(diagnostics, metrics_path)
        mean_mask, mean_world, mean_image = summarise_diagnostics(diagnostics)
        print(
            "Diagnostics summary – "
            f"mask coverage: {mean_mask:.1%}, "
            f"visible 3D landmarks: {mean_world:.1%}, "
            f"visible 2D landmarks: {mean_image:.1%}."
        )
        print(f"Per-frame visibility metrics saved to {metrics_path}.")
    else:
        print("No pose frames retained; skipping metrics export.")

    print(
        f"Processed {payload['num_frames']} frames. Pose data saved to {output_path}."
    )


if __name__ == "__main__":
    main()