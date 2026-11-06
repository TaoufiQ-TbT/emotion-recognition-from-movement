"""Run MediaPipe Pose on sampled frames."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - defer to caller with guidance.
    raise SystemExit(
        "mediapipe is required. Install dependencies with "
        "`python3 -m pip install mediapipe opencv-python`."
    ) from exc

from config import PoseEstimationConfig
from preprocessing import get_pipeline
from preprocessing.base import empty_report
from .types import FrameDiagnostics, FramePose, FrameSample, Landmark


def _iter_landmarks(
    pose_result: "mp.framework.formats.landmark_pb2.LandmarkList",
    landmark_names: List[str],
) -> Iterable[Landmark]:
    """Convert a MediaPipe landmark list into Landmark objects."""
    for idx, landmark in enumerate(pose_result.landmark):
        yield Landmark(
            index=idx,
            name=landmark_names[idx],
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
            visibility=float(landmark.visibility),
        )


def _reproject_image_landmarks(
    landmarks: Optional[List[Landmark]],
    crop_box: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
) -> None:
    """Move 2D landmarks from the cropped frame back to the full frame, in place."""
    if not landmarks:
        return

    height, width = original_size
    x1, y1, x2, y2 = crop_box
    crop_width = max(1, x2 - x1)
    crop_height = max(1, y2 - y1)

    for landmark in landmarks:
        pixel_x = landmark.x * crop_width + x1
        pixel_y = landmark.y * crop_height + y1
        landmark.x = float(pixel_x) / float(width)
        landmark.y = float(pixel_y) / float(height)


def _compute_visibility_stats(
    landmarks: Optional[List[Landmark]], threshold: float
) -> Tuple[int, int, float]:
    """Return the landmark count, how many are visible, and their mean visibility."""
    if not landmarks:
        return 0, 0, 0.0

    total = len(landmarks)
    visible = sum(1 for lm in landmarks if lm.visibility >= threshold)
    mean_visibility = sum(lm.visibility for lm in landmarks) / float(total)
    return total, visible, mean_visibility


def estimate_pose_landmarks(
    samples: Iterable[FrameSample],
    settings: PoseEstimationConfig,
    *,
    frame_paths: Optional[Dict[int, Path]] = None,
) -> Tuple[List[FramePose], List[FrameDiagnostics]]:
    """
    Run MediaPipe Pose on the sampled frames and return FramePose entries.
    """

    landmark_names = [lm.name for lm in mp.solutions.pose.PoseLandmark]
    recorded_frames: List[FramePose] = []
    diagnostics: List[FrameDiagnostics] = []

    pipeline = get_pipeline(settings.preprocess_mode)
    segmentation_enabled = pipeline.requires_segmentation
    segmentation_context = (
        mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        if segmentation_enabled
        else nullcontext()
    )

    # static_image_mode=False tracks the pose from one frame to the next, so frames must stay in order.
    with segmentation_context as segmenter, mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=settings.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=settings.detection_confidence,
        min_tracking_confidence=settings.tracking_confidence,
    ) as pose_detector:
        for sample in samples:
            default_report = empty_report(
                sample.image.shape[0], sample.image.shape[1]
            )

            frame_for_pose, preprocess_report = pipeline.processor(
                sample.image, segmenter, settings
            )
            if preprocess_report is None:
                preprocess_report = default_report
            else:
                merged = default_report
                merged.update(preprocess_report)
                preprocess_report = merged

            rgb_frame = cv2.cvtColor(frame_for_pose, cv2.COLOR_BGR2RGB)
            result = pose_detector.process(rgb_frame)

            world_landmarks = None
            image_landmarks = None

            if result.pose_world_landmarks and result.pose_world_landmarks.landmark:
                world_landmarks = list(
                    _iter_landmarks(result.pose_world_landmarks, landmark_names)
                )

            if result.pose_landmarks and result.pose_landmarks.landmark:
                image_landmarks = list(
                    _iter_landmarks(result.pose_landmarks, landmark_names)
                )

            if settings.capture_world is False:
                world_landmarks = None

            # Keep 2D landmarks either if explicitly requested or if 3D is missing.
            keep_image = settings.capture_image or not world_landmarks
            if not keep_image:
                image_landmarks = None

            # World landmarks are relative to the hips, so cropping only affects the 2D ones.
            if (
                preprocess_report["crop_applied"]
                and image_landmarks
                and preprocess_report["crop_box"]
            ):
                _reproject_image_landmarks(
                    image_landmarks,
                    preprocess_report["crop_box"],
                    preprocess_report["original_size"],
                )

            total_world, visible_world, mean_world = _compute_visibility_stats(
                world_landmarks, settings.visibility_threshold
            )
            total_image, visible_image, mean_image = _compute_visibility_stats(
                image_landmarks, settings.visibility_threshold
            )

            diagnostics.append(
                FrameDiagnostics(
                    frame_index=sample.frame_index,
                    timestamp_s=sample.timestamp_s,
                    preprocess_applied=bool(preprocess_report["preprocess_applied"]),
                    mask_coverage=float(preprocess_report["mask_coverage"]),
                    mask_confidence=float(preprocess_report["mask_confidence"]),
                    crop_applied=bool(preprocess_report["crop_applied"]),
                    crop_area_ratio=float(preprocess_report["crop_area_ratio"]),
                    background_blur_applied=bool(
                        preprocess_report["background_blur_applied"]
                    ),
                    fallback_reason=str(preprocess_report["fallback_reason"]),
                    world_landmarks_total=total_world,
                    world_landmarks_visible=visible_world,
                    world_visibility_mean=mean_world,
                    image_landmarks_total=total_image,
                    image_landmarks_visible=visible_image,
                    image_visibility_mean=mean_image,
                )
            )

            # Metrics cover every sampled frame, but a frame without a pose stops here.
            if not world_landmarks and not image_landmarks:
                continue

            image_path = ""
            if frame_paths:
                image_path = str(frame_paths.get(sample.frame_index, ""))

            recorded_frames.append(
                FramePose(
                    frame_index=sample.frame_index,
                    timestamp_s=sample.timestamp_s,
                    image_path=image_path,
                    landmarks_world=world_landmarks,
                    landmarks_image=image_landmarks,
                )
            )

    return recorded_frames, diagnostics