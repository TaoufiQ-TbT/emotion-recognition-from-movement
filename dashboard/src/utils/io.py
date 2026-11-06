"""Read and write the pose JSON format used between the pipeline stages."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .types import FrameDiagnostics, FramePose, Landmark


def load_pose_run(json_path: Path) -> Tuple[
    Dict[str, object], List[FramePose], List[FrameDiagnostics]
]:
    """
    Load a pose estimation JSON file produced by ``serialise_pose_run``.

    Returns a tuple of metadata (video path, fps, stride, num frames),
    the list of FramePose entries, and optional diagnostics captured during
    pose estimation.
    """

    with Path(json_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames: List[FramePose] = []
    for frame in payload.get("frames", []):
        def _decode_landmarks(entries):
            if not entries:
                return None
            return [
                Landmark(
                    index=int(lm["index"]),
                    name=str(lm["name"]),
                    x=float(lm["x"]),
                    y=float(lm["y"]),
                    z=float(lm["z"]),
                    visibility=float(lm.get("visibility", 0.0)),
                )
                for lm in entries
            ]

        landmarks_world = _decode_landmarks(frame.get("landmarks_world"))
        landmarks_image = _decode_landmarks(frame.get("landmarks_image"))

        # Backwards compatibility with the initial schema.
        if not landmarks_world and not landmarks_image and "landmarks" in frame:
            space = str(frame.get("landmark_space", "world"))
            decoded = _decode_landmarks(frame.get("landmarks"))
            if space == "world":
                landmarks_world = decoded
            else:
                landmarks_image = decoded

        frames.append(
            FramePose(
                frame_index=int(frame["frame_index"]),
                timestamp_s=float(frame["timestamp_s"]),
                image_path=str(frame.get("image_path", "")),
                landmarks_world=landmarks_world,
                landmarks_image=landmarks_image,
            )
        )

    diagnostics_payload = payload.get("frame_metrics", [])
    frame_diagnostics: List[FrameDiagnostics] = []
    for entry in diagnostics_payload:
        frame_diagnostics.append(
            FrameDiagnostics(
                frame_index=int(entry.get("frame_index", 0)),
                timestamp_s=float(entry.get("timestamp_s", 0.0)),
                preprocess_applied=bool(entry.get("preprocess_applied", False)),
                mask_coverage=float(entry.get("mask_coverage", 0.0)),
                mask_confidence=float(entry.get("mask_confidence", 0.0)),
                crop_applied=bool(entry.get("crop_applied", False)),
                crop_area_ratio=float(entry.get("crop_area_ratio", 1.0)),
                background_blur_applied=bool(
                    entry.get("background_blur_applied", False)
                ),
                fallback_reason=str(entry.get("fallback_reason", "")),
                world_landmarks_total=int(entry.get("world_landmarks_total", 0)),
                world_landmarks_visible=int(entry.get("world_landmarks_visible", 0)),
                world_visibility_mean=float(entry.get("world_visibility_mean", 0.0)),
                image_landmarks_total=int(entry.get("image_landmarks_total", 0)),
                image_landmarks_visible=int(entry.get("image_landmarks_visible", 0)),
                image_visibility_mean=float(entry.get("image_visibility_mean", 0.0)),
            )
        )

    metadata = {
        "video_path": payload.get("video_path", ""),
        "fps": float(payload.get("fps", 0.0)),
        "frame_stride": int(payload.get("frame_stride", 1)),
        "num_frames": int(payload.get("num_frames", len(frames))),
        "visibility_threshold": float(payload.get("visibility_threshold", 0.0)),
    }

    return metadata, frames, frame_diagnostics


def serialise_pose_run(
    *,
    video_path: Path,
    stride: int,
    fps: float,
    frames: List[FramePose],
    frame_metrics: Optional[List[FrameDiagnostics]] = None,
    visibility_threshold: float = 0.0,
) -> Dict[str, object]:
    """Convert pose results into a JSON-friendly structure."""

    def _encode_landmarks(landmarks: List[Landmark]) -> List[Dict[str, object]]:
        return [asdict(landmark) for landmark in landmarks]

    payload = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_stride": stride,
        "num_frames": len(frames),
        "visibility_threshold": visibility_threshold,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp_s": frame.timestamp_s,
                "image_path": frame.image_path,
                **(
                    {"landmarks_world": _encode_landmarks(frame.landmarks_world)}
                    if frame.landmarks_world
                    else {}
                ),
                **(
                    {"landmarks_image": _encode_landmarks(frame.landmarks_image)}
                    if frame.landmarks_image
                    else {}
                ),
            }
            for frame in frames
        ],
    }

    if frame_metrics:
        payload["frame_metrics"] = [asdict(metric) for metric in frame_metrics]

    return payload