"""Data structures shared by the pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - only used for static typing.
    import numpy as np


@dataclass
class FrameSample:
    """Container for a sampled frame pulled from a video stream."""

    frame_index: int
    timestamp_s: float
    image: "np.ndarray"


@dataclass
class Landmark:
    """One body point, with MediaPipe's confidence that it is visible (0 to 1).

    World landmarks are in metres around the hips; image landmarks are relative to the frame (0 to 1).
    """
    index: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FramePose:
    """The landmarks of one frame: the 3D pose (world) and the 2D pose (image)."""
    frame_index: int
    timestamp_s: float
    image_path: str
    landmarks_world: Optional[List[Landmark]] = None
    landmarks_image: Optional[List[Landmark]] = None


@dataclass
class FrameDiagnostics:
    """Per-frame analytics emitted alongside pose estimation results."""

    frame_index: int
    timestamp_s: float
    preprocess_applied: bool
    mask_coverage: float
    mask_confidence: float
    crop_applied: bool
    crop_area_ratio: float
    background_blur_applied: bool
    fallback_reason: str
    world_landmarks_total: int
    world_landmarks_visible: int
    world_visibility_mean: float
    image_landmarks_total: int
    image_landmarks_visible: int
    image_visibility_mean: float