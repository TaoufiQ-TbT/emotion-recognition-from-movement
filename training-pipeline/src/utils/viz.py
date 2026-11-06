"""Helpers shared by the skeleton visualization scripts."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import FramePose
from .skeleton import NUM_LANDMARKS


def has_landmarks(frame: FramePose, space: str) -> bool:
    """Tell whether a frame has landmarks in this space ("world" or "image")."""
    if space == "world":
        return bool(frame.landmarks_world)
    return bool(frame.landmarks_image)


def choose_space(frames: Sequence[FramePose], requested: str) -> str:
    """Pick the landmark space based on availability and user preference."""

    if requested == "auto":
        if any(has_landmarks(frame, "world") for frame in frames):
            return "world"
        if any(has_landmarks(frame, "image") for frame in frames):
            return "image"
    else:
        if any(has_landmarks(frame, requested) for frame in frames):
            return requested

    raise SystemExit(
        "No frames available in the requested coordinate space. "
        "Re-run pose extraction with the desired landmarks enabled."
    )


def orient_landmarks(points: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe coordinates into a viewer-friendly orientation.

    - X axis: left/right (unchanged).
    - Y axis: forward/backward (MediaPipe Z).
    - Z axis: vertical, positive upwards (negated MediaPipe Y).
    """

    x = points[:, 0]
    y = points[:, 2]
    z = -points[:, 1]
    oriented = np.stack([x, y, z], axis=1)

    hip_indices = [23, 24]
    if oriented.shape[0] > max(hip_indices):
        hips = oriented[hip_indices]
        valid = ~np.isnan(hips).any(axis=1)
        if valid.any():
            root = np.nanmean(hips[valid], axis=0)
            oriented = oriented - root
    return oriented


def frames_to_tensor(
    frames: Sequence[FramePose], space: str
) -> Tuple[np.ndarray, List[FramePose]]:
    """Convert landmarks into a tensor [T, N, 3] and return paired frames."""

    coords = []
    retained_frames: List[FramePose] = []
    for frame in frames:
        landmarks = (
            frame.landmarks_world if space == "world" else frame.landmarks_image
        )
        if not landmarks:
            continue
        pts = np.full((NUM_LANDMARKS, 3), np.nan, dtype=float)
        for lm in landmarks:
            if 0 <= lm.index < NUM_LANDMARKS:
                pts[lm.index] = (lm.x, lm.y, lm.z)
        oriented = orient_landmarks(pts)
        coords.append(oriented)
        retained_frames.append(frame)

    if not coords:
        raise SystemExit("No landmarks found in the pose JSON for the selected space.")

    return np.stack(coords, axis=0), retained_frames


def compute_axes_limits(points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Return axis bounds that keep the skeleton centered."""
    valid = ~np.isnan(points)
    mins = np.where(valid, points, np.inf).min(axis=(0, 1))
    maxs = np.where(valid, points, -np.inf).max(axis=(0, 1))

    # Expand bounds slightly for padding.
    ranges = maxs - mins
    padding = np.maximum(ranges * 0.1, 0.05)
    mins -= padding
    maxs += padding
    return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]


def resolve_image_path(image_path: str, pose_json_path: Path) -> Path:
    """Resolve frame image paths relative to the pose JSON location."""
    path = Path(image_path)
    if not path.is_absolute():
        path = pose_json_path.parent / path
    return path


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Read an image from disk and convert to RGB for Matplotlib display."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read frame image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def extract_image_landmarks(
    frame: FramePose, image: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (coords, mask) for image-space landmarks in pixel units."""
    if not frame.landmarks_image:
        return None
    height, width = image.shape[:2]
    coords = np.full((NUM_LANDMARKS, 2), np.nan, dtype=float)
    mask = np.zeros(NUM_LANDMARKS, dtype=bool)
    for lm in frame.landmarks_image:
        if 0 <= lm.index < NUM_LANDMARKS:
            coords[lm.index] = (lm.x * width, lm.y * height)
            mask[lm.index] = True
    if not mask.any():
        return None
    return coords, mask