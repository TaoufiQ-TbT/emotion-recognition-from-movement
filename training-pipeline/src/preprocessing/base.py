"""Helpers shared by the preprocessing pipelines."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from config import PoseEstimationConfig


def empty_report(height: int, width: int) -> Dict[str, object]:
    """Return the report of a frame that was not preprocessed."""
    return {
        "preprocess_applied": False,
        "mask_coverage": 0.0,
        "mask_confidence": 0.0,
        "crop_applied": False,
        "crop_area_ratio": 1.0,
        "background_blur_applied": False,
        "fallback_reason": "segmentation_disabled",
        "crop_box": (0, 0, width, height),
        "original_size": (height, width),
    }


def apply_background_blur(
    image_bgr: np.ndarray, binary_mask: np.ndarray, settings: PoseEstimationConfig
) -> Tuple[np.ndarray, bool]:
    """Apply Gaussian blur to the background while keeping the subject sharp."""

    kernel = max(3, settings.background_blur_kernel)
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(image_bgr, (kernel, kernel), 0)
    mask_bool = binary_mask.astype(bool)
    mask_3c = np.repeat(mask_bool[:, :, None], 3, axis=2)
    processed = np.where(mask_3c, image_bgr, blurred)
    return processed, True


def padding_offsets(width: int, height: int, padding: float) -> Tuple[int, int]:
    """Convert the relative crop padding into pixels."""
    pad_x = int(padding * width)
    pad_y = int(padding * height)
    return pad_x, pad_y