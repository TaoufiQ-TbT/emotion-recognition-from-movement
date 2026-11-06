"""Correct the lighting of the frame, then crop it with the segmentation pipeline."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from config import PoseEstimationConfig

from . import register_pipeline, PreprocessPipeline
from .segmentation_pipeline import process as segmentation_process


def _apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Even out local contrast on the brightness channel only, so colors don't change."""
    y_cr_cb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _apply_gamma(image_bgr: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """Brighten the image slightly (gamma above 1)."""
    inv = 1.0 / max(gamma, 1e-3)
    table = np.array(
        [(i / 255.0) ** inv * 255 for i in np.arange(0, 256)], dtype=np.uint8
    )
    return cv2.LUT(image_bgr, table)


def process(
    image_bgr: np.ndarray,
    segmenter: "object | None",
    settings: PoseEstimationConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Correct the lighting, then crop the frame around the person."""
    enhanced = _apply_clahe(image_bgr)
    enhanced = _apply_gamma(enhanced, gamma=1.1)
    return segmentation_process(enhanced, segmenter, settings)


register_pipeline(
    PreprocessPipeline(
        name="illumination",
        processor=process,
        requires_segmentation=True,
    )
)