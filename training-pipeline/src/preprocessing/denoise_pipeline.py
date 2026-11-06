"""Denoise and sharpen the frame, then crop it with the segmentation pipeline."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from config import PoseEstimationConfig

from . import register_pipeline, PreprocessPipeline
from .segmentation_pipeline import process as segmentation_process


def _bilateral_filter(image_bgr: np.ndarray) -> np.ndarray:
    """Reduce noise while keeping edges sharp."""
    return cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)


def _unsharp_mask(image_bgr: np.ndarray, amount: float = 0.4) -> np.ndarray:
    """Sharpen the image by subtracting a blurred copy."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image_bgr, 1 + amount, blurred, -amount, 0)
    return sharpened


def process(
    image_bgr: np.ndarray,
    segmenter: "object | None",
    settings: PoseEstimationConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Clean the frame, then crop it around the person."""
    denoised = _bilateral_filter(image_bgr)
    sharpened = _unsharp_mask(denoised)
    return segmentation_process(sharpened, segmenter, settings)


register_pipeline(
    PreprocessPipeline(
        name="denoise_sharpen",
        processor=process,
        requires_segmentation=True,
    )
)