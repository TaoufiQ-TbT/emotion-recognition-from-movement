"""Crop the frame around the person, using a segmentation mask.

If no crop is made, the background can be blurred instead. The denoise and illumination pipelines
apply their own filter first, then hand over to this one.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from config import PoseEstimationConfig

from .base import apply_background_blur, empty_report, padding_offsets
from . import register_pipeline, PreprocessPipeline


def _run_segmentation(
    image_bgr: np.ndarray,
    segmenter: "object | None",
    settings: PoseEstimationConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the frame to analyze and a report. The input frame is the fallback."""
    height, width = image_bgr.shape[:2]
    report = empty_report(height, width)
    if segmenter is None:
        return image_bgr, report

    report["preprocess_applied"] = True

    rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb_frame)
    segmentation_mask = getattr(result, "segmentation_mask", None)
    if segmentation_mask is None:
        report["fallback_reason"] = "no_mask"
        return image_bgr, report

    mask = np.array(segmentation_mask, dtype=np.float32)
    report["mask_confidence"] = float(mask.mean())
    binary_mask = (mask > settings.segmentation_confidence).astype(np.uint8)
    coverage = float(binary_mask.mean())
    report["mask_coverage"] = coverage

    if coverage <= 0.0:
        report["fallback_reason"] = "empty_mask"
        return image_bgr, report

    mask_for_ops = (binary_mask * 255).astype(np.uint8)
    crop_applied = False
    crop_box = (0, 0, width, height)
    fallback_reason = ""

    if settings.enable_person_crop:
        contours, _ = cv2.findContours(
            mask_for_ops, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Only trust the largest region, and only if it is big enough to be a person.
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest) / float(width * height)
            if area_ratio >= settings.min_crop_area:
                x, y, w, h = cv2.boundingRect(largest)
                pad_x, pad_y = padding_offsets(width, height, settings.crop_padding)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(width, x + w + pad_x)
                y2 = min(height, y + h + pad_y)
                crop_box = (x1, y1, x2, y2)
                crop_applied = True
            else:
                fallback_reason = "crop_area_below_min"
        else:
            fallback_reason = "no_contours"
    else:
        fallback_reason = "crop_disabled"

    background_blur_applied = False
    processed = image_bgr

    if crop_applied:
        processed = image_bgr[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    elif settings.background_blur:
        processed, background_blur_applied = apply_background_blur(
            image_bgr, binary_mask, settings
        )

    crop_area_ratio = (
        float((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
        / float(width * height)
        if crop_applied
        else 1.0
    )

    report.update(
        {
            "crop_applied": crop_applied,
            "crop_area_ratio": crop_area_ratio,
            "background_blur_applied": background_blur_applied,
            "fallback_reason": fallback_reason,
            "crop_box": crop_box,
            "original_size": (height, width),
        }
    )

    return processed, report


def process(
    image_bgr: np.ndarray,
    segmenter: "object | None",
    settings: PoseEstimationConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Entry point of the pipeline, also called by the denoise and illumination pipelines."""
    return _run_segmentation(image_bgr, segmenter, settings)


register_pipeline(
    PreprocessPipeline(
        name="segmentation",
        processor=process,
        requires_segmentation=True,
    )
)