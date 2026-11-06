"""Registry of the frame preprocessing pipelines.

A pipeline takes a BGR frame, the segmentation model and the settings. It returns the frame to
give to MediaPipe and a report of what it did. Modules register themselves when imported, so a
new pipeline only needs a new module, imported at the end of this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from config import PoseEstimationConfig


PipelineFn = Callable[
    [np.ndarray, object | None, PoseEstimationConfig],
    Tuple[np.ndarray, Dict[str, object]],
]


@dataclass(frozen=True)
class PreprocessPipeline:
    """A named pipeline. requires_segmentation asks the caller to load the segmentation model."""
    name: str
    processor: PipelineFn
    requires_segmentation: bool = False


_PIPELINES: Dict[str, PreprocessPipeline] = {}


def register_pipeline(pipeline: PreprocessPipeline) -> None:
    """Register a pipeline so it can be resolved by name at runtime."""

    if pipeline.name in _PIPELINES:
        raise ValueError(f"Pipeline '{pipeline.name}' is already registered.")
    _PIPELINES[pipeline.name] = pipeline


def get_pipeline(name: str) -> PreprocessPipeline:
    """Return the pipeline with this name, or raise KeyError listing the valid names."""
    try:
        return _PIPELINES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PIPELINES))
        raise KeyError(f"Unknown preprocessing pipeline '{name}'. Options: {available}") from exc


def list_pipelines() -> tuple[str, ...]:
    """Return the registered pipeline names."""

    return tuple(sorted(_PIPELINES))


# Import pipeline modules to trigger registration.
from . import segmentation_pipeline  # noqa: E402,F401
from . import illumination_pipeline  # noqa: E402,F401
from . import denoise_pipeline  # noqa: E402,F401