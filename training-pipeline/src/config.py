"""Project paths and the settings of the pose estimation stage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Resolve directories relative to the project root.
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = BASE_DIR / "assets"
OUTPUT_DIR: Path = BASE_DIR / "outputs"

# Convenience handles for the sample run shipped with the repository.
DEFAULT_VIDEO: Path = ASSETS_DIR / "improvisation_05_happines.avi"
DEFAULT_OUTPUT_DIR: Path = OUTPUT_DIR / "sample_pose"


@dataclass
class PoseEstimationConfig:
    """Settings that control the sparse 3D pose extraction pass."""

    stride: int = 4  # Process every Nth video frame; the models of this project were trained on 4.
    max_frames: Optional[int] = 25  # Set to None for unlimited.
    save_frames: bool = True
    # MediaPipe Pose: complexity 2 is the slowest and most accurate model.
    model_complexity: int = 2
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    capture_world: bool = True
    capture_image: bool = False
    # Preprocessing: crop around the person, or blur the background when there is no crop.
    enable_person_crop: bool = True
    crop_padding: float = 0.15
    min_crop_area: float = 0.05
    segmentation_confidence: float = 0.1
    background_blur: bool = True
    background_blur_kernel: int = 21
    # Landmarks at or above this score count as visible in the metrics that drive blending.
    visibility_threshold: float = 0.5
    preprocess_mode: str = "segmentation"

    def resolved_max_frames(self) -> Optional[int]:
        """Treat values <= 0 as unlimited for friendlier CLI defaults."""
        if self.max_frames is None or self.max_frames <= 0:
            return None
        return self.max_frames

    def validate(self) -> None:
        """Ensure at least one landmark space is requested."""
        if not self.capture_world and not self.capture_image:
            raise ValueError(
                "PoseEstimationConfig must capture world and/or image landmarks."
            )
        if self.background_blur and self.background_blur_kernel < 3:
            raise ValueError("background_blur_kernel must be >= 3 when background blur is enabled.")
        if self.background_blur_kernel % 2 == 0:
            raise ValueError("background_blur_kernel must be an odd number.")
        if not 0.0 <= self.crop_padding <= 0.5:
            raise ValueError("crop_padding should be between 0.0 and 0.5.")
        if not 0.0 <= self.min_crop_area <= 1.0:
            raise ValueError("min_crop_area should be between 0.0 and 1.0.")
        if not 0.0 < self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be between 0 and 1.")
        if not self.preprocess_mode:
            raise ValueError("preprocess_mode must be a non-empty string.")


def ensure_output_layout(
    base_output: Path, *, create_frames_dir: bool = True
) -> tuple[Path, Path]:
    """
    Prepare output folders and return (frames_dir, poses_dir).

    This helper keeps directory creation logic in one place so both scripts and
    notebooks produce a consistent layout.
    """

    base_output.mkdir(parents=True, exist_ok=True)
    frames_dir = base_output / "frames"
    poses_dir = base_output / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    if create_frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir, poses_dir