"""Stage 2: clean a pose sequence (quality control and denoising).

The steps run in this order: filter landmarks by visibility, fill short gaps, limit joint speed,
then drop the frames that still have too few landmarks. "Frame" means a sampled frame here, so
gaps and speeds depend on the sampling stride. Distances are in metres.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, List, Optional, Sequence, Tuple

from utils.skeleton import NUM_LANDMARKS
from utils.types import FramePose, Landmark


@dataclass
class QualityControlConfig:
    """Tunable parameters for the quality-control pass."""

    min_visibility_world: float = 0.5
    min_visibility_image: float = 0.5
    min_valid_landmarks_ratio: float = 0.6
    max_joint_speed: float = 0.0  # metres per frame; <=0 disables
    gap_fill_frames: int = 2

    def validate(self) -> None:
        """Raise ValueError if a setting is out of range."""
        if not (0.0 <= self.min_visibility_world <= 1.0):
            raise ValueError("min_visibility_world must be between 0 and 1.")
        if not (0.0 <= self.min_visibility_image <= 1.0):
            raise ValueError("min_visibility_image must be between 0 and 1.")
        if not (0.0 < self.min_valid_landmarks_ratio <= 1.0):
            raise ValueError("min_valid_landmarks_ratio must be in (0, 1].")
        if self.max_joint_speed < 0.0:
            raise ValueError("max_joint_speed must be >= 0.")
        if self.gap_fill_frames < 0:
            raise ValueError("gap_fill_frames must be >= 0.")


@dataclass
class QualityReport:
    """What quality control changed in one sequence."""
    total_frames: int
    kept_frames: int
    dropped_frames: int
    mean_valid_landmarks_world: Optional[float]
    mean_valid_landmarks_image: Optional[float]
    velocity_clamps: int
    gap_fill_count: int


def apply_quality_control(
    frames: Sequence[FramePose], config: QualityControlConfig
) -> Tuple[List[FramePose], QualityReport]:
    """
    Clean per-frame landmarks, enforce biomechanical constraints,
    and discard frames that remain too noisy.
    """

    config.validate()
    if not frames:
        raise ValueError("No frames provided for quality control.")

    processed_frames: List[FramePose] = []
    valid_counts_world: List[int] = []
    valid_counts_image: List[int] = []

    previous_world: Dict[int, Landmark] = {}
    previous_image: Dict[int, Landmark] = {}

    velocity_clamps = 0
    gap_fills_world = 0
    gap_fills_image = 0

    world_series = [
        _prepare_landmarks_dict(frame.landmarks_world, config.min_visibility_world)
        for frame in frames
    ]
    image_series = [
        _prepare_landmarks_dict(frame.landmarks_image, config.min_visibility_image)
        for frame in frames
    ]

    # Gaps are filled first, over the whole sequence, so the checks below see the repaired frames.
    if config.gap_fill_frames > 0:
        gap_fills_world = _interpolate_landmarks(world_series, config.gap_fill_frames)
        gap_fills_image = _interpolate_landmarks(image_series, config.gap_fill_frames)

    for idx, frame in enumerate(frames):
        filled_world = world_series[idx]
        filled_image = image_series[idx]

        # Only the 3D landmarks are speed-limited.
        if config.max_joint_speed > 0 and previous_world:
            velocity_clamps += _limit_joint_velocity(
                filled_world,
                previous_world,
                config.max_joint_speed,
            )

        ratio_world = len(filled_world) / NUM_LANDMARKS if filled_world else 0.0
        ratio_image = len(filled_image) / NUM_LANDMARKS if filled_image else 0.0

        # A frame survives if either its 3D or its 2D landmarks are complete enough.
        keep_frame = False
        if filled_world and ratio_world >= config.min_valid_landmarks_ratio:
            keep_frame = True
        if filled_image and ratio_image >= config.min_valid_landmarks_ratio:
            keep_frame = True

        # After a dropped frame, the next one has no previous position to be limited against.
        if not keep_frame:
            _update_previous_state(previous_world, None)
            _update_previous_state(previous_image, None)
            continue

        sorted_world = (
            [filled_world[idx] for idx in sorted(filled_world)]
            if filled_world
            else None
        )
        sorted_image = (
            [filled_image[idx] for idx in sorted(filled_image)]
            if filled_image
            else None
        )

        valid_counts_world.append(len(sorted_world) if sorted_world else 0)
        valid_counts_image.append(len(sorted_image) if sorted_image else 0)

        processed_frames.append(
            FramePose(
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                image_path=frame.image_path,
                landmarks_world=sorted_world,
                landmarks_image=sorted_image,
            )
        )

        _update_previous_state(previous_world, filled_world)
        _update_previous_state(previous_image, filled_image)

    report = QualityReport(
        total_frames=len(frames),
        kept_frames=len(processed_frames),
        dropped_frames=len(frames) - len(processed_frames),
        mean_valid_landmarks_world=_mean_or_none(valid_counts_world),
        mean_valid_landmarks_image=_mean_or_none(valid_counts_image),
        velocity_clamps=velocity_clamps,
        gap_fill_count=gap_fills_world + gap_fills_image,
    )

    return processed_frames, report


# ---------------------------------------------------------------------------
# Helpers


def _prepare_landmarks_dict(
    landmarks: Optional[Sequence[Landmark]], min_visibility: float
) -> Dict[int, Landmark]:
    """Copy the landmarks above min_visibility into a dict keyed by landmark index."""
    if not landmarks:
        return {}
    result: Dict[int, Landmark] = {}
    for landmark in landmarks:
        if landmark.visibility < min_visibility:
            continue
        result[landmark.index] = replace(landmark)
    return result


def _interpolate_landmarks(
    series: List[Dict[int, Landmark]],
    gap_limit: int,
) -> int:
    """Fill gaps of up to gap_limit frames in place and return the number of landmarks added.

    A gap between two detections is interpolated linearly. A short gap at the end of the sequence
    repeats the last known position, and a gap at the very start stays empty.
    """
    fills = 0
    if gap_limit <= 0:
        return fills

    for landmark_idx in range(NUM_LANDMARKS):
        pending_indices: List[int] = []
        last_landmark: Optional[Landmark] = None

        for frame_idx, frame_landmarks in enumerate(series):
            if landmark_idx in frame_landmarks:
                current_landmark = frame_landmarks[landmark_idx]
                if (
                    pending_indices
                    and last_landmark is not None
                    and len(pending_indices) <= gap_limit
                ):
                    total = len(pending_indices) + 1
                    for gap_pos, missing_idx in enumerate(pending_indices, start=1):
                        alpha = gap_pos / total
                        series[missing_idx][landmark_idx] = _lerp_landmark(
                            last_landmark, current_landmark, alpha
                        )
                        fills += 1
                pending_indices.clear()
                last_landmark = current_landmark
            else:
                pending_indices.append(frame_idx)

        if (
            pending_indices
            and last_landmark is not None
            and len(pending_indices) <= gap_limit
        ):
            for missing_idx in pending_indices:
                series[missing_idx][landmark_idx] = replace(last_landmark)
                fills += 1

    return fills


def _lerp_landmark(
    start: Landmark,
    end: Landmark,
    alpha: float,
) -> Landmark:
    """Blend two landmarks: alpha 0 gives start and alpha 1 gives end."""
    alpha = max(0.0, min(1.0, alpha))
    return Landmark(
        index=start.index,
        name=start.name,
        x=start.x + (end.x - start.x) * alpha,
        y=start.y + (end.y - start.y) * alpha,
        z=start.z + (end.z - start.z) * alpha,
        visibility=min(max(start.visibility, end.visibility), 1.0),
    )


def _limit_joint_velocity(
    current: Dict[int, Landmark],
    previous: Dict[int, Landmark],
    max_distance: float,
) -> int:
    """Pull back in place the joints that moved more than max_distance, and count them.

    A clamped joint keeps its direction of motion and ends up max_distance from its last position.
    """
    adjustments = 0
    for idx, landmark in current.items():
        if idx not in previous:
            continue
        prev = previous[idx]
        dx = landmark.x - prev.x
        dy = landmark.y - prev.y
        dz = landmark.z - prev.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= max_distance or distance == 0:
            continue
        scale = max_distance / distance
        landmark.x = prev.x + dx * scale
        landmark.y = prev.y + dy * scale
        landmark.z = prev.z + dz * scale
        adjustments += 1
    return adjustments


def _update_previous_state(
    previous: Dict[int, Landmark],
    current: Optional[Dict[int, Landmark]],
) -> None:
    """Remember these landmarks for the next frame. None clears the state."""
    previous.clear()
    if not current:
        return
    for idx, landmark in current.items():
        previous[idx] = replace(landmark)


def _mean_or_none(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))