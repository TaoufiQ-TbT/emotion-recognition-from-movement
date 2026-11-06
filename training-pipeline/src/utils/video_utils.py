"""Read frames from a video file."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cv2

from .types import FrameSample


def _open_video(video_path: Path) -> Tuple[cv2.VideoCapture, float]:
    """Open a video and return it with its FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0  # Fallback for files with missing metadata.
    return cap, fps


def read_video_fps(video_path: Path) -> float:
    """Return the FPS of a video."""
    cap, fps = _open_video(video_path)
    cap.release()
    return fps


def iter_video_frames(
    video_path: Path, step: float, max_samples: Optional[int] = None
) -> Iterator[FrameSample]:
    """
    Yield one frame every `step` video frames. `step` can be fractional.

    Frames are decoded one at a time, so memory use doesn't grow with the length of the video.
    """

    cap, fps = _open_video(video_path)
    frame_index = -1
    next_index = 0
    count = 0
    try:
        while max_samples is None or count < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index < next_index:
                continue

            yield FrameSample(frame_index=frame_index, timestamp_s=frame_index / fps, image=frame)
            count += 1
            next_index = round(count * step)
    finally:
        cap.release()


def sample_video_frames(
    video_path: Path, stride: int, max_samples: Optional[int] = None
) -> Tuple[List[FrameSample], float]:
    """
    Sample frames from the video at the given stride.

    Returns a list of FrameSample objects and the source video's FPS. The helper
    keeps decoding responsibilities close to OpenCV so the rest of the pipeline
    can work with plain Python data structures.
    """

    cap, fps = _open_video(video_path)

    stride = max(1, stride)
    max_samples = None if max_samples is not None and max_samples <= 0 else max_samples

    samples: List[FrameSample] = []
    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        if frame_index % stride != 0:
            continue

        timestamp_s = frame_index / fps if fps else 0.0
        samples.append(FrameSample(frame_index=frame_index, timestamp_s=timestamp_s, image=frame))

        if max_samples is not None and len(samples) >= max_samples:
            break

    cap.release()
    return samples, fps


def save_frame_images(
    samples: Iterable[FrameSample], target_dir: Path
) -> Dict[int, Path]:
    """
    Persist sampled frames to disk and return a mapping of frame index to path.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[int, Path] = {}
    for sample in samples:
        output_path = target_dir / f"frame_{sample.frame_index:06d}.jpg"
        cv2.imwrite(str(output_path), sample.image)
        saved_paths[sample.frame_index] = output_path
    return saved_paths