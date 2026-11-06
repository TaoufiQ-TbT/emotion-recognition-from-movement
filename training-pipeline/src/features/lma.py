"""Stage 3: movement features inspired by Laban Movement Analysis (LMA).

Six values are computed per frame. Each one comes directly from the pose landmarks, never from
another feature; the two energy features also read the previous frame's pose.

Coordinates are MediaPipe world landmarks: metres, origin midway between the hips, y pointing down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence

from utils.types import FramePose, Landmark

HAND_LANDMARKS = ("LEFT_WRIST", "RIGHT_WRIST")
FOOT_LANDMARKS = ("LEFT_ANKLE", "RIGHT_ANKLE")
HIP_LANDMARKS = ("LEFT_HIP", "RIGHT_HIP")
SHOULDER_LANDMARKS = ("LEFT_SHOULDER", "RIGHT_SHOULDER")


@dataclass
class FrameFeatures:
    """The six LMA features of one frame.

    active_ratio: share of the 33 landmarks present in this frame.
    body_expansion: volume of the box around the body, in cubic metres.
    vertical_level: mean y of the hips. It stays close to 0, because the world origin is at the hips.
    hand_energy, foot_energy: mean speed of the wrists or the ankles, in metres per second.
    torso_inclination: angle between the torso and the +y axis, in degrees. An upright torso is
        close to 180, because y points down.
    """
    frame_index: int
    timestamp_s: float
    active_ratio: float
    body_expansion: float
    vertical_level: float
    hand_energy: float
    foot_energy: float
    torso_inclination: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_lma_features(frames: Sequence[FramePose]) -> List[FrameFeatures]:
    """Return LMA-inspired features for each frame (expects world landmarks)."""

    if not frames:
        return []

    feature_list: List[FrameFeatures] = []
    prev_world: Dict[str, Landmark] | None = None
    prev_timestamp: float | None = None

    for frame in frames:
        world_landmarks = _landmarks_by_name(frame.landmarks_world or [])

        active_ratio = (
            len(world_landmarks) / 33.0 if world_landmarks else 0.0
        )
        body_expansion = _body_expansion(world_landmarks)
        vertical_level = _root_vertical_level(world_landmarks)

        # The first frame has no previous pose, so its energies are 0.
        dt = (
            frame.timestamp_s - prev_timestamp
            if prev_timestamp is not None
            else 0.0
        )
        hand_energy = _aggregate_velocity(
            world_landmarks,
            prev_world,
            HAND_LANDMARKS,
            dt,
        )
        foot_energy = _aggregate_velocity(
            world_landmarks,
            prev_world,
            FOOT_LANDMARKS,
            dt,
        )
        torso_inclination = _torso_inclination(world_landmarks)

        feature_list.append(
            FrameFeatures(
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                active_ratio=active_ratio,
                body_expansion=body_expansion,
                vertical_level=vertical_level,
                hand_energy=hand_energy,
                foot_energy=foot_energy,
                torso_inclination=torso_inclination,
            )
        )

        prev_world = world_landmarks
        prev_timestamp = frame.timestamp_s

    return feature_list


def _landmarks_by_name(landmarks: Iterable[Landmark]) -> Dict[str, Landmark]:
    return {landmark.name: landmark for landmark in landmarks}


def _body_expansion(world: Dict[str, Landmark]) -> float:
    """Return the volume of the axis-aligned box around all landmarks."""
    if not world:
        return 0.0
    xs = [lm.x for lm in world.values()]
    ys = [lm.y for lm in world.values()]
    zs = [lm.z for lm in world.values()]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    return dx * dy * dz


def _root_vertical_level(world: Dict[str, Landmark]) -> float:
    """Return the mean y of the hips."""
    hips = [world[key] for key in HIP_LANDMARKS if key in world]
    if not hips:
        return 0.0
    return sum(lm.y for lm in hips) / len(hips)


def _aggregate_velocity(
    world: Dict[str, Landmark],
    prev_world: Dict[str, Landmark] | None,
    names: Sequence[str],
    dt: float,
) -> float:
    """Return the mean speed of the named landmarks since the previous frame."""
    if prev_world is None or dt <= 0:
        return 0.0
    velocities = []
    for name in names:
        if name in world and name in prev_world:
            velocities.append(
                _distance(world[name], prev_world[name]) / dt
            )
    if not velocities:
        return 0.0
    return sum(velocities) / len(velocities)


def _distance(a: Landmark, b: Landmark) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    )


def _torso_inclination(world: Dict[str, Landmark]) -> float:
    """Return the angle in degrees between the hips-to-shoulders vector and the +y axis."""
    hips = [world[key] for key in HIP_LANDMARKS if key in world]
    shoulders = [world[key] for key in SHOULDER_LANDMARKS if key in world]
    if len(hips) < 2 or len(shoulders) < 2:
        return 0.0
    hip_center = _centroid(hips)
    shoulder_center = _centroid(shoulders)
    vector = (
        shoulder_center.x - hip_center.x,
        shoulder_center.y - hip_center.y,
        shoulder_center.z - hip_center.z,
    )
    vertical = (0.0, 1.0, 0.0)
    dot = sum(v * w for v, w in zip(vector, vertical))
    norm_vector = math.sqrt(sum(v * v for v in vector))
    if norm_vector == 0:
        return 0.0
    cos_angle = dot / norm_vector
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle


def _centroid(points: Sequence[Landmark]) -> Landmark:
    """Return the mean position of the points, as a synthetic landmark."""
    n = len(points)
    return Landmark(
        index=-1,
        name="CENTROID",
        x=sum(p.x for p in points) / n,
        y=sum(p.y for p in points) / n,
        z=sum(p.z for p in points) / n,
        visibility=1.0,
    )