"""Play a 3D skeleton animation from a pose JSON file."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from config import DEFAULT_OUTPUT_DIR
from utils.io import load_pose_run
from utils.skeleton import POSE_CONNECTIONS
from utils.viz import choose_space, compute_axes_limits, frames_to_tensor as build_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D skeleton animation from pose JSON output."
    )
    parser.add_argument(
        "pose_json",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "poses" / "poses.json",
        help="Path to the pose JSON file (default: latest sample output).",
    )
    parser.add_argument(
        "--space",
        choices=("auto", "world", "image"),
        default="auto",
        help="Which coordinate space to use (default: prefer world, fall back to image).",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=12.0,
        help="Playback frame rate for the animation.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the animation (e.g., out.mp4 or out.gif).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip showing the interactive window (useful when only saving to disk).",
    )
    return parser.parse_args()


def create_animation(
    frames,
    coords: np.ndarray,
    connections: Sequence[Tuple[int, int]],
    playback_fps: float,
    space_label: str,
) -> FuncAnimation:
    """Build the Matplotlib animation object."""

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (left/right)")
    ax.set_ylabel("Y (forward/back)")
    ax.set_zlabel("Z (up)")
    ax.set_title("MediaPipe Pose – 3D Skeleton")

    x_min, x_max, y_min, y_max, z_min, z_max = compute_axes_limits(coords)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

    scatter = ax.scatter([], [], [], color="crimson", s=25)
    bones = [ax.plot([], [], [], color="navy", linewidth=2)[0] for _ in connections]
    frame_label = ax.text2D(0.02, 0.92, "", transform=ax.transAxes)

    def init():
        # Matplotlib has no public setter for the points of a 3D scatter.
        scatter._offsets3d = ([], [], [])
        for bone in bones:
            bone.set_data([], [])
            bone.set_3d_properties([])
        frame_label.set_text("")
        return [scatter, *bones, frame_label]

    def update(idx: int):
        points = coords[idx]
        valid = ~np.isnan(points).any(axis=1)
        xs, ys, zs = points[valid, 0], points[valid, 1], points[valid, 2]
        scatter._offsets3d = (xs, ys, zs)

        for bone, (i, j) in zip(bones, connections):
            if valid[i] and valid[j]:
                bone.set_data([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]])
                bone.set_3d_properties([points[i, 2], points[j, 2]])
            else:
                bone.set_data([], [])
                bone.set_3d_properties([])

        frame = frames[idx]
        frame_label.set_text(
            f"Frame {frame.frame_index}  |  t={frame.timestamp_s:.2f}s  |  space={space_label}"
        )
        return [scatter, *bones, frame_label]

    interval_ms = max(1, int(1000 / max(playback_fps, 0.1)))
    return FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(coords),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )


def main() -> None:
    args = parse_args()

    metadata, frames, _diagnostics = load_pose_run(args.pose_json)
    space = choose_space(frames, args.space)
    coords, frames = build_tensor(frames, space)

    animation = create_animation(
        frames, coords, POSE_CONNECTIONS, args.playback_fps, space
    )

    if args.save:
        save_path = args.save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Matplotlib picks the writer from the file extension.
        print(f"Saving animation to {save_path} ...")
        animation.save(str(save_path), fps=args.playback_fps)
        print("Save complete.")

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()