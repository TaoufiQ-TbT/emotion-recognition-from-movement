"""Play the video frames with their 2D pose, side by side with the 3D skeleton."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from config import DEFAULT_OUTPUT_DIR
from utils.io import load_pose_run
from utils.skeleton import POSE_CONNECTIONS
from utils.viz import (
    choose_space,
    compute_axes_limits,
    extract_image_landmarks,
    frames_to_tensor,
    load_rgb_image,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display RGB frames alongside the corresponding 3D skeleton."
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
        help="Coordinate space to animate (default: prefer world, fall back to image).",
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


def prepare_image_data(
    frames, pose_json_path: Path
) -> Tuple[List[np.ndarray], List[Optional[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Load RGB frames and corresponding 2D landmarks (in pixel coordinates).

    Returns two lists indexed by animation frame: the image array and a matrix
    of 2D points (shape [N, 2]). Entries may be ``None`` when data is missing.
    """

    images: List[np.ndarray] = []
    image_landmarks: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    missing_images = 0
    missing_landmarks = 0

    for frame in frames:
        if not frame.image_path:
            images.append(None)
            image_landmarks.append(None)
            missing_images += 1
            continue

        image_path = resolve_image_path(frame.image_path, pose_json_path)
        if not image_path.exists():
            images.append(None)
            image_landmarks.append(None)
            missing_images += 1
            continue

        image = load_rgb_image(image_path)
        images.append(image)

        landmarks = extract_image_landmarks(frame, image)
        if landmarks is None:
            image_landmarks.append(None)
            missing_landmarks += 1
        else:
            image_landmarks.append(landmarks)

    if missing_images:
        print(f"[visualize] Warning: {missing_images} frames have no RGB snapshot.")
    if missing_landmarks:
        print(f"[visualize] Warning: {missing_landmarks} frames lack 2D landmarks.")

    return images, image_landmarks


def create_animation(
    frames,
    coords_3d: np.ndarray,
    images: List[np.ndarray],
    image_landmarks: List[np.ndarray],
    connections: Sequence[Tuple[int, int]],
    playback_fps: float,
    space_label: str,
) -> FuncAnimation:
    """Build the Matplotlib animation object (RGB + 3D skeleton)."""

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_title("RGB Frame + 2D Pose")
    ax_img.axis("off")

    # Use first available frame as base canvas.
    default_image = next((img for img in images if img is not None), None)
    if default_image is None:
        default_image = np.zeros((480, 640, 3), dtype=np.uint8)

    img_artist = ax_img.imshow(default_image)
    height, width = default_image.shape[:2]
    ax_img.set_xlim(0, width)
    ax_img.set_ylim(height, 0)

    scatter2d = ax_img.scatter([], [], color="lime", s=25)
    lines2d = [ax_img.plot([], [], color="cyan", linewidth=2)[0] for _ in connections]

    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.set_xlabel("X (left/right)")
    ax_3d.set_ylabel("Y (forward/back)")
    ax_3d.set_zlabel("Z (up)")
    ax_3d.set_title("3D Skeleton")

    x_min, x_max, y_min, y_max, z_min, z_max = compute_axes_limits(coords_3d)
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(z_min, z_max)
    ax_3d.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

    scatter3d = ax_3d.scatter([], [], [], color="crimson", s=25)
    lines3d = [ax_3d.plot([], [], [], color="navy", linewidth=2)[0] for _ in connections]
    frame_label = fig.text(0.02, 0.95, "", transform=fig.transFigure)

    blank_image = np.zeros_like(default_image)

    def init():
        scatter2d.set_offsets(np.empty((0, 2)))
        for line in lines2d:
            line.set_data([], [])
        scatter3d._offsets3d = ([], [], [])
        for line in lines3d:
            line.set_data([], [])
            line.set_3d_properties([])
        frame_label.set_text("")
        return [img_artist, scatter2d, *lines2d, scatter3d, *lines3d, frame_label]

    def update(idx: int):
        frame = frames[idx]

        image = images[idx] if idx < len(images) else None
        if image is None:
            img_artist.set_data(blank_image)
        else:
            img_artist.set_data(image)
            h, w = image.shape[:2]
            if h != height or w != width:
                ax_img.set_xlim(0, w)
                ax_img.set_ylim(h, 0)

        # 2D overlay
        landmarks_entry = image_landmarks[idx] if idx < len(image_landmarks) else None
        if landmarks_entry is not None:
            coords2d, mask2d = landmarks_entry
            valid_coords = coords2d[mask2d]
            scatter2d.set_offsets(valid_coords)
            for line, (i, j) in zip(lines2d, connections):
                if mask2d[i] and mask2d[j]:
                    line.set_data(
                        [coords2d[i, 0], coords2d[j, 0]],
                        [coords2d[i, 1], coords2d[j, 1]],
                    )
                else:
                    line.set_data([], [])
        else:
            scatter2d.set_offsets(np.empty((0, 2)))
            for line in lines2d:
                line.set_data([], [])

        # 3D skeleton
        points3d = coords_3d[idx]
        valid3d = ~np.isnan(points3d).any(axis=1)
        xs, ys, zs = points3d[valid3d, 0], points3d[valid3d, 1], points3d[valid3d, 2]
        scatter3d._offsets3d = (xs, ys, zs)
        for line, (i, j) in zip(lines3d, connections):
            if valid3d[i] and valid3d[j]:
                line.set_data(
                    [points3d[i, 0], points3d[j, 0]],
                    [points3d[i, 1], points3d[j, 1]],
                )
                line.set_3d_properties([points3d[i, 2], points3d[j, 2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        frame_label.set_text(
            f"Frame {frame.frame_index}  |  t={frame.timestamp_s:.2f}s  |  space={space_label}"
        )
        return [img_artist, scatter2d, *lines2d, scatter3d, *lines3d, frame_label]

    interval_ms = max(1, int(1000 / max(playback_fps, 0.1)))
    return FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(coords_3d),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )


def main() -> None:
    args = parse_args()

    metadata, frames, _diagnostics = load_pose_run(args.pose_json)
    space = choose_space(frames, args.space)
    coords_3d, frames = frames_to_tensor(frames, space)

    images, image_landmarks = prepare_image_data(frames, args.pose_json)

    animation = create_animation(
        frames,
        coords_3d,
        images,
        image_landmarks,
        POSE_CONNECTIONS,
        args.playback_fps,
        space,
    )

    if args.save:
        save_path = args.save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving animation to {save_path} ...")
        animation.save(str(save_path), fps=args.playback_fps)
        print("Save complete.")

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()