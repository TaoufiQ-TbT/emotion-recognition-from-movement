"""Run stages 1 to 3 of the training pipeline on every video of a folder.

For each video: estimate the pose on four preprocessed versions of the frames, blend them frame
by frame, apply quality control, then compute the LMA features. Each step is a separate script
started by `run_command`, so it can also be run and debugged on its own.

The final files are outputs/run_blended_qc/<clip>_poses.json and <clip>_features.json.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_INTERMEDIATE_ROOT = PROJECT_ROOT / "outputs" / "batch_runs"
DEFAULT_FINAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "run_blended_qc"
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
TIME_EMOJI = "⏱️"
# What it usually means when a step of the pipeline fails on a video.
FAILURE_REASONS = {
    "pose_estimation.py": "the video could not be read",
    "combine_pose_runs.py": "no person was detected",
    "process_quality.py": "too few usable frames",
    "feature_extraction.py": "the features could not be computed",
}


@dataclass(frozen=True)
class PoseRunSpec:
    """One preprocessing variant, with the pose_estimation.py options that select it."""
    label: str
    folder: str
    extra_args: Sequence[str]


# The four versions of each frame. "raw" uses the segmentation mode with its crop and blur off,
# so the frame reaches MediaPipe unchanged.
POSE_RUNS: tuple[PoseRunSpec, ...] = (
    PoseRunSpec(
        label="raw",
        folder="run_raw",
        extra_args=(
            "--preprocess-mode",
            "segmentation",
            "--no-person-crop",
            "--no-background-blur",
        ),
    ),
    PoseRunSpec(
        label="segmentation",
        folder="run_segmentation",
        extra_args=(
            "--preprocess-mode",
            "segmentation",
            "--min-crop-area",
            "0.02",
            "--mask-threshold",
            "0.05",
            "--crop-padding",
            "0.2",
        ),
    ),
    PoseRunSpec(
        label="denoise",
        folder="run_denoise",
        extra_args=(
            "--preprocess-mode",
            "denoise_sharpen",
            "--min-crop-area",
            "0.02",
            "--mask-threshold",
            "0.05",
            "--crop-padding",
            "0.2",
        ),
    ),
    PoseRunSpec(
        label="illumination",
        folder="run_illumination",
        extra_args=(
            "--preprocess-mode",
            "illumination",
            "--min-crop-area",
            "0.02",
            "--mask-threshold",
            "0.05",
            "--crop-padding",
            "0.2",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each video, run pose estimation across preprocessing variants,"
            " blend the results, quality-control the merged poses, and derive LMA features."
        )
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Directory containing source videos (default: ./assets).",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        type=Path,
        help="Optional explicit list of videos to process."
        " Paths are resolved relative to --assets-dir when not absolute.",
    )
    parser.add_argument(
        "--intermediate-root",
        type=Path,
        default=DEFAULT_INTERMEDIATE_ROOT,
        help="Where to place per-video pose estimation runs (default: outputs/batch_runs).",
    )
    parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_DIR,
        help="Directory for final *_poses.json and *_features.json (default: outputs/run_blended_qc).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3000,
        help="Maximum frames sampled per pose estimation run (default: 3000).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Process every Nth frame during pose estimation (default: 4, as used to train the models of this project).",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Also save the sampled frames as images, for visualize_pose_multiview.py (about 12 MB per second of video).",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip videos whose final pose+feature JSON already exist (default).",
    )
    skip_group.add_argument(
        "--reprocess-existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-running videos even if final outputs already exist.",
    )
    return parser.parse_args()


def discover_videos(assets_dir: Path, explicit: Iterable[Path] | None) -> list[Path]:
    """Return the videos given explicitly, or else every video directly inside assets_dir."""
    if explicit:
        videos: list[Path] = []
        for entry in explicit:
            path = entry if entry.is_absolute() else assets_dir / entry
            path = path.expanduser().resolve()
            if not path.exists():
                raise SystemExit(f"Video not found: {path}")
            videos.append(path)
        return videos

    assets_dir = assets_dir.expanduser().resolve()
    if not assets_dir.is_dir():
        raise SystemExit(f"Folder not found: {assets_dir}. Create it and put your videos in it, or choose another folder with --assets-dir.")

    videos = [
        path
        for path in sorted(assets_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    ignored = [
        path.name
        for path in sorted(assets_dir.iterdir())
        if path.is_file() and path.suffix.lower() not in VIDEO_EXTENSIONS and not path.name.startswith(".")
    ]
    if ignored:
        print(f"Ignoring {len(ignored)} unsupported file(s), expected {', '.join(VIDEO_EXTENSIONS)}: {', '.join(ignored)}")
    if not videos:
        raise SystemExit(f"No videos found in {assets_dir}. Add your {', '.join(VIDEO_EXTENSIONS)} files there, then run this again.")
    return videos


def run_command(label: str, command: Sequence[str]) -> None:
    """Print a command and run it from the project root. Raises if it fails."""
    pretty = " ".join(shlex.quote(part) for part in command)
    print(f"\n[{label}] {pretty}\n")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_pose_variant(
    video: Path,
    spec: PoseRunSpec,
    *,
    max_frames: int,
    output_dir: Path,
    python_bin: str,
    stride: int,
    keep_frames: bool,
) -> Path:
    """Stage 1: estimate the pose on one preprocessed version and return its pose JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "poses" / "metrics.csv"
    command = [
        python_bin,
        "src/pose_estimation.py",
        str(video),
        "--output-dir",
        str(output_dir),
        "--stride",
        str(stride),
        "--max-frames",
        str(max_frames),
        "--include-2d",
        "--metrics-csv",
        str(metrics_csv),
    ]
    # The frames only serve the visual check, and they take far more disk than the poses.
    if not keep_frames:
        command.append("--skip-frame-export")
    command.extend(spec.extra_args)
    run_command(f"pose:{spec.label}", command)
    pose_json = output_dir / "poses" / "poses.json"
    if not pose_json.exists():
        raise FileNotFoundError(f"Pose JSON missing after {spec.label} run: {pose_json}")
    return pose_json


def combine_runs(
    run_paths: dict[str, Path],
    *,
    python_bin: str,
    output_dir: Path,
) -> Path:
    """Stage 1: blend the versions frame by frame and return the merged pose JSON."""
    poses_dir = output_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    blended_json = poses_dir / "poses.json"
    metrics_csv = poses_dir / "metrics.csv"
    selection_csv = poses_dir / "selection.csv"

    command = [python_bin, "src/tools/combine_pose_runs.py"]
    for label, path in run_paths.items():
        command.extend(["--run", f"{label}={path}"])
    command.extend(
        [
            "--output-json",
            str(blended_json),
            "--output-metrics",
            str(metrics_csv),
            "--selection-report",
            str(selection_csv),
        ]
    )
    run_command("combine", command)
    if not blended_json.exists():
        raise FileNotFoundError(f"Combined pose JSON missing: {blended_json}")
    return blended_json


def process_quality(
    blended_json: Path,
    *,
    python_bin: str,
    output_json: Path,
) -> None:
    """Stage 2: clean the blended poses and write them to output_json.

    A visibility of 0.0 keeps every landmark at this stage. Low-visibility landmarks are only left
    out when the features are computed (see extract_features).
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        "src/process_quality.py",
        str(blended_json),
        "--output-json",
        str(output_json),
        "--min-visibility-world",
        "0.0",
        "--min-visibility-image",
        "0.0",
        "--gap-fill-frames",
        "5",
        "--max-joint-speed",
        "0.38",
    ]
    run_command("quality", command)
    if not output_json.exists():
        raise FileNotFoundError(f"QC pose JSON missing: {output_json}")


def extract_features(
    qc_pose_json: Path,
    *,
    python_bin: str,
    output_json: Path,
) -> None:
    """Stage 3: write the LMA features, leaving out landmarks under 0.5 visibility."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        "src/feature_extraction.py",
        str(qc_pose_json),
        "--min-visibility-world",
        "0.5",
        "--output-json",
        str(output_json),
    ]
    run_command("features", command)
    if not output_json.exists():
        raise FileNotFoundError(f"Feature JSON missing: {output_json}")


def process_video(
    video: Path,
    args: argparse.Namespace,
    *,
    python_bin: str,
    final_dir: Path,
    intermediate_root: Path,
) -> None:
    """Run stages 1 to 3 on one video."""
    stem = video.stem
    qc_output = final_dir / f"{stem}_poses.json"
    feature_output = final_dir / f"{stem}_features.json"

    # Long extractions can be resumed: finished videos are skipped.
    if args.skip_existing and qc_output.exists() and feature_output.exists():
        print(f"Skipping {video} (final outputs already exist). Use --reprocess-existing to force.")
        return

    print(f"\n=== Processing {video} ===")
    start_time = perf_counter()
    per_video_root = intermediate_root / stem
    run_paths: dict[str, Path] = {}
    for spec in POSE_RUNS:
        run_dir = per_video_root / spec.folder
        pose_json = run_pose_variant(
            video,
            spec,
            max_frames=args.max_frames,
            output_dir=run_dir,
            python_bin=python_bin,
            stride=args.stride,
            keep_frames=args.keep_frames,
        )
        run_paths[spec.label] = pose_json

    blended_json = combine_runs(
        run_paths,
        python_bin=python_bin,
        output_dir=per_video_root / "run_blended",
    )

    process_quality(
        blended_json,
        python_bin=python_bin,
        output_json=qc_output,
    )

    extract_features(
        qc_output,
        python_bin=python_bin,
        output_json=feature_output,
    )

    elapsed = perf_counter() - start_time
    minutes, seconds = divmod(elapsed, 60)
    duration = f"{int(minutes)}m {seconds:04.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    print(
        f"Completed {video.name} {TIME_EMOJI} {duration}: poses -> {qc_output.name}, features -> {feature_output.name}."
    )


def main() -> None:
    args = parse_args()
    python_bin = sys.executable or "python3"

    videos = discover_videos(args.assets_dir, args.videos)
    final_dir = args.final_output_dir.expanduser().resolve()
    intermediate_root = args.intermediate_root.expanduser().resolve()
    intermediate_root.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # One bad video must not stop a batch that can run for hours: keep going and report it at the end.
    failed: list[tuple[str, str]] = []
    for video in videos:
        try:
            process_video(
                video,
                args,
                python_bin=python_bin,
                final_dir=final_dir,
                intermediate_root=intermediate_root,
            )
        except subprocess.CalledProcessError as exc:
            script = Path(exc.cmd[1]).name
            failed.append((video.name, FAILURE_REASONS.get(script, f"{script} failed")))
        except FileNotFoundError as exc:
            failed.append((video.name, str(exc)))

    if failed:
        print(f"\n{len(failed)} of {len(videos)} video(s) could not be processed and are left out of the dataset:")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()