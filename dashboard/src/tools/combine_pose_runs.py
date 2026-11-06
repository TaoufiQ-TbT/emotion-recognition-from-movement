"""Blend several pose runs of one video into one, frame by frame (end of stage 1).

For each frame, the run with the best visibility metric wins and its whole skeleton is kept:
landmarks from different runs are never mixed within a frame.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Ensure project modules are importable when the script is executed directly.
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from utils.io import load_pose_run, serialise_pose_run  # noqa: E402
from utils.types import FrameDiagnostics, FramePose  # noqa: E402


class RunBundle:
    """Container for the data loaded from a single pose estimation run."""

    def __init__(
        self,
        label: str,
        json_path: Path,
        metadata: Dict[str, object],
        frames: List[FramePose],
        diagnostics: List[FrameDiagnostics],
    ) -> None:
        self.label = label
        self.json_path = json_path
        self.metadata = metadata
        self.frames = frames
        self.diagnostics = diagnostics

        self.pose_by_index: Dict[int, FramePose] = {frame.frame_index: frame for frame in frames}
        self.diag_by_index: Dict[int, FrameDiagnostics] = {
            diag.frame_index: diag for diag in diagnostics
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple pose JSON runs by selecting, per frame, the pipeline with the best "
            "visibility metric."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=POSE_JSON",
        help="Pose JSON produced by pose_estimation.py (provide multiple --run entries).",
    )
    parser.add_argument(
        "--metric",
        default="world_visible_ratio",
        choices=(
            "world_visible_ratio",
            "world_visibility_mean",
            "image_visible_ratio",
            "image_visibility_mean",
            "mask_coverage",
        ),
        help="Frame-level metric used to choose the winning pipeline (default: world_visible_ratio).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Destination for the merged pose JSON.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        help="Optional CSV containing diagnostics for the merged sequence.",
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Optional CSV logging which pipeline was chosen per frame.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help=(
            "Only switch pipelines when the winning metric exceeds the current best by this margin "
            "(helps avoid jitter). Default: 0.0."
        ),
    )
    return parser.parse_args()


def parse_run_argument(entry: str) -> Tuple[str, Path]:
    """Split a LABEL=POSE_JSON argument and check that the file exists."""
    if "=" not in entry:
        raise ValueError(f"Run specification must be LABEL=PATH, got '{entry}'.")
    label, path = entry.split("=", 1)
    label = label.strip()
    json_path = Path(path).expanduser().resolve()
    if not label:
        raise ValueError(f"Run label cannot be empty (entry: '{entry}').")
    if not json_path.exists():
        raise FileNotFoundError(f"Pose JSON not found: {json_path}")
    return label, json_path


def load_runs(entries: Iterable[str]) -> List[RunBundle]:
    """Load every run given on the command line."""
    runs: List[RunBundle] = []
    for entry in entries:
        label, json_path = parse_run_argument(entry)
        metadata, frames, diagnostics = load_pose_run(json_path)
        runs.append(RunBundle(label, json_path, metadata, frames, diagnostics))
    if not runs:
        raise SystemExit("No valid runs provided.")
    return runs


def compute_metric(diag: FrameDiagnostics, metric: str) -> float:
    """Return the score of one frame for the chosen metric. Higher is better."""
    if metric == "world_visible_ratio":
        return (
            diag.world_landmarks_visible / diag.world_landmarks_total
            if diag.world_landmarks_total
            else 0.0
        )
    if metric == "image_visible_ratio":
        return (
            diag.image_landmarks_visible / diag.image_landmarks_total
            if diag.image_landmarks_total
            else 0.0
        )
    if metric == "world_visibility_mean":
        return diag.world_visibility_mean
    if metric == "image_visibility_mean":
        return diag.image_visibility_mean
    if metric == "mask_coverage":
        return diag.mask_coverage
    raise ValueError(f"Unsupported metric: {metric}")


def merge_runs(
    runs: List[RunBundle],
    metric: str,
    min_delta: float,
) -> Tuple[List[FramePose], List[FrameDiagnostics], List[Dict[str, object]]]:
    """Return merged frames, diagnostics, and selection log rows."""

    frame_indices = sorted(
        {index for run in runs for index in run.diag_by_index.keys()}
    )

    merged_frames: List[FramePose] = []
    merged_diagnostics: List[FrameDiagnostics] = []
    selection_rows: List[Dict[str, object]] = []

    current_label: str | None = None
    current_metric: float = 0.0

    for index in frame_indices:
        candidates = []
        for run in runs:
            diag = run.diag_by_index.get(index)
            if diag is None:
                continue
            pose = run.pose_by_index.get(index)
            score = compute_metric(diag, metric)
            candidates.append((score, pose is not None, run.label, diag, pose))

        if not candidates:
            continue

        # Best score first. On a tie, prefer a run that has a pose for this frame.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        best_candidate = next((c for c in candidates if c[1]), None)
        if best_candidate is None:
            continue

        # min_delta adds hysteresis: stay on the current run unless another one is clearly better.
        current_candidate = (
            next((c for c in candidates if c[2] == current_label and c[1]), None)
            if current_label is not None
            else None
        )

        if current_candidate is None:
            chosen_score, _has_pose, chosen_label, chosen_diag, chosen_pose = best_candidate
        else:
            best_score = best_candidate[0]
            current_score = current_candidate[0]
            if best_candidate[2] == current_label or (best_score - current_score) >= min_delta:
                chosen_score, _has_pose, chosen_label, chosen_diag, chosen_pose = best_candidate
            else:
                chosen_score, _has_pose, chosen_label, chosen_diag, chosen_pose = current_candidate

        if chosen_pose is None or chosen_diag is None:
            continue

        # Copy the pose so we do not accidentally mutate the source run.
        merged_frames.append(
            FramePose(
                frame_index=chosen_pose.frame_index,
                timestamp_s=chosen_pose.timestamp_s,
                image_path=str(Path(chosen_pose.image_path).resolve())
                if chosen_pose.image_path
                else "",
                landmarks_world=chosen_pose.landmarks_world,
                landmarks_image=chosen_pose.landmarks_image,
            )
        )

        merged_diag = replace(
            chosen_diag,
            fallback_reason=f"{chosen_diag.fallback_reason or 'ok'}|selected={chosen_label}",
        )
        merged_diagnostics.append(merged_diag)

        selection_rows.append(
            {
                "frame_index": chosen_diag.frame_index,
                "timestamp_s": chosen_diag.timestamp_s,
                "selected_pipeline": chosen_label,
                metric: chosen_score,
                "world_landmarks_visible": chosen_diag.world_landmarks_visible,
                "world_landmarks_total": chosen_diag.world_landmarks_total,
                "image_landmarks_visible": chosen_diag.image_landmarks_visible,
                "image_landmarks_total": chosen_diag.image_landmarks_total,
            }
        )

        current_label = chosen_label
        current_metric = chosen_score

    return merged_frames, merged_diagnostics, selection_rows


def write_selection_csv(path: Path, rows: List[Dict[str, object]], metric: str) -> None:
    """Log which run won each frame."""
    fieldnames = [
        "frame_index",
        "timestamp_s",
        "selected_pipeline",
        metric,
        "world_landmarks_visible",
        "world_landmarks_total",
        "image_landmarks_visible",
        "image_landmarks_total",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics_csv(path: Path, diagnostics: List[FrameDiagnostics]) -> None:
    """Write the visibility metrics of the merged sequence."""
    fieldnames = [
        "frame_index",
        "timestamp_s",
        "world_landmarks_total",
        "world_landmarks_visible",
        "world_visibility_mean",
        "image_landmarks_total",
        "image_landmarks_visible",
        "image_visibility_mean",
        "mask_coverage",
        "mask_confidence",
        "crop_applied",
        "crop_area_ratio",
        "background_blur_applied",
        "fallback_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for diag in diagnostics:
            writer.writerow(
                {
                    "frame_index": diag.frame_index,
                    "timestamp_s": diag.timestamp_s,
                    "world_landmarks_total": diag.world_landmarks_total,
                    "world_landmarks_visible": diag.world_landmarks_visible,
                    "world_visibility_mean": diag.world_visibility_mean,
                    "image_landmarks_total": diag.image_landmarks_total,
                    "image_landmarks_visible": diag.image_landmarks_visible,
                    "image_visibility_mean": diag.image_visibility_mean,
                    "mask_coverage": diag.mask_coverage,
                    "mask_confidence": diag.mask_confidence,
                    "crop_applied": diag.crop_applied,
                    "crop_area_ratio": diag.crop_area_ratio,
                    "background_blur_applied": diag.background_blur_applied,
                    "fallback_reason": diag.fallback_reason,
                }
            )


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run)

    merged_frames, merged_diagnostics, selection_rows = merge_runs(
        runs, args.metric, args.min_delta
    )
    if not merged_frames:
        raise SystemExit("No frames were selected. Check metric thresholds and inputs.")

    base_metadata = runs[0].metadata
    payload = serialise_pose_run(
        video_path=Path(base_metadata.get("video_path", "")),
        stride=int(base_metadata.get("frame_stride", 1)),
        fps=float(base_metadata.get("fps", 0.0)),
        frames=merged_frames,
        frame_metrics=merged_diagnostics,
        visibility_threshold=float(base_metadata.get("visibility_threshold", 0.0)),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)

    print(
        f"Merged {len(selection_rows)} frames using metric '{args.metric}'. "
        f"Pose JSON written to {args.output_json}."
    )

    if args.output_metrics:
        write_metrics_csv(args.output_metrics, merged_diagnostics)
        print(f"Diagnostics CSV saved to {args.output_metrics}.")

    if args.selection_report:
        write_selection_csv(args.selection_report, selection_rows, args.metric)
        print(f"Selection report saved to {args.selection_report}.")


if __name__ == "__main__":
    main()