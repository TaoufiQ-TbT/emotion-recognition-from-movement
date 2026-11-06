"""Compare the average landmark visibility of one or more pose runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

# Ensure local imports resolve when executed directly.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_pose_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average visibility metrics from pose JSON outputs."
    )
    parser.add_argument(
        "pose_jsons",
        nargs="+",
        help="Pose JSON path or LABEL=PATH (provide multiple to compare).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write a CSV summary.",
    )
    return parser.parse_args()


def resolve_entry(entry: str) -> Tuple[str, Path]:
    """Split an optional LABEL= prefix from a pose JSON path."""
    if "=" in entry:
        label, raw_path = entry.split("=", 1)
        label = label.strip()
        json_path = Path(raw_path.strip()).expanduser().resolve()
    else:
        json_path = Path(entry).expanduser().resolve()
        label = json_path.stem

    if not json_path.exists():
        raise FileNotFoundError(f"Pose JSON not found: {json_path}")

    if not label:
        label = json_path.stem

    return label, json_path


def summarise(label: str, json_path: Path) -> Tuple[str, int, float, float, float, float]:
    """Return the table row of one pose run."""
    metadata, frames, diagnostics = load_pose_run(json_path)

    if not diagnostics:
        raise SystemExit(
            f"No diagnostics found in {json_path}. Re-run pose_estimation.py with metrics enabled."
        )

    world_ratios: List[float] = []
    world_means: List[float] = []
    image_ratios: List[float] = []
    image_means: List[float] = []

    for diag in diagnostics:
        if diag.world_landmarks_total:
            world_ratios.append(diag.world_landmarks_visible / diag.world_landmarks_total)
        if diag.image_landmarks_total:
            image_ratios.append(diag.image_landmarks_visible / diag.image_landmarks_total)
        world_means.append(diag.world_visibility_mean)
        image_means.append(diag.image_visibility_mean)

    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    return (
        label,
        len(frames),
        _mean(world_ratios),
        _mean(world_means),
        _mean(image_ratios),
        _mean(image_means),
    )


def main() -> None:
    args = parse_args()
    rows = []
    for entry in args.pose_jsons:
        label, json_path = resolve_entry(entry)
        rows.append(summarise(label, json_path))

    headers = [
        "label",
        "frames",
        "world_visible_ratio_mean",
        "world_visibility_mean",
        "image_visible_ratio_mean",
        "image_visibility_mean",
    ]

    print("\t".join(headers))
    for label, frames, w_ratio, w_mean, i_ratio, i_mean in rows:
        print(
            f"{label}\t{frames}\t{w_ratio:.4f}\t{w_mean:.4f}\t{i_ratio:.4f}\t{i_mean:.4f}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
        print(f"Summary written to {args.csv}")


if __name__ == "__main__":
    main()