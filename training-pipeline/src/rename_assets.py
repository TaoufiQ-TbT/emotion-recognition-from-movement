"""Rename dataset files so the emotion comes first, as the pipeline expects.

Turns gesture_10_happiness.avi into happiness_10_gesture.avi. Nothing is renamed without --apply.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Accept both .avi videos (assets) and .json outputs from run_blended_qc
# We keep the same transformation, but allow for extra suffixes after the emotion,
# e.g. gesture_10_happiness_ts_poses.json -> happiness_10_gesture_ts_poses.json
PATTERN = re.compile(
    r"^(gesture|improvisation)_(\d+)_([a-z0-9]+)(.*)\.(avi|json)$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename gesture/improvisation files to emotion_number_prefix.*"
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Directory containing the video files (default: assets).",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("outputs/run_blended_qc"),
        help="Directory containing generated JSONs to rename (default: outputs/run_blended_qc).",
    )
    parser.add_argument(
        "--assets",
        action="store_true",
        help="Rename video assets in --assets-dir (default if no target flag is given).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Rename JSON outputs in --json-dir.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the renames. Without this flag, a dry-run is shown.",
    )
    return parser.parse_args()


def _rename_dir(directory: Path, apply: bool) -> int:
    """Rename matching files in a directory. Returns count of renamed/matched files."""
    matched = 0
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        m = PATTERN.match(path.name)
        if not m:
            continue

        prefix, number, emotion, tail, _ext = m.groups()
        new_name = f"{emotion.lower()}_{number}_{prefix.lower()}{tail}{path.suffix}"
        new_path = path.with_name(new_name)

        if new_path.exists():
            print(f"SKIP (target exists): {path.name} -> {new_path.name}")
            continue

        matched += 1
        action = "RENAME" if apply else "DRY-RUN"
        print(f"{action}: {path.name} -> {new_path.name}")
        if apply:
            path.rename(new_path)

    return matched


def main() -> None:
    args = parse_args()
    # default behavior (backward compatible): rename assets only
    targets = []
    if not args.assets and not args.json:
        targets = ["assets"]
    else:
        if args.assets:
            targets.append("assets")
        if args.json:
            targets.append("json")

    if "assets" in targets:
        assets_dir = args.assets_dir.expanduser().resolve()
        if not assets_dir.is_dir():
            raise SystemExit(f"Assets directory not found: {assets_dir}")
        matched = _rename_dir(assets_dir, args.apply)
        if matched == 0:
            print("No gesture/improvisation assets matched the expected pattern.")

    if "json" in targets:
        json_dir = args.json_dir.expanduser().resolve()
        if not json_dir.is_dir():
            raise SystemExit(f"JSON directory not found: {json_dir}")
        matched = _rename_dir(json_dir, args.apply)
        if matched == 0:
            print("No gesture/improvisation JSONs matched the expected pattern.")


if __name__ == "__main__":
    main()