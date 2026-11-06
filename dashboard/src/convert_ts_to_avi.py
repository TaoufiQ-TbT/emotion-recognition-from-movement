"""Convert .ts recordings to .avi with ffmpeg.

The new file gets a "_ts" suffix, which keeps it apart from the .avi of the same performance.
Each source file is deleted once its conversion succeeds. Without --apply, the commands are only
printed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .ts files to .avi in a folder.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Directory containing the .ts files (default: assets).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run ffmpeg conversions. Without this flag, shows what would happen.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .avi files if present.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable to use (default: ffmpeg on PATH).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ffmpeg_path = shutil.which(args.ffmpeg)
    if ffmpeg_path is None:
        raise SystemExit(f"ffmpeg not found on PATH (looked for: {args.ffmpeg}).")

    assets_dir = args.assets_dir.expanduser().resolve()
    if not assets_dir.is_dir():
        raise SystemExit(f"Assets directory not found: {assets_dir}")

    ts_files = sorted(assets_dir.glob("*.ts"))
    if not ts_files:
        print("No .ts files found.")
        return

    for src in ts_files:
        dst = src.with_name(f"{src.stem}_ts.avi")
        if dst.exists() and not args.overwrite:
            print(f"SKIP (exists): {dst.name}")
            continue

        cmd = [
            ffmpeg_path,
            "-y" if args.overwrite else "-n",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            str(dst),
        ]
        if not args.apply:
            print("DRY-RUN:", " ".join(cmd))
            continue

        print(f"CONVERT: {src.name} -> {dst.name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR (kept source): {src.name} -> {dst.name} ({exc})")
            continue

        try:
            src.unlink()
            print(f"DELETE SOURCE: {src.name}")
        except OSError as exc:
            print(f"WARNING: Converted but could not delete {src.name}: {exc}")


if __name__ == "__main__":
    main()