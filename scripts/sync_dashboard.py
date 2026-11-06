"""Copy the shared code from training-pipeline/ into dashboard/, so the dashboard keeps running on its own.

The dashboard ships its own copy of training-pipeline/src and training-pipeline/training. Edit the code in
training-pipeline/, then run from the repository root:

    python scripts/sync_dashboard.py          # copy changed files into dashboard/
    python scripts/sync_dashboard.py --check  # only report differences (exit code 1 if out of sync)
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ("src", "training")
IGNORED = {"__pycache__", ".DS_Store"}


def tracked_files(base: Path) -> set[Path]:
    """Return the files under base, relative to it, leaving out caches."""
    return {
        p.relative_to(base)
        for p in base.rglob("*")
        if p.is_file() and not IGNORED & set(p.relative_to(base).parts)
    }


def main() -> int:
    """Copy or compare the shared folders. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Report differences without copying.")
    args = parser.parse_args()

    out_of_sync = 0
    for folder in SHARED:
        source, copy = ROOT / "training-pipeline" / folder, ROOT / "dashboard" / folder
        source_files, copy_files = tracked_files(source), tracked_files(copy)

        for rel in sorted(source_files):
            src, dst = source / rel, copy / rel
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            out_of_sync += 1
            print(f"{'differs' if args.check else 'copied '}  dashboard/{folder}/{rel}")
            if not args.check:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # Never delete automatically: a file only in the copy may be work in progress.
        for rel in sorted(copy_files - source_files):
            out_of_sync += 1
            print(f"extra    dashboard/{folder}/{rel} (not in training-pipeline/{folder}; remove it by hand if obsolete)")

    if out_of_sync == 0:
        print("dashboard/ is in sync with training-pipeline/.")
    return 1 if args.check and out_of_sync else 0


if __name__ == "__main__":
    sys.exit(main())
