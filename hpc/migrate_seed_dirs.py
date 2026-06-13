"""One-off migration: flatten seeded run dirs into per-experiment subfolders.

Moves    results/<base>_s<seed>/   ->   results/<base>/s<seed>/

so the comparison dashboard lists one entry per experiment (and its existing
algorithm_id grouping aggregates the seeds). Matches the new naming produced by
hpc/generate_registry.py.

Usage:
    python hpc/migrate_seed_dirs.py            # dry run: print planned moves
    python hpc/migrate_seed_dirs.py --apply     # actually move the directories
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_RESULTS = _project_root / "results"

# <base>_s<digits> at the end of the directory name.
_SEED_DIR = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")


def plan_moves() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for child in sorted(_RESULTS.iterdir()):
        if not child.is_dir():
            continue
        m = _SEED_DIR.match(child.name)
        if not m:
            continue
        dest = _RESULTS / m.group("base") / f"s{m.group('seed')}"
        moves.append((child, dest))
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Perform the moves (default is a dry run).")
    args = parser.parse_args()

    if not _RESULTS.exists():
        print(f"No results/ directory at {_RESULTS}")
        sys.exit(0)

    moves = plan_moves()
    if not moves:
        print("No flat '<base>_s<seed>' directories found. Nothing to do.")
        return

    print(f"{len(moves)} directory(ies) to migrate:\n")
    conflicts = []
    for src, dest in moves:
        flag = ""
        if dest.exists():
            flag = "  [SKIP: destination exists]"
            conflicts.append((src, dest))
        print(f"  {src.relative_to(_project_root).as_posix()}"
              f"  ->  {dest.relative_to(_project_root).as_posix()}{flag}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to perform the moves.")
        return

    print()
    moved = 0
    for src, dest in moves:
        if dest.exists():
            print(f"  [skip] {dest.relative_to(_project_root).as_posix()} already exists")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1
        print(f"  [moved] {src.name} -> {dest.relative_to(_project_root).as_posix()}")

    print(f"\nDone. {moved} moved, {len(conflicts)} skipped.")


if __name__ == "__main__":
    main()
