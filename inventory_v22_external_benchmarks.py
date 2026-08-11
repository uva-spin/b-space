#!/usr/bin/env python3
"""Locate the exact MCFM/DYTurbo artifacts used for the earlier tail benchmark.

Run from ~/work/bT-TMD.  The script is read-only.  It searches the project,
MCFM, and DYTurbo trees for likely input cards, cut files, logs, tables, and
comparison scripts, then writes a compact inventory with relevant excerpts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable


DEFAULT_ROOTS = [
    Path.cwd(),
    Path.home() / "work" / "MCFM-10.3",
    Path.home() / "src" / "dyturbo-1.4.2",
    Path.home() / "work" / "dyturbo-1.4.2",
]

FILENAME_RE = re.compile(
    r"(mcfm|dyturbo|dy[_-]?turbo|onepoint|secondpoint|bridgecut|"
    r"nearcut|midcut|widecut|gencuts|input_v1[45678]|external.*benchmark)",
    re.IGNORECASE,
)

CONTENT_RE = re.compile(
    r"(v15_onepoint|v15_secondpoint|Value of integral|external/v15|"
    r"qT/Q|m34min|m34max|sqrts|qt_bins|y_bins|m_bins|runstring|"
    r"NNPDF40_nnlo_as_01180|E288_400:80|0\.553|0\.600)",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {
    ".txt", ".log", ".out", ".dat", ".csv", ".json", ".yaml", ".yml",
    ".in", ".ini", ".toml", ".f", ".f90", ".py", ".sh", ".md",
}

SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules",
    "site-packages", "build", "CMakeFiles",
}


def walk_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in SKIP_DIRS for part in path.parts):
            continue

        yield path


def read_matching_lines(path: Path, max_bytes: int, max_matches: int) -> list[str]:
    try:
        if path.stat().st_size > max_bytes:
            return []
    except OSError:
        return []

    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []

    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []

    matches: list[str] = []

    for number, line in enumerate(text.splitlines(), start=1):
        if CONTENT_RE.search(line):
            matches.append(f"{number}: {line.rstrip()}")
            if len(matches) >= max_matches:
                break

    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_ROOTS],
        help="Trees to search.",
    )
    parser.add_argument(
        "--out",
        default="v22/outputs/external_benchmark_inventory",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--max-matches-per-file",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    roots = []
    seen_roots = set()

    for raw in args.roots:
        root = Path(raw).expanduser().resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        roots.append(root)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    records = []
    seen_paths = set()

    for root in roots:
        if not root.exists():
            continue

        for path in walk_files(root):
            resolved = path.resolve()

            if resolved in seen_paths:
                continue

            filename_hit = bool(FILENAME_RE.search(path.name))
            matches = read_matching_lines(
                path,
                max_bytes=max_bytes,
                max_matches=int(args.max_matches_per_file),
            )

            if not filename_hit and not matches:
                continue

            seen_paths.add(resolved)

            try:
                stat = path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
                size = stat.st_size
            except OSError:
                modified = ""
                size = -1

            records.append({
                "root": str(root),
                "path": str(resolved),
                "size_bytes": size,
                "modified_local": modified,
                "filename_match": filename_hit,
                "n_content_matches": len(matches),
                "matches": matches,
            })

    records.sort(
        key=lambda item: (
            -int(item["n_content_matches"] > 0),
            -int(item["filename_match"]),
            item["path"],
        )
    )

    csv_path = out / "external_benchmark_inventory.csv"
    txt_path = out / "external_benchmark_inventory.txt"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "root",
                "path",
                "size_bytes",
                "modified_local",
                "filename_match",
                "n_content_matches",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: record[key]
                for key in writer.fieldnames
            })

    with txt_path.open("w") as handle:
        handle.write("=== Search roots ===\n")
        for root in roots:
            handle.write(
                f"{'FOUND' if root.exists() else 'MISSING'}: {root}\n"
            )

        handle.write("\n=== Candidate artifacts ===\n")

        for record in records:
            handle.write(f"\n--- {record['path']} ---\n")
            handle.write(
                f"size={record['size_bytes']} "
                f"modified={record['modified_local']} "
                f"filename_match={record['filename_match']} "
                f"content_matches={record['n_content_matches']}\n"
            )
            for line in record["matches"]:
                handle.write(line + "\n")

    print("search roots:")
    for root in roots:
        print(f"  {'FOUND' if root.exists() else 'MISSING'}: {root}")

    print(f"\ncandidate files: {len(records)}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {txt_path}")

    high_value = [
        record
        for record in records
        if record["n_content_matches"] > 0
    ][:20]

    print("\n=== Highest-value candidates ===")
    for record in high_value:
        print(record["path"])
        for line in record["matches"][:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
