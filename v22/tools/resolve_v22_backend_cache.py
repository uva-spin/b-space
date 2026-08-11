#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def score_w(path: Path, columns: set[str]) -> int:
    lower = {c.lower() for c in columns}
    score = 0
    name = path.name.lower()
    if "row_id" in lower:
        score += 10
    if "bt" in lower or "b" in lower:
        score += 10
    if "wpert_cs" in lower:
        score += 50
    if "w_cs" in lower:
        score += 40
    if "w" in lower:
        score += 20
    if "w" in name:
        score += 5
    if "grid" in name or "cache" in name:
        score += 2
    return score


def score_y(path: Path, columns: set[str]) -> int:
    lower = {c.lower() for c in columns}
    score = 0
    name = path.name.lower()
    if "row_id" in lower:
        score += 10
    if "y_cs" in lower:
        score += 60
    if "y" in lower:
        score += 30
    if "y" in name:
        score += 5
    if "summary" in name or "diagnostic" in name:
        score -= 30
    return score


def read_columns(path: Path) -> set[str]:
    try:
        frame = pd.read_csv(path, nrows=5)
    except Exception:
        return set()
    return set(frame.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-env", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    root = Path(args.cache_root)
    if not root.exists():
        raise SystemExit(f"Missing cache root: {root}")

    records = []
    for path in sorted(root.rglob("*.csv")):
        columns = read_columns(path)
        if not columns:
            continue
        records.append({
            "path": path,
            "columns": sorted(columns),
            "w_score": score_w(path, columns),
            "y_score": score_y(path, columns),
        })

    if not records:
        raise SystemExit(f"No readable CSV files found under {root}")

    w_record = max(records, key=lambda r: r["w_score"])
    y_record = max(records, key=lambda r: r["y_score"])

    if w_record["w_score"] < 40:
        raise SystemExit(
            "Could not identify W cache. Candidates:\n"
            + "\n".join(f"{r['path']}: {r['columns']} score={r['w_score']}" for r in records)
        )

    if y_record["y_score"] < 30:
        raise SystemExit(
            "Could not identify Y cache. Candidates:\n"
            + "\n".join(f"{r['path']}: {r['columns']} score={r['y_score']}" for r in records)
        )

    w_path = w_record["path"].resolve()
    y_path = y_record["path"].resolve()

    out = {
        "cache_root": str(root.resolve()),
        "w_grid": str(w_path),
        "y_grid": str(y_path),
        "w_columns": w_record["columns"],
        "y_columns": y_record["columns"],
        "all_candidates": [
            {
                "path": str(r["path"].resolve()),
                "columns": r["columns"],
                "w_score": r["w_score"],
                "y_score": r["y_score"],
            }
            for r in records
        ],
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")

    env = (
        f"export W_GRID='{w_path}'\n"
        f"export Y_GRID='{y_path}'\n"
        f"export BACKEND_CACHE_ROOT='{root.resolve()}'\n"
    )
    Path(args.out_env).write_text(env)

    print("Resolved backend cache:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
