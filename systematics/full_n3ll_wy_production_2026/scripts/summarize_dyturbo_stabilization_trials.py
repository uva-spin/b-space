#!/usr/bin/env python3
"""Consolidate every isolated E288 high-Q DYTurbo stabilization trial."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = ROOT / "reports/dyturbo_stabilization_pilot_e288_200_30"
DYROOT = Path("/home/dustin/src/dyturbo-1.4.2")
TARGET_A = 9.0121831
Q2_WIDTH = 0.2**2 - 0.02**2
Y_WIDTH = 0.431469072 - 0.372722813
DATA_A = 0.399


def parse_result(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) >= 2:
            try:
                return float(fields[0]), float(fields[1])
            except ValueError:
                pass
    return None, None


def main() -> None:
    rows = []
    for card in sorted((TRIAL_DIR / "cards").glob("e288_200_30_*.in")):
        variant = card.stem.removeprefix("e288_200_30_")
        text = card.read_text(errors="replace")
        overrides = {}
        for key, value in re.findall(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE):
            if key in {"blim", "bintaccuracy", "sumlogs", "bprescription", "qtcut", "xqtcut", "qt_bins", "modlog", "bstar_pdf", "bstar_sudakov", "bstar_expc"}:
                overrides[key] = value
        log = TRIAL_DIR / "logs" / f"e288_200_30_{variant}.log"
        log_text = log.read_text(errors="replace") if log.exists() else ""
        output = DYROOT / f"e288_200_30_stab_{variant}.txt"
        result, uncertainty = parse_result(output)
        predicted_a = None if result is None else result / (1000.0 * TARGET_A * math.pi * Q2_WIDTH * Y_WIDTH)
        ratio = None if predicted_a is None else predicted_a / DATA_A
        rows.append({
            "variant": variant,
            "overrides": overrides,
            "warning_count": log_text.count("dequad abnormal termination"),
            "output_exists": output.exists(),
            "result": result,
            "uncertainty": uncertainty,
            "result_finite": result is not None,
            "result_positive": result is not None and result > 0.0,
            "predicted_A": predicted_a,
            "predicted_A_to_data": ratio,
            "acceptable": (
                result is not None and result > 0.0
                and log_text.count("dequad abnormal termination") == 0
                and ratio is not None and 0.01 < ratio < 100.0
            ),
            "card": str(card),
            "log": str(log),
            "output": str(output),
        })
    summary = {
        "status": "isolated_dyturbo_stabilization_trials_consolidated",
        "row": "E288_200:30",
        "calls_per_component": 100000,
        "acceptance": "finite positive output with zero inverse-Bessel abnormal-termination warnings",
        "trial_count": len(rows),
        "acceptable_trials": [row["variant"] for row in rows if row["acceptable"]],
        "trials": rows,
        "frozen_production_modified": False,
    }
    out = TRIAL_DIR / "all_trials_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
