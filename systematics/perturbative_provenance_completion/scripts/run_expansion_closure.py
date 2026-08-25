#!/usr/bin/env python3
"""Run the published v22 source checks and record an isolated closure report."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "systematics" / "perturbative_provenance_completion"
V22 = ROOT / "b-space-public" / "v22"
REPORT = WORK / "reports" / "expansion_closure.json"
CHECKS = (
    "run_convolution_smoke.py",
    "run_css2_ope_nlo_smoke.py",
    "run_css2_ope_nlo_general_smoke.py",
    "run_dy_hard_nlo_smoke.py",
    "run_dy_w_nlo_reference_smoke.py",
    "run_small_b_profile_smoke.py",
)


def main() -> None:
    records = []
    env = {"PYTHONPATH": str(ROOT), **__import__("os").environ}
    for name in CHECKS:
        path = V22 / "tests" / name
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        records.append(
            {
                "check": name,
                "returncode": int(result.returncode),
                "passed": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
    report = {
        "status": "expansion_closure_passed" if all(r["passed"] for r in records) else "expansion_closure_failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree": "b-space-public/v22",
        "checks": records,
        "scope": "source-level closure only; not an external fixed-order observable validation",
        "production_outputs_modified": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "expansion_closure_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
