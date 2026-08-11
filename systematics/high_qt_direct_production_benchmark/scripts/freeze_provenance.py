#!/usr/bin/env python3
"""Freeze hashes and environment metadata for the external-code campaign."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "systematics" / "high_qt_direct_production_benchmark"
PATHS = {
    "dyturbo_executable": Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo"),
    "mcfm_executable": Path("/home/dustin/work/MCFM-10.3/Bin/mcfm"),
    "python": Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python3.11"),
    "source_gate": ROOT / "systematics/finite_y_tail_benchmark/summaries/tail_benchmark_row_gate.csv",
    "policy": STUDY / "config/promotion_policy.json",
    "tevatron_dyturbo_runner": ROOT / "v23/tools/run_tevatron_dyturbo_benchmark.py",
    "tevatron_mcfm_runner": ROOT / "v23/tools/run_tevatron_mcfm_benchmark.py",
    "lhcb_dyturbo_runner": ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py",
    "lhcb_mcfm_runner": ROOT / "systematics/finite_y_tail_benchmark/scripts/run_lhcb7_mcfm_benchmark.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required provenance inputs: {missing}")
    pdf_root = Path("/home/dustin/work/MCFM-10.3/Bin/PDFs")
    pdf_matches = sorted(str(path) for path in pdf_root.glob("NNPDF40_nnlo_as_01180*"))
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "campaign": "high_qt_direct_production_benchmark",
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in PATHS.items()},
        "pdf": {"set": "NNPDF40_nnlo_as_01180", "member": 0, "mcfm_pdf_matches": pdf_matches},
        "git": "workspace is not a Git worktree",
        "cpu": subprocess.run(["lscpu"], check=True, text=True, capture_output=True).stdout,
    }
    out = STUDY / "config" / "campaign_provenance.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
