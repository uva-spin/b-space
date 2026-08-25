#!/usr/bin/env python3
"""Summarize isolated fixed-target integration-method probes."""

from __future__ import annotations

import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "reports/fixed_target_quadrature_probes"
OUT = BASE / "reports/fixed_target_quadrature_probe_summary.json"


def main() -> None:
    records = []
    for status_path in sorted(ROOT.glob("*/probe_status.json")):
        status = json.loads(status_path.read_text())
        log_path = Path(status["log"])
        total = None
        uncertainty = None
        if log_path.exists():
            text = log_path.read_text()
            match = re.search(r"Total cross section\s+([+-]?\d+(?:\.\d+)?)\s*\+-\s*([+-]?\d+(?:\.\d+)?)\s+fb", text)
            if match:
                total, uncertainty = float(match.group(1)), float(match.group(2))
        records.append({
            "name": status_path.parent.name,
            "status": status["status"],
            "intDimVJ": status.get("integration", {}).get("intDimVJ"),
            "VJquad": status.get("integration", {}).get("VJquad"),
            "makecuts": status.get("integration", {}).get("makecuts"),
            "return_code": status.get("return_code"),
            "target": status.get("target"),
            "total_fb_per_bin": total,
            "total_unc_fb_per_bin": uncertainty,
            "status_file": str(status_path),
        })
    result = {
        "status": "isolated_fixed_target_quadrature_method_summary_complete_not_production",
        "row_tested": "E288_200:0",
        "records": records,
        "interpretation": "no-cut quadrature is a promising numerical route; cut-aware quadrature failed and no 329-row closure is claimed",
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
