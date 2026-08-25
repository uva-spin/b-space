#!/usr/bin/env python3
"""Check N3LL exponentiation regularization on an existing Tevatron card."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "reports/tevatron_n3ll_nnlo_wy_final_g1_1p017/cards/CDF_RUN_2_full_n3ll_nnlo_grid_g1_1p017_seed_20260823.in"
OUT = BASE / "reports/expcreg_tevatron_stress"
DYTURBO = Path("/home/dustin/src/dyturbo-1.4.2/bin/dyturbo")
DYROOT = DYTURBO.parent.parent
VARIANTS = {"expcreg05": "0.5", "expcreg075": "0.75", "default1": "1.0", "expcreg15": "1.5", "expcreg2": "2.0"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    source_text = SOURCE.read_text()
    for name, value in VARIANTS.items():
        text = re.sub(r"vegasncallsBORN\s*=\s*\d+", "vegasncallsBORN   = 100000", source_text)
        text = re.sub(r"vegasncallsCT\s*=\s*\d+", "vegasncallsCT     = 100000", text)
        text = re.sub(r"vegasncallsVJLO\s*=\s*\d+", "vegasncallsVJLO   = 100000", text)
        text = re.sub(r"vegasncallsVJREAL\s*=\s*\d+", "vegasncallsVJREAL = 100000", text)
        text = re.sub(r"vegasncallsVJVIRT\s*=\s*\d+", "vegasncallsVJVIRT = 100000", text)
        text = re.sub(r"output_filename\s*=\s*\S+", f"output_filename = cdf_run2_expcreg_{name}", text)
        text += f"\n# Isolated exponentiation-regularization probe.\nexpcreg = {value}\n"
        card = OUT / f"cdf_run2_{name}.in"
        log = OUT / f"cdf_run2_{name}.log"
        table = DYROOT / f"cdf_run2_expcreg_{name}.txt"
        card.write_text(text)
        table.unlink(missing_ok=True)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join(x for x in ("/home/dustin/miniforge3/envs/pdf-fit/lib", str(DYROOT / "lib"), env.get("LD_LIBRARY_PATH", "")) if x)
        with log.open("w") as handle:
            try:
                proc = subprocess.run([str(DYTURBO), str(card)], cwd=DYROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=600, check=False)
                return_code, timed_out = proc.returncode, False
            except subprocess.TimeoutExpired:
                return_code, timed_out = None, True
        warning_count = log.read_text(errors="replace").count("dequad abnormal termination")
        first = None
        if table.exists():
            for line in table.read_text(errors="replace").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                try:
                    first = [float(x) for x in line.split()[:2]]
                    break
                except ValueError:
                    pass
        results.append({"variant": name, "expcreg": float(value), "return_code": return_code, "timed_out": timed_out,
                        "warning_count": warning_count, "output_exists": table.exists(),
                        "first_qT_value_unc": first, "card": str(card), "log": str(log), "table": str(table)})
    status = {"status": "isolated_expcreg_tevatron_stress", "source_card": str(SOURCE), "calls_per_component": 100000,
              "results": results, "frozen_production_modified": False}
    (OUT / "stress_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
