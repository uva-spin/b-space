#!/usr/bin/env python3
"""Build b/k projections and crossed bands for adjacent cross-fit candidates."""

from __future__ import annotations

from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/dustin/miniforge3/envs/pdf-fit/bin/python")
BUILD_B = BASE / "scripts/build_bspace_tmd_ensemble.py"
TRANSFORM = BASE / "scripts/transform_bspace_ensemble_to_kspace.py"
CROSS = BASE / "scripts/cross_matched_flength_with_experimental.py"
SEEDS = tuple(range(303, 327))
CANDIDATES = ((1.5, "2e00"), (2.0, "2p0"))


def fold(seed: int) -> str:
    return "evenref" if seed % 2 else "oddref"


def main() -> None:
    for strength, token in CANDIDATES:
        prefix = f"crossfit_reference_distance_lam{str(strength).replace('.', 'p')}_full24"
        command = [str(PYTHON), str(BUILD_B), "--target-name", f"{prefix}_bspace"]
        for seed in SEEDS:
            tag = (f"exactbaseline_matched_reference_distance_{fold(seed)}_"
                   f"b0p1_2p0_lam{token}_s{seed}")
            command.extend(["--run-tag", tag])
        subprocess.run(command, check=True)
        subprocess.run([
            str(PYTHON), str(TRANSFORM),
            "--bspace-ensemble", str(BASE / "summaries" / f"{prefix}_bspace" / "bspace_tmd_ensemble_long.csv"),
            "--target-name", f"{prefix}_kspace",
        ], check=True)
        subprocess.run([
            str(PYTHON), str(CROSS), "--token", "1e00", "--prefix", prefix,
        ], check=True)


if __name__ == "__main__":
    main()
