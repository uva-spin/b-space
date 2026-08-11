#!/usr/bin/env python3
"""Build the v23a fixed-target plus fit-ready absolute Tevatron data directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


FIXED_TARGET_DATASETS = ["E288_200", "E288_300", "E288_400", "E605", "E772"]
TEVATRON_ABSOLUTE_DATASETS = ["CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1"]
EXCLUDED_NORMALIZED_DATASETS = ["D0_RUN_2", "D0_RUN_2N"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def copy_dataset(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": str(dst),
        "source": str(src),
        "rows": row_count(dst),
        "sha256": sha256(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-target-source",
        default="Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400",
    )
    parser.add_argument(
        "--tevatron-reviewed",
        default="v23/outputs/tevatron_reviewed_table/tevatron_reviewed_absolute_diagnostic_qToQ_le_0p5.csv",
    )
    parser.add_argument("--out-data-dir", default="Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready")
    parser.add_argument("--manifest-name", default="manifest.json")
    args = parser.parse_args()

    fixed_source = Path(args.fixed_target_source)
    tevatron_path = Path(args.tevatron_reviewed)
    out = Path(args.out_data_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "purpose": "Fit-ready v23a data directory combining fixed-target DY with absolute Tevatron Z qT spectra.",
        "status": "fit_ready_absolute_subset",
        "created_from": {
            "fixed_target_source_dir": str(fixed_source),
            "tevatron_reviewed_absolute_table": str(tevatron_path),
        },
        "datasets": FIXED_TARGET_DATASETS + TEVATRON_ABSOLUTE_DATASETS,
        "excluded_normalized_datasets": [
            {
                "dataset": name,
                "reason": "published as normalized spectrum; requires normalized-theory observable before production inclusion",
            }
            for name in EXCLUDED_NORMALIZED_DATASETS
        ],
        "recommended_backend": "v23/backends/bt_internal_css_backend_v22_tevatron.py",
        "recommended_fit_cut": {
            "mode": "matched",
            "qT_max_over_Q": 0.5,
            "tmd_qT_max_over_Q": 0.2,
            "note": "The trainer applies a strict qT/Q < cut when loading rows.",
        },
        "uncertainty_treatment": {
            "absolute_tevatron": "released table errors as diagonal point-to-point uncertainties plus dataset normalization nuisance from sysNorm_rel",
            "fixed_target": "source directory already contains explicit CSV normalization and point-to-point priors",
            "known_limitation": "CDF_RUN_2 publication states the efficiency systematic is fully correlated across bins; current trainer has no separate correlated-shape nuisance, so this component remains included in diagonal dA for the first fit.",
        },
        "files": {},
    }

    for dataset in FIXED_TARGET_DATASETS:
        src = fixed_source / f"{dataset}.csv"
        dst = out / f"{dataset}.csv"
        manifest["files"][dataset] = copy_dataset(src, dst)

    tevatron_rows = read_csv(tevatron_path)
    if not tevatron_rows:
        raise ValueError(f"No rows in {tevatron_path}")
    fieldnames = list(tevatron_rows[0].keys())
    for dataset in TEVATRON_ABSOLUTE_DATASETS:
        ds_rows = [r for r in tevatron_rows if r.get("dataset") == dataset]
        if not ds_rows:
            raise ValueError(f"No reviewed absolute rows found for {dataset}")
        dst = out / f"{dataset}.csv"
        write_csv(dst, ds_rows, fieldnames)
        manifest["files"][dataset] = {
            "path": str(dst),
            "source": str(tevatron_path),
            "rows": len(ds_rows),
            "sha256": sha256(dst),
        }

    manifest["n_rows_written"] = sum(info["rows"] for info in manifest["files"].values())
    manifest_path = out / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
