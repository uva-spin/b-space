#!/usr/bin/env python3
"""Extract the conventional N3LL+NNLO W/ASY/FO/Y terms from DYTurbo logs.

The full-grid runner retains DYTurbo's term table in each dataset log.  This
postprocessor reconstructs the explicit conventional pieces without another
integration:

    W = RES, ASY = -CT, FO = VJREAL + VJVIRTUAL, Y = FO - ASY = VJ + CT.

It is fail-closed and writes only inside the isolated campaign directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(__file__).resolve().parents[2].parent / "Data/v23a_tevatron_plus_lhcb7_fiducial_candidate"
DATASETS = ("CDF_RUN_1", "CDF_RUN_2", "D0_RUN_1")
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
TERM_RE = re.compile(rf"(?P<v>{NUM})\s*\+\-\s*(?P<u>{NUM})(?:\s*e(?P<e>[+-]?\d+))?")


def parse_term(text: str) -> tuple[float, float]:
    m = TERM_RE.search(text)
    if m is None:
        raise ValueError(f"term cell is not parseable: {text!r}")
    exponent = int(m.group("e") or 0)
    scale = 10.0 ** exponent
    return float(m.group("v")) * scale, abs(float(m.group("u")) * scale)


def parse_log(path: Path, edges: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.count("|") < 6:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # A single-row precision-refinement log omits the qT-bin label from
        # its term table.  The caller supplies that row's unique edge pair.
        if len(cells) == 5 and len(edges) == 1 \
                and re.match(rf"^{NUM}\s*\+\-", cells[0]):
            lo, hi = map(float, edges[0])
            res, res_u = parse_term(cells[0])
            ct, ct_u = parse_term(cells[1])
            vjreal, vjreal_u = parse_term(cells[2])
            vjvirt, vjvirt_u = parse_term(cells[3])
            total, total_u = parse_term(cells[4])
            rows.append({
                "qT_low": lo, "qT_high": hi,
                "res_raw_fb_per_bin": res, "res_unc_fb_per_bin": res_u,
                "ct_raw_fb_per_bin": ct, "ct_unc_fb_per_bin": ct_u,
                "vjreal_raw_fb_per_bin": vjreal, "vjreal_unc_fb_per_bin": vjreal_u,
                "vjvirt_raw_fb_per_bin": vjvirt, "vjvirt_unc_fb_per_bin": vjvirt_u,
                "total_log_raw_fb_per_bin": total, "total_log_unc_fb_per_bin": total_u,
            })
            continue
        if len(cells) != 6 or not re.match(rf"^{NUM}\s*-\s*{NUM}$", cells[0]):
            continue
        lo_s, hi_s = re.split(r"\s*-\s*", cells[0], maxsplit=1)
        lo, hi = float(lo_s), float(hi_s)
        if not any(abs(lo - a) < 1.0e-8 and abs(hi - b) < 1.0e-8 for a, b in edges):
            continue
        res, res_u = parse_term(cells[1])
        ct, ct_u = parse_term(cells[2])
        vjreal, vjreal_u = parse_term(cells[3])
        vjvirt, vjvirt_u = parse_term(cells[4])
        total, total_u = parse_term(cells[5])
        rows.append({
            "qT_low": lo, "qT_high": hi,
            "res_raw_fb_per_bin": res, "res_unc_fb_per_bin": res_u,
            "ct_raw_fb_per_bin": ct, "ct_unc_fb_per_bin": ct_u,
            "vjreal_raw_fb_per_bin": vjreal, "vjreal_unc_fb_per_bin": vjreal_u,
            "vjvirt_raw_fb_per_bin": vjvirt, "vjvirt_unc_fb_per_bin": vjvirt_u,
            "total_log_raw_fb_per_bin": total, "total_log_unc_fb_per_bin": total_u,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"no DYTurbo term rows parsed from {path}")
    result = result.drop_duplicates(["qT_low", "qT_high"], keep="last")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed when reading a historical grid status without random_seed")
    ap.add_argument("--precision-dir", default=None,
                    help="optional precision-refinement root; matching row logs override the base grid terms")
    args = ap.parse_args()
    grid_dir = Path(args.grid_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    precision_dir = (Path(args.precision_dir).resolve() if args.precision_dir
                     else grid_dir / "precision_refinement")
    precision_logs_dir = precision_dir / "logs"
    grid = pd.read_csv(grid_dir / "tevatron_full_wy_grid.csv")
    status = json.loads((grid_dir / "grid_status.json").read_text())
    seed = int(args.seed if args.seed is not None else status["random_seed"])
    records = []
    log_paths = {}
    precision_overrides = []
    for dataset in DATASETS:
        source = pd.read_csv(DATA_ROOT / f"{dataset}.csv").sort_values("qT_low").reset_index(drop=True)
        edges = np.r_[source.qT_low.iloc[0], source.qT_high.to_numpy(float)]
        pattern = f"{dataset}_full_n3ll_nnlo_grid_g1_*_seed_{seed}.log"
        matches = sorted((grid_dir / "logs").glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one log for {dataset} seed {seed}, found {matches}")
        log_paths[dataset] = str(matches[0])
        terms = parse_log(matches[0], np.c_[edges[:-1], edges[1:]])
        # The final candidate grid may contain high-statistic replacements for
        # cancellation-sensitive rows.  The original dataset log still holds
        # the 100M-call term table, so splice in the matching 300M-call term
        # row whenever it exists.  This keeps the explicit Y decomposition
        # numerically tied to the final refined W+Y grid.
        if precision_logs_dir.exists():
            refined_terms = []
            for _, source_row in source.iterrows():
                row_key = str(source_row.row_id).replace(":", "_")
                candidates = sorted(precision_logs_dir.glob(
                    f"{row_key}_g1_*_refined_*.log"))
                if len(candidates) > 1:
                    raise RuntimeError(
                        f"multiple precision logs for {source_row.row_id}: {candidates}")
                if not candidates:
                    continue
                refined = parse_log(
                    candidates[0],
                    np.array([[float(source_row.qT_low), float(source_row.qT_high)]],
                              dtype=float))
                refined_terms.append(refined)
                precision_overrides.append({
                    "dataset": dataset, "row_id": str(source_row.row_id),
                    "log": str(candidates[0]),
                })
            if refined_terms:
                refined_frame = pd.concat(refined_terms, ignore_index=True)
                terms = pd.concat([
                    terms[~terms.set_index(["qT_low", "qT_high"]).index.isin(
                        refined_frame.set_index(["qT_low", "qT_high"]).index)],
                    refined_frame,
                ], ignore_index=True)
        merged = source[["dataset", "row_id", "qT_low", "qT_high", "CS", "error"]].merge(
            terms, on=["qT_low", "qT_high"], how="left", validate="one_to_one"
        )
        if len(merged) != len(source):
            raise RuntimeError(f"{dataset}: term rows do not match source bins")
        width = merged.qT_high - merged.qT_low
        merged["W_RES_pb_per_GeV"] = merged.res_raw_fb_per_bin / width / 1000.0
        merged["W_RES_unc_pb_per_GeV"] = merged.res_unc_fb_per_bin / width / 1000.0
        merged["ASY_minus_pb_per_GeV"] = -merged.ct_raw_fb_per_bin / width / 1000.0
        merged["ASY_minus_unc_pb_per_GeV"] = merged.ct_unc_fb_per_bin / width / 1000.0
        merged["FO_VJ_pb_per_GeV"] = (merged.vjreal_raw_fb_per_bin + merged.vjvirt_raw_fb_per_bin) / width / 1000.0
        merged["FO_VJ_unc_pb_per_GeV"] = np.sqrt(merged.vjreal_unc_fb_per_bin**2 + merged.vjvirt_unc_fb_per_bin**2) / width / 1000.0
        merged["Y_CS"] = (merged.ct_raw_fb_per_bin + merged.vjreal_raw_fb_per_bin + merged.vjvirt_raw_fb_per_bin) / width / 1000.0
        merged["Y_CS_unc_pb_per_GeV"] = np.sqrt(merged.ct_unc_fb_per_bin**2 + merged.vjreal_unc_fb_per_bin**2 + merged.vjvirt_unc_fb_per_bin**2) / width / 1000.0
        merged["reconstructed_full_wy_pb_per_GeV"] = merged.W_RES_pb_per_GeV + merged.Y_CS
        merged["source_grid_full_wy_pb_per_GeV"] = grid.set_index("row_id").loc[merged.row_id, "full_wy_pb_per_GeV"].to_numpy(float)
        merged["term_reconstruction_difference_pb_per_GeV"] = merged.reconstructed_full_wy_pb_per_GeV - merged.source_grid_full_wy_pb_per_GeV
        records.append(merged)
    result = pd.concat(records, ignore_index=True)
    numeric = ["W_RES_pb_per_GeV", "ASY_minus_pb_per_GeV", "FO_VJ_pb_per_GeV", "Y_CS", "reconstructed_full_wy_pb_per_GeV"]
    finite = bool(np.isfinite(result[numeric].to_numpy(float)).all())
    positive_full = bool((result.reconstructed_full_wy_pb_per_GeV > 0).all())
    if not finite or not positive_full:
        raise RuntimeError("term grid contains nonfinite or nonpositive reconstructed W+Y values")
    result.to_csv(out / "tevatron_y_grid.csv", index=False)
    diff = result.term_reconstruction_difference_pb_per_GeV.to_numpy(float)
    summary = {
        "status": "isolated_tevatron_conventional_y_grid_extracted_not_promoted",
        "formula": "W=RES; ASY=-CT; FO=VJREAL+VJVIRTUAL; Y=FO-ASY=VJ+CT",
        "engine": "/home/dustin/src/dyturbo-1.4.2/bin/dyturbo",
        "order": 3, "primed": False, "g1_GeV2": float(status["g1_GeV2"]),
        "random_seed": seed, "row_count": int(len(result)),
        "precision_override_count": len(precision_overrides),
        "precision_overrides": precision_overrides,
        "checks": {
            "all_finite": finite, "all_positive_reconstructed_full_wy": positive_full,
            "max_abs_term_reconstruction_difference_pb_per_GeV": float(np.max(np.abs(diff))),
            "median_abs_term_reconstruction_difference_pb_per_GeV": float(np.median(np.abs(diff))),
        },
        "logs": log_paths,
        "artifact": str(out / "tevatron_y_grid.csv"),
        "frozen_baseline_unchanged": True,
        "production_outputs_modified": False,
        "promotion_authorized": False,
    }
    (out / "y_grid_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
