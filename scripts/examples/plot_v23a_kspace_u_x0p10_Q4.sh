#!/usr/bin/env bash
set -euo pipefail

# Produce the paper-style regularized k_T-space TMDPDF plot
# for the u-quark at x=0.10, Q=4 GeV:
#
#   f_1^u(x=0.10, k_T; Q=4 GeV)
#
# This is an exp+PDF-overlay k_T companion figure.  It uses:
#   1. existing experimental-replica F_NP runs;
#   2. PDF-member overlay in the TMD reconstruction;
#   3. regularized finite-b_T Hankel transform, default expb2;
#   4. optional dashed central PDF0 curve generated at the same Q=4.
#
# The output plot is:
#   plots/v23a_traditional_kspace_TMDPDF_u_x0p10_Q4.pdf
#
# Notes:
#   * This does NOT rebuild W/Y cross-section caches.
#   * This does NOT retrain replicas.
#   * Q=4 GeV is slightly below the fixed-target fit's lowest Q bin
#     (nominally Q>=4.5 GeV), so treat this as a near-boundary companion
#     plot unless you intentionally include it in a production grid.

ROOT="${ROOT:-$(pwd)}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---- User-adjustable defaults ------------------------------------------------
XVAL="${XVAL:-0.10}"
QVAL="${QVAL:-4}"
FLAVOR="${FLAVOR:-u}"
PID="${PID:-2}"                      # u-quark
KMAX="${KMAX:-4}"
BMAX="${BMAX:-8}"
NB="${NB:-321}"
NKT="${NKT:-401}"
BTRANS_MAX="${BTRANS_MAX:-24}"
NBTRANS="${NBTRANS:-6001}"
TAIL_MODE="${TAIL_MODE:-expb2}"
REBUILD="${REBUILD:-0}"

PDF_SET="${PDF_SET:-NNPDF40_nnlo_as_01180}"
BACKEND="${BACKEND:-v22/backends/bt_internal_css_backend_v22_full.py}"
CENTRAL_RUN="${CENTRAL_RUN:-outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors15_p2p5_E772_E288400_s303}"

# Existing 50 experimental replicas used for the exp+PDF overlay.
RUN_GLOB="${RUN_GLOB:-replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/outputs/v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda_s*}"

# Overlay plan. If missing, this script will make it.
OVERLAY_ROOT="${OVERLAY_ROOT:-replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep}"
PLAN="${PLAN:-${OVERLAY_ROOT}/replica_plan.csv}"
PDF_MEMBERS="${PDF_MEMBERS:-1-50}"

# Output directories for this one-curve Q=4 construction.
TAG="${TAG:-u_x0p10_Q4}"
BSPACE_OUT="${BSPACE_OUT:-${OVERLAY_ROOT}/tmd_bspace_bands_expPDF_overlay_${TAG}}"
KSPACE_OUT="${KSPACE_OUT:-${OVERLAY_ROOT}/kspace_regularized_expPDF_overlay_${TAIL_MODE}_${TAG}}"
CENTRAL_GRID_OUT="${CENTRAL_GRID_OUT:-plots/v23a_central_bspace_${TAG}}"
PLOT_OUT="${PLOT_OUT:-plots/v23a_traditional_kspace_TMDPDF_${FLAVOR}_x0p10_Q4.pdf}"

# ---- Checks ------------------------------------------------------------------
need_file() {
  [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }
}

need_file "v23/tools/make_v23a_pdf_overlay_plan_from_runs.py"
need_file "v23/tools/construct_v23a_data_pdf_bspace_tmd_bands_v2.py"
need_file "v23/tools/construct_v23a_regularized_kspace_tmd.py"
need_file "v23/tools/plot_v23a_traditional_kspace_tmd.py"
need_file "v22/tools/construct_v22_scheme_tmd_grid.py"
need_file "${BACKEND}"
need_file "${CENTRAL_RUN}"

mkdir -p "${OVERLAY_ROOT}" plots

echo
echo "=== v23a kT paper plot: ${FLAVOR}, x=${XVAL}, Q=${QVAL} GeV ==="
echo "ROOT:          ${ROOT}"
echo "overlay root:  ${OVERLAY_ROOT}"
echo "plan:          ${PLAN}"
echo "bspace out:    ${BSPACE_OUT}"
echo "kspace out:    ${KSPACE_OUT}"
echo "central grid:  ${CENTRAL_GRID_OUT}"
echo "plot out:      ${PLOT_OUT}"

# ---- Step 1: make overlay plan if needed -------------------------------------
if [[ ! -f "${PLAN}" ]]; then
  echo
  echo "=== Creating exp+PDF overlay plan ==="
  PYTHONPATH=. python3 v23/tools/make_v23a_pdf_overlay_plan_from_runs.py \
    --run-glob "${RUN_GLOB}" \
    --pdf-set "${PDF_SET}" \
    --pdf-members "${PDF_MEMBERS}" \
    --member-strategy cycle \
    --out-root "${OVERLAY_ROOT}" \
    --out "${PLAN}"
fi

# ---- Optional rebuild cleanup -------------------------------------------------
if [[ "${REBUILD}" == "1" ]]; then
  rm -rf "${BSPACE_OUT}" "${KSPACE_OUT}" "${CENTRAL_GRID_OUT}"
fi

# ---- Step 2: construct b-space exp+PDF overlay TMD grid at Q=4 ---------------
if [[ ! -f "${BSPACE_OUT}/v23a_dataPDF_tmd_replica_bspace_long.csv" ]]; then
  echo
  echo "=== Constructing b-space exp+PDF overlay grid for ${FLAVOR}, x=${XVAL}, Q=${QVAL} ==="
  PYTHONPATH=. python3 v23/tools/construct_v23a_data_pdf_bspace_tmd_bands_v2.py \
    --plan "${PLAN}" \
    --out "${BSPACE_OUT}" \
    --backend-script "${BACKEND}" \
    --pdf-set "${PDF_SET}" \
    --resum-order n3llp \
    --pids "${PID}" \
    --x-values "${XVAL}" \
    --Q-values "${QVAL}" \
    --b-min 0 \
    --b-max "${BMAX}" \
    --n-b "${NB}" \
    --pdf-member-source plan
else
  echo
  echo "=== Reusing existing b-space overlay grid: ${BSPACE_OUT} ==="
fi

# ---- Step 3: construct regularized kT transform ------------------------------
if [[ ! -f "${KSPACE_OUT}/v23a_regularized_kspace_bands.csv" ]]; then
  echo
  echo "=== Constructing regularized kT transform (${TAIL_MODE}) ==="
  PYTHONPATH=. python3 v23/tools/construct_v23a_regularized_kspace_tmd.py \
    --bspace-long "${BSPACE_OUT}/v23a_dataPDF_tmd_replica_bspace_long.csv" \
    --out "${KSPACE_OUT}" \
    --quantities ftilde \
    --tail-mode "${TAIL_MODE}" \
    --b-transform-max "${BTRANS_MAX}" \
    --n-b-transform "${NBTRANS}" \
    --k-max "${KMAX}" \
    --n-k "${NKT}"
else
  echo
  echo "=== Reusing existing kT transform: ${KSPACE_OUT} ==="
fi

# ---- Step 4: central PDF0 b-space grid at Q=4, for dashed central line --------
if [[ ! -f "${CENTRAL_GRID_OUT}/v22_scheme_tmd_bspace_long.csv" ]]; then
  echo
  echo "=== Constructing central PDF0 b-space grid for dashed central curve ==="
  PYTHONPATH=. python3 v22/tools/construct_v22_scheme_tmd_grid.py \
    --run "${CENTRAL_RUN}" \
    --backend-script "${BACKEND}" \
    --pdf-set "${PDF_SET}" \
    --pdf-member 0 \
    --resum-order n3llp \
    --pids "${PID}" \
    --x-values "${XVAL}" \
    --Q-values "${QVAL}" \
    --b-min 0 \
    --b-max "${BMAX}" \
    --n-b "${NB}" \
    --out "${CENTRAL_GRID_OUT}"
else
  echo
  echo "=== Reusing existing central grid: ${CENTRAL_GRID_OUT} ==="
fi

# ---- Step 5: paper-style plot -------------------------------------------------
echo
echo "=== Plotting traditional kT-space TMDPDF figure ==="
PYTHONPATH=. python3 v23/tools/plot_v23a_traditional_kspace_tmd.py \
  --band-dir "${KSPACE_OUT}" \
  --central-bspace-grid "${CENTRAL_GRID_OUT}/v22_scheme_tmd_bspace_long.csv" \
  --quantity ftilde \
  --flavor "${FLAVOR}" \
  --pid "${PID}" \
  --x "${XVAL}" \
  --Q "${QVAL}" \
  --k-max "${KMAX}" \
  --title "TMD PDFs" \
  --label "v23a FT-DY" \
  --band-label "68% exp+PDF overlay" \
  --central-label "central fit, PDF0" \
  --thin-replicas 0 \
  --show-zero \
  --out "${PLOT_OUT}"

echo
echo "=== Done ==="
echo "PDF:  ${PLOT_OUT}"
echo "PNG:  ${PLOT_OUT%.pdf}.png"
echo "CSV:  ${PLOT_OUT%.pdf}.curve.csv"
echo "JSON: ${PLOT_OUT%.pdf}.diagnostics.json"
