#!/usr/bin/env bash
set -euo pipefail

# Build the missing Q=7.5 GeV exp+PDF-overlay TMD reconstruction and make
# separate paper-style 3D u- and d-quark plots in both b_T and k_T space.
#
# This does NOT retrain the 50 DNN replicas and does NOT rebuild cross-section
# W/Y caches. It:
#   1. reuses the existing experimental-replica model checkpoints;
#   2. reconstructs u/d b-space TMDs at Q=7.5 with the planned PDF members;
#   3. constructs expb2/expb/taper regularized k-space transforms;
#   4. audits regularization-mode stability at Q=7.5;
#   5. creates separate beautified u and d figures with shared color scales.
#
# Run from the bT-TMD repository root:
#
#   chmod +x build_v23a_q7p5_3d_paper_plots.sh
#   ./build_v23a_q7p5_3d_paper_plots.sh
#
# To erase and rebuild the Q=7.5-only products:
#
#   REBUILD=1 ./build_v23a_q7p5_3d_paper_plots.sh

ROOT="${ROOT:-$(pwd)}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

QVAL="${QVAL:-7.5}"
QTAG="${QTAG:-Q7p5}"

PDF_SET="${PDF_SET:-NNPDF40_nnlo_as_01180}"
BACKEND="${BACKEND:-v22/backends/bt_internal_css_backend_v22_full.py}"
RESUM_ORDER="${RESUM_ORDER:-n3llp}"

RUN_GLOB="${RUN_GLOB:-replica_pilot_v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda/outputs/v23a_lambda3_normpriors15_p2p5_E772_E288400_cached_cuda_s*}"
OVERLAY_ROOT="${OVERLAY_ROOT:-replica_v23a_expPDF_overlay_lambda3_normpriors15_p2p5_50rep}"
PLAN="${PLAN:-${OVERLAY_ROOT}/replica_plan.csv}"
PDF_MEMBERS="${PDF_MEMBERS:-1-50}"

# Use a new additive directory. Do not overwrite the frozen Q={5,10} product.
BSPACE_OUT="${BSPACE_OUT:-${OVERLAY_ROOT}/tmd_bspace_bands_expPDF_overlay_${QTAG}_ud}"
KSPACE_EXPB2="${KSPACE_EXPB2:-${OVERLAY_ROOT}/kspace_regularized_expPDF_overlay_expb2_${QTAG}_ud}"
KSPACE_EXPB="${KSPACE_EXPB:-${OVERLAY_ROOT}/kspace_regularized_expPDF_overlay_expb_${QTAG}_ud}"
KSPACE_TAPER="${KSPACE_TAPER:-${OVERLAY_ROOT}/kspace_regularized_expPDF_overlay_taper_${QTAG}_ud}"
KSPACE_COMPARE="${KSPACE_COMPARE:-${OVERLAY_ROOT}/kspace_regularized_comparison_${QTAG}_ud}"
PLOT_DIR="${PLOT_DIR:-plots/v23a_paper_${QTAG}_3D}"

BMAX="${BMAX:-8}"
NB="${NB:-321}"
B_PLOT_MAX="${B_PLOT_MAX:-4}"

KMAX_TRANSFORM="${KMAX_TRANSFORM:-4}"
K_PLOT_MAX="${K_PLOT_MAX:-3}"
NK="${NK:-401}"
BTRANS_MAX="${BTRANS_MAX:-24}"
NBTRANS="${NBTRANS:-6001}"

NX_PLOT="${NX_PLOT:-61}"
NB_PLOT="${NB_PLOT:-241}"
NK_PLOT="${NK_PLOT:-241}"
X_RIDGES="${X_RIDGES:-13}"
CROSS_LINES="${CROSS_LINES:-11}"
VIEW_ELEV="${VIEW_ELEV:-27}"
VIEW_AZIM="${VIEW_AZIM:--56}"

REBUILD="${REBUILD:-0}"

need_file() {
  [[ -e "$1" ]] || {
    echo "Missing required path: $1" >&2
    exit 1
  }
}

need_file "v23/tools/make_v23a_pdf_overlay_plan_from_runs.py"
need_file "v23/tools/construct_v23a_data_pdf_bspace_tmd_bands_v2.py"
need_file "v23/tools/construct_v23a_regularized_kspace_tmd.py"
need_file "v23/tools/compare_v23a_regularized_kspace_modes.py"
need_file "v23/tools/plot_v23a_paper_bspace_3d_tmd.py"
need_file "v23/tools/plot_v23a_paper_kspace_3d_tmd.py"
need_file "${BACKEND}"

mkdir -p "${OVERLAY_ROOT}" "${PLOT_DIR}"

echo
echo "=== Q=${QVAL} GeV fixed-target paper-surface build ==="
echo "Plan:              ${PLAN}"
echo "b_T overlay:       ${BSPACE_OUT}"
echo "k_T expb2:         ${KSPACE_EXPB2}"
echo "k_T comparison:    ${KSPACE_COMPARE}"
echo "plots:             ${PLOT_DIR}"

if [[ "${REBUILD}" == "1" ]]; then
  echo
  echo "Removing prior Q=${QVAL} products..."
  rm -rf \
    "${BSPACE_OUT}" \
    "${KSPACE_EXPB2}" \
    "${KSPACE_EXPB}" \
    "${KSPACE_TAPER}" \
    "${KSPACE_COMPARE}"
  rm -f "${PLOT_DIR}"/*"${QTAG}"*
fi

# -----------------------------------------------------------------------------
# 1. Overlay plan
# -----------------------------------------------------------------------------
if [[ ! -f "${PLAN}" ]]; then
  echo
  echo "=== Creating the exp+PDF overlay plan ==="
  PYTHONPATH=. python3 v23/tools/make_v23a_pdf_overlay_plan_from_runs.py \
    --run-glob "${RUN_GLOB}" \
    --pdf-set "${PDF_SET}" \
    --pdf-members "${PDF_MEMBERS}" \
    --member-strategy cycle \
    --out-root "${OVERLAY_ROOT}" \
    --out "${PLAN}"
else
  echo
  echo "Reusing overlay plan: ${PLAN}"
fi

# -----------------------------------------------------------------------------
# 2. Reconstruct only u and d at Q=7.5, exact x={0.1,0.2,0.3,0.5}
# -----------------------------------------------------------------------------
BSPACE_LONG="${BSPACE_OUT}/v23a_dataPDF_tmd_replica_bspace_long.csv"
BSPACE_BANDS="${BSPACE_OUT}/v23a_dataPDF_tmd_replica_bspace_bands.csv"

if [[ ! -f "${BSPACE_LONG}" || ! -f "${BSPACE_BANDS}" ]]; then
  echo
  echo "=== Constructing Q=${QVAL} exp+PDF-overlay b_T ensemble (u,d only) ==="
  PYTHONPATH=. python3 v23/tools/construct_v23a_data_pdf_bspace_tmd_bands_v2.py \
    --plan "${PLAN}" \
    --out "${BSPACE_OUT}" \
    --backend-script "${BACKEND}" \
    --pdf-set "${PDF_SET}" \
    --resum-order "${RESUM_ORDER}" \
    --pids 2 1 \
    --x-values 0.10 0.20 0.30 0.50 \
    --Q-values "${QVAL}" \
    --b-min 0 \
    --b-max "${BMAX}" \
    --n-b "${NB}" \
    --pdf-member-source plan
else
  echo
  echo "Reusing Q=${QVAL} b_T ensemble: ${BSPACE_OUT}"
fi

# Sanity-check that the requested scale is genuinely present.
python3 - "${BSPACE_BANDS}" "${QVAL}" <<'PY'
import sys
import numpy as np
import pandas as pd

p, q = sys.argv[1], float(sys.argv[2])
df = pd.read_csv(p)
vals = sorted(pd.to_numeric(df["Q"], errors="coerce").dropna().unique().tolist())
print("b-space Q values:", vals)
if not any(np.isclose(vals, q, rtol=0, atol=1e-10)):
    raise SystemExit(f"Q={q} is missing from {p}")
PY

# -----------------------------------------------------------------------------
# 3. Regularized k_T transforms at the exact same Q
# -----------------------------------------------------------------------------
build_kspace() {
  local mode="$1"
  local out="$2"

  if [[ -f "${out}/v23a_regularized_kspace_bands.csv" ]]; then
    echo "Reusing ${mode} k_T transform: ${out}"
    return
  fi

  echo
  echo "=== Constructing Q=${QVAL} regularized k_T transform: ${mode} ==="
  PYTHONPATH=. python3 v23/tools/construct_v23a_regularized_kspace_tmd.py \
    --bspace-long "${BSPACE_LONG}" \
    --out "${out}" \
    --quantities ftilde x_ftilde \
    --tail-mode "${mode}" \
    --b-transform-max "${BTRANS_MAX}" \
    --n-b-transform "${NBTRANS}" \
    --k-max "${KMAX_TRANSFORM}" \
    --n-k "${NK}"
}

build_kspace expb2 "${KSPACE_EXPB2}"
build_kspace expb  "${KSPACE_EXPB}"
build_kspace taper "${KSPACE_TAPER}"

echo
echo "=== Comparing Q=${QVAL} k_T regularization modes ==="
rm -rf "${KSPACE_COMPARE}"
PYTHONPATH=. python3 v23/tools/compare_v23a_regularized_kspace_modes.py \
  --dirs \
    "${KSPACE_EXPB2}" \
    "${KSPACE_EXPB}" \
    "${KSPACE_TAPER}" \
  --reference expb2 \
  --out "${KSPACE_COMPARE}"

# Verify Q=7.5 is in the transform bands.
python3 - "${KSPACE_EXPB2}/v23a_regularized_kspace_bands.csv" "${QVAL}" <<'PY'
import sys
import numpy as np
import pandas as pd

p, q = sys.argv[1], float(sys.argv[2])
df = pd.read_csv(p)
vals = sorted(pd.to_numeric(df["Q"], errors="coerce").dropna().unique().tolist())
print("k-space Q values:", vals)
if not any(np.isclose(vals, q, rtol=0, atol=1e-10)):
    raise SystemExit(f"Q={q} is missing from {p}")
PY

# -----------------------------------------------------------------------------
# 4. First plotting pass: discover sensible uncertainty colorbar maxima
# -----------------------------------------------------------------------------
plot_bspace() {
  local flavor="$1"
  local vmax="${2:-}"
  local extra=()
  [[ -n "${vmax}" ]] && extra+=(--uncertainty-vmax "${vmax}")

  PYTHONPATH=. python3 v23/tools/plot_v23a_paper_bspace_3d_tmd.py \
    --band-dir "${BSPACE_OUT}" \
    --quantity x_ftilde \
    --flavors "${flavor}" \
    --Q "${QVAL}" \
    --b-max "${B_PLOT_MAX}" \
    --x-min 0.10 \
    --x-max 0.50 \
    --n-x "${NX_PLOT}" \
    --n-b "${NB_PLOT}" \
    --x-ridges "${X_RIDGES}" \
    --b-cross-lines "${CROSS_LINES}" \
    --view-elev "${VIEW_ELEV}" \
    --view-azim "${VIEW_AZIM}" \
    "${extra[@]}" \
    --out "${PLOT_DIR}/v23a_paper_bspace_3D_xftilde_${flavor}_${QTAG}.pdf"
}

plot_kspace() {
  local flavor="$1"
  local vmax="${2:-}"
  local extra=()
  [[ -n "${vmax}" ]] && extra+=(--uncertainty-vmax "${vmax}")

  PYTHONPATH=. python3 v23/tools/plot_v23a_paper_kspace_3d_tmd.py \
    --band-dir "${KSPACE_EXPB2}" \
    --quantity x_ftilde \
    --flavors "${flavor}" \
    --Q "${QVAL}" \
    --k-max "${K_PLOT_MAX}" \
    --x-min 0.10 \
    --x-max 0.50 \
    --n-x "${NX_PLOT}" \
    --n-k "${NK_PLOT}" \
    --x-ridges "${X_RIDGES}" \
    --k-cross-lines "${CROSS_LINES}" \
    --view-elev "${VIEW_ELEV}" \
    --view-azim "${VIEW_AZIM}" \
    "${extra[@]}" \
    --out "${PLOT_DIR}/v23a_paper_kspace_3D_xftilde_${flavor}_${QTAG}.pdf"
}

echo
echo "=== First-pass separate u/d plots ==="
plot_bspace u
plot_bspace d
plot_kspace u
plot_kspace d

# -----------------------------------------------------------------------------
# 5. Rerun each u/d pair with one shared uncertainty color scale
# -----------------------------------------------------------------------------
read -r BSPACE_VMAX KSPACE_VMAX < <(
python3 - "${PLOT_DIR}" "${QTAG}" <<'PY'
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
tag = sys.argv[2]

def vmax(prefix):
    vals = []
    for flavor in ("u", "d"):
        p = d / f"{prefix}_{flavor}_{tag}.diagnostics.json"
        obj = json.loads(p.read_text())
        vals.append(float(obj["uncertainty_colorbar_vmax_percent"]))
    return max(vals)

print(f"{vmax('v23a_paper_bspace_3D_xftilde'):g} "
      f"{vmax('v23a_paper_kspace_3D_xftilde'):g}")
PY
)

echo
echo "Shared b_T uncertainty scale: 0-${BSPACE_VMAX}%"
echo "Shared k_T uncertainty scale: 0-${KSPACE_VMAX}%"

echo
echo "=== Final separate u/d plots with shared scales ==="
plot_bspace u "${BSPACE_VMAX}"
plot_bspace d "${BSPACE_VMAX}"
plot_kspace u "${KSPACE_VMAX}"
plot_kspace d "${KSPACE_VMAX}"

echo
echo "=== Complete ==="
echo "b_T u: ${PLOT_DIR}/v23a_paper_bspace_3D_xftilde_u_${QTAG}.pdf"
echo "b_T d: ${PLOT_DIR}/v23a_paper_bspace_3D_xftilde_d_${QTAG}.pdf"
echo "k_T u: ${PLOT_DIR}/v23a_paper_kspace_3D_xftilde_u_${QTAG}.pdf"
echo "k_T d: ${PLOT_DIR}/v23a_paper_kspace_3D_xftilde_d_${QTAG}.pdf"
echo "regularization audit: ${KSPACE_COMPARE}/regularization_mode_comparison_summary.json"
