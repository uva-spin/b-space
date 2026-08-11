#!/usr/bin/env bash
set -euo pipefail

# Construct and audit the v23a central b-space TMD grid.
#
# Run from ~/work/bT-TMD after the v23a central refit passes.
#
# This is b-space only. kT-space remains diagnostic and is not produced here.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

RUN="${RUN:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_15pct_s303}"
BACKEND="${BACKEND:-${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py}"
OUT_GRID="${OUT_GRID:-${ROOT}/plots/v23a_fixed_target_lowQ_normpriors15_central_exactx}"
AUDIT_OUT="${AUDIT_OUT:-${ROOT}/v23/outputs/v23a_fixed_target_lowQ_normpriors15_central_tmd_audit}"
REF_GRID="${REF_GRID:-${ROOT}/plots/v22_scheme_tmd_stage1_s303_bandgrid_exactx/v22_scheme_tmd_bspace_long.csv}"

die() {
  echo "$*" >&2
  exit 1
}

for path in \
  "${RUN}/model_state.pt" \
  "${BACKEND}" \
  "${ROOT}/v22/tools/construct_v22_scheme_tmd_grid.py" \
  "${ROOT}/v23/tools/audit_v23a_central_tmd_grid.py"
do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

if [[ -e "${OUT_GRID}" ]]; then
  die "Refusing to overwrite existing grid dir: ${OUT_GRID}"
fi

mkdir -p "$(dirname "${OUT_GRID}")" "$(dirname "${AUDIT_OUT}")"

echo "Constructing v23a central exact-x b-space TMD grid..."
PYTHONPATH=. python3 v22/tools/construct_v22_scheme_tmd_grid.py \
  --run "${RUN}" \
  --backend-script "${BACKEND}" \
  --pdf-set NNPDF40_nnlo_as_01180 \
  --pdf-member 0 \
  --resum-order n3llp \
  --pids 2 1 -2 -1 \
  --x-values 0.10 0.20 0.30 0.50 \
  --Q-values 5 10 \
  --b-min 0 \
  --b-max 8 \
  --n-b 321 \
  --out "${OUT_GRID}"

GRID="${OUT_GRID}/v22_scheme_tmd_bspace_long.csv"
[[ -f "${GRID}" ]] || die "Expected grid not found: ${GRID}"

echo
echo "Auditing v23a central b-space TMD grid..."
if [[ -f "${REF_GRID}" ]]; then
  REF_ARG=(--reference-grid "${REF_GRID}")
else
  echo "Reference v22 grid not found, skipping comparison: ${REF_GRID}"
  REF_ARG=()
fi

PYTHONPATH=. python3 v23/tools/audit_v23a_central_tmd_grid.py \
  --grid "${GRID}" \
  "${REF_ARG[@]}" \
  --exact-x-values 0.10 0.20 0.30 0.50 \
  --out "${AUDIT_OUT}"

echo
echo "Done."
echo "Grid:  ${OUT_GRID}"
echo "Audit: ${AUDIT_OUT}"
