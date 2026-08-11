#!/usr/bin/env bash
set -euo pipefail

# Postprocess/audit v23a lambda=3 replica pilot.
#
# Run from ~/work/bT-TMD after append_v23a_lambda3_normpriors15_cached_cuda_replicas.sh.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_pilot_v23a_lambda3_normpriors15_cached_cuda}"
RUN_PREFIX="${RUN_PREFIX:-v23a_lambda3_normpriors15_cached_cuda}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_15pct_s303}"
CENTRAL_GRID="${CENTRAL_GRID:-${ROOT}/plots/v23a_fixed_target_lowQ_normpriors15_central_exactx/v22_scheme_tmd_bspace_long.csv}"
BAND_DIR="${BAND_DIR:-${OUT_ROOT}/tmd_bspace_bands_exactx}"
MIN_REPLICAS="${MIN_REPLICAS:-10}"

die() {
  echo "$*" >&2
  exit 1
}

for path in \
  "${CENTRAL_RUN}/metrics.json" \
  "${CENTRAL_RUN}/predictions.csv" \
  "${CENTRAL_GRID}" \
  "${ROOT}/v22/tools/audit_v22_replica_pilot_basic.py" \
  "${ROOT}/v22/tools/construct_v22_bspace_tmd_bands_from_replicas.py" \
  "${ROOT}/v22/tools/audit_v22_bspace_tmd_bands.py" \
  "${ROOT}/v22/tools/audit_v22_lambda3_ensemble_convergence_q95.py"
do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

RUN_GLOB="${OUT_ROOT}/outputs/${RUN_PREFIX}_s*"

N_RUNS="$(find "${OUT_ROOT}/outputs" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PREFIX}_s*" 2>/dev/null | wc -l | tr -d ' ')"
echo "Found ${N_RUNS} replica output dirs matching ${RUN_GLOB}"
if [[ "${N_RUNS}" -lt "${MIN_REPLICAS}" ]]; then
  die "Need at least ${MIN_REPLICAS} replicas before postprocessing."
fi

echo
echo "=== Basic replica audit ==="
PYTHONPATH=. python3 v22/tools/audit_v22_replica_pilot_basic.py \
  --glob "${RUN_GLOB}" \
  --central-run "${CENTRAL_RUN}" \
  --out "${OUT_ROOT}/audit_basic"

echo
echo "=== Constructing b-space TMD bands ==="
rm -rf "${BAND_DIR}"
PYTHONPATH=. python3 v22/tools/construct_v22_bspace_tmd_bands_from_replicas.py \
  --central-grid "${CENTRAL_GRID}" \
  --replica-glob "${RUN_GLOB}" \
  --out "${BAND_DIR}"

echo
echo "=== Auditing b-space TMD bands ==="
PYTHONPATH=. python3 v22/tools/audit_v22_bspace_tmd_bands.py \
  --band-dir "${BAND_DIR}" \
  --central-grid "${CENTRAL_GRID}" \
  --exact-x-values 0.10 0.20 0.30 0.50 \
  --min-useful-band-p90 0.02 \
  --out "${BAND_DIR}/audit"

echo
echo "=== q95 ensemble convergence audit ==="
PYTHONPATH=. python3 v22/tools/audit_v22_lambda3_ensemble_convergence_q95.py \
  --run-glob "${RUN_GLOB}" \
  --band-dir "${BAND_DIR}" \
  --band-audit-dir "${BAND_DIR}/audit" \
  --min-replicas "${MIN_REPLICAS}" \
  --out "${OUT_ROOT}/audit_convergence_q95"

echo
echo "Postprocessing complete."
echo "Inspect:"
echo "  ${OUT_ROOT}/audit_basic"
echo "  ${BAND_DIR}/audit"
echo "  ${OUT_ROOT}/audit_convergence_q95"
