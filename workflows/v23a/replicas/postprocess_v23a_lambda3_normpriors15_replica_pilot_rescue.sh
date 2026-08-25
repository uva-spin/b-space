#!/usr/bin/env bash
set -u -o pipefail

# Rescue/continue postprocessing for the v23a lambda=3 normpriors15 replica pilot.
#
# The first postprocess script stopped after the "basic" audit because that
# audit is an all-replica/three-replica gate and returned nonzero.  This script
# keeps going so we can still build b-space bands and run q95 convergence
# diagnostics for the 10-rep pilot.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_pilot_v23a_lambda3_normpriors15_cached_cuda}"
RUN_PREFIX="${RUN_PREFIX:-v23a_lambda3_normpriors15_cached_cuda}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_15pct_s303}"
CENTRAL_GRID="${CENTRAL_GRID:-${ROOT}/plots/v23a_fixed_target_lowQ_normpriors15_central_exactx/v22_scheme_tmd_bspace_long.csv}"
BAND_DIR="${BAND_DIR:-${OUT_ROOT}/tmd_bspace_bands_exactx}"
MIN_REPLICAS="${MIN_REPLICAS:-10}"

RUN_GLOB="${OUT_ROOT}/outputs/${RUN_PREFIX}_s*"

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

N_RUNS="$(find "${OUT_ROOT}/outputs" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PREFIX}_s*" 2>/dev/null | wc -l | tr -d ' ')"
echo "Found ${N_RUNS} replica output dirs matching ${RUN_GLOB}"
if [[ "${N_RUNS}" -lt "${MIN_REPLICAS}" ]]; then
  die "Need at least ${MIN_REPLICAS} replicas before postprocessing."
fi

mkdir -p "${OUT_ROOT}/audit_basic" "${OUT_ROOT}/audit_convergence_q95"

echo
echo "=== Basic replica audit; nonzero exit is allowed for this pilot ==="
PYTHONPATH=. python3 v22/tools/audit_v22_replica_pilot_basic.py \
  --glob "${RUN_GLOB}" \
  --central-run "${CENTRAL_RUN}" \
  --out "${OUT_ROOT}/audit_basic"
BASIC_STATUS=$?
echo "basic audit exit status: ${BASIC_STATUS}"
echo "basic audit files:"
find "${OUT_ROOT}/audit_basic" -maxdepth 2 -type f -printf '  %p\n' | sort

echo
echo "=== Constructing b-space TMD bands ==="
rm -rf "${BAND_DIR}"
PYTHONPATH=. python3 v22/tools/construct_v22_bspace_tmd_bands_from_replicas.py \
  --central-grid "${CENTRAL_GRID}" \
  --replica-glob "${RUN_GLOB}" \
  --out "${BAND_DIR}"
BAND_BUILD_STATUS=$?
echo "band construction exit status: ${BAND_BUILD_STATUS}"
if [[ "${BAND_BUILD_STATUS}" -ne 0 ]]; then
  die "Band construction failed; inspect logs above."
fi

echo
echo "=== Auditing b-space TMD bands; nonzero exit is allowed so outputs remain inspectable ==="
PYTHONPATH=. python3 v22/tools/audit_v22_bspace_tmd_bands.py \
  --band-dir "${BAND_DIR}" \
  --central-grid "${CENTRAL_GRID}" \
  --exact-x-values 0.10 0.20 0.30 0.50 \
  --min-useful-band-p90 0.02 \
  --out "${BAND_DIR}/audit"
BAND_AUDIT_STATUS=$?
echo "band audit exit status: ${BAND_AUDIT_STATUS}"
echo "band audit files:"
find "${BAND_DIR}/audit" -maxdepth 2 -type f -printf '  %p\n' | sort

echo
echo "=== q95 ensemble convergence audit; nonzero exit is allowed so outputs remain inspectable ==="
PYTHONPATH=. python3 v22/tools/audit_v22_lambda3_ensemble_convergence_q95.py \
  --run-glob "${RUN_GLOB}" \
  --band-dir "${BAND_DIR}" \
  --band-audit-dir "${BAND_DIR}/audit" \
  --min-replicas "${MIN_REPLICAS}" \
  --out "${OUT_ROOT}/audit_convergence_q95"
Q95_STATUS=$?
echo "q95 audit exit status: ${Q95_STATUS}"
echo "q95 audit files:"
find "${OUT_ROOT}/audit_convergence_q95" -maxdepth 2 -type f -printf '  %p\n' | sort

echo
echo "=== Suggested inspection commands ==="
echo "find '${OUT_ROOT}/audit_basic' -maxdepth 2 -type f | sort"
echo "find '${BAND_DIR}/audit' -maxdepth 2 -type f | sort"
echo "find '${OUT_ROOT}/audit_convergence_q95' -maxdepth 2 -type f | sort"
echo
echo "Then cat the summary JSONs that actually exist. Common names are:"
echo "  ${BAND_DIR}/audit/bspace_band_audit_summary.json"
echo "  ${BAND_DIR}/audit/bspace_band_by_quantity.csv"
echo "  ${OUT_ROOT}/audit_convergence_q95/lambda3_ensemble_q95_summary.json"
echo
echo "Finished rescue postprocess."
echo "Statuses: basic=${BASIC_STATUS}, band_audit=${BAND_AUDIT_STATUS}, q95=${Q95_STATUS}"

# Exit zero if the technical band construction succeeded.  The diagnostic
# pass/fail decisions are in the output summaries.
exit 0
