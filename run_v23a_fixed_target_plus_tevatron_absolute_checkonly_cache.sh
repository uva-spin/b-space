#!/usr/bin/env bash
set -euo pipefail

# Check-only backend cache for v23a fixed-target plus fit-ready absolute
# Tevatron CDF/D0 rows.
#
# This builds W/Y grids only.  The Tevatron absolute subset uses confirmed
# publication units, released table errors as diagonal point-to-point errors,
# explicit normalization nuisances, and a pbar-p-aware v22 backend wrapper.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/dustin/miniforge3/envs/pdf-fit/bin/python}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py}"
BACKEND="${BACKEND:-${ROOT}/v23/backends/bt_internal_css_backend_v22_tevatron.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready}"
OUT="${OUT:-${ROOT}/outputs/v23a_fixed_target_plus_tevatron_absolute_fit_ready_ewpty_checkonly_cache}"
CACHE_TAG="${CACHE_TAG:-v23a_fixedtarget_plus_tevatron_absolute_fitready_v22tev_ewpty_n3llp_nloQ96_b160}"
DEVICE="${DEVICE:-cpu}"
WGRID_WORKERS="${WGRID_WORKERS:-12}"
export TEVATRON_WGRID_WORKERS="${TEVATRON_WGRID_WORKERS:-${WGRID_WORKERS}}"
Y_WORKERS="${Y_WORKERS:-${WGRID_WORKERS}}"
export TEVATRON_Y_WORKERS="${TEVATRON_Y_WORKERS:-${Y_WORKERS}}"

die() {
  echo "$*" >&2
  exit 1
}

for path in "${PYTHON}" "${TRAIN}" "${BACKEND}" "${DATA_DIR}"; do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

if [[ -e "${OUT}" ]]; then
  die "Refusing to overwrite existing output: ${OUT}"
fi

echo "Python:    ${PYTHON}"
echo "Trainer:   ${TRAIN}"
echo "Backend:   ${BACKEND}"
echo "Data dir:  ${DATA_DIR}"
echo "Output:    ${OUT}"
echo "Device:    ${DEVICE}"
echo "W workers: ${TEVATRON_WGRID_WORKERS}"
echo "Y workers: ${TEVATRON_Y_WORKERS}"

"${PYTHON}" "${TRAIN}" \
  --backend-script "${BACKEND}" \
  --data-dir "${DATA_DIR}" \
  --datasets E288_200 E288_300 E288_400 E605 E772 CDF_RUN_1 CDF_RUN_2 D0_RUN_1 \
  --mode matched \
  --qT-max-over-Q 0.5 \
  --tmd-qT-max-over-Q 0.2 \
  --w-backend internal_css \
  --pdf-set NNPDF40_nnlo_as_01180 \
  --pdf-member 0 \
  --resum-order n3llp \
  --match-order nlo \
  --nlo-singular-mode asymptotic_damped \
  --nlo-singular-rsub 0.10 \
  --nlo-singular-power 2 \
  --nlo-singular-damp-kind exp \
  --nlo-real-quad 96 \
  --nlo-real-tail-repair mcfm_logistic \
  --nlo-real-tail-r0 0.530 \
  --nlo-real-tail-width 0.008 \
  --nlo-real-tail-rinf 0.180 \
  --target-mode nuclear_isospin \
  --prefactor-scheme oldA_to_CS \
  --y-mode zero \
  --n-b 160 \
  --b-min 1e-4 \
  --b-max 8 \
  --b-star-max 1.5 \
  --mu-min 1.3 \
  --n-sudakov-quad 32 \
  --q0 2.0 \
  --np-width 48 \
  --np-cond-width 32 \
  --np-blocks 3 \
  --np-a0 0.05 \
  --np-min-a 0 \
  --np-a-mode positive \
  --np-shape-mode monotone \
  --fnp-exponent-clip 40 \
  --np-a-smooth-sigma 0.45 \
  --np-a-tail-amp 0.019 \
  --np-a-tail-b0 3.5 \
  --np-a-tail-width 0.25 \
  --soft-q-evolution none \
  --fit-dataset-norms \
  --lambda-dataset-norm 1.0 \
  --norm-source csv \
  --ptp-source csv \
  --dtype float32 \
  --device "${DEVICE}" \
  --num-threads 4 \
  --cache-backend-grids \
  --cache-tag "${CACHE_TAG}" \
  --check-only \
  --out "${OUT}"

if [[ -f v22/tools/resolve_v22_backend_cache.py ]]; then
  "${PYTHON}" v22/tools/resolve_v22_backend_cache.py \
    --cache-root "${OUT}/backend_cache" \
    --out-env "${OUT}/backend_cache/cache_paths.env" \
    --out-json "${OUT}/backend_cache/cache_paths.json"
else
  echo "WARNING: v22/tools/resolve_v22_backend_cache.py not found; cache was written but paths were not resolved."
fi

echo
echo "Fixed-target plus Tevatron absolute fit-ready check-only cache build complete."
echo "Inspect:"
echo "  ${OUT}/metrics.json"
echo "  ${OUT}/backend_cache"
echo "  ${OUT}/backend_y_summary_by_dataset.csv"
