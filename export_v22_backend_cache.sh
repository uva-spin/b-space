#!/usr/bin/env bash
set -euo pipefail

# Build the v22 perturbative W/Y cache once.
#
# This is expected to take about as long as one old "W grid" build, but after
# that all replica fits can use --w-backend external with --w-grid/--y-grid.
#
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py}"
BACKEND="${BACKEND:-${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data}"
OUT="${OUT:-${ROOT}/outputs/v22_full_backend_cache_export}"
CACHE_TAG="${CACHE_TAG:-v22full_n3llp_nloQ96_b160_qToQ05}"

die() {
  echo "$*" >&2
  exit 1
}

for path in "${TRAIN}" "${BACKEND}" "${DATA_DIR}"; do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

if [[ -e "${OUT}" ]]; then
  die "Refusing to overwrite existing output: ${OUT}"
fi

echo "Trainer:   ${TRAIN}"
echo "Backend:   ${BACKEND}"
echo "Data:      ${DATA_DIR}"
echo "Out:       ${OUT}"
echo "Cache tag: ${CACHE_TAG}"

python3 "${TRAIN}" \
  --backend-script "${BACKEND}" \
  --data-dir "${DATA_DIR}" \
  --datasets E288_200 E288_300 E288_400 E605 \
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
  --norm-source paper \
  --ptp-source paper \
  --dtype float32 \
  --device cpu \
  --num-threads 4 \
  --quiet-backend \
  --cache-backend-grids \
  --cache-tag "${CACHE_TAG}" \
  --check-only \
  --out "${OUT}"

python3 v22/tools/resolve_v22_backend_cache.py \
  --cache-root "${OUT}/backend_cache" \
  --out-env "${OUT}/backend_cache/cache_paths.env" \
  --out-json "${OUT}/backend_cache/cache_paths.json"

echo
echo "Cache export complete."
echo "Source the cache paths with:"
echo "  source '${OUT}/backend_cache/cache_paths.env'"
echo
cat "${OUT}/backend_cache/cache_paths.env"
