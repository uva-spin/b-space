#!/usr/bin/env bash
set -euo pipefail

# Validate that external cached W/Y grids plus CUDA reproduce the v22 central
# refit predictions at zero learning rate.
#
# Run after workflows/v22/utilities/export_v22_backend_cache.sh.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v22_full_backend_cache_export/backend_cache/cache_paths.env}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v22_full_backend_central_refit_stage1_s303}"
OUT="${OUT:-${ROOT}/outputs/v22_cache_cuda_smoketest_s303}"
DEVICE="${DEVICE:-cuda}"

die() {
  echo "$*" >&2
  exit 1
}

for path in "${CACHE_ENV}" "${TRAIN}" "${CENTRAL_RUN}/model_state.pt" "${CENTRAL_RUN}/dataset_norms.csv"; do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

# shellcheck disable=SC1090
source "${CACHE_ENV}"

for path in "${W_GRID}" "${Y_GRID}"; do
  [[ -f "${path}" ]] || die "Missing cache file: ${path}"
done

if [[ -e "${OUT}" ]]; then
  die "Refusing to overwrite existing output: ${OUT}"
fi

echo "Trainer:     ${TRAIN}"
echo "Central run: ${CENTRAL_RUN}"
echo "W grid:      ${W_GRID}"
echo "Y grid:      ${Y_GRID}"
echo "Device:      ${DEVICE}"
echo "Out:         ${OUT}"

python3 "${TRAIN}" \
  --data-dir "${ROOT}/Data" \
  --datasets E288_200 E288_300 E288_400 E605 \
  --mode matched \
  --qT-max-over-Q 0.5 \
  --tmd-qT-max-over-Q 0.2 \
  --w-backend external \
  --w-grid "${W_GRID}" \
  --y-grid "${Y_GRID}" \
  --pdf-set NNPDF40_nnlo_as_01180 \
  --pdf-member 0 \
  --resum-order n3llp \
  --match-order nlo \
  --target-mode nuclear_isospin \
  --prefactor-scheme oldA_to_CS \
  --y-mode zero \
  --n-b 160 \
  --b-min 1e-4 \
  --b-max 8 \
  --b-star-max 1.5 \
  --q0 2.0 \
  --epochs 1 \
  --batch-size 10000 \
  --lr 0 \
  --dataset-norm-lr 0 \
  --weight-decay 0 \
  --grad-clip 10 \
  --patience 1 \
  --min-delta 1e-12 \
  --dtype float32 \
  --device "${DEVICE}" \
  --num-threads 4 \
  --log-every 1 \
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
  --init-model-state "${CENTRAL_RUN}/model_state.pt" \
  --init-dataset-norms-from "${CENTRAL_RUN}/dataset_norms.csv" \
  --seed 303 \
  --out "${OUT}"

python3 v22/tools/audit_v22_cache_cuda_smoketest.py \
  --reference-run "${CENTRAL_RUN}" \
  --test-run "${OUT}" \
  --out "${OUT}/cache_cuda_audit"

echo
echo "Cache/CUDA smoke test complete."
