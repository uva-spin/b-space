#!/usr/bin/env bash
set -euo pipefail

# v23a central refit using explicit CSV normalization priors.
#
# Before running, create DATA_DIR with make_v23a_explicit_csv_norm_prior_dir.py.
#
# Example:
#   DATA_DIR=$PWD/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors_trial \
#   OUT=$PWD/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_trial_s303 \
#   DEVICE=cuda ./workflows/v23a/runs/run_v23a_fixed_target_corrected_central_refit_explicit_csvnorm.sh

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py}"
BACKEND="${BACKEND:-${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to an explicit-prior data directory}"
CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_checkonly_cache/backend_cache/cache_paths.env}"
INIT_STATE="${INIT_STATE:-${ROOT}/outputs/v22_full_backend_central_refit_stage1_s303/model_state.pt}"
OUT="${OUT:?Set OUT for this run}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-303}"

die() {
  echo "$*" >&2
  exit 1
}

for path in "${TRAIN}" "${BACKEND}" "${DATA_DIR}" "${CACHE_ENV}" "${INIT_STATE}"; do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

# shellcheck disable=SC1090
source "${CACHE_ENV}"

for path in "${W_GRID}" "${Y_GRID}"; do
  [[ -f "${path}" ]] || die "Missing cache file from CACHE_ENV: ${path}"
done

if [[ -e "${OUT}" ]]; then
  die "Refusing to overwrite existing output: ${OUT}"
fi

echo "Trainer:    ${TRAIN}"
echo "Data dir:   ${DATA_DIR}"
echo "W grid:     ${W_GRID}"
echo "Y grid:     ${Y_GRID}"
echo "Init state: ${INIT_STATE}"
echo "Device:     ${DEVICE}"
echo "Output:     ${OUT}"
echo "Norm/P2P:   csv"

python3 "${TRAIN}" \
  --backend-script "${BACKEND}" \
  --data-dir "${DATA_DIR}" \
  --datasets E288_200 E288_300 E288_400 E605 E772 \
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
  --epochs 1800 \
  --batch-size 10000 \
  --lr 2e-5 \
  --weight-decay 0 \
  --grad-clip 10 \
  --patience 300 \
  --min-delta 1e-7 \
  --dtype float32 \
  --device "${DEVICE}" \
  --num-threads 4 \
  --log-every 100 \
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
  --init-model-state "${INIT_STATE}" \
  --seed "${SEED}" \
  --out "${OUT}"
