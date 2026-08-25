#!/usr/bin/env bash
set -euo pipefail

# Production central refit for the validated v23a fixed-target + Tevatron core set.
#
# Included Tevatron datasets:
#   CDF_RUN_1, CDF_RUN_2, D0_RUN_1
#
# Excluded for this production central:
#   D0_RUN_2   - normalized spectrum; needs normalized-theory observable
#   D0_RUN_2N  - normalized spectrum; needs normalized-theory observable
#   qT/Q > 0.10 collider/fixed-target rows - outside the currently validated
#             low-qT TMD extraction domain; high-qT collider rows need
#             independent finite-tail/Y benchmarking

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/dustin/miniforge3/envs/pdf-fit/bin/python}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py}"
BACKEND="${BACKEND:-${ROOT}/v23/backends/bt_internal_css_backend_v22_tevatron.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready}"
DATASETS="${DATASETS:-E288_200 E288_300 E288_400 E605 E772 CDF_RUN_1 CDF_RUN_2 D0_RUN_1}"
CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v23a_fixed_target_plus_tevatron_absolute_fit_ready_ewpty_checkonly_cache/backend_cache/cache_paths.env}"
INIT_STATE="${INIT_STATE:-${ROOT}/outputs/v23a_fixed_target_plus_tevatron_absolute_fit_ready_ewpty_core_no_CDF_RUN_2_central_refit_s303/model_state.pt}"
OUT="${OUT:-${ROOT}/outputs/v23a_tevatron_lowqt010_allchecked_tailpass_central_s303}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-303}"
EPOCHS="${EPOCHS:-5000}"
LR="${LR:-1e-5}"
QTMQ="${QTMQ:-0.10}"
NP_A_TAIL_AMP="${NP_A_TAIL_AMP:-0.08}"
LAMBDA_FNP_TAIL="${LAMBDA_FNP_TAIL:-0}"
FNP_TAIL_BMIN="${FNP_TAIL_BMIN:-6.0}"
FNP_TAIL_TARGET="${FNP_TAIL_TARGET:-0.35}"

die() {
  echo "$*" >&2
  exit 1
}

for path in "${PYTHON}" "${TRAIN}" "${BACKEND}" "${DATA_DIR}" "${CACHE_ENV}" "${INIT_STATE}"; do
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

echo "Python:     ${PYTHON}"
echo "Trainer:    ${TRAIN}"
echo "Backend:    ${BACKEND}"
echo "Data dir:   ${DATA_DIR}"
echo "Datasets:   ${DATASETS}"
echo "W grid:     ${W_GRID}"
echo "Y grid:     ${Y_GRID}"
echo "Init state: ${INIT_STATE}"
echo "Device:     ${DEVICE}"
echo "Epochs:     ${EPOCHS}"
echo "LR:         ${LR}"
echo "qT/Q cut:   ${QTMQ}"
echo "Tail amp:   ${NP_A_TAIL_AMP}"
echo "Tail loss:  ${LAMBDA_FNP_TAIL} above b=${FNP_TAIL_BMIN}, target=${FNP_TAIL_TARGET}"
echo "Output:     ${OUT}"

"${PYTHON}" "${TRAIN}" \
  --backend-script "${BACKEND}" \
  --data-dir "${DATA_DIR}" \
  --datasets ${DATASETS} \
  --mode matched \
  --qT-max-over-Q "${QTMQ}" \
  --tmd-qT-max-over-Q "${QTMQ}" \
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
  --epochs "${EPOCHS}" \
  --batch-size 10000 \
  --lr "${LR}" \
  --weight-decay 0 \
  --grad-clip 10 \
  --patience 600 \
  --min-delta 1e-8 \
  --dtype float32 \
  --device "${DEVICE}" \
  --num-threads 4 \
  --log-every 250 \
  --np-width 48 \
  --np-cond-width 32 \
  --np-blocks 3 \
  --np-a0 0.05 \
  --np-min-a 0 \
  --np-a-mode positive \
  --np-shape-mode monotone \
  --fnp-exponent-clip 40 \
  --np-a-smooth-sigma 0.45 \
  --np-a-tail-amp "${NP_A_TAIL_AMP}" \
  --np-a-tail-b0 3.5 \
  --np-a-tail-width 0.25 \
  --lambda-fnp-tail "${LAMBDA_FNP_TAIL}" \
  --fnp-tail-bmin "${FNP_TAIL_BMIN}" \
  --fnp-tail-target "${FNP_TAIL_TARGET}" \
  --soft-q-evolution none \
  --fit-dataset-norms \
  --lambda-dataset-norm 1.0 \
  --norm-source csv \
  --ptp-source csv \
  --init-model-state "${INIT_STATE}" \
  --seed "${SEED}" \
  --out "${OUT}"
