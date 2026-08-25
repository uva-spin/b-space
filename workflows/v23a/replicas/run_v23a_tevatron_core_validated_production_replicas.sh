#!/usr/bin/env bash
set -euo pipefail

# Production experimental-replica ensemble for the validated v23a fixed-target
# + Tevatron low-qT set. This includes CDF_RUN_2 only in the validated
# qT/Q <= 0.10 domain and intentionally excludes normalized D0 Run II spectra.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/dustin/miniforge3/envs/pdf-fit/bin/python}"
CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v23a_fixed_target_plus_tevatron_absolute_fit_ready_ewpty_checkonly_cache/backend_cache/cache_paths.env}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py}"
PREP="${PREP:-${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py}"
BACKEND="${BACKEND:-${ROOT}/v23/backends/bt_internal_css_backend_v22_tevatron.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready}"
DATASETS="${DATASETS:-E288_200 E288_300 E288_400 E605 E772 CDF_RUN_1 CDF_RUN_2 D0_RUN_1}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v23a_tevatron_lowqt010_allchecked_tailpass_central_s303}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_v23a_tevatron_lowqt010_allchecked_tailpass_lambda3_50rep}"
RUN_PREFIX="${RUN_PREFIX:-v23a_tevatron_lowqt010_allchecked_tailpass_lambda3}"
SEEDS=(${SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050})
DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
QTMQ="${QTMQ:-0.10}"
NP_A_TAIL_AMP="${NP_A_TAIL_AMP:-0.08}"

die() {
  echo "$*" >&2
  exit 1
}

if [[ ! -f "${PREP}" ]]; then
  PREP="${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py"
fi

for path in \
  "${PYTHON}" \
  "${CACHE_ENV}" \
  "${TRAIN}" \
  "${PREP}" \
  "${BACKEND}" \
  "${DATA_DIR}" \
  "${CENTRAL_RUN}/model_state.pt" \
  "${CENTRAL_RUN}/predictions.csv"
do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

# shellcheck disable=SC1090
source "${CACHE_ENV}"

for path in "${W_GRID}" "${Y_GRID}"; do
  [[ -f "${path}" ]] || die "Missing cache file: ${path}"
done

mkdir -p "${OUT_ROOT}/outputs" "${OUT_ROOT}/logs" "${OUT_ROOT}/manifests"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NORM_INIT_ROOT="${OUT_ROOT}/norm_inits_${STAMP}"
mkdir -p "${NORM_INIT_ROOT}"

cat > "${OUT_ROOT}/manifests/run_${STAMP}.json" <<EOF
{
  "role": "production experimental replicas for validated v23a fixed-target plus Tevatron core data",
  "central_run": "${CENTRAL_RUN}",
  "data_dir": "${DATA_DIR}",
  "datasets": "$(printf '%s' "${DATASETS}")",
  "excluded": {
    "D0_RUN_2": "normalized spectrum; needs normalized-theory observable",
    "D0_RUN_2N": "normalized spectrum; needs normalized-theory observable",
    "qT/Q > production cut": "outside validated low-qT TMD extraction domain; high-qT collider rows need independent finite-tail/Y benchmarking"
  },
  "w_grid": "${W_GRID}",
  "y_grid": "${Y_GRID}",
  "backend": "${BACKEND}",
  "train_script": "${TRAIN}",
  "prep_script": "${PREP}",
  "device": "${DEVICE}",
  "lambda_replica_logf_anchor": 3,
  "replica_anchor_x_values": [0.10, 0.20, 0.30, 0.50],
  "seeds": [$(printf '%s\n' "${SEEDS[@]}" | paste -sd, -)],
  "max_parallel": ${MAX_PARALLEL},
  "norm_init_root": "${NORM_INIT_ROOT}",
  "norm_source": "csv",
  "ptp_source": "csv",
  "qT_max_over_Q": ${QTMQ},
  "tmd_qT_max_over_Q": ${QTMQ},
  "np_a_tail_amp": ${NP_A_TAIL_AMP}
}
EOF

echo "Preparing normalization initializations..."
"${PYTHON}" "${PREP}" \
  --central-predictions "${CENTRAL_RUN}/predictions.csv" \
  --seeds "${SEEDS[@]}" \
  --out "${NORM_INIT_ROOT}"

run_seed() {
  local S="$1"
  local OUT="${OUT_ROOT}/outputs/${RUN_PREFIX}_s${S}"
  local LOG="${OUT_ROOT}/logs/${RUN_PREFIX}_s${S}.log"
  local NORM_INIT="${NORM_INIT_ROOT}/s${S}/dataset_norms.csv"

  if [[ -f "${OUT}/metrics.json" && -f "${OUT}/predictions.csv" && -f "${OUT}/model_state.pt" ]]; then
    echo "seed ${S}: already complete, skipping"
    return 0
  fi

  [[ ! -e "${OUT}" ]] || {
    echo "Output directory exists but is incomplete/refused: ${OUT}" >&2
    exit 1
  }

  [[ -f "${NORM_INIT}" ]] || {
    echo "Missing norm init ${NORM_INIT}" >&2
    exit 1
  }

  echo "Running validated Tevatron core replica seed ${S}"

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
    --epochs 500 \
    --batch-size 10000 \
    --lr 5e-5 \
    --dataset-norm-lr 1e-3 \
    --weight-decay 0 \
    --grad-clip 10 \
    --patience 150 \
    --min-delta 1e-8 \
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
    --np-a-tail-amp "${NP_A_TAIL_AMP}" \
    --np-a-tail-b0 3.5 \
    --np-a-tail-width 0.25 \
    --replica-train-scope head \
    --lambda-replica-logf-anchor 3 \
    --replica-anchor-x-values 0.10 0.20 0.30 0.50 \
    --replica-anchor-bmin 0 \
    --replica-anchor-bmax 8 \
    --lambda-a-l2 0 \
    --lambda-gk-l2 0 \
    --lambda-fnp-mono 0 \
    --lambda-fnp-bcurv 0 \
    --lambda-fnp-xcurv 0 \
    --lambda-fnp-pair-bcurv 0 \
    --lambda-fnp-local-bcurv 0 \
    --lambda-fnp-lowpass 0 \
    --lambda-fnp-ratecurv 0 \
    --lambda-fnp-tail 0 \
    --soft-q-evolution none \
    --fit-dataset-norms \
    --lambda-dataset-norm 1.0 \
    --norm-source csv \
    --ptp-source csv \
    --init-model-state "${CENTRAL_RUN}/model_state.pt" \
    --init-dataset-norms-from "${NORM_INIT}" \
    --replica-seed "${S}" \
    --seed "${S}" \
    --out "${OUT}" \
    > "${LOG}" 2>&1

  echo "seed ${S}: complete"
}

export -f run_seed
export PYTHON ROOT OUT_ROOT RUN_PREFIX TRAIN BACKEND W_GRID Y_GRID DEVICE CENTRAL_RUN NORM_INIT_ROOT DATA_DIR DATASETS QTMQ NP_A_TAIL_AMP

printf '%s\n' "${SEEDS[@]}" \
  | xargs -n 1 -P "${MAX_PARALLEL}" bash -c 'run_seed "$1"' _

echo
echo "Validated Tevatron core production replicas complete."
echo "Output root: ${OUT_ROOT}"
