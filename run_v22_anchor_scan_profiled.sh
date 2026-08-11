#!/usr/bin/env bash
set -euo pipefail

# v22 profiled-replica anchor-strength scan.
#
# Default scan:
#   lambda_logF = 30, 10, 3
#   seeds = 1001,1002,1003
#
# Run from ~/work/bT-TMD.
#
# Optional:
#   ANCHORS="30 10" SEEDS="1001 1002 1003" MAX_PARALLEL=1 ./run_v22_anchor_scan_profiled.sh

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

die() {
  echo "$*" >&2
  exit 1
}

CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v22_full_backend_central_refit_stage1_s303}"
BACKEND="${BACKEND:-${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_scan_v22_stage1_logf_anchor}"
ANCHORS=(${ANCHORS:-30 10 3})
SEEDS=(${SEEDS:-1001 1002 1003})
MAX_PARALLEL="${MAX_PARALLEL:-1}"

TRAIN_DEFAULT="${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py"
TRAIN_FALLBACK="${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py"
PREP_DEFAULT="${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py"
PREP_FALLBACK="${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py"

TRAIN="${TRAIN:-}"
PREP="${PREP:-}"

if [[ -z "${TRAIN}" ]]; then
  if [[ -f "${TRAIN_DEFAULT}" ]]; then
    TRAIN="${TRAIN_DEFAULT}"
  elif [[ -f "${TRAIN_FALLBACK}" ]]; then
    TRAIN="${TRAIN_FALLBACK}"
  else
    TRAIN="$(find "${ROOT}" -maxdepth 7 -name train_bt_dnn_v21_replica_stable.py -type f 2>/dev/null | sort | head -1 || true)"
  fi
fi

if [[ -z "${PREP}" ]]; then
  if [[ -f "${PREP_DEFAULT}" ]]; then
    PREP="${PREP_DEFAULT}"
  elif [[ -f "${PREP_FALLBACK}" ]]; then
    PREP="${PREP_FALLBACK}"
  else
    PREP="$(find "${ROOT}" -maxdepth 7 -name prepare_v21_replica_norm_inits.py -type f 2>/dev/null | sort | head -1 || true)"
  fi
fi

CENTRAL_STATE="${CENTRAL_RUN}/model_state.pt"
CENTRAL_PRED="${CENTRAL_RUN}/predictions.csv"

for path in \
  "${TRAIN}" \
  "${PREP}" \
  "${BACKEND}" \
  "${CENTRAL_STATE}" \
  "${CENTRAL_PRED}" \
  "${DATA_DIR}"
do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

if [[ -e "${OUT_ROOT}" ]]; then
  die "Refusing to overwrite existing output root: ${OUT_ROOT}"
fi

mkdir -p "${OUT_ROOT}"

cat > "${OUT_ROOT}/ANCHOR_SCAN_MANIFEST.json" <<EOF
{
  "central_run": "${CENTRAL_RUN}",
  "backend": "${BACKEND}",
  "train_script": "${TRAIN}",
  "prep_script": "${PREP}",
  "anchors": [$(printf '%s\n' "${ANCHORS[@]}" | paste -sd, -)],
  "seeds": [$(printf '%s\n' "${SEEDS[@]}" | paste -sd, -)],
  "max_parallel": ${MAX_PARALLEL},
  "purpose": "Find weakest log-F_NP anchor that preserves fit quality while giving non-collapsed b-space TMD bands"
}
EOF

sanitize_anchor() {
  echo "$1" | sed 's/\./p/g; s/-/m/g'
}

run_one_seed() {
  local LAMBDA="$1"
  local S="$2"
  local TAG
  TAG="$(sanitize_anchor "${LAMBDA}")"
  local SCAN="${OUT_ROOT}/lambda_${TAG}"
  local OUT="${SCAN}/outputs/v22_stage1_profiled_lambda${TAG}_s${S}"
  local LOG="${SCAN}/logs/v22_stage1_profiled_lambda${TAG}_s${S}.log"
  local NORM_INIT="${SCAN}/norm_inits/s${S}/dataset_norms.csv"

  [[ -f "${NORM_INIT}" ]] || {
    echo "Missing norm init: ${NORM_INIT}" >&2
    exit 1
  }
  [[ ! -e "${OUT}" ]] || {
    echo "Refusing to overwrite existing replica output: ${OUT}" >&2
    exit 1
  }

  echo
  echo "============================================================"
  echo "v22 profiled replica: lambda_logF=${LAMBDA}, seed=${S}"
  echo "Output: ${OUT}"
  echo "Log:    ${LOG}"
  echo "============================================================"

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
    --epochs 300 \
    --batch-size 10000 \
    --lr 1e-4 \
    --dataset-norm-lr 1e-3 \
    --weight-decay 0 \
    --grad-clip 10 \
    --patience 100 \
    --min-delta 1e-7 \
    --dtype float32 \
    --device cpu \
    --num-threads 4 \
    --log-every 50 \
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
    --replica-train-scope head \
    --lambda-replica-logf-anchor "${LAMBDA}" \
    --replica-anchor-x-values 0.15 0.20 0.30 0.40 0.50 \
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
    --norm-source paper \
    --ptp-source paper \
    --init-model-state "${CENTRAL_STATE}" \
    --init-dataset-norms-from "${NORM_INIT}" \
    --replica-seed "${S}" \
    --seed "${S}" \
    --out "${OUT}" \
    > "${LOG}" 2>&1

  echo "lambda ${LAMBDA}, seed ${S}: complete"
}

export -f run_one_seed sanitize_anchor
export OUT_ROOT TRAIN BACKEND DATA_DIR CENTRAL_STATE

for LAMBDA in "${ANCHORS[@]}"; do
  TAG="$(sanitize_anchor "${LAMBDA}")"
  SCAN="${OUT_ROOT}/lambda_${TAG}"
  mkdir -p "${SCAN}/outputs" "${SCAN}/logs" "${SCAN}/norm_inits"

  echo
  echo "Preparing normalization initializations for lambda=${LAMBDA}..."
  python3 "${PREP}" \
    --central-predictions "${CENTRAL_PRED}" \
    --seeds "${SEEDS[@]}" \
    --out "${SCAN}/norm_inits"

  printf '%s\n' "${SEEDS[@]}" \
    | xargs -n 1 -P "${MAX_PARALLEL}" bash -c 'run_one_seed "$1" "$2"' _ "${LAMBDA}"

  EXPECTED="${#SEEDS[@]}"
  FOUND=0
  for S in "${SEEDS[@]}"; do
    OUT="${SCAN}/outputs/v22_stage1_profiled_lambda${TAG}_s${S}"
    if [[ -f "${OUT}/metrics.json" && -f "${OUT}/predictions.csv" && -f "${OUT}/model_state.pt" ]]; then
      FOUND=$((FOUND + 1))
    fi
  done

  if [[ "${FOUND}" -ne "${EXPECTED}" ]]; then
    echo "lambda=${LAMBDA}: expected ${EXPECTED}, found ${FOUND} successful outputs" >&2
    exit 1
  fi

  echo "lambda=${LAMBDA}: all ${FOUND} replicas complete"
done

echo
echo "Anchor scan complete."
echo "Postprocess with:"
echo "  ./postprocess_v22_anchor_scan.sh"
