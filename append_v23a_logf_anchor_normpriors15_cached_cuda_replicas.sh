#!/usr/bin/env bash
set -euo pipefail

# Generic v23a cached-CUDA replica runner for log-F anchor scans.
#
# Default use:
#   LAMBDA_LOGF=1 \
#   SEEDS="1001 1002 ... 1010" \
#   MAX_PARALLEL=1 \
#   DEVICE=cuda \
#   ./append_v23a_logf_anchor_normpriors15_cached_cuda_replicas.sh
#
# Scope:
#   v23a fixed-target low-Q DY, corrected E288_300:99, explicit 15% norm priors.
#
# Output default:
#   replica_pilot_v23a_lambda${LAMBDA_LOGF}_normpriors15_cached_cuda/outputs/
#
# This is for pilot/scan use, not final production freeze.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

LAMBDA_LOGF="${LAMBDA_LOGF:-1}"
LAMBDA_LABEL="${LAMBDA_LABEL:-lambda${LAMBDA_LOGF}}"

CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_checkonly_cache/backend_cache/cache_paths.env}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py}"
PREP="${PREP:-${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_15pct_s303}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_explicit_normpriors_15pct}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_pilot_v23a_${LAMBDA_LABEL}_normpriors15_cached_cuda}"
RUN_PREFIX="${RUN_PREFIX:-v23a_${LAMBDA_LABEL}_normpriors15_cached_cuda}"

SEEDS=(${SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010})
DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

die() {
  echo "$*" >&2
  exit 1
}

if [[ ! -f "${PREP}" ]]; then
  PREP="${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py"
fi

for path in \
  "${CACHE_ENV}" \
  "${TRAIN}" \
  "${PREP}" \
  "${CENTRAL_RUN}/model_state.pt" \
  "${CENTRAL_RUN}/predictions.csv" \
  "${DATA_DIR}"
do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

# shellcheck disable=SC1090
source "${CACHE_ENV}"

for path in "${W_GRID}" "${Y_GRID}"; do
  [[ -f "${path}" ]] || die "Missing cache file: ${path}"
done

mkdir -p "${OUT_ROOT}/outputs" "${OUT_ROOT}/logs" "${OUT_ROOT}/append_manifests"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NORM_INIT_ROOT="${OUT_ROOT}/norm_inits_append_${STAMP}"
mkdir -p "${NORM_INIT_ROOT}"

cat > "${OUT_ROOT}/append_manifests/append_${STAMP}.json" <<EOF
{
  "central_run": "${CENTRAL_RUN}",
  "data_dir": "${DATA_DIR}",
  "w_grid": "${W_GRID}",
  "y_grid": "${Y_GRID}",
  "train_script": "${TRAIN}",
  "prep_script": "${PREP}",
  "device": "${DEVICE}",
  "lambda_replica_logf_anchor": ${LAMBDA_LOGF},
  "replica_anchor_x_values": [0.10, 0.20, 0.30, 0.50],
  "seeds": [$(printf '%s\n' "${SEEDS[@]}" | paste -sd, -)],
  "max_parallel": ${MAX_PARALLEL},
  "norm_init_root": "${NORM_INIT_ROOT}",
  "norm_source": "csv",
  "ptp_source": "csv"
}
EOF

echo "Preparing normalization initializations for v23a seeds..."
python3 "${PREP}" \
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

  echo "Running v23a cached CUDA ${LAMBDA_LABEL} replica seed ${S}"

  python3 "${TRAIN}" \
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
    --target-mode nuclear_isospin \
    --prefactor-scheme oldA_to_CS \
    --y-mode zero \
    --n-b 160 \
    --b-min 1e-4 \
    --b-max 8 \
    --b-star-max 1.5 \
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
    --device "${DEVICE}" \
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
    --lambda-replica-logf-anchor "${LAMBDA_LOGF}" \
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
export ROOT OUT_ROOT RUN_PREFIX TRAIN W_GRID Y_GRID DEVICE CENTRAL_RUN NORM_INIT_ROOT DATA_DIR LAMBDA_LOGF LAMBDA_LABEL

printf '%s\n' "${SEEDS[@]}" \
  | xargs -n 1 -P "${MAX_PARALLEL}" bash -c 'run_seed "$1"' _

echo
echo "v23a ${LAMBDA_LABEL} replica append/run complete."
echo "Output root: ${OUT_ROOT}"
echo
echo "Postprocess with:"
echo "  OUT_ROOT='${OUT_ROOT}' \\"
echo "  RUN_PREFIX='${RUN_PREFIX}' \\"
echo "  BAND_DIR='${OUT_ROOT}/tmd_bspace_bands_exactx' \\"
echo "  MIN_REPLICAS=${#SEEDS[@]} \\"
echo "  ./postprocess_v23a_lambda3_normpriors15_replica_pilot_rescue.sh"
