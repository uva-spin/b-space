#!/usr/bin/env bash
set -euo pipefail

# Continue the existing v23a lambda=3 normpriors15 cached-CUDA replica pilot.
#
# This is not a physics change.  It starts from the existing seed checkpoints
# and trains the same pseudo-data replicas for additional epochs.
#
# Default input:
#   replica_pilot_v23a_lambda3_normpriors15_cached_cuda/outputs/v23a_lambda3_normpriors15_cached_cuda_s1001...
#
# Default output:
#   replica_pilot_v23a_lambda3_normpriors15_cached_cuda_cont300/outputs/v23a_lambda3_normpriors15_cached_cuda_cont300_s1001...
#
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_checkonly_cache/backend_cache/cache_paths.env}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_explicit_normpriors_15pct}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors_15pct_s303}"

IN_ROOT="${IN_ROOT:-${ROOT}/replica_pilot_v23a_lambda3_normpriors15_cached_cuda}"
IN_PREFIX="${IN_PREFIX:-v23a_lambda3_normpriors15_cached_cuda}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_pilot_v23a_lambda3_normpriors15_cached_cuda_cont300}"
OUT_PREFIX="${OUT_PREFIX:-v23a_lambda3_normpriors15_cached_cuda_cont300}"

SEEDS=(${SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010})
DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
EPOCHS_CONT="${EPOCHS_CONT:-300}"

die() {
  echo "$*" >&2
  exit 1
}

for path in \
  "${CACHE_ENV}" \
  "${TRAIN}" \
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

mkdir -p "${OUT_ROOT}/outputs" "${OUT_ROOT}/logs" "${OUT_ROOT}/continue_manifests"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cat > "${OUT_ROOT}/continue_manifests/continue_${STAMP}.json" <<EOF
{
  "input_root": "${IN_ROOT}",
  "input_prefix": "${IN_PREFIX}",
  "output_root": "${OUT_ROOT}",
  "output_prefix": "${OUT_PREFIX}",
  "central_run": "${CENTRAL_RUN}",
  "data_dir": "${DATA_DIR}",
  "w_grid": "${W_GRID}",
  "y_grid": "${Y_GRID}",
  "train_script": "${TRAIN}",
  "device": "${DEVICE}",
  "epochs_continuation": ${EPOCHS_CONT},
  "lambda_replica_logf_anchor": 3,
  "seeds": [$(printf '%s\n' "${SEEDS[@]}" | paste -sd, -)],
  "norm_source": "csv",
  "ptp_source": "csv"
}
EOF

run_seed() {
  local S="$1"
  local OLD="${IN_ROOT}/outputs/${IN_PREFIX}_s${S}"
  local OUT="${OUT_ROOT}/outputs/${OUT_PREFIX}_s${S}"
  local LOG="${OUT_ROOT}/logs/${OUT_PREFIX}_s${S}.log"

  for path in \
    "${OLD}/model_state.pt" \
    "${OLD}/dataset_norms.csv" \
    "${OLD}/metrics.json" \
    "${OLD}/predictions.csv"
  do
    [[ -e "${path}" ]] || {
      echo "seed ${S}: missing required input ${path}" >&2
      exit 1
    }
  done

  if [[ -f "${OUT}/metrics.json" && -f "${OUT}/predictions.csv" && -f "${OUT}/model_state.pt" ]]; then
    echo "seed ${S}: continuation already complete, skipping"
    return 0
  fi

  [[ ! -e "${OUT}" ]] || {
    echo "seed ${S}: output exists but incomplete/refused: ${OUT}" >&2
    exit 1
  }

  echo "Continuing v23a seed ${S} for ${EPOCHS_CONT} more epochs"

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
    --epochs "${EPOCHS_CONT}" \
    --batch-size 10000 \
    --lr 5e-5 \
    --dataset-norm-lr 5e-4 \
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
    --init-model-state "${OLD}/model_state.pt" \
    --init-dataset-norms-from "${OLD}/dataset_norms.csv" \
    --replica-seed "${S}" \
    --seed "${S}" \
    --out "${OUT}" \
    > "${LOG}" 2>&1

  echo "seed ${S}: continuation complete"
}

export -f run_seed
export ROOT OUT_ROOT OUT_PREFIX IN_ROOT IN_PREFIX TRAIN W_GRID Y_GRID DEVICE CENTRAL_RUN DATA_DIR EPOCHS_CONT

printf '%s\n' "${SEEDS[@]}" \
  | xargs -n 1 -P "${MAX_PARALLEL}" bash -c 'run_seed "$1"' _

echo
echo "Continuation complete."
echo
echo "Postprocess with:"
echo "  OUT_ROOT='${OUT_ROOT}' \\"
echo "  RUN_PREFIX='${OUT_PREFIX}' \\"
echo "  BAND_DIR='${OUT_ROOT}/tmd_bspace_bands_exactx' \\"
echo "  MIN_REPLICAS=10 \\"
echo "  ./postprocess_v23a_lambda3_normpriors15_replica_pilot_rescue.sh"
