#!/usr/bin/env bash
set -euo pipefail

# Append more cached-CUDA v22 lambda=3 replicas to an existing ensemble.
#
# Default appends seeds 1011-1020 into:
#   replica_pilot_v22_lambda3_cached_cuda/outputs/
#
# Run from ~/work/bT-TMD after cache/CUDA validation.
#
# Examples:
#   SEEDS="1011 1012 1013 1014 1015 1016 1017 1018 1019 1020" ./workflows/v22/replicas/append_v22_lambda3_cached_cuda_replicas.sh
#   DEVICE=cuda MAX_PARALLEL=1 ./workflows/v22/replicas/append_v22_lambda3_cached_cuda_replicas.sh

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CACHE_ENV="${CACHE_ENV:-${ROOT}/outputs/v22_full_backend_cache_export/backend_cache/cache_paths.env}"
TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py}"
PREP="${PREP:-${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py}"
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v22_full_backend_central_refit_stage1_s303}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_pilot_v22_lambda3_cached_cuda}"
SEEDS=(${SEEDS:-1011 1012 1013 1014 1015 1016 1017 1018 1019 1020})
DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

die() {
  echo "$*" >&2
  exit 1
}

if [[ ! -f "${PREP}" ]]; then
  PREP="${ROOT}/v21_tail_release_amp0p019_candidate/prepare_v21_replica_norm_inits.py"
fi

for path in "${CACHE_ENV}" "${TRAIN}" "${PREP}" "${CENTRAL_RUN}/model_state.pt" "${CENTRAL_RUN}/predictions.csv"; do
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
  "w_grid": "${W_GRID}",
  "y_grid": "${Y_GRID}",
  "train_script": "${TRAIN}",
  "prep_script": "${PREP}",
  "device": "${DEVICE}",
  "lambda_replica_logf_anchor": 3,
  "seeds": [$(printf '%s\n' "${SEEDS[@]}" | paste -sd, -)],
  "max_parallel": ${MAX_PARALLEL},
  "norm_init_root": "${NORM_INIT_ROOT}"
}
EOF

echo "Preparing normalization initializations for appended seeds..."
python3 "${PREP}" \
  --central-predictions "${CENTRAL_RUN}/predictions.csv" \
  --seeds "${SEEDS[@]}" \
  --out "${NORM_INIT_ROOT}"

run_seed() {
  local S="$1"
  local OUT="${OUT_ROOT}/outputs/v22_lambda3_cached_cuda_s${S}"
  local LOG="${OUT_ROOT}/logs/v22_lambda3_cached_cuda_s${S}.log"
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

  echo "Running cached CUDA lambda=3 replica seed ${S}"

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
    --lambda-replica-logf-anchor 3 \
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
    --init-model-state "${CENTRAL_RUN}/model_state.pt" \
    --init-dataset-norms-from "${NORM_INIT}" \
    --replica-seed "${S}" \
    --seed "${S}" \
    --out "${OUT}" \
    > "${LOG}" 2>&1

  echo "seed ${S}: complete"
}

export -f run_seed
export ROOT OUT_ROOT TRAIN W_GRID Y_GRID DEVICE CENTRAL_RUN NORM_INIT_ROOT

printf '%s\n' "${SEEDS[@]}" \
  | xargs -n 1 -P "${MAX_PARALLEL}" bash -c 'run_seed "$1"' _

echo
echo "Append run complete."
echo
echo "Re-audit with:"
echo "  python3 v22/tools/audit_v22_replica_pilot_basic.py \\"
echo "    --glob '${OUT_ROOT}/outputs/v22_lambda3_cached_cuda_s*' \\"
echo "    --central-run '${CENTRAL_RUN}' \\"
echo "    --out '${OUT_ROOT}/audit_basic_all'"
