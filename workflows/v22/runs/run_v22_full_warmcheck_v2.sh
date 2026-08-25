#!/usr/bin/env bash
set -euo pipefail

# Robust zero-learning-rate warm check of the full v22 W+Y backend.
#
# Run from ~/work/bT-TMD, where v22/backends/bt_internal_css_backend_v22_full.py exists.
#
# Optional overrides:
#   FROZEN=/path/to/production_frozen/PRIMARY_ONE_REPLICA ./workflows/v22/runs/run_v22_full_warmcheck_v2.sh
#   TRAIN=/path/to/train_bt_dnn_v21_smoothedA_tail.py ./workflows/v22/runs/run_v22_full_warmcheck_v2.sh
#   DATA_DIR=/path/to/Data ./workflows/v22/runs/run_v22_full_warmcheck_v2.sh
#   OUT=outputs/custom_name ./workflows/v22/runs/run_v22_full_warmcheck_v2.sh

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BACKEND="${BACKEND:-${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py}"
DATA_DIR="${DATA_DIR:-${ROOT}/Data}"
OUT="${OUT:-${ROOT}/outputs/v22_full_backend_warmcheck_s303}"

die() {
  echo "$*" >&2
  exit 1
}

has_frozen_payload() {
  local d="$1"
  [[ -f "${d}/run/model_state.pt" && -f "${d}/run/dataset_norms.csv" ]]
}

resolve_frozen() {
  if [[ -n "${FROZEN:-}" ]]; then
    local f
    f="$(realpath -m "${FROZEN}")"
    has_frozen_payload "${f}" || die "FROZEN was set but does not contain run/model_state.pt and run/dataset_norms.csv: ${f}"
    echo "${f}"
    return 0
  fi

  local candidates=(
    "${ROOT}/production_frozen/PRIMARY_ONE_REPLICA"
    "${ROOT}/v21_tail_release_amp0p019_candidate/production_frozen/PRIMARY_ONE_REPLICA"
    "${ROOT}/v21_tail_release_amp0p019_candidate/production_frozen/v21_smoothedA_tail_release_amp0p019_s303"
    "${ROOT}/production_frozen/v21_smoothedA_tail_release_amp0p019_s303"
  )

  local c
  for c in "${candidates[@]}"; do
    if has_frozen_payload "${c}"; then
      realpath -m "${c}"
      return 0
    fi
  done

  while IFS= read -r c; do
    if has_frozen_payload "${c}"; then
      realpath -m "${c}"
      return 0
    fi
  done < <(
    find "${ROOT}" -maxdepth 6 \( -type d -o -type l \) \
      \( -name PRIMARY_ONE_REPLICA -o -name 'v21_smoothedA_tail_release_amp0p019_s303' \) \
      2>/dev/null | sort
  )

  die "Could not locate frozen central reference. Set FROZEN=/path/to/production_frozen/PRIMARY_ONE_REPLICA"
}

resolve_train() {
  local frozen="$1"

  if [[ -n "${TRAIN:-}" ]]; then
    local t
    t="$(realpath -m "${TRAIN}")"
    [[ -f "${t}" ]] || die "TRAIN was set but does not exist: ${t}"
    echo "${t}"
    return 0
  fi

  local candidates=(
    "${ROOT}/train_bt_dnn_v21_smoothedA_tail.py"
    "${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py"
    "${frozen}/code/train_bt_dnn_v21_smoothedA_tail.py"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      realpath -m "${c}"
      return 0
    fi
  done

  while IFS= read -r c; do
    if [[ -f "${c}" ]]; then
      realpath -m "${c}"
      return 0
    fi
  done < <(
    find "${ROOT}" -maxdepth 7 -name train_bt_dnn_v21_smoothedA_tail.py -type f 2>/dev/null | sort
  )

  die "Could not find train_bt_dnn_v21_smoothedA_tail.py. Set TRAIN=/path/to/train_bt_dnn_v21_smoothedA_tail.py"
}

FROZEN_RESOLVED="$(resolve_frozen)"
TRAIN_RESOLVED="$(resolve_train "${FROZEN_RESOLVED}")"

for path in \
  "${TRAIN_RESOLVED}" \
  "${BACKEND}" \
  "${FROZEN_RESOLVED}/run/model_state.pt" \
  "${FROZEN_RESOLVED}/run/dataset_norms.csv" \
  "${DATA_DIR}"
do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

if [[ -e "${OUT}" ]]; then
  die "Refusing to overwrite existing output: ${OUT}"
fi

echo "ROOT:         ${ROOT}"
echo "Trainer:      ${TRAIN_RESOLVED}"
echo "Backend:      ${BACKEND}"
echo "Frozen:       ${FROZEN_RESOLVED}"
echo "Data:         ${DATA_DIR}"
echo "Output:       ${OUT}"

python3 "${TRAIN_RESOLVED}" \
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
  --epochs 1 \
  --batch-size 10000 \
  --lr 0 \
  --weight-decay 0 \
  --grad-clip 10 \
  --patience 1 \
  --min-delta 1e-12 \
  --dtype float32 \
  --device cpu \
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
  --init-model-state "${FROZEN_RESOLVED}/run/model_state.pt" \
  --init-dataset-norms-from "${FROZEN_RESOLVED}/run/dataset_norms.csv" \
  --seed 303 \
  --out "${OUT}"

echo
echo "Warm check complete."
echo
echo "Analyze with:"
echo "  python3 v22/tools/audit_v22_warmcheck_shift.py \\"
echo "    --central-run '${FROZEN_RESOLVED}/run' \\"
echo "    --v22-run '${OUT}' \\"
echo "    --out v22/outputs/v22_full_backend_warmcheck_audit"
