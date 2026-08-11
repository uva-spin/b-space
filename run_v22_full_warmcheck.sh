#!/usr/bin/env bash
set -euo pipefail

# Zero-learning-rate warm check of the full v22 W+Y backend using the
# frozen v21 central NP state and central dataset normalizations.
#
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

FROZEN="${FROZEN:-production_frozen/PRIMARY_ONE_REPLICA}"
DATA_DIR="${DATA_DIR:-./Data}"
BACKEND="${BACKEND:-v22/backends/bt_internal_css_backend_v22_full.py}"
OUT="${OUT:-outputs/v22_full_backend_warmcheck_s303}"

if [[ -f ./train_bt_dnn_v21_smoothedA_tail.py ]]; then
  TRAIN="./train_bt_dnn_v21_smoothedA_tail.py"
elif [[ -f "${FROZEN}/code/train_bt_dnn_v21_smoothedA_tail.py" ]]; then
  TRAIN="${FROZEN}/code/train_bt_dnn_v21_smoothedA_tail.py"
else
  echo "Could not find train_bt_dnn_v21_smoothedA_tail.py" >&2
  exit 1
fi

for path in \
  "${TRAIN}" \
  "${BACKEND}" \
  "${FROZEN}/run/model_state.pt" \
  "${FROZEN}/run/dataset_norms.csv" \
  "${DATA_DIR}"
do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT}" >&2
  exit 1
fi

echo "Trainer: ${TRAIN}"
echo "Backend: ${BACKEND}"
echo "Frozen state: ${FROZEN}/run/model_state.pt"
echo "Output: ${OUT}"

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
  --init-model-state "${FROZEN}/run/model_state.pt" \
  --init-dataset-norms-from "${FROZEN}/run/dataset_norms.csv" \
  --seed 303 \
  --out "${OUT}"

echo
echo "Warm check complete."
echo "Analyze with:"
echo "  python3 v22/tools/audit_v22_warmcheck_shift.py \\"
echo "    --central-run '${FROZEN}/run' \\"
echo "    --v22-run '${OUT}' \\"
echo "    --out v22/outputs/v22_full_backend_warmcheck_audit"
