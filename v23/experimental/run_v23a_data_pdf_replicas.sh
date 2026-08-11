# EXPERIMENTAL: true PDF-through-refit route; high CPU cost for backend caches. Prefer overlay route first.
#!/usr/bin/env bash
set -euo pipefail

# Run v23a data-replica x PDF-replica fits.
#
# This is the missing piece between the existing data-replica workflow and a
# full experimental+PDF TMD ensemble.  For every row in PLAN it:
#   1. builds/reuses a W/Y backend cache for the row's PDF member;
#   2. trains one experimental pseudo-data replica with that same cache/member;
#   3. writes a pdf_replica_meta.json sidecar into the run directory.
#
# The script intentionally uses the already successful v23a lambda=3,
# normpriors15, p2p5-style training settings from the cached-CUDA branch, but
# replaces fixed --pdf-member 0 with per-replica PDF members.

ROOT="${ROOT:-$(pwd)}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PLAN="${PLAN:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/replica_pilot_v23a_dataPDF_lambda3_normpriors15_p2p5}"
RUN_PREFIX="${RUN_PREFIX:-v23a_dataPDF_lambda3_normpriors15_p2p5}"
PDF_SET="${PDF_SET:-NNPDF40_nnlo_as_01180}"
PDF_MEMBERS="${PDF_MEMBERS:-1-20}"
N_REPLICAS="${N_REPLICAS:-20}"
SEED_START="${SEED_START:-1001}"
MEMBER_STRATEGY="${MEMBER_STRATEGY:-cycle}"

TRAIN="${TRAIN:-${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py}"
BACKEND="${BACKEND:-${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py}"

# Set DATA_DIR to the exact p2p5/normpriors15 table you want.
DATA_DIR="${DATA_DIR:-${ROOT}/Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400}"

# A good default for this branch, but override if your central run has a different path.
CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v23a_fixed_target_lowQ_corrected_central_refit_normpriors15_p2p5_E772_E288400_s303}"
NORM_INIT_ROOT="${NORM_INIT_ROOT:-${CENTRAL_RUN}}"
INIT_STATE="${INIT_STATE:-${CENTRAL_RUN}/model_state.pt}"
NORM_INIT="${NORM_INIT:-${NORM_INIT_ROOT}/dataset_norms.csv}"

DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL_CACHE="${MAX_PARALLEL_CACHE:-1}"
MAX_PARALLEL_TRAIN="${MAX_PARALLEL_TRAIN:-1}"
EPOCHS="${EPOCHS:-300}"

# Reuse existing outputs if present.
REUSE_CACHES="${REUSE_CACHES:-1}"
REUSE_RUNS="${REUSE_RUNS:-1}"

# Extra user arguments can be appended without editing this script.
CACHE_EXTRA_ARGS="${CACHE_EXTRA_ARGS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

die() { echo "$*" >&2; exit 1; }

for path in "${TRAIN}" "${BACKEND}" "${DATA_DIR}" "${INIT_STATE}" "${NORM_INIT}"; do
  [[ -e "${path}" ]] || die "Missing required path: ${path}"
done

mkdir -p "${OUT_ROOT}"/{outputs,logs,pdf_caches}

if [[ -z "${PLAN}" ]]; then
  PLAN="${OUT_ROOT}/replica_plan.csv"
fi

if [[ ! -f "${PLAN}" ]]; then
  echo "Creating replica plan: ${PLAN}"
  python3 v23/tools/make_v23a_data_pdf_replica_plan.py \
    --n-replicas "${N_REPLICAS}" \
    --seed-start "${SEED_START}" \
    --pdf-set "${PDF_SET}" \
    --pdf-members "${PDF_MEMBERS}" \
    --member-strategy "${MEMBER_STRATEGY}" \
    --out-root "${OUT_ROOT}" \
    --run-prefix "${RUN_PREFIX}" \
    --out "${PLAN}"
fi

echo "Plan: ${PLAN}"
echo "Out root: ${OUT_ROOT}"
echo "Device: ${DEVICE}"

build_cache_for_member() {
  local member="$1"
  local cache_dir="$2"
  local cache_tag="$3"
  local done_file="${cache_dir}/CACHE_DONE"
  local cache_root="${cache_dir}/checkonly/backend_cache"

  if [[ "${REUSE_CACHES}" == "1" && -f "${done_file}" ]]; then
    echo "pdf member ${member}: cache already done"
    return 0
  fi

  rm -rf "${cache_dir}/checkonly"
  mkdir -p "${cache_dir}"

  local log="${OUT_ROOT}/logs/cache_pdf$(printf '%04d' "${member}").log"
  echo "pdf member ${member}: building backend cache -> ${cache_dir}"

  # shellcheck disable=SC2086
  python3 "${TRAIN}" \
    --backend-script "${BACKEND}" \
    --data-dir "${DATA_DIR}" \
    --datasets E288_200 E288_300 E288_400 E605 E772 \
    --mode matched \
    --qT-max-over-Q 0.5 \
    --tmd-qT-max-over-Q 0.2 \
    --w-backend internal_css \
    --pdf-set "${PDF_SET}" \
    --pdf-member "${member}" \
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
    --dtype float32 \
    --device cpu \
    --num-threads 4 \
    --quiet-backend \
    --cache-backend-grids \
    --cache-tag "${cache_tag}" \
    --check-only \
    --out "${cache_dir}/checkonly" \
    ${CACHE_EXTRA_ARGS} \
    > "${log}" 2>&1

  local w_grid y_grid
  w_grid="$(find "${cache_root}" -maxdepth 1 -type f -name 'wpert_*.csv' | sort | head -1)"
  y_grid="$(find "${cache_root}" -maxdepth 1 -type f -name 'y_*.csv' | sort | head -1)"
  [[ -f "${w_grid}" ]] || die "pdf member ${member}: missing W grid in ${cache_root}"
  [[ -f "${y_grid}" ]] || die "pdf member ${member}: missing Y grid in ${cache_root}"

  cat > "${cache_dir}/cache_paths.env" <<EOF
W_GRID='${w_grid}'
Y_GRID='${y_grid}'
PDF_MEMBER='${member}'
PDF_SET='${PDF_SET}'
EOF
  touch "${done_file}"
  echo "pdf member ${member}: cache complete"
}

run_one_replica() {
  local seed="$1"
  local member="$2"
  local run_dir="$3"
  local cache_dir="$4"
  local cache_tag="$5"

  local meta="${run_dir}/pdf_replica_meta.json"
  if [[ "${REUSE_RUNS}" == "1" && -f "${run_dir}/metrics.json" && -f "${run_dir}/model_state.pt" && -f "${meta}" ]]; then
    echo "seed ${seed}, pdf ${member}: run already done"
    return 0
  fi

  [[ -f "${cache_dir}/cache_paths.env" ]] || die "missing cache env: ${cache_dir}/cache_paths.env"
  # shellcheck source=/dev/null
  source "${cache_dir}/cache_paths.env"

  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  local log="${OUT_ROOT}/logs/fit_pdf$(printf '%04d' "${member}")_s${seed}.log"

  echo "seed ${seed}, pdf ${member}: training -> ${run_dir}"

  # shellcheck disable=SC2086
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
    --pdf-set "${PDF_SET}" \
    --pdf-member "${member}" \
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
    --epochs "${EPOCHS}" \
    --batch-size 10000 \
    --lr 1e-4 \
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
    --init-model-state "${INIT_STATE}" \
    --init-dataset-norms-from "${NORM_INIT}" \
    --replica-seed "${seed}" \
    --seed "${seed}" \
    --out "${run_dir}" \
    ${TRAIN_EXTRA_ARGS} \
    > "${log}" 2>&1

  python3 - <<PY
import json
from pathlib import Path
meta = {
  "seed": int("${seed}"),
  "pdf_set": "${PDF_SET}",
  "pdf_member": int("${member}"),
  "w_grid": "${W_GRID}",
  "y_grid": "${Y_GRID}",
  "cache_dir": "${cache_dir}",
  "cache_tag": "${cache_tag}",
  "data_dir": "${DATA_DIR}",
  "run_dir": "${run_dir}",
  "uncertainty_role": "one joint experimental-pseudodata x PDF-member replica"
}
Path("${meta}").write_text(json.dumps(meta, indent=2) + "\n")
PY

  echo "seed ${seed}, pdf ${member}: complete"
}

export -f build_cache_for_member run_one_replica die
export ROOT OUT_ROOT TRAIN BACKEND DATA_DIR PDF_SET DEVICE EPOCHS INIT_STATE NORM_INIT CACHE_EXTRA_ARGS TRAIN_EXTRA_ARGS REUSE_CACHES REUSE_RUNS

echo
echo "=== Building unique PDF-member caches ==="
python3 - "${PLAN}" <<'PY' > "${OUT_ROOT}/cache_tasks.tsv"
import pandas as pd, sys
df = pd.read_csv(sys.argv[1]).drop_duplicates("pdf_member").sort_values("pdf_member")
for _, r in df.iterrows():
    print(f"{int(r.pdf_member)}\t{r.cache_dir}\t{r.cache_tag}")
PY

xargs -a "${OUT_ROOT}/cache_tasks.tsv" -d '\n' -P "${MAX_PARALLEL_CACHE}" -I{} bash -lc '
  IFS=$'\''\t'\'' read -r member cache_dir cache_tag <<< "{}"
  build_cache_for_member "$member" "$cache_dir" "$cache_tag"
'

echo
echo "=== Training data x PDF replicas ==="
python3 - "${PLAN}" <<'PY' > "${OUT_ROOT}/fit_tasks.tsv"
import pandas as pd, sys
df = pd.read_csv(sys.argv[1]).sort_values("replica_index")
for _, r in df.iterrows():
    print(f"{int(r.seed)}\t{int(r.pdf_member)}\t{r.run_dir}\t{r.cache_dir}\t{r.cache_tag}")
PY

xargs -a "${OUT_ROOT}/fit_tasks.tsv" -d '\n' -P "${MAX_PARALLEL_TRAIN}" -I{} bash -lc '
  IFS=$'\''\t'\'' read -r seed member run_dir cache_dir cache_tag <<< "{}"
  run_one_replica "$seed" "$member" "$run_dir" "$cache_dir" "$cache_tag"
'

echo
echo "Data x PDF replica training complete."
echo "Plan: ${PLAN}"
echo "Output root: ${OUT_ROOT}"
echo
echo "Next:"
echo "  PYTHONPATH=. python3 v23/tools/construct_v23a_data_pdf_bspace_tmd_bands.py \\"
echo "    --plan '${PLAN}' \\"
echo "    --out '${OUT_ROOT}/tmd_bspace_bands_dataPDF' \\"
echo "    --x-values 0.10 0.20 0.30 0.50 \\"
echo "    --Q-values 5 10"
