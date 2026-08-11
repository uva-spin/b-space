#!/usr/bin/env bash
set -euo pipefail

# Freeze the v22 lambda=3 50-replica DY-only b-space TMD ensemble.
#
# Run from ~/work/bT-TMD after the 50-rep audits pass.
#
# Output:
#   production_frozen/v22_lambda3_50rep_DYonly_bspace/
#   production_frozen/v22_lambda3_50rep_DYonly_bspace.tgz

ROOT="$(pwd)"
TAG="${TAG:-v22_lambda3_50rep_DYonly_bspace}"
FREEZE_PARENT="${FREEZE_PARENT:-${ROOT}/production_frozen}"
FREEZE_DIR="${FREEZE_PARENT}/${TAG}"
TARBALL="${FREEZE_PARENT}/${TAG}.tgz"

CENTRAL_RUN="${CENTRAL_RUN:-${ROOT}/outputs/v22_full_backend_central_refit_stage1_s303}"
CACHE_EXPORT="${CACHE_EXPORT:-${ROOT}/outputs/v22_full_backend_cache_export}"
REPLICA_ROOT="${REPLICA_ROOT:-${ROOT}/replica_pilot_v22_lambda3_cached_cuda}"
CENTRAL_GRID_EXACT="${CENTRAL_GRID_EXACT:-${ROOT}/plots/v22_scheme_tmd_stage1_s303_bandgrid_exactx}"
BAND_DIR="${BAND_DIR:-${REPLICA_ROOT}/tmd_bspace_bands_exactx_50rep}"
BAND_AUDIT="${BAND_AUDIT:-${BAND_DIR}/audit}"
Q95_AUDIT="${Q95_AUDIT:-${REPLICA_ROOT}/audit_convergence_q95_50rep}"
BOOTSTRAP_AUDIT="${BOOTSTRAP_AUDIT:-${REPLICA_ROOT}/audit_bootstrap_split_norms_50rep}"

die() {
  echo "$*" >&2
  exit 1
}

copy_path() {
  local src="$1"
  local dst="$2"
  [[ -e "${src}" ]] || die "Missing required path: ${src}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -d "${src}" ]]; then
    rsync -a "${src}/" "${dst}/"
  else
    cp -a "${src}" "${dst}"
  fi
}

copy_optional() {
  local src="$1"
  local dst="$2"
  if [[ -e "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    if [[ -d "${src}" ]]; then
      rsync -a "${src}/" "${dst}/"
    else
      cp -a "${src}" "${dst}"
    fi
  fi
}

for path in \
  "${CENTRAL_RUN}/metrics.json" \
  "${CENTRAL_RUN}/model_state.pt" \
  "${CACHE_EXPORT}/backend_cache/cache_paths.json" \
  "${BAND_DIR}/v22_tmd_replica_bspace_bands.csv" \
  "${BAND_DIR}/v22_tmd_replica_bspace_long.csv" \
  "${BAND_AUDIT}/bspace_band_audit_summary.json" \
  "${Q95_AUDIT}/lambda3_ensemble_q95_summary.json" \
  "${BOOTSTRAP_AUDIT}/bootstrap_split_norm_summary.json"
do
  [[ -e "${path}" ]] || die "Missing required freeze input: ${path}"
done

if [[ -e "${FREEZE_DIR}" || -e "${TARBALL}" ]]; then
  die "Refusing to overwrite existing frozen output: ${FREEZE_DIR} or ${TARBALL}"
fi

mkdir -p "${FREEZE_DIR}"

echo "Freezing to ${FREEZE_DIR}"

# Core code
copy_path "${ROOT}/v22/backends/bt_internal_css_backend_v22_full.py" "${FREEZE_DIR}/code/v22/backends/bt_internal_css_backend_v22_full.py"
copy_path "${ROOT}/v22/src" "${FREEZE_DIR}/code/v22/src"
copy_optional "${ROOT}/v22/tools" "${FREEZE_DIR}/code/v22/tools"

# Key runner scripts if present
for f in \
  export_v22_backend_cache.sh \
  run_v22_cache_cuda_smoketest.sh \
  run_v22_lambda3_cached_cuda_replicas.sh \
  append_v22_lambda3_cached_cuda_replicas.sh \
  run_v22_central_refit_stage1.sh
do
  copy_optional "${ROOT}/${f}" "${FREEZE_DIR}/code/${f}"
done

copy_optional "${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py" "${FREEZE_DIR}/code/train_bt_dnn_v21_replica_stable.py"
copy_optional "${ROOT}/v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py" "${FREEZE_DIR}/code/train_bt_dnn_v21_smoothedA_tail.py"
copy_optional "${ROOT}/v21_tail_release_amp0p019_candidate/production_frozen/REPLICA_PROTOCOL/prepare_v21_replica_norm_inits.py" "${FREEZE_DIR}/code/prepare_v21_replica_norm_inits.py"

# Central run and cache
copy_path "${CENTRAL_RUN}" "${FREEZE_DIR}/central_run"
copy_path "${CACHE_EXPORT}/backend_cache" "${FREEZE_DIR}/backend_cache"

# Replica outputs and audits
copy_path "${REPLICA_ROOT}/outputs" "${FREEZE_DIR}/replicas/outputs"
copy_optional "${REPLICA_ROOT}/logs" "${FREEZE_DIR}/replicas/logs"
copy_path "${BAND_DIR}" "${FREEZE_DIR}/tmd_bspace_bands_exactx_50rep"
copy_path "${Q95_AUDIT}" "${FREEZE_DIR}/audit_convergence_q95_50rep"
copy_path "${BOOTSTRAP_AUDIT}" "${FREEZE_DIR}/audit_bootstrap_split_norms_50rep"

# Central exact-x grid used for formal band construction
copy_path "${CENTRAL_GRID_EXACT}" "${FREEZE_DIR}/central_tmd_grid_exactx"

# Optional provenance artifacts
copy_optional "${ROOT}/v22/reference/external_tail_benchmark" "${FREEZE_DIR}/external_benchmark/reference_external_tail_benchmark"
copy_optional "${ROOT}/v22/outputs/external_tail_benchmark_reduced" "${FREEZE_DIR}/external_benchmark/reduced_external_tail_benchmark"
copy_optional "${ROOT}/plots/v22_scheme_tmd_stage1_s303_bandgrid" "${FREEZE_DIR}/plots/central_bandgrid_interpx"
copy_optional "${ROOT}/replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx" "${FREEZE_DIR}/pilot_history/tmd_bspace_bands_exactx_10rep"
copy_optional "${ROOT}/replica_pilot_v22_lambda3_cached_cuda/tmd_bspace_bands_exactx_20rep" "${FREEZE_DIR}/pilot_history/tmd_bspace_bands_exactx_20rep"

# Environment capture
mkdir -p "${FREEZE_DIR}/environment"
{
  echo "python: $(python3 --version 2>&1)"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "root: ${ROOT}"
} > "${FREEZE_DIR}/environment/runtime.txt"

python3 -m pip freeze > "${FREEZE_DIR}/environment/pip-freeze.txt" || true
conda env export --from-history > "${FREEZE_DIR}/environment/conda-env-from-history.yml" 2>/dev/null || true
nvidia-smi > "${FREEZE_DIR}/environment/nvidia-smi.txt" 2>/dev/null || true

python3 - <<'PY' > "${FREEZE_DIR}/environment/torch_cuda.txt"
import json
try:
    import torch
    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    info = {"error": repr(exc)}
print(json.dumps(info, indent=2))
PY

# Write manifest and notes from audited JSON files.
FREEZE_DIR_ENV="${FREEZE_DIR}" python3 - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path

root = Path(os.environ["FREEZE_DIR_ENV"])

def load(rel: str):
    p = root / rel
    if not p.exists():
        return {}
    with p.open() as handle:
        return json.load(handle)

band = load("tmd_bspace_bands_exactx_50rep/audit/bspace_band_audit_summary.json")
q95 = load("audit_convergence_q95_50rep/lambda3_ensemble_q95_summary.json")
boot = load("audit_bootstrap_split_norms_50rep/bootstrap_split_norm_summary.json")
central = load("central_run/metrics.json")
cache = load("backend_cache/cache_paths.json")

manifest = {
    "tag": root.name,
    "status": "frozen_v22_lambda3_50rep_DYonly_bspace_TMD_ensemble",
    "scope": {
        "physics": "Drell-Yan-only b-space unpolarized TMDPDF ensemble",
        "excluded": [
            "kT-space production TMDs",
            "SIDIS/e+e- constraints",
            "PDF/member uncertainty",
            "scale/profile variation uncertainty",
            "nuclear-model uncertainty beyond selected target mode"
        ],
        "formal_x_grid": [0.20, 0.30, 0.50],
        "visual_interpolated_x_values_if_plotted": [0.15, 0.40],
        "Q_values_GeV": [5.0, 10.0],
        "flavors": ["u", "d", "ubar", "dbar"]
    },
    "central_refit": {
        "run_dir": "central_run",
        "chi2_total": central.get("chi2_total"),
        "per_dataset": central.get("per_dataset"),
    },
    "ensemble": {
        "lambda_replica_logf_anchor": 3,
        "n_replicas": q95.get("n_replicas"),
        "seed_range": "1001-1050",
        "chi2_median": q95.get("chi2_median"),
        "chi2_q95": q95.get("chi2_q95"),
        "chi2_max": q95.get("chi2_max"),
        "norm_pull_q95": q95.get("norm_pull_q95"),
        "norm_pull_max": q95.get("norm_pull_max"),
        "fit_distribution_pass": q95.get("fit_distribution_pass"),
        "norm_distribution_pass": q95.get("norm_distribution_pass"),
    },
    "bspace_band_audit": {
        "technical_pass": band.get("BSPACE_TMD_BAND_TECHNICAL_PASS"),
        "uncertainty_useful_pass": band.get("BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS"),
        "max_relative_68_halfwidth_active": band.get("max_relative_68_halfwidth_active"),
        "max_central_vs_replica_median_rel_p90_active": band.get("max_central_vs_replica_median_rel_p90_active"),
        "interpolated_x_values": band.get("interpolated_x_values"),
    },
    "convergence": {
        "q95_gate_pass": q95.get("V22_LAMBDA3_BSPACE_ENSEMBLE_Q95_PASS"),
        "strict_freeze_pass": boot.get("STRICT_20REP_FREEZE_PASS"),
        "conditional_pass": boot.get("CONDITIONAL_20REP_BSPACE_PILOT_PASS"),
        "random_split_width_q90": boot.get("random_split_width_q90"),
        "random_split_center_q90": boot.get("random_split_center_q90"),
        "note": "Bootstrap script key still says 20REP, but this frozen run has n_replicas_from_runs=50 and n_replicas_in_band_long=50."
    },
    "backend_cache": {
        "cache_paths": cache,
        "note": "Cached W/Y grids were used for replica production."
    },
    "top_norm_pull_rows": boot.get("top_norm_pull_rows", [])[:12],
    "production_claim": (
        "The frozen artifact supports DY-only b-space TMDPDF central curves and 68% replica bands "
        "at the specified exact-x grid. kT-space TMDs remain diagnostic only."
    )
}
(root / "PRODUCTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

notes = f"""# {root.name}

Frozen v22 lambda=3 50-replica DY-only **b-space** TMD ensemble.

## Status

PASS for the DY-only b-space ensemble freeze.

Important audited values:

- n replicas: {q95.get('n_replicas')}
- chi2 q95: {q95.get('chi2_q95')}
- norm-pull q95: {q95.get('norm_pull_q95')}
- b-space band technical pass: {band.get('BSPACE_TMD_BAND_TECHNICAL_PASS')}
- b-space band uncertainty-useful pass: {band.get('BSPACE_TMD_BAND_UNCERTAINTY_USEFUL_PASS')}
- q95 ensemble pass: {q95.get('V22_LAMBDA3_BSPACE_ENSEMBLE_Q95_PASS')}
- bootstrap/random-split strict pass: {boot.get('STRICT_20REP_FREEZE_PASS')}
- random split width q90: {boot.get('random_split_width_q90')}
- random split center q90: {boot.get('random_split_center_q90')}

## Scope

This artifact is for b-space DY-only TMDPDFs. It does **not** freeze kT-space TMDs for production use.

Formal exact-x grid:
- x = 0.20, 0.30, 0.50

Interpolated visual grid, if used:
- x = 0.15, 0.40

## Caveats

- The fitted nonperturbative factor is flavor independent in this model.
- Full TMDPDF flavor dependence enters through the collinear PDFs and OPE.
- Cross-section bands include fitted random normalization draws.
- kT-space transforms remain diagnostic because high-k closure was not stable enough.
- One E605 normalization outlier remains visible, but the norm-pull q95 passes.
"""
(root / "FREEZE_NOTES.md").write_text(notes)
PY

# Verify replica count inside frozen artifact.
N_REP="$(find "${FREEZE_DIR}/replicas/outputs" -mindepth 1 -maxdepth 1 -type d -name 'v22_lambda3_cached_cuda_s*' | wc -l | tr -d ' ')"
if [[ "${N_REP}" != "50" ]]; then
  die "Expected 50 frozen replica output directories; found ${N_REP}"
fi

# Checksums. Exclude SHA256SUMS itself.
(
  cd "${FREEZE_DIR}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

# Tarball.
(
  cd "${FREEZE_PARENT}"
  tar -czf "${TARBALL}" "${TAG}"
)

echo
echo "Frozen artifact created:"
echo "  ${FREEZE_DIR}"
echo "  ${TARBALL}"
echo
echo "Verify with:"
echo "  cd '${FREEZE_DIR}'"
echo "  sha256sum -c SHA256SUMS"
echo "  cat PRODUCTION_MANIFEST.json"
