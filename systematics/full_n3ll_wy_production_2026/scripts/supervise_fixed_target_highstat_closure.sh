#!/usr/bin/env bash
# Restart-safe, isolated fixed-target W+Y convention/cancellation campaign.
# This never writes the frozen production package and does not authorize the
# 353-row scope. It waits for the Tevatron gates so it cannot contend with the
# primary production calculation indefinitely.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
REPORTS="$ROOT/reports"
TEV="$REPORTS/tevatron_n3ll_nnlo_wy_final_g1_1p017"
SCALE="$REPORTS/tevatron_scale_variations_g1_1p017"
OUT="$REPORTS/fixed_target_highstat_wy_closure_20260819"
LOG="$OUT/campaign_supervisor.log"
GRID="$TEV/grid_status.json"
SCALE_STATUS="$SCALE/scale_variation_status.json"
SCALE_REFINE="$SCALE/scale_variation_refinement_status.json"
PROBE="$ROOT/scripts/run_fixed_target_n3ll_nnlo_probe.py"
mkdir -p "$OUT"

good_grid() {
  [[ -f "$GRID" ]] || return 1
  "$PY" - "$GRID" <<'PY'
import json, sys
s=json.load(open(sys.argv[1])); c=s.get("checks", {})
raise SystemExit(0 if s.get("row_count") == 122 and c.get("all_finite") and c.get("all_positive") else 1)
PY
}

good_scale() {
  [[ -f "$SCALE_STATUS" && -f "$SCALE_REFINE" ]] || return 1
  "$PY" - "$SCALE_STATUS" "$SCALE_REFINE" <<'PY'
import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
raise SystemExit(0 if a.get("row_count") == 122 and a.get("all_finite")
                 and a.get("all_positive") and b.get("all_finite")
                 and b.get("all_positive") else 1)
PY
}

echo "[$(date -Is)] waiting for Tevatron finite/positive and scale gates" >> "$LOG"
while ! good_grid || ! good_scale; do sleep 60; done
echo "[$(date -Is)] Tevatron gates passed; starting high-stat fixed-target closure" >> "$LOG"

# One representative row per experiment, repeated at three independent seeds.
# Proton comparisons are limited to E288 rows; nuclear/per-nucleon cards cover
# all five families.
declare -a specs=(
  "E288_200|E288_200:0|nuc|4|9.0121831|be"
  "E288_300|E288_300:0|nuc|4|9.0121831|be"
  "E288_400|E288_400:0|nuc|4|9.0121831|be"
  "E605|E605:0|nuc|29|63.546|cu"
  "E772|E772:0|nuc|0.5|1|isoscalar"
  "E288_200|E288_200:0|pp|||pp"
  "E288_400|E288_400:0|pp|||pp"
)

for spec in "${specs[@]}"; do
  IFS='|' read -r dataset row mode z a label <<< "$spec"
  for seed_offset in 0 1 2; do
    tag="${dataset}_${mode}_100M_s$((20264000 + seed_offset))"
    status="$OUT/${dataset}_0_full_n3ll_nnlo_probe_${tag}_status.json"
    if [[ -f "$status" ]]; then
      echo "[$(date -Is)] reusing $status" >> "$LOG"
      continue
    fi
    cmd=("$PY" "$PROBE" --dataset "$dataset" --row "$row" --calls 100000000
         --seed "$((20264000 + seed_offset))" --tag "$tag" --out "$OUT")
    if [[ "$mode" == nuc ]]; then
      cmd+=(--target-z "$z" --target-a "$a")
    fi
    echo "[$(date -Is)] running ${dataset} ${mode} seed=$((20264000 + seed_offset))" >> "$LOG"
    "${cmd[@]}" >> "$LOG" 2>&1
  done
done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); rows=[]
for p in sorted(out.glob('*_status.json')):
    if p.name == 'closure_summary.json': continue
    try: rows.append(json.loads(p.read_text()))
    except Exception: continue
groups={}
for r in rows:
    key=(r.get('dataset'), 'nuc' if r.get('target_A') is not None else 'pp')
    groups.setdefault(key, []).append(r)
summary={'status':'isolated_fixed_target_highstat_closure_complete_not_production',
         'calls_per_vegas_component':100000000,'independent_seed_count':3,
         'groups':{},'production_outputs_modified':False,
         'promotion_authorized':False}
for key, vals in sorted(groups.items()):
    raw=np.array([v.get('raw_full_wy_fb_per_bin') for v in vals],float)
    unc=np.array([v.get('raw_full_wy_unc_fb_per_bin') for v in vals],float)
    per=np.array([v.get('per_nucleon_full_wy_fb_per_GeV') for v in vals
                  if v.get('per_nucleon_full_wy_fb_per_GeV') is not None],float)
    summary['groups'][f'{key[0]}_{key[1]}']={
        'n':len(vals),'finite':bool(np.isfinite(raw).all()),
        'raw_values_fb_per_bin':raw.tolist(),
        'raw_uncertainties_fb_per_bin':unc.tolist(),
        'raw_coefficient_of_variation':float(np.std(raw)/max(abs(np.mean(raw)),1e-30)) if len(raw)>1 else None,
        'per_nucleon_values_fb_per_GeV':per.tolist(),
        'per_nucleon_coefficient_of_variation':float(np.std(per)/max(abs(np.mean(per)),1e-30)) if len(per)>1 else None,
        'statuses':[v.get('status') for v in vals],
    }
(out/'closure_summary.json').write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary,indent=2))
PY
echo "[$(date -Is)] fixed-target high-stat closure complete; promotion remains fail-closed" >> "$LOG"
