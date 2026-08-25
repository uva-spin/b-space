#!/usr/bin/env bash
# Keep the existing scale campaign alive across terminal disconnects.
# If the original terminal-scoped supervisor is still alive, wait for it;
# otherwise resume the same restartable supervisor in this user service.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/scripts/supervise_tevatron_scale_campaign.sh"
PY="/home/dustin/miniforge3/envs/pdf-fit/bin/python"
STATUS="$ROOT/reports/tevatron_scale_variations_g1_1p017/scale_variation_status.json"
REFINEMENT="$ROOT/reports/tevatron_scale_variations_g1_1p017/scale_variation_refinement_status.json"

valid_scale_status() {
  [[ -f "$STATUS" ]] || return 1
  "$PY" - "$STATUS" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
raise SystemExit(0 if (s.get("row_count") == 122 and s.get("all_finite") and s.get("all_positive")) else 1)
PY
}

valid_refinement_status() {
  [[ -f "$REFINEMENT" ]] || return 1
  "$PY" - "$REFINEMENT" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
raise SystemExit(0 if (s.get("all_finite") and s.get("all_positive")) else 1)
PY
}

has_terminal_supervisor() {
  pgrep -f 'supervise_tevatron_scale_campaign\.sh$' >/dev/null 2>&1
}

while ! valid_scale_status || ! valid_refinement_status; do
  if ! has_terminal_supervisor; then
    exec /bin/bash "$TARGET"
  fi
  sleep 30
done
exit 0
