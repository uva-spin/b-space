#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-.}"
rsync -av ./ "$DEST"/
echo "Staged b-space GitHub files into $DEST"
