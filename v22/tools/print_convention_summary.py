#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "conventions.json"
with path.open() as handle:
    data = json.load(handle)

print(json.dumps(data, indent=2))
