"""Isolated unitary finite-Y construction.

This is deliberately distinct from conventional additive ``FO - ASY``
matching.  It defines an additive correction that guarantees the desired
limits when the resummed W is not numerically close to its asymptotic
expansion in the transition region.
"""

from __future__ import annotations

import math
import numpy as np


def smootherstep_profile(r, *, r_start: float = 0.20, r_end: float = 0.30):
    if not (math.isfinite(r_start) and math.isfinite(r_end) and r_end > r_start):
        raise ValueError("profile endpoints must be finite and ordered")
    values = np.asarray(r, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("r must be finite")
    t = np.clip((values - r_start) / (r_end - r_start), 0.0, 1.0)
    out = t**3 * (t * (6.0 * t - 15.0) + 10.0)
    return float(out) if values.ndim == 0 else out


def unitary_y(*, w, fixed_order, r, r_start: float = 0.20, r_end: float = 0.30):
    """Return ``Y_unitary = p(r) * (FO - W)`` with strict validation."""
    wv, fv, rv = np.broadcast_arrays(
        np.asarray(w, dtype=float), np.asarray(fixed_order, dtype=float), np.asarray(r, dtype=float)
    )
    if not (np.isfinite(wv).all() and np.isfinite(fv).all() and np.isfinite(rv).all()):
        raise ValueError("unitary inputs must be finite")
    profile = smootherstep_profile(rv, r_start=r_start, r_end=r_end)
    return profile * (fv - wv)


def unitary_matched(*, w, fixed_order, r, r_start: float = 0.20, r_end: float = 0.30):
    wv, fv, rv = np.broadcast_arrays(
        np.asarray(w, dtype=float), np.asarray(fixed_order, dtype=float), np.asarray(r, dtype=float)
    )
    y = unitary_y(w=wv, fixed_order=fv, r=rv, r_start=r_start, r_end=r_end)
    out = wv + y
    return float(out) if out.ndim == 0 else out
