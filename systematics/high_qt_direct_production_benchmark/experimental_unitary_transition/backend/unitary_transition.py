"""Pure, isolated unitary transition utilities."""

from __future__ import annotations

import math
import numpy as np


def smootherstep_profile(r, *, r_start: float = 0.20, r_end: float = 0.30):
    """C2 profile equal to zero below r_start and one above r_end."""
    if not (math.isfinite(r_start) and math.isfinite(r_end) and r_end > r_start):
        raise ValueError("profile endpoints must be finite and ordered")
    values = np.asarray(r, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("r must be finite")
    t = np.clip((values - r_start) / (r_end - r_start), 0.0, 1.0)
    out = t**3 * (t * (6.0 * t - 15.0) + 10.0)
    return float(out) if values.ndim == 0 else out


def bin_averaged_profile(qT_low: float, qT_high: float, Q: float, *, r_start=0.20, r_end=0.30, n=16) -> float:
    """Average the profile over a qT bin using Gauss-Legendre quadrature."""
    if not (Q > 0.0 and qT_high > qT_low >= 0.0 and n >= 2):
        raise ValueError("invalid bin-average inputs")
    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (qT_high - qT_low)
    qT = 0.5 * (qT_high + qT_low) + half * nodes
    integral = half * np.sum(weights * smootherstep_profile(qT / Q, r_start=r_start, r_end=r_end))
    return float(integral / (qT_high - qT_low))


def unitary_transition(w, fixed_order, profile):
    """Return the convex W-to-FO transition with strict validation."""
    wv, fv, pv = np.broadcast_arrays(
        np.asarray(w, dtype=float), np.asarray(fixed_order, dtype=float), np.asarray(profile, dtype=float)
    )
    if not (np.isfinite(wv).all() and np.isfinite(fv).all() and np.isfinite(pv).all()):
        raise ValueError("transition inputs must be finite")
    if ((pv < 0.0) | (pv > 1.0)).any():
        raise ValueError("profile must lie in [0,1]")
    out = (1.0 - pv) * wv + pv * fv
    return float(out) if out.ndim == 0 else out
