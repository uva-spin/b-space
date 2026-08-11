"""Pure additive-matching composition for the isolated experimental track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchedComponents:
    w: np.ndarray
    fixed_order: np.ndarray
    asymptotic: np.ndarray
    profile: np.ndarray
    y: np.ndarray
    matched: np.ndarray


def _component(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional row-aligned array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def compose_additive_matched(*, w, fixed_order, asymptotic, profile) -> MatchedComponents:
    """Compose row/bin-aligned W + profile * (FO - ASY).

    This function performs no physics approximation and no implicit unit or
    normalization conversion. Callers must provide already canonicalized,
    identically ordered bin-level components.
    """
    w_array = _component("w", w)
    fo_array = _component("fixed_order", fixed_order)
    asym_array = _component("asymptotic", asymptotic)
    profile_array = _component("profile", profile)
    shapes = {array.shape for array in (w_array, fo_array, asym_array, profile_array)}
    if len(shapes) != 1:
        raise ValueError(f"component shapes differ: {sorted(shapes)}")
    if ((profile_array < 0.0) | (profile_array > 1.0)).any():
        raise ValueError("profile must lie in [0, 1]")
    y_array = profile_array * (fo_array - asym_array)
    matched_array = w_array + y_array
    return MatchedComponents(
        w=w_array.copy(), fixed_order=fo_array.copy(), asymptotic=asym_array.copy(),
        profile=profile_array.copy(), y=y_array, matched=matched_array,
    )
