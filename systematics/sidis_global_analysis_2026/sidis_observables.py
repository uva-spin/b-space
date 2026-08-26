"""Convention-explicit SIDIS PDF-times-TMDFF structure-function building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.special import j0

Array = np.ndarray
TMDValue = Callable[[Array, float, float], Array]


@dataclass(frozen=True)
class SidisKinematics:
    x: float
    q2: float
    z: float
    pht: float
    y: float | None = None

    def validate(self) -> None:
        if not 0.0 < self.x < 1.0 or self.q2 <= 0.0 or not 0.0 < self.z < 1.0 or self.pht < 0.0:
            raise ValueError("invalid SIDIS kinematics")
        if self.y is not None and not 0.0 < self.y < 1.0:
            raise ValueError("y must lie in (0,1)")


@dataclass(frozen=True)
class RadialConvolutionConvention:
    qT_definition: str = "P_hT / z"
    measure: str = "b db / (2 pi)"

    def qT(self, pht: Array | float, z: float) -> Array:
        if z <= 0.0:
            raise ValueError("z must be positive")
        return np.asarray(pht, dtype=float) / z


def uu_structure_function(
    b_grid: Array,
    pht: Array | float,
    z: float,
    pdfs: Mapping[str, TMDValue],
    tmdffs: Mapping[str, TMDValue],
    charge_squared: Mapping[str, float],
    *,
    q2: float,
    x: float,
    convention: RadialConvolutionConvention | None = None,
    hard_factor: Callable[[float, float], float] | None = None,
) -> Array:
    """Return only the radial ``F_UU`` convolution, not a final multiplicity."""
    b = np.asarray(b_grid, dtype=float)
    if b.ndim != 1 or len(b) < 2 or np.any(~np.isfinite(b)) or np.any(b < 0.0) or np.any(np.diff(b) <= 0.0):
        raise ValueError("b_grid must be a finite, strictly increasing non-negative grid")
    if not np.isfinite(z) or not 0.0 < z < 1.0:
        raise ValueError("z must lie in (0,1)")
    pht_array = np.atleast_1d(np.asarray(pht, dtype=float))
    if np.any(~np.isfinite(pht_array)) or np.any(pht_array < 0.0):
        raise ValueError("pht must be finite and non-negative")
    qT = (convention or RadialConvolutionConvention()).qT(pht_array, z)
    result = np.zeros_like(pht_array)
    for flavor, pdf in pdfs.items():
        if flavor not in tmdffs or flavor not in charge_squared:
            raise KeyError(f"missing TMDFF or charge for flavor {flavor!r}")
        f = np.asarray(pdf(b, x, q2), dtype=float)
        d = np.asarray(tmdffs[flavor](b, z, q2), dtype=float)
        if f.shape != b.shape or d.shape != b.shape or np.any(~np.isfinite(f)) or np.any(~np.isfinite(d)):
            raise ValueError(f"invalid b-space value for flavor {flavor!r}")
        integrand = (b * f * d)[None, :] * j0(qT[:, None] * b[None, :])
        result += float(charge_squared[flavor]) * np.trapezoid(integrand, b, axis=1) / (2.0 * np.pi)
    if hard_factor is not None:
        result *= float(hard_factor(x, q2))
    return result


def multiplicity_ratio(sidis: Array | float, dis: Array | float) -> Array:
    numerator = np.asarray(sidis, dtype=float)
    denominator = np.asarray(dis, dtype=float)
    if np.any(~np.isfinite(numerator)) or np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("SIDIS numerator and positive finite DIS denominator are required")
    return numerator / denominator
