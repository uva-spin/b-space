"""Exact bin quadrature and collider-aware v22 asymptotic adapter.

This experimental module does not modify the imported production backend. It
temporarily supplies the collider-aware luminosity to the isolated v22 strict
one-loop expansion and evaluates row copies at explicit qT/y nodes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import j0
from scipy.integrate import simpson


PointEvaluator = Callable[[pd.Series], float]
AcceptanceEvaluator = Callable[[pd.Series], float]


@dataclass(frozen=True)
class BinAsymptoticResult:
    value_pb_per_GeV: float
    qT_low: float
    qT_high: float
    y_low: float
    y_high: float
    n_qT: int
    n_y: int
    acceptance_mode: str


def _gauss_interval(low: float, high: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"invalid integration interval [{low}, {high}]")
    if n < 2:
        raise ValueError("quadrature order must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (high - low)
    return 0.5 * (high + low) + half * nodes, half * weights


def rapidity_interval(row: pd.Series) -> tuple[float, float]:
    y_low = float(row.get("y_Low", np.nan))
    y_high = float(row.get("y_High", np.nan))
    if math.isfinite(y_low) and math.isfinite(y_high) and y_high > y_low:
        return y_low, y_high
    q = float(row["QM"])
    sqrts = float(row["SqrtS"])
    if not (q > 0.0 and sqrts > q):
        raise ValueError("cannot derive the leading-power rapidity interval")
    bound = math.log(sqrts / q)
    return -bound, bound


def node_row(row: pd.Series, *, qT: float, y: float) -> pd.Series:
    out = row.copy()
    q = float(out["QM"])
    sqrts = float(out["SqrtS"])
    tau = q / sqrts
    x1 = tau * math.exp(y)
    x2 = tau * math.exp(-y)
    if not (0.0 < x1 < 1.0 and 0.0 < x2 < 1.0):
        raise ValueError(f"unphysical node x1={x1}, x2={x2}, y={y}")
    out["qT"] = float(qT)
    out["y"] = float(y)
    out["x1"] = float(x1)
    out["x2"] = float(x2)
    # Force the collider prefactor to use the point differential 2*pi*qT
    # Jacobian. Bin averaging is performed explicitly outside the evaluator.
    out["qT_low"] = np.nan
    out["qT_high"] = np.nan
    out["qT_bin_width"] = np.nan
    return out


def integrate_exact_bin(
    row: pd.Series,
    *,
    point_evaluator: PointEvaluator,
    n_qT: int = 5,
    n_y: int = 9,
    acceptance: AcceptanceEvaluator | None = None,
) -> BinAsymptoticResult:
    qT_low = float(row["qT_low"])
    qT_high = float(row["qT_high"])
    y_low, y_high = rapidity_interval(row)
    target = str(row.get("target", "")).strip().lower()
    is_fiducial = target == "pp" or "fiducial" in str(row.get("observable_name", "")).lower()
    if is_fiducial and acceptance is None:
        raise ValueError(
            "fiducial collider row requires an explicit node-level acceptance/decay evaluator"
        )
    qt_nodes, qt_weights = _gauss_interval(qT_low, qT_high, n_qT)
    y_nodes, y_weights = _gauss_interval(y_low, y_high, n_y)
    total = 0.0
    for qT, q_weight in zip(qt_nodes, qt_weights):
        for y, y_weight in zip(y_nodes, y_weights):
            current = node_row(row, qT=float(qT), y=float(y))
            factor = float(acceptance(current)) if acceptance is not None else 1.0
            if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
                raise ValueError(f"invalid acceptance {factor}")
            value = float(point_evaluator(current))
            if not math.isfinite(value):
                raise FloatingPointError("point asymptotic evaluator returned non-finite value")
            total += float(q_weight * y_weight) * factor * value
    return BinAsymptoticResult(
        value_pb_per_GeV=float(total / (qT_high - qT_low)),
        qT_low=qT_low, qT_high=qT_high, y_low=y_low, y_high=y_high,
        n_qT=int(n_qT), n_y=int(n_y),
        acceptance_mode="explicit_node_acceptance" if acceptance is not None else "inclusive",
    )


@contextmanager
def collider_luminosity_patch(backend):
    """Patch only the in-memory scheme module, restoring it on exit."""
    scheme = backend._scheme
    original = scheme._v22_luminosity
    scheme._v22_luminosity = backend._TEVATRON_V22_LUMINOSITY
    try:
        yield scheme
    finally:
        scheme._v22_luminosity = original


def make_v22_point_evaluator(*, backend, pdf, cfg) -> PointEvaluator:
    def evaluate(row: pd.Series) -> float:
        with collider_luminosity_patch(backend) as scheme:
            return float(
                scheme.singular_nlo_v22_wexp_numeric_for_row(
                    row, pdf, cfg, positive=False
                )
            )
    return evaluate


def make_resummed_w_point_evaluators(
    *, backend, pdf, cfg, np_pair_factor=None,
    remove_inclusive_rapidity_approximation: bool = False,
    integration_rule: str = "trapezoid",
):
    """Return cached perturbative and fitted resummed-W point evaluators.

    `np_pair_factor`, when supplied, must return the row-aligned pair factor on
    the provided b grid. Both evaluators share the expensive perturbative
    b-space kernels, ensuring that their difference is not a recomputation
    artifact.
    """
    b_grid = np.asarray(backend.make_b_grid(cfg), dtype=float)
    rule = str(integration_rule).strip().lower()
    if rule not in {"trapezoid", "simpson"}:
        raise ValueError("integration_rule must be trapezoid or simpson")
    cache: dict[tuple[float, float, float, float], np.ndarray] = {}

    def transform(values: np.ndarray, qT: float) -> float:
        integrand_values = b_grid * j0(qT * b_grid) * values
        if rule == "simpson":
            return float(simpson(integrand_values, x=b_grid))
        return float(np.trapezoid(integrand_values, x=b_grid))

    def integrand(row: pd.Series) -> np.ndarray:
        key = (float(row["qT"]), float(row["y"]), float(row["x1"]), float(row["x2"]))
        if key not in cache:
            values = np.asarray(backend.wpert_cs_for_row(row, b_grid, pdf, cfg), dtype=float)
            if remove_inclusive_rapidity_approximation:
                rapidity_factor_fn = getattr(backend, "_tevatron_rapidity_factor", None)
                if rapidity_factor_fn is None:
                    raise AttributeError("backend lacks the explicit Tevatron rapidity-factor hook")
                factor = float(rapidity_factor_fn(row))
                if not math.isfinite(factor) or factor <= 0.0:
                    raise FloatingPointError("invalid inclusive rapidity approximation factor")
                values = values / factor
            if values.shape != b_grid.shape or not np.isfinite(values).all():
                raise FloatingPointError("invalid resummed-W b-space kernel")
            cache[key] = values
        return cache[key]

    def perturbative(row: pd.Series) -> float:
        values = integrand(row)
        return transform(values, float(row["qT"]))

    def fitted(row: pd.Series) -> float:
        values = integrand(row)
        if np_pair_factor is None:
            factor = np.ones_like(b_grid)
        else:
            factor = np.asarray(np_pair_factor(float(row["x1"]), float(row["x2"]), b_grid), dtype=float)
            if factor.shape != b_grid.shape or not np.isfinite(factor).all() or (factor < 0.0).any():
                raise FloatingPointError("invalid fitted nonperturbative pair factor")
        return transform(values * factor, float(row["qT"]))

    return perturbative, fitted
