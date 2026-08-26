"""Strict covariance operations for the isolated SIDIS likelihood boundary.

The ingestion layer keeps statistical, systematic, normalization, and theory
components separate. This module only evaluates a caller-supplied covariance
matrix; it never invents correlations or combines uncertainty components.
"""

from __future__ import annotations

import numpy as np


def _validated_inputs(residual: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(residual, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if vector.ndim != 1:
        raise ValueError("residual must be a one-dimensional vector")
    if matrix.ndim != 2 or matrix.shape != (vector.size, vector.size):
        raise ValueError(f"covariance shape {matrix.shape} does not match residual size {vector.size}")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
        raise ValueError("residual and covariance must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("covariance matrix is not symmetric")
    return vector, matrix


def whiten_residual(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return ``L^-1 residual`` for ``covariance=L L^T``."""

    vector, matrix = _validated_inputs(residual, covariance)
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return np.linalg.solve(factor, vector)


def correlated_chi2(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Evaluate the correlated quadratic form ``residual.T cov^-1 residual``."""

    whitened = whiten_residual(residual, covariance)
    result = float(np.dot(whitened, whitened))
    if not np.isfinite(result):
        raise ValueError("correlated chi2 is non-finite")
    return result


def covariance_condition_number(covariance: np.ndarray) -> float:
    """Return the 2-norm condition number after strict symmetry validation."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite square matrix")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("covariance matrix is not symmetric")
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition):
        raise ValueError("covariance condition number is non-finite")
    return condition
