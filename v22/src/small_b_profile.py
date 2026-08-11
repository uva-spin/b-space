"""Small-b perturbative-coordinate profiles for the v22 OPE."""

from __future__ import annotations

import math

from v22.src.css2_ope_nlo import B0


def b_min_from_Q(
    *,
    Q_GeV: float,
    C5: float = 1.0,
) -> float:
    """Return b_min=b0/(C5 Q), in GeV^-1."""

    Q = float(Q_GeV)
    C5 = float(C5)

    if not math.isfinite(Q) or Q <= 0.0:
        raise ValueError("Q must be finite and positive")

    if not math.isfinite(C5) or C5 <= 0.0:
        raise ValueError("C5 must be finite and positive")

    return B0 / (C5 * Q)


def smooth_max_power(
    a: float,
    b: float,
    *,
    power: float = 16.0,
) -> float:
    """Stable positive p-norm approximation to max(a,b)."""

    a = float(a)
    b = float(b)
    p = float(power)

    if not math.isfinite(a) or a < 0.0:
        raise ValueError("a must be finite and nonnegative")

    if not math.isfinite(b) or b < 0.0:
        raise ValueError("b must be finite and nonnegative")

    if not math.isfinite(p) or p <= 1.0:
        raise ValueError("power must be finite and greater than one")

    scale = max(a, b)

    if scale == 0.0:
        return 0.0

    return scale * (
        (a / scale) ** p
        + (b / scale) ** p
    ) ** (1.0 / p)


def b_ope_profile(
    *,
    b_star_GeV_inv: float,
    Q_GeV: float,
    C5: float = 1.0,
    power: float = 16.0,
    kind: str = "smooth",
) -> float:
    """Return the perturbative coordinate entering the OPE coefficients."""

    b_star = float(b_star_GeV_inv)

    if not math.isfinite(b_star) or b_star < 0.0:
        raise ValueError(
            "b_star must be finite and nonnegative"
        )

    b_min = b_min_from_Q(
        Q_GeV=Q_GeV,
        C5=C5,
    )

    mode = str(kind).strip().lower()

    if mode == "hard":
        return max(b_star, b_min)

    if mode == "smooth":
        return smooth_max_power(
            b_star,
            b_min,
            power=power,
        )

    raise ValueError(
        "kind must be 'hard' or 'smooth'"
    )


def matching_log_Lb(
    *,
    b_pert_GeV_inv: float,
    mu_GeV: float,
) -> float:
    """Return L_b=ln(mu^2 b_pert^2/b0^2)."""

    b = float(b_pert_GeV_inv)
    mu = float(mu_GeV)

    if not math.isfinite(b) or b <= 0.0:
        raise ValueError(
            "b_pert must be finite and positive"
        )

    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError(
            "mu must be finite and positive"
        )

    return 2.0 * math.log(mu * b / B0)
