"""One-loop CSS2 Drell--Yan hard factor.

The electric-charge factor e_q^2 is deliberately excluded.  It belongs
to the flavor luminosity in the v22 observable convention.
"""

from __future__ import annotations

import math

from v22.src.css2_ope_nlo import CF, a_s_from_alpha_s


def hard_log_T(*, Q_GeV: float, mu_GeV: float) -> float:
    """Return T = ln(Q^2/mu^2)."""

    Q = float(Q_GeV)
    mu = float(mu_GeV)

    if not math.isfinite(Q) or Q <= 0.0:
        raise ValueError("Q must be finite and positive")

    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")

    return 2.0 * math.log(Q / mu)


def dy_hard_coefficient_1(
    *,
    Q_GeV: float,
    mu_GeV: float,
) -> float:
    """Coefficient H^(1) in H=1+a_s H^(1)+O(a_s^2)."""

    T = hard_log_T(
        Q_GeV=Q_GeV,
        mu_GeV=mu_GeV,
    )

    return CF * (
        -16.0
        + 7.0 * math.pi**2 / 3.0
        + 6.0 * T
        - 2.0 * T**2
    )


def dy_hard_nlo(
    *,
    Q_GeV: float,
    mu_GeV: float,
    alpha_s: float,
) -> float:
    """Return the one-loop hard factor without e_q^2."""

    return (
        1.0
        + a_s_from_alpha_s(alpha_s)
        * dy_hard_coefficient_1(
            Q_GeV=Q_GeV,
            mu_GeV=mu_GeV,
        )
    )


def dy_hard_nlo_at_Q(
    *,
    Q_GeV: float,
    alpha_s_at_Q: float,
) -> float:
    """Return H_DY(Q,mu=Q)."""

    return dy_hard_nlo(
        Q_GeV=Q_GeV,
        mu_GeV=Q_GeV,
        alpha_s=alpha_s_at_Q,
    )
