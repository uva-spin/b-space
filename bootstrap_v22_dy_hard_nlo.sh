#!/usr/bin/env bash
set -euo pipefail

# Add the one-loop CSS2 Drell--Yan hard factor.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
SRC="${ROOT}/v22/src/dy_hard_nlo.py"
TEST="${ROOT}/v22/tests/run_dy_hard_nlo_smoke.py"
DOC="${ROOT}/v22/DY_HARD_FACTOR.md"

for required in \
  "${ROOT}/v22/CONVENTIONS.md" \
  "${ROOT}/v22/CSS2_SCHEME.md" \
  "${ROOT}/v22/src/css2_ope_nlo.py"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required v22 file: ${required}" >&2
    exit 1
  fi
done

for target in "${SRC}" "${TEST}" "${DOC}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing ${target}" >&2
    exit 1
  fi
done

cat > "${DOC}" <<'MD'
# v22 one-loop Drell--Yan hard factor

The CSS2 Drell--Yan hard factor is the absolute square of the
time-like quark form-factor hard part.  The electric charge factor is
**not** included here because the v22 flavor luminosity already contains
\(e_q^2\).

The internal perturbative expansion parameter is

\[
a_s(\mu)=\frac{\alpha_s(\mu)}{4\pi}.
\]

Define

\[
T=\ln\frac{Q^2}{\mu^2}.
\]

At one loop,

\[
H_{\rm DY}(Q,\mu)
=
1
+
a_s(\mu)\,C_F
\left[
-16+\frac{7\pi^2}{3}+6T-2T^2
\right]
+\mathcal O(a_s^2).
\]

The default hard scale is \(\mu_H=Q\), so

\[
H_{\rm DY}(Q,Q)
=
1+
a_s(Q)\,C_F
\left[
-16+\frac{7\pi^2}{3}
\right].
\]

The hard factor is separate from:

- the fixed-target electromagnetic and unit-conversion prefactor;
- the TMD OPE coefficients;
- the Sudakov evolution;
- the finite \(Y\) term.

For strict NLO bookkeeping,

\[
W_{\rm NLO}
=
W_{\rm Born}
+
\delta W_H
+
\delta W_{\rm OPE}.
\]

Multiplying the complete NLO hard factor by two complete NLO legs is a
valid resummed implementation choice, but it contains formally
higher-order cross terms.  v22 audits those terms explicitly before
using the multiplicative form.
MD

cat > "${SRC}" <<'PY'
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
PY

cat > "${TEST}" <<'PY'
from __future__ import annotations

import math

from v22.src.css2_ope_nlo import CF
from v22.src.dy_hard_nlo import (
    dy_hard_coefficient_1,
    dy_hard_nlo,
    dy_hard_nlo_at_Q,
    hard_log_T,
)


def assert_close(
    actual: float,
    expected: float,
    *,
    rel: float = 2.0e-12,
    abs_tol: float = 2.0e-13,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=rel,
        abs_tol=abs_tol,
    ):
        raise AssertionError(
            f"actual={actual:.16e}, "
            f"expected={expected:.16e}"
        )


checks = 0


# 1: T=0 at mu=Q.
assert_close(
    hard_log_T(Q_GeV=10.0, mu_GeV=10.0),
    0.0,
)
checks += 1


# 2: T=ln 4 at mu=Q/2.
assert_close(
    hard_log_T(Q_GeV=10.0, mu_GeV=5.0),
    math.log(4.0),
)
checks += 1


# 3: canonical hard coefficient.
expected_h1 = CF * (
    -16.0 + 7.0 * math.pi**2 / 3.0
)

assert_close(
    dy_hard_coefficient_1(
        Q_GeV=10.0,
        mu_GeV=10.0,
    ),
    expected_h1,
)
checks += 1


# 4: alpha_s=0 gives H=1.
assert_close(
    dy_hard_nlo(
        Q_GeV=10.0,
        mu_GeV=10.0,
        alpha_s=0.0,
    ),
    1.0,
)
checks += 1


# 5: direct one-loop formula at mu=Q.
alpha_s = 0.2
expected = (
    1.0
    + alpha_s
    / (4.0 * math.pi)
    * expected_h1
)

assert_close(
    dy_hard_nlo_at_Q(
        Q_GeV=10.0,
        alpha_s_at_Q=alpha_s,
    ),
    expected,
)
checks += 1


# 6: general-scale implementation.
Q = 12.0
mu = 7.0
alpha_s = 0.19
T = math.log(Q**2 / mu**2)
expected_general = (
    1.0
    + alpha_s
    / (4.0 * math.pi)
    * CF
    * (
        -16.0
        + 7.0 * math.pi**2 / 3.0
        + 6.0 * T
        - 2.0 * T**2
    )
)

assert_close(
    dy_hard_nlo(
        Q_GeV=Q,
        mu_GeV=mu,
        alpha_s=alpha_s,
    ),
    expected_general,
)
checks += 1


# 7-8: invalid scales are rejected.
for bad_Q, bad_mu in [
    (0.0, 1.0),
    (1.0, 0.0),
]:
    try:
        hard_log_T(
            Q_GeV=bad_Q,
            mu_GeV=bad_mu,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"invalid scales were not rejected: "
            f"Q={bad_Q}, mu={bad_mu}"
        )

    checks += 1


print(f"{checks} Drell--Yan hard-factor checks passed")
PY

touch \
  "${ROOT}/v22/__init__.py" \
  "${ROOT}/v22/src/__init__.py"

python3 -m py_compile \
  "${SRC}" \
  "${TEST}"

cd "${ROOT}"

PYTHONPATH=. \
python3 \
  v22/tests/run_dy_hard_nlo_smoke.py

echo
echo "Created:"
echo "  v22/DY_HARD_FACTOR.md"
echo "  v22/src/dy_hard_nlo.py"
echo "  v22/tests/run_dy_hard_nlo_smoke.py"
