#!/usr/bin/env bash
set -euo pipefail

# Install the v22 small-b perturbative-coordinate profile.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
DOC="${ROOT}/v22/SMALL_B_PROFILE.md"
SRC="${ROOT}/v22/src/small_b_profile.py"
TEST="${ROOT}/v22/tests/run_small_b_profile_smoke.py"

for required in \
  "${ROOT}/v22/CONVENTIONS.md" \
  "${ROOT}/v22/src/css2_ope_nlo.py" \
  "${ROOT}/v22/src/css2_ope_nlo_general.py"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required v22 file: ${required}" >&2
    exit 1
  fi
done

for target in "${DOC}" "${SRC}" "${TEST}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing ${target}" >&2
    exit 1
  fi
done

cat > "${DOC}" <<'MD'
# v22 small-\(b_T\) matching-coordinate profile

## Problem exposed by the full-profile audit

The existing scale profile caps the matching scale at the hard scale,

\[
\mu_b \le Q.
\]

If the coefficient functions continue to use the unmodified
\(b_*\to0\) coordinate, then

\[
L_b=\ln\frac{\mu_b^2b_*^2}{b_0^2}
\]

becomes arbitrarily large and negative as \(b_T\to0\). A finite-order
coefficient expansion is then not meaningful point by point.

## Profiled perturbative coordinate

For the OPE coefficients only, define

\[
b_{\min}(Q)=\frac{b_0}{C_5Q},
\]

and a smooth maximum

\[
b_{\rm OPE}
=
\left[
b_*^p+b_{\min}^p
\right]^{1/p}.
\]

The default is

\[
C_5=1,\qquad p=16.
\]

Thus:

- \(b_{\rm OPE}\to b_0/Q\) as \(b_T\to0\);
- \(b_{\rm OPE}\to b_*\) away from the small-\(b_T\) cap;
- with \(\mu_b=Q\), \(L_b\to0\) as \(b_T\to0\);
- the large-\(b_T\) infrared-floor treatment is unchanged.

This is a perturbative profile choice, not a fitted nonperturbative
constraint. It must later be varied as part of the theory/profile
uncertainty.
MD

cat > "${SRC}" <<'PY'
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
PY

cat > "${TEST}" <<'PY'
from __future__ import annotations

import math

from v22.src.css2_ope_nlo import B0
from v22.src.small_b_profile import (
    b_min_from_Q,
    b_ope_profile,
    matching_log_Lb,
    smooth_max_power,
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


# 1: b_min definition.
Q = 10.0
assert_close(
    b_min_from_Q(Q_GeV=Q),
    B0 / Q,
)
checks += 1


# 2: hard profile is an exact maximum.
b_min = B0 / Q
assert_close(
    b_ope_profile(
        b_star_GeV_inv=0.1 * b_min,
        Q_GeV=Q,
        kind="hard",
    ),
    b_min,
)
checks += 1


# 3: smooth profile reaches b_min at b_star=0.
assert_close(
    b_ope_profile(
        b_star_GeV_inv=0.0,
        Q_GeV=Q,
        power=16.0,
        kind="smooth",
    ),
    b_min,
)
checks += 1


# 4: the small-b cap gives L_b=0 when mu=Q.
assert_close(
    matching_log_Lb(
        b_pert_GeV_inv=b_min,
        mu_GeV=Q,
    ),
    0.0,
)
checks += 1


# 5: smooth maximum is monotonic.
values = [
    b_ope_profile(
        b_star_GeV_inv=value,
        Q_GeV=Q,
        power=16.0,
        kind="smooth",
    )
    for value in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
]

if not all(
    right >= left
    for left, right in zip(values[:-1], values[1:])
):
    raise AssertionError("smooth profile is not monotonic")

checks += 1


# 6: far above b_min, the profile approaches b_star.
b_star = 20.0 * b_min
profiled = b_ope_profile(
    b_star_GeV_inv=b_star,
    Q_GeV=Q,
    power=16.0,
    kind="smooth",
)

assert_close(
    profiled,
    b_star,
    rel=1.0e-12,
)
checks += 1


# 7: at equality, the p-norm enhancement is known analytically.
assert_close(
    smooth_max_power(
        b_min,
        b_min,
        power=16.0,
    ),
    b_min * 2.0 ** (1.0 / 16.0),
)
checks += 1


# 8: invalid profile mode is rejected.
try:
    b_ope_profile(
        b_star_GeV_inv=0.1,
        Q_GeV=Q,
        kind="unknown",
    )
except ValueError:
    pass
else:
    raise AssertionError("invalid profile kind was not rejected")

checks += 1


print(f"{checks} small-b profile checks passed")
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
  v22/tests/run_small_b_profile_smoke.py

echo
echo "Created:"
echo "  v22/SMALL_B_PROFILE.md"
echo "  v22/src/small_b_profile.py"
echo "  v22/tests/run_small_b_profile_smoke.py"
