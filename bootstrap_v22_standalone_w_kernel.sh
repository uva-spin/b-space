#!/usr/bin/env bash
set -euo pipefail

# Install the standalone scalar reference implementation of the v22 DY W kernel.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
DOC="${ROOT}/v22/STANDALONE_W_KERNEL.md"
SRC="${ROOT}/v22/src/dy_w_nlo_reference.py"
TEST="${ROOT}/v22/tests/run_dy_w_nlo_reference_smoke.py"

for required in \
  "${ROOT}/v22/src/css2_ope_nlo_general.py" \
  "${ROOT}/v22/src/dy_hard_nlo.py" \
  "${ROOT}/v22/src/small_b_profile.py"
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
# v22 standalone reference DY \(W\) kernel

This layer assembles the already tested ingredients into one reusable
scalar reference implementation.

For each quark or antiquark leg,

\[
F_q^{\rm OPE}
=
f_q
+
a_s(\mu_b)
\left[
C_{q\leftarrow q}^{(1)}\otimes f_q
+
C_{q\leftarrow g}^{(1)}\otimes f_g
\right].
\]

The DY luminosity is built from the charge-weighted
\(q\bar q+\bar q q\) sum.

The strict one-loop result is

\[
W_{\rm strict}
=
{\cal N}\,x_1x_2 e^{-S}
\left[
L_0
+
a_s(\mu_b)\Delta L_{\rm OPE}
+
\delta_H(Q)L_0
\right],
\]

where

\[
{\cal N}
=
\frac{\text{observable prefactor}}{2\pi}.
\]

The multiplicative reference is

\[
W_{\rm mult}
=
{\cal N}\,x_1x_2 e^{-S}
H_{\rm DY}^{\rm NLO}
L_{\rm product}^{\rm NLO}.
\]

`W_mult` contains formally higher-order hard--OPE and leg--leg cross
terms. Both forms are exposed so that the fixed-order expansion and the
eventual resummed implementation cannot be confused.

This module contains no fitted nonperturbative factor and no \(Y\) term.
It is intentionally scalar and slow. A later vectorized implementation
must reproduce it.
MD

cat > "${SRC}" <<'PY'
"""Standalone scalar reference for the v22 one-loop DY W kernel."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

from v22.src.css2_ope_nlo_general import (
    general_scale_quark_ope_nlo_components,
)


PDF = Callable[[float], float]

FOURIER_NORM = 1.0 / (2.0 * math.pi)


@dataclass(frozen=True)
class QuarkLegNLO:
    pid: int
    born: float
    delta_qq_coefficient: float
    delta_qg_coefficient: float
    a_s: float
    matched: float
    L_b: float
    l_zeta: float


@dataclass(frozen=True)
class DYLuminosityNLO:
    born: float
    delta_qq_coefficient: float
    delta_qg_coefficient: float
    a_s: float
    strict_ope: float
    naive_product: float


@dataclass(frozen=True)
class DYWKernelNLO:
    common_factor: float
    hard_factor: float
    hard_fraction: float
    born: float
    strict_nlo: float
    multiplicative_nlo: float
    strict_ratio_to_born: float
    multiplicative_ratio_to_born: float
    beyond_nlo_fraction_of_born: float


def build_quark_leg_nlo(
    *,
    pid: int,
    x: float,
    alpha_s_mu: float,
    b_pert_GeV_inv: float,
    mu_GeV: float,
    zeta_GeV2: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> QuarkLegNLO:
    """Build one signed-flavor perturbative OPE leg."""

    components = general_scale_quark_ope_nlo_components(
        x=float(x),
        alpha_s=float(alpha_s_mu),
        b_pert_GeV_inv=float(b_pert_GeV_inv),
        mu_GeV=float(mu_GeV),
        zeta_GeV2=float(zeta_GeV2),
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
    )

    return QuarkLegNLO(
        pid=int(pid),
        born=float(components.born_quark),
        delta_qq_coefficient=float(
            components.one_loop_qq_total
        ),
        delta_qg_coefficient=float(
            components.one_loop_qg_total
        ),
        a_s=float(components.a_s),
        matched=float(components.matched_tmd),
        L_b=float(components.L_b),
        l_zeta=float(components.l_zeta),
    )


def build_dy_luminosity_nlo(
    *,
    legs_a: Mapping[int, QuarkLegNLO],
    legs_b: Mapping[int, QuarkLegNLO],
    charge_squared: Mapping[int, float],
    flavors: tuple[int, ...] = (1, 2, 3),
) -> DYLuminosityNLO:
    """Build the charge-weighted q qbar luminosity through one loop."""

    born = 0.0
    delta_qq = 0.0
    delta_qg = 0.0
    naive_product = 0.0
    a_s_values: list[float] = []

    for flavor in flavors:
        pid = abs(int(flavor))

        if pid not in legs_a or -pid not in legs_a:
            raise KeyError(f"beam-A legs missing flavor +/-{pid}")

        if pid not in legs_b or -pid not in legs_b:
            raise KeyError(f"beam-B legs missing flavor +/-{pid}")

        q_a = legs_a[pid]
        qb_a = legs_a[-pid]
        q_b = legs_b[pid]
        qb_b = legs_b[-pid]

        charge2 = float(charge_squared[pid])

        a_s_values.extend(
            [q_a.a_s, qb_a.a_s, q_b.a_s, qb_b.a_s]
        )

        born_channel = (
            q_a.born * qb_b.born
            + qb_a.born * q_b.born
        )

        delta_qq_channel = (
            q_a.delta_qq_coefficient * qb_b.born
            + q_a.born * qb_b.delta_qq_coefficient
            + qb_a.delta_qq_coefficient * q_b.born
            + qb_a.born * q_b.delta_qq_coefficient
        )

        delta_qg_channel = (
            q_a.delta_qg_coefficient * qb_b.born
            + q_a.born * qb_b.delta_qg_coefficient
            + qb_a.delta_qg_coefficient * q_b.born
            + qb_a.born * q_b.delta_qg_coefficient
        )

        naive_channel = (
            q_a.matched * qb_b.matched
            + qb_a.matched * q_b.matched
        )

        born += charge2 * born_channel
        delta_qq += charge2 * delta_qq_channel
        delta_qg += charge2 * delta_qg_channel
        naive_product += charge2 * naive_channel

    if not a_s_values:
        raise ValueError("no active flavor channels")

    a_s = float(a_s_values[0])

    if max(abs(value - a_s) for value in a_s_values) > 1.0e-13:
        raise ValueError("all legs must use the same a_s(mu_b)")

    strict_ope = born + a_s * (delta_qq + delta_qg)

    return DYLuminosityNLO(
        born=float(born),
        delta_qq_coefficient=float(delta_qq),
        delta_qg_coefficient=float(delta_qg),
        a_s=a_s,
        strict_ope=float(strict_ope),
        naive_product=float(naive_product),
    )


def assemble_dy_w_nlo(
    *,
    luminosity: DYLuminosityNLO,
    hard_factor: float,
    observable_prefactor: float,
    x1: float,
    x2: float,
    sudakov_pair_exponent: float,
    fourier_norm: float = FOURIER_NORM,
) -> DYWKernelNLO:
    """Assemble Born, strict-NLO and multiplicative-NLO W integrands."""

    hard = float(hard_factor)
    prefactor = float(observable_prefactor)
    x1 = float(x1)
    x2 = float(x2)
    sudakov = float(sudakov_pair_exponent)
    fourier = float(fourier_norm)

    values = [hard, prefactor, x1, x2, sudakov, fourier]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("all scalar inputs must be finite")

    if x1 <= 0.0 or x2 <= 0.0:
        raise ValueError("x1 and x2 must be positive")

    common = (
        prefactor
        * fourier
        * x1
        * x2
        * math.exp(-sudakov)
    )

    hard_fraction = hard - 1.0

    born_lum = float(luminosity.born)

    strict_lum = (
        born_lum
        + luminosity.a_s
        * (
            luminosity.delta_qq_coefficient
            + luminosity.delta_qg_coefficient
        )
        + hard_fraction * born_lum
    )

    multiplicative_lum = (
        hard * luminosity.naive_product
    )

    born = common * born_lum
    strict = common * strict_lum
    multiplicative = common * multiplicative_lum

    scale = max(abs(born), 1.0e-300)

    return DYWKernelNLO(
        common_factor=float(common),
        hard_factor=hard,
        hard_fraction=hard_fraction,
        born=float(born),
        strict_nlo=float(strict),
        multiplicative_nlo=float(multiplicative),
        strict_ratio_to_born=float(strict / scale),
        multiplicative_ratio_to_born=float(
            multiplicative / scale
        ),
        beyond_nlo_fraction_of_born=float(
            (multiplicative - strict) / scale
        ),
    )
PY

cat > "${TEST}" <<'PY'
from __future__ import annotations

import math

from v22.src.dy_w_nlo_reference import (
    QuarkLegNLO,
    assemble_dy_w_nlo,
    build_dy_luminosity_nlo,
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
            f"actual={actual:.16e}, expected={expected:.16e}"
        )


def leg(
    pid: int,
    born: float,
    dqq: float,
    dqg: float,
    a_s: float,
) -> QuarkLegNLO:
    return QuarkLegNLO(
        pid=pid,
        born=born,
        delta_qq_coefficient=dqq,
        delta_qg_coefficient=dqg,
        a_s=a_s,
        matched=born + a_s * (dqq + dqg),
        L_b=0.0,
        l_zeta=0.0,
    )


checks = 0

a_s = 0.02

legs_a = {
    1: leg(1, 2.0, -0.4, 0.1, a_s),
    -1: leg(-1, 0.3, -0.05, 0.07, a_s),
}

legs_b = {
    1: leg(1, 1.5, -0.2, 0.08, a_s),
    -1: leg(-1, 0.5, -0.09, 0.06, a_s),
}

charges = {1: 1.0 / 9.0}

lum = build_dy_luminosity_nlo(
    legs_a=legs_a,
    legs_b=legs_b,
    charge_squared=charges,
    flavors=(1,),
)

# 1: Born luminosity.
born_manual = (1.0 / 9.0) * (
    2.0 * 0.5 + 0.3 * 1.5
)

assert_close(lum.born, born_manual)
checks += 1


# 2: strict OPE decomposition.
dqq_manual = (1.0 / 9.0) * (
    -0.4 * 0.5
    + 2.0 * -0.09
    + -0.05 * 1.5
    + 0.3 * -0.2
)

dqg_manual = (1.0 / 9.0) * (
    0.1 * 0.5
    + 2.0 * 0.06
    + 0.07 * 1.5
    + 0.3 * 0.08
)

assert_close(
    lum.strict_ope,
    born_manual + a_s * (dqq_manual + dqg_manual),
)
checks += 1


# 3: beam-exchange symmetry.
lum_ba = build_dy_luminosity_nlo(
    legs_a=legs_b,
    legs_b=legs_a,
    charge_squared=charges,
    flavors=(1,),
)

assert_close(lum_ba.born, lum.born)
assert_close(lum_ba.strict_ope, lum.strict_ope)
assert_close(lum_ba.naive_product, lum.naive_product)
checks += 1


# 4: alpha_s=0 and H=1 recover Born W.
zero_a = {
    pid: leg(
        pid,
        value.born,
        value.delta_qq_coefficient,
        value.delta_qg_coefficient,
        0.0,
    )
    for pid, value in legs_a.items()
}

zero_b = {
    pid: leg(
        pid,
        value.born,
        value.delta_qq_coefficient,
        value.delta_qg_coefficient,
        0.0,
    )
    for pid, value in legs_b.items()
}

zero_lum = build_dy_luminosity_nlo(
    legs_a=zero_a,
    legs_b=zero_b,
    charge_squared=charges,
    flavors=(1,),
)

zero_w = assemble_dy_w_nlo(
    luminosity=zero_lum,
    hard_factor=1.0,
    observable_prefactor=3.0,
    x1=0.2,
    x2=0.3,
    sudakov_pair_exponent=0.7,
)

assert_close(zero_w.strict_nlo, zero_w.born)
assert_close(zero_w.multiplicative_nlo, zero_w.born)
checks += 1


# 5: strict hard+OPE formula.
hard = 1.12

w = assemble_dy_w_nlo(
    luminosity=lum,
    hard_factor=hard,
    observable_prefactor=3.0,
    x1=0.2,
    x2=0.3,
    sudakov_pair_exponent=0.7,
)

expected_ratio = (
    1.0
    + (hard - 1.0)
    + a_s
    * (
        dqq_manual + dqg_manual
    )
    / born_manual
)

assert_close(
    w.strict_ratio_to_born,
    expected_ratio,
)
checks += 1


# 6: beyond-NLO difference scales quadratically.
def scaled_case(scale: float) -> float:
    scaled_a = {
        pid: leg(
            pid,
            value.born,
            value.delta_qq_coefficient,
            value.delta_qg_coefficient,
            a_s * scale,
        )
        for pid, value in legs_a.items()
    }

    scaled_b = {
        pid: leg(
            pid,
            value.born,
            value.delta_qq_coefficient,
            value.delta_qg_coefficient,
            a_s * scale,
        )
        for pid, value in legs_b.items()
    }

    scaled_lum = build_dy_luminosity_nlo(
        legs_a=scaled_a,
        legs_b=scaled_b,
        charge_squared=charges,
        flavors=(1,),
    )

    scaled_w = assemble_dy_w_nlo(
        luminosity=scaled_lum,
        hard_factor=1.0 + 0.12 * scale,
        observable_prefactor=1.0,
        x1=0.2,
        x2=0.3,
        sudakov_pair_exponent=0.0,
    )

    return abs(scaled_w.beyond_nlo_fraction_of_born)


d1 = scaled_case(1.0)
dhalf = scaled_case(0.5)

if not (d1 > 0.0 and dhalf > 0.0):
    raise AssertionError("beyond-NLO differences must be nonzero")

assert_close(
    dhalf / d1,
    0.25,
    rel=2.0e-2,
)
checks += 1


print(f"{checks} standalone DY W-kernel checks passed")
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
  v22/tests/run_dy_w_nlo_reference_smoke.py

echo
echo "Created:"
echo "  v22/STANDALONE_W_KERNEL.md"
echo "  v22/src/dy_w_nlo_reference.py"
echo "  v22/tests/run_dy_w_nlo_reference_smoke.py"
