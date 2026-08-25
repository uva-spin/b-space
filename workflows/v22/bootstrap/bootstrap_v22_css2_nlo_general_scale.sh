#!/usr/bin/env bash
set -euo pipefail

# Install the general-scale one-loop quark-TMD OPE layer.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
DOC="${ROOT}/v22/GENERAL_SCALE_OPE.md"
SRC="${ROOT}/v22/src/css2_ope_nlo_general.py"
TEST="${ROOT}/v22/tests/run_css2_ope_nlo_general_smoke.py"

for required in \
  "${ROOT}/v22/CONVENTIONS.md" \
  "${ROOT}/v22/CSS2_SCHEME.md" \
  "${ROOT}/v22/src/convolution.py" \
  "${ROOT}/v22/src/css2_ope_nlo.py"
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
# v22 general-scale one-loop TMD OPE

## Why this layer is required

The existing profile uses a perturbative transverse coordinate \(b_*\)
and a profiled matching scale \(\mu_b\).  Over much of the fixed-target
support,

\[
\mu_b \ne \frac{b_0}{b_*},
\qquad
b_0=2e^{-\gamma_E}.
\]

Therefore the canonical \(L_b=0\) matching coefficients are not enough.

This does **not** mean the OPE is evaluated at the physical large
\(b_T\).  The perturbative coefficients use \(b_*\); the fitted
nonperturbative factor carries the remaining physical-\(b_T\)
dependence.

## Logarithms

\[
L_b = \ln\left(\frac{\mu^2 b_*^2}{b_0^2}\right),
\qquad
\ell_\zeta = \ln\left(\frac{\mu^2}{\zeta}\right).
\]

The default v22 boundary choice is

\[
\zeta_b=\mu_b^2,
\]

so \(\ell_\zeta=0\), but the implementation keeps general \(\zeta\).

## One-loop coefficients

The expansion parameter is

\[
a_s=\frac{\alpha_s}{4\pi}.
\]

For the quark TMDPDF,

\[
C_{q\leftarrow q}^{(1)}(z)
=
C_F\left[
-2L_b\left(\frac{1+z^2}{1-z}\right)_+
+2(1-z)
+\delta(1-z)
\left(
-L_b^2+2L_b\ell_\zeta-\frac{\pi^2}{6}
\right)
\right],
\]

\[
C_{q\leftarrow g}^{(1)}(z)
=
T_R\left[
-2L_b\left(z^2+(1-z)^2\right)
+4z(1-z)
\right].
\]

At \(L_b=\ell_\zeta=0\), these reduce exactly to the canonical module
already tested in `v22/src/css2_ope_nlo.py`.

## Scope

This module is a scalar high-accuracy reference implementation.  It is
not yet the vectorized fit-time implementation and does not yet assemble
the full DY \(W\) term.
MD

cat > "${SRC}" <<'PY'
r"""General-scale one-loop matching of a quark TMDPDF onto collinear PDFs.

Conventions:

    a_s = alpha_s/(4*pi)

    L_b = ln(mu^2 b_pert^2 / b0^2)
    l_zeta = ln(mu^2 / zeta)

The perturbative coordinate is `b_pert`, normally the backend's b_star,
not the physical large-b coordinate.

The input PDF callables return unweighted f(x,mu), not x*f.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from v22.src.convolution import (
    convolve_plus,
    convolve_regular,
)
from v22.src.css2_ope_nlo import (
    B0,
    CF,
    CanonicalOPEComponents,
    a_s_from_alpha_s,
    canonical_css2_quark_ope_nlo_components,
)


PDF = Callable[[float], float]

TR = 0.5


@dataclass(frozen=True)
class GeneralScaleOPEComponents:
    x: float
    alpha_s: float
    a_s: float
    b_pert_GeV_inv: float
    mu_GeV: float
    zeta_GeV2: float
    L_b: float
    l_zeta: float
    born_quark: float
    one_loop_qq_plus_log: float
    one_loop_qq_regular: float
    one_loop_qq_delta: float
    one_loop_qg_log: float
    one_loop_qg_regular: float
    one_loop_qq_total: float
    one_loop_qg_total: float
    one_loop_total: float
    matched_tmd: float


def _validate_x(x: float) -> float:
    x = float(x)
    if not 0.0 < x < 1.0:
        raise ValueError("x must lie strictly between 0 and 1")
    return x


def _checked_density(pdf: PDF, x: float) -> float:
    value = float(pdf(float(x)))
    if not math.isfinite(value):
        raise FloatingPointError(
            f"collinear density returned nonfinite value at x={x}"
        )
    return value


def matching_logs(
    *,
    b_pert_GeV_inv: float,
    mu_GeV: float,
    zeta_GeV2: float,
) -> tuple[float, float]:
    b = float(b_pert_GeV_inv)
    mu = float(mu_GeV)
    zeta = float(zeta_GeV2)

    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("b_pert must be finite and positive")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    if not math.isfinite(zeta) or zeta <= 0.0:
        raise ValueError("zeta must be finite and positive")

    L_b = math.log((mu * b / B0) ** 2)
    l_zeta = math.log(mu * mu / zeta)

    return L_b, l_zeta


def p_qq_singular(z: float) -> float:
    """The ordinary function underlying [(1+z^2)/(1-z)]_+."""

    z = float(z)
    return (1.0 + z * z) / (1.0 - z)


def integral_p_qq_zero_to_x(x: float) -> float:
    """Integral from 0 to x of (1+z^2)/(1-z), for x<1."""

    x = _validate_x(x)

    return (
        -0.5 * x * x
        - x
        - 2.0 * math.log1p(-x)
    )


def p_qg(z: float) -> float:
    """Color-stripped g -> q DGLAP shape z^2+(1-z)^2."""

    z = float(z)
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must lie in [0,1]")

    return z * z + (1.0 - z) ** 2


def c_qq_1_delta_general(
    *,
    L_b: float,
    l_zeta: float,
) -> float:
    return CF * (
        -L_b * L_b
        + 2.0 * L_b * l_zeta
        - math.pi * math.pi / 6.0
    )


def general_scale_quark_ope_nlo_components(
    *,
    x: float,
    alpha_s: float,
    b_pert_GeV_inv: float,
    mu_GeV: float,
    zeta_GeV2: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> GeneralScaleOPEComponents:
    """Return the general-scale one-loop quark-TMD OPE decomposition."""

    x = _validate_x(x)
    a_s = a_s_from_alpha_s(alpha_s)

    L_b, l_zeta = matching_logs(
        b_pert_GeV_inv=b_pert_GeV_inv,
        mu_GeV=mu_GeV,
        zeta_GeV2=zeta_GeV2,
    )

    born = _checked_density(quark_pdf, x)

    qq_plus_log = (
        -2.0
        * L_b
        * CF
        * convolve_plus(
            quark_pdf,
            x=x,
            kernel=p_qq_singular,
            integral_zero_to_x=integral_p_qq_zero_to_x,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    )

    qq_regular = convolve_regular(
        quark_pdf,
        x=x,
        kernel=lambda z: 2.0 * CF * (1.0 - z),
        epsabs=epsabs,
        epsrel=epsrel,
    )

    qq_delta = (
        c_qq_1_delta_general(
            L_b=L_b,
            l_zeta=l_zeta,
        )
        * born
    )

    qg_log = (
        -2.0
        * L_b
        * TR
        * convolve_regular(
            gluon_pdf,
            x=x,
            kernel=p_qg,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    )

    qg_regular = (
        TR
        * convolve_regular(
            gluon_pdf,
            x=x,
            kernel=lambda z: 4.0 * z * (1.0 - z),
            epsabs=epsabs,
            epsrel=epsrel,
        )
    )

    qq_total = qq_plus_log + qq_regular + qq_delta
    qg_total = qg_log + qg_regular
    one_loop_total = qq_total + qg_total
    matched = born + a_s * one_loop_total

    return GeneralScaleOPEComponents(
        x=x,
        alpha_s=float(alpha_s),
        a_s=a_s,
        b_pert_GeV_inv=float(b_pert_GeV_inv),
        mu_GeV=float(mu_GeV),
        zeta_GeV2=float(zeta_GeV2),
        L_b=L_b,
        l_zeta=l_zeta,
        born_quark=born,
        one_loop_qq_plus_log=qq_plus_log,
        one_loop_qq_regular=qq_regular,
        one_loop_qq_delta=qq_delta,
        one_loop_qg_log=qg_log,
        one_loop_qg_regular=qg_regular,
        one_loop_qq_total=qq_total,
        one_loop_qg_total=qg_total,
        one_loop_total=one_loop_total,
        matched_tmd=matched,
    )


def general_scale_quark_ope_nlo(**kwargs) -> float:
    return general_scale_quark_ope_nlo_components(
        **kwargs
    ).matched_tmd


def canonical_recovery_difference(
    *,
    x: float,
    alpha_s: float,
    b_pert_GeV_inv: float,
    quark_pdf: PDF,
    gluon_pdf: PDF,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-11,
) -> tuple[GeneralScaleOPEComponents, CanonicalOPEComponents]:
    """Evaluate both modules at mu=b0/b and zeta=mu^2."""

    mu = B0 / float(b_pert_GeV_inv)

    general = general_scale_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        b_pert_GeV_inv=b_pert_GeV_inv,
        mu_GeV=mu,
        zeta_GeV2=mu * mu,
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    canonical = canonical_css2_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        quark_pdf=quark_pdf,
        gluon_pdf=gluon_pdf,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    return general, canonical
PY

cat > "${TEST}" <<'PY'
from __future__ import annotations

import math

from v22.src.css2_ope_nlo import B0, CF
from v22.src.css2_ope_nlo_general import (
    TR,
    canonical_recovery_difference,
    general_scale_quark_ope_nlo_components,
    integral_p_qq_zero_to_x,
    matching_logs,
    p_qg,
)


def assert_close(
    actual: float,
    expected: float,
    *,
    rel: float = 3.0e-9,
    abs_tol: float = 3.0e-11,
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


# 1-3: canonical-scale recovery for nontrivial toy PDFs.
for x in [0.08, 0.31, 0.73]:
    quark = lambda y: y**0.4 * (1.0 - y)**2
    gluon = lambda y: 2.1 * y**-0.15 * (1.0 - y)**5

    general, canonical = canonical_recovery_difference(
        x=x,
        alpha_s=0.22,
        b_pert_GeV_inv=0.47,
        quark_pdf=quark,
        gluon_pdf=gluon,
    )

    assert_close(general.L_b, 0.0, abs_tol=2.0e-15)
    assert_close(general.l_zeta, 0.0, abs_tol=2.0e-15)
    assert_close(
        general.one_loop_qq_total,
        (
            canonical.one_loop_qq_delta
            + canonical.one_loop_qq_regular
        ),
    )
    assert_close(
        general.one_loop_qg_total,
        canonical.one_loop_qg_regular,
    )
    assert_close(
        general.matched_tmd,
        canonical.matched_tmd,
    )

    checks += 1


# 4: matching-log definition.
b = 1.5
mu = 1.3
zeta = mu * mu
L_expected = math.log((mu * b / B0) ** 2)

L_b, l_zeta = matching_logs(
    b_pert_GeV_inv=b,
    mu_GeV=mu,
    zeta_GeV2=zeta,
)

assert_close(L_b, L_expected)
assert_close(l_zeta, 0.0)
checks += 1


# 5: p_qg shape.
assert_close(
    p_qg(0.3),
    0.3**2 + 0.7**2,
)
checks += 1


# 6: analytic primitive of p_qq.
x = 0.42
primitive = (
    -0.5 * x * x
    - x
    - 2.0 * math.log1p(-x)
)

assert_close(
    integral_p_qq_zero_to_x(x),
    primitive,
)
checks += 1


# 7-10: constant quark, zero gluon, general L and l_zeta.
for x in [0.1, 0.25, 0.6, 0.9]:
    b = 1.5
    mu = 1.3
    zeta = 7.0
    alpha_s = 0.24
    a_s = alpha_s / (4.0 * math.pi)

    L = math.log((mu * b / B0) ** 2)
    lz = math.log(mu * mu / zeta)

    # For f(y)=1:
    # (p_qq)_+ tensor f =
    # -ln(x) + 1/2 + x + 2 ln(1-x).
    pqq_conv = (
        -math.log(x)
        + 0.5
        + x
        + 2.0 * math.log1p(-x)
    )

    regular_conv = 2.0 * CF * (
        -math.log(x) - 1.0 + x
    )

    delta = CF * (
        -L * L
        + 2.0 * L * lz
        - math.pi**2 / 6.0
    )

    one_loop = (
        -2.0 * L * CF * pqq_conv
        + regular_conv
        + delta
    )

    expected = 1.0 + a_s * one_loop

    result = general_scale_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        b_pert_GeV_inv=b,
        mu_GeV=mu,
        zeta_GeV2=zeta,
        quark_pdf=lambda y: 1.0,
        gluon_pdf=lambda y: 0.0,
    )

    assert_close(result.matched_tmd, expected)
    checks += 1


# 11-14: zero quark, constant gluon, general L.
for x in [0.1, 0.25, 0.6, 0.9]:
    b = 1.5
    mu = 1.3
    zeta = mu * mu
    alpha_s = 0.24
    a_s = alpha_s / (4.0 * math.pi)
    L = math.log((mu * b / B0) ** 2)

    # p_qg tensor 1 = -ln(x) - (1-x)^2.
    pqg_conv = (
        -math.log(x)
        - (1.0 - x) ** 2
    )

    finite_conv = 2.0 * TR * (1.0 - x) ** 2

    one_loop = (
        -2.0 * L * TR * pqg_conv
        + finite_conv
    )

    expected = a_s * one_loop

    result = general_scale_quark_ope_nlo_components(
        x=x,
        alpha_s=alpha_s,
        b_pert_GeV_inv=b,
        mu_GeV=mu,
        zeta_GeV2=zeta,
        quark_pdf=lambda y: 0.0,
        gluon_pdf=lambda y: 1.0,
    )

    assert_close(result.matched_tmd, expected)
    checks += 1


# 15: decomposition closes exactly.
result = general_scale_quark_ope_nlo_components(
    x=0.37,
    alpha_s=0.21,
    b_pert_GeV_inv=1.2,
    mu_GeV=1.5,
    zeta_GeV2=2.7,
    quark_pdf=lambda y: y**0.3 * (1.0 - y)**3,
    gluon_pdf=lambda y: 1.7 * y**-0.1 * (1.0 - y)**4,
)

assert_close(
    result.one_loop_qq_total,
    (
        result.one_loop_qq_plus_log
        + result.one_loop_qq_regular
        + result.one_loop_qq_delta
    ),
)
assert_close(
    result.one_loop_qg_total,
    (
        result.one_loop_qg_log
        + result.one_loop_qg_regular
    ),
)
assert_close(
    result.one_loop_total,
    (
        result.one_loop_qq_total
        + result.one_loop_qg_total
    ),
)
assert_close(
    result.matched_tmd,
    (
        result.born_quark
        + result.a_s * result.one_loop_total
    ),
)

checks += 1


print(
    f"{checks} general-scale CSS2 OPE checks passed"
)
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
  v22/tests/run_css2_ope_nlo_general_smoke.py

echo
echo "Created:"
echo "  v22/GENERAL_SCALE_OPE.md"
echo "  v22/src/css2_ope_nlo_general.py"
echo "  v22/tests/run_css2_ope_nlo_general_smoke.py"
