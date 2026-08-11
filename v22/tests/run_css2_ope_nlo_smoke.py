from __future__ import annotations

import math

from scipy.integrate import quad

from v22.src.css2_ope_nlo import (
    B0,
    CF,
    a_s_from_alpha_s,
    c_qg_1_regular,
    c_qq_1_delta_coefficient,
    c_qq_1_regular,
    canonical_css2_quark_ope_nlo,
    canonical_css2_quark_ope_nlo_components,
    canonical_css2_scales,
)


def assert_close(
    actual: float,
    expected: float,
    *,
    rel: float = 2.0e-9,
    abs_tol: float = 2.0e-11,
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


# 1-3: canonical scales.
for bT in [0.1, 1.0, 3.0]:
    scales = canonical_css2_scales(bT)

    assert_close(
        scales.mu_GeV,
        B0 / bT,
    )
    assert_close(
        scales.zeta_GeV2,
        scales.mu_GeV**2,
    )
    checks += 1


# 4: expansion parameter.
alpha_s = 0.24
assert_close(
    a_s_from_alpha_s(alpha_s),
    alpha_s / (4.0 * math.pi),
)
checks += 1


# 5-7: coefficient constants and endpoint behavior.
assert_close(
    c_qq_1_delta_coefficient(),
    -CF * math.pi**2 / 6.0,
)
checks += 1

assert_close(
    c_qq_1_regular(0.25),
    2.0 * CF * 0.75,
)
checks += 1

assert_close(
    c_qg_1_regular(0.25),
    2.0 * 0.25 * 0.75,
)
checks += 1


# 8: alpha_s=0 gives the Born collinear density.
x = 0.31
born = x**0.4 * (1.0 - x)**2

result = canonical_css2_quark_ope_nlo(
    x=x,
    alpha_s=0.0,
    quark_pdf=lambda y: y**0.4 * (1.0 - y)**2,
    gluon_pdf=lambda y: 7.0 * y**-0.1 * (1.0 - y)**5,
)

assert_close(result, born)
checks += 1


# 9: constant quark PDF and zero gluon PDF have a closed form.
x = 0.2
alpha_s = 0.27
a_s = alpha_s / (4.0 * math.pi)

expected_qq_regular = (
    2.0
    * CF
    * (
        -math.log(x)
        - 1.0
        + x
    )
)

expected_one_loop = (
    -CF * math.pi**2 / 6.0
    + expected_qq_regular
)

expected = 1.0 + a_s * expected_one_loop

components = canonical_css2_quark_ope_nlo_components(
    x=x,
    alpha_s=alpha_s,
    quark_pdf=lambda y: 1.0,
    gluon_pdf=lambda y: 0.0,
)

assert_close(
    components.one_loop_qq_regular,
    expected_qq_regular,
)
assert_close(
    components.matched_tmd,
    expected,
)
checks += 1


# 10: zero quark PDF and constant gluon PDF have qg=(1-x)^2.
x = 0.37
alpha_s = 0.19
a_s = alpha_s / (4.0 * math.pi)

components = canonical_css2_quark_ope_nlo_components(
    x=x,
    alpha_s=alpha_s,
    quark_pdf=lambda y: 0.0,
    gluon_pdf=lambda y: 1.0,
)

assert_close(
    components.one_loop_qg_regular,
    (1.0 - x)**2,
)
assert_close(
    components.matched_tmd,
    a_s * (1.0 - x)**2,
)
checks += 1


# 11: independent direct quadrature for nontrivial toy PDFs.
x = 0.43
alpha_s = 0.21
a_s = alpha_s / (4.0 * math.pi)

quark = lambda y: y**0.6 * (1.0 - y)**2
gluon = lambda y: 2.3 * y**-0.15 * (1.0 - y)**4

qq_regular_direct, _ = quad(
    lambda z: (
        2.0
        * CF
        * (1.0 - z)
        * quark(x / z)
        / z
    ),
    x,
    1.0,
    epsabs=1.0e-12,
    epsrel=1.0e-12,
    limit=400,
)

qg_direct, _ = quad(
    lambda z: (
        2.0
        * z
        * (1.0 - z)
        * gluon(x / z)
        / z
    ),
    x,
    1.0,
    epsabs=1.0e-12,
    epsrel=1.0e-12,
    limit=400,
)

direct = (
    quark(x)
    + a_s
    * (
        -CF
        * math.pi**2
        / 6.0
        * quark(x)
        + qq_regular_direct
        + qg_direct
    )
)

reference = canonical_css2_quark_ope_nlo(
    x=x,
    alpha_s=alpha_s,
    quark_pdf=quark,
    gluon_pdf=gluon,
)

assert_close(reference, direct)
checks += 1


# 12: decomposition sums exactly.
components = canonical_css2_quark_ope_nlo_components(
    x=0.28,
    alpha_s=0.22,
    quark_pdf=lambda y: y**0.3 * (1.0 - y)**3,
    gluon_pdf=lambda y: 1.8 * y**-0.2 * (1.0 - y)**5,
)

assert_close(
    components.one_loop_total,
    (
        components.one_loop_qq_delta
        + components.one_loop_qq_regular
        + components.one_loop_qg_regular
    ),
)
assert_close(
    components.matched_tmd,
    (
        components.born_quark
        + components.a_s
        * components.one_loop_total
    ),
)
checks += 1


# 13: invalid x is rejected.
try:
    canonical_css2_quark_ope_nlo(
        x=1.0,
        alpha_s=0.2,
        quark_pdf=lambda y: 1.0,
        gluon_pdf=lambda y: 1.0,
    )
except ValueError:
    pass
else:
    raise AssertionError("x=1 was not rejected")

checks += 1


print(f"{checks} canonical CSS2 NLO OPE checks passed")
