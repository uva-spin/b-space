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
