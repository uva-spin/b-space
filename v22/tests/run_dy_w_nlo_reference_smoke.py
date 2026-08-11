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
