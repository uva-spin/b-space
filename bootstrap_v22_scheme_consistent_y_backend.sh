#!/usr/bin/env bash
set -euo pipefail

# Create a non-destructive v22 backend wrapper that replaces only the
# singular subtraction used in Y with the complete v22 one-loop W expansion.
# Run from ~/work/bT-TMD.

ROOT="$(pwd)"
OUTDIR="${ROOT}/v22/backends"
OUT="${OUTDIR}/bt_internal_css_backend_v22_scheme_y.py"

for required in \
  "${ROOT}/bt_internal_css_backend_v19_smoothprofile.py" \
  "${ROOT}/v22/src/dy_hard_nlo.py" \
  "${ROOT}/v22/src/dy_w_nlo_reference.py" \
  "${ROOT}/v22/src/small_b_profile.py"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing ${OUT}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"

cat > "${OUT}" <<'PY'
"""v22 scheme-consistent Y-subtraction wrapper.

This module loads the existing v19 smooth-profile backend unchanged, then
patches only `singular_nlo_dev_for_row`.

For this wrapper:

  * `wexp_numeric` means the complete v22 strict one-loop W expansion:
        Sudakov expansion + DY hard factor + q<-q/q<-g OPE matching.
  * `wexp_positive` is its positive part.
  * `asymptotic_damped` means the same v22 expansion multiplied by the
    existing subtraction-localization profile S_sub(qT/Q).

The existing `y_nlo_dev_for_rows` then automatically forms

    Y_v22 = S_Y [FO_real_repaired - singular_v22],

because its function globals belong to the loaded base module and that module
is patched below.

The old backend file is not modified.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.special import j0

from v22.src.dy_hard_nlo import dy_hard_nlo_at_Q
from v22.src.dy_w_nlo_reference import (
    build_dy_luminosity_nlo,
    build_quark_leg_nlo,
)
from v22.src.small_b_profile import b_ope_profile


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "bt_internal_css_backend_v19_smoothprofile.py"

if not BASE_PATH.exists():
    raise FileNotFoundError(BASE_PATH)

_spec = importlib.util.spec_from_file_location(
    "_v22_base_smoothprofile",
    str(BASE_PATH),
)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import {BASE_PATH}")

_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

_ORIGINAL_SINGULAR_SELECTOR = _base.singular_nlo_dev_for_row

# Re-export the ordinary backend API. Dynamic LAST_* diagnostics are delegated
# through __getattr__ instead of copied.
for _name, _value in vars(_base).items():
    if _name.startswith("__") or _name.startswith("LAST_"):
        continue
    globals()[_name] = _value


V22_SCHEME_CONSISTENT_Y_ACTIVE = True
V22_BASE_BACKEND = str(BASE_PATH)
V22_FOURIER_NORM = 1.0 / (2.0 * math.pi)


def _proton_density(
    pdf,
    pid: int,
    mu: float,
) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        x = float(x)
        if not 0.0 < x < 1.0:
            return 0.0
        return float(pdf.xf_proton(int(pid), x, mu)) / x

    return evaluate


def _target_density(
    pdf,
    pid: int,
    mu: float,
    *,
    dataset: str,
    target_mode: str,
) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        x = float(x)
        if not 0.0 < x < 1.0:
            return 0.0

        return (
            float(
                pdf.xf_target(
                    int(pid),
                    x,
                    mu,
                    dataset=dataset,
                    target_mode=target_mode,
                )
            )
            / x
        )

    return evaluate


def _v22_luminosity(
    *,
    row,
    pdf,
    cfg,
    b_pert: float,
    mu: float,
    zeta: float,
    alpha_s_mu: float,
):
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    dataset = str(row["dataset"])

    gluon_a = _proton_density(pdf, 21, mu)
    gluon_b = _target_density(
        pdf,
        21,
        mu,
        dataset=dataset,
        target_mode=cfg.target_mode,
    )

    legs_a = {}
    legs_b = {}

    flavors = tuple(
        sorted(
            {
                abs(int(pid))
                for pid in cfg.flavors
                if int(pid) != 0
            }
        )
    )

    for pid in flavors:
        for signed_pid in (pid, -pid):
            legs_a[signed_pid] = build_quark_leg_nlo(
                pid=signed_pid,
                x=x1,
                alpha_s_mu=alpha_s_mu,
                b_pert_GeV_inv=b_pert,
                mu_GeV=mu,
                zeta_GeV2=zeta,
                quark_pdf=_proton_density(
                    pdf,
                    signed_pid,
                    mu,
                ),
                gluon_pdf=gluon_a,
                epsabs=float(
                    getattr(
                        cfg,
                        "v22_ope_epsabs",
                        1.0e-8,
                    )
                ),
                epsrel=float(
                    getattr(
                        cfg,
                        "v22_ope_epsrel",
                        1.0e-7,
                    )
                ),
            )

            legs_b[signed_pid] = build_quark_leg_nlo(
                pid=signed_pid,
                x=x2,
                alpha_s_mu=alpha_s_mu,
                b_pert_GeV_inv=b_pert,
                mu_GeV=mu,
                zeta_GeV2=zeta,
                quark_pdf=_target_density(
                    pdf,
                    signed_pid,
                    mu,
                    dataset=dataset,
                    target_mode=cfg.target_mode,
                ),
                gluon_pdf=gluon_b,
                epsabs=float(
                    getattr(
                        cfg,
                        "v22_ope_epsabs",
                        1.0e-8,
                    )
                ),
                epsrel=float(
                    getattr(
                        cfg,
                        "v22_ope_epsrel",
                        1.0e-7,
                    )
                ),
            )

    return build_dy_luminosity_nlo(
        legs_a=legs_a,
        legs_b=legs_b,
        charge_squared=_base.CHARGE2,
        flavors=flavors,
    )


def singular_nlo_v22_wexp_numeric_for_row(
    row,
    pdf,
    cfg,
    *,
    positive: bool = False,
) -> float:
    """Complete strict one-loop expansion of the v22 W term."""

    Q = float(row["QM"])
    qT = float(row["qT"])

    if Q <= 0.0 or qT <= 0.0:
        return 0.0

    x1 = float(row["x1"])
    x2 = float(row["x2"])

    b_grid = np.asarray(
        _base.make_b_grid(cfg),
        dtype=float,
    )

    hard_fraction = (
        dy_hard_nlo_at_Q(
            Q_GeV=Q,
            alpha_s_at_Q=float(pdf.alphas(Q)),
        )
        - 1.0
    )

    prefactor = float(
        _base.fixed_target_prefactor_cs(
            row,
            cfg,
        )
    )

    C5 = float(
        getattr(
            cfg,
            "v22_ope_C5",
            1.0,
        )
    )

    profile_power = float(
        getattr(
            cfg,
            "v22_ope_profile_power",
            16.0,
        )
    )

    profile_kind = str(
        getattr(
            cfg,
            "v22_ope_profile_kind",
            "smooth",
        )
    )

    values = np.empty_like(
        b_grid,
        dtype=float,
    )

    for index, bT in enumerate(b_grid):
        b_star = float(
            np.asarray(
                _base.bstar(
                    float(bT),
                    float(cfg.bstar_bmax),
                )
            )
        )

        b_pert = b_ope_profile(
            b_star_GeV_inv=b_star,
            Q_GeV=Q,
            C5=C5,
            power=profile_power,
            kind=profile_kind,
        )

        mu = float(
            _base.mu_b_of_b(
                float(bT),
                Q,
                cfg,
            )
        )

        zeta = mu * mu
        alpha_s_mu = float(pdf.alphas(mu))

        luminosity = _v22_luminosity(
            row=row,
            pdf=pdf,
            cfg=cfg,
            b_pert=b_pert,
            mu=mu,
            zeta=zeta,
            alpha_s_mu=alpha_s_mu,
        )

        sudakov_one_loop = float(
            _base.sudakov_s_one_loop(
                float(bT),
                Q,
                pdf,
                cfg,
            )
        )

        delta_luminosity = (
            -luminosity.born
            * sudakov_one_loop
            + hard_fraction
            * luminosity.born
            + luminosity.a_s
            * (
                luminosity.delta_qq_coefficient
                + luminosity.delta_qg_coefficient
            )
        )

        values[index] = (
            prefactor
            * V22_FOURIER_NORM
            * x1
            * x2
            * delta_luminosity
        )

    integral = float(
        np.trapezoid(
            b_grid
            * j0(qT * b_grid)
            * values,
            x=b_grid,
        )
    )

    value = integral

    if positive:
        value = max(value, 0.0)

    value *= float(
        getattr(
            cfg,
            "nlo_singular_norm",
            1.0,
        )
    )

    convention_function = getattr(
        _base,
        "_nlo_convention_multiplier",
        None,
    )

    if convention_function is not None:
        value *= float(
            convention_function(
                getattr(
                    cfg,
                    "nlo_singular_convention",
                    "base",
                ),
                row,
            )
        )

    if not np.isfinite(value):
        return 0.0

    return float(value)


def singular_nlo_dev_for_row(
    row,
    pdf,
    cfg,
) -> float:
    """Select the v22 expansion while preserving the existing CLI modes."""

    mode = (
        str(
            getattr(
                cfg,
                "nlo_singular_mode",
                "asymptotic_damped",
            )
        )
        .lower()
        .replace("-", "_")
    )

    if mode in {
        "asymptotic_damped",
        "analytic_damped",
        "stable",
        "stable_asymptotic",
        "damped",
    }:
        raw = singular_nlo_v22_wexp_numeric_for_row(
            row,
            pdf,
            cfg,
            positive=False,
        )

        return float(
            raw
            * _base.nlo_singular_damping_factor(
                row,
                cfg,
            )
        )

    if mode in {
        "wexp",
        "wexp_numeric",
        "same_scheme",
        "same_scheme_numeric",
    }:
        return singular_nlo_v22_wexp_numeric_for_row(
            row,
            pdf,
            cfg,
            positive=False,
        )

    if mode in {
        "wexp_positive",
        "wexp_pos",
        "same_scheme_positive",
    }:
        return singular_nlo_v22_wexp_numeric_for_row(
            row,
            pdf,
            cfg,
            positive=True,
        )

    # Retain analytic, none and any other explicitly supported legacy modes
    # for diagnostics.
    return _ORIGINAL_SINGULAR_SELECTOR(
        row,
        pdf,
        cfg,
    )


# Patch the base module. Its `compute_backend_grids` and
# `y_nlo_dev_for_rows` functions resolve globals in `_base.__dict__`, so this
# makes the existing Y plumbing use the v22 subtraction without modifying the
# original backend file.
_base.singular_nlo_v22_wexp_numeric_for_row = (
    singular_nlo_v22_wexp_numeric_for_row
)

_base.singular_nlo_dev_for_row = (
    singular_nlo_dev_for_row
)

# Ensure these names in the wrapper itself point to the patched versions.
globals()[
    "singular_nlo_v22_wexp_numeric_for_row"
] = singular_nlo_v22_wexp_numeric_for_row

globals()[
    "singular_nlo_dev_for_row"
] = singular_nlo_dev_for_row


def __getattr__(name: str):
    """Delegate dynamic backend state, including LAST_* diagnostics."""

    return getattr(_base, name)
PY

python3 -m py_compile "${OUT}"

PYTHONPATH=. python3 - <<'PY'
from pathlib import Path
import importlib.util
import sys

path = Path(
    "v22/backends/"
    "bt_internal_css_backend_v22_scheme_y.py"
).resolve()

spec = importlib.util.spec_from_file_location(
    "v22_scheme_y_smoke",
    str(path),
)

if spec is None or spec.loader is None:
    raise SystemExit("could not import wrapper")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module.V22_SCHEME_CONSISTENT_Y_ACTIVE
assert hasattr(module, "compute_backend_grids")
assert hasattr(module, "singular_nlo_dev_for_row")
assert hasattr(module, "y_nlo_dev_for_rows")

print("v22 scheme-consistent Y backend import passed")
print("base backend:", module.V22_BASE_BACKEND)
PY

echo
echo "Created:"
echo "  v22/backends/bt_internal_css_backend_v22_scheme_y.py"
