"""Complete v22 perturbative backend wrapper.

Starting from the already validated scheme-consistent-Y wrapper, this module
also replaces the resummed W integrand by

    H_DY^NLO(Q)
    * [C^NLO tensor f]_A
    * [C^NLO tensor f]_B
    * exp[-S(b,Q)].

The W term uses the multiplicative NLO organization. Its singular subtraction
remains the strict one-loop expansion already installed by the scheme-Y
wrapper. Therefore hard--OPE and leg--leg products are present only as formally
higher-order content of the resummed W, not in the NLO subtraction.

The original v19 backend remains unchanged.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

from v22.src.dy_hard_nlo import dy_hard_nlo_at_Q
from v22.src.small_b_profile import b_ope_profile


ROOT = Path(__file__).resolve().parents[2]
SCHEME_Y_PATH = (
    ROOT
    / "v22"
    / "backends"
    / "bt_internal_css_backend_v22_scheme_y.py"
)

if not SCHEME_Y_PATH.exists():
    raise FileNotFoundError(SCHEME_Y_PATH)

_spec = importlib.util.spec_from_file_location(
    "_v22_scheme_y_backend",
    str(SCHEME_Y_PATH),
)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import {SCHEME_Y_PATH}")

_scheme = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _scheme
_spec.loader.exec_module(_scheme)

# The scheme-Y wrapper exposes its loaded v19 module here. Functions such as
# compute_backend_grids resolve globals in this module, so patching it updates
# the existing backend plumbing without editing the original source file.
_base = _scheme._base

_ORIGINAL_WPERT_CS_FOR_ROW = _base.wpert_cs_for_row

for _name, _value in vars(_scheme).items():
    if _name.startswith("__") or _name.startswith("LAST_"):
        continue
    globals()[_name] = _value


V22_FULL_PERTURBATIVE_BACKEND_ACTIVE = True
V22_W_ORGANIZATION = "multiplicative_nlo"
V22_BASE_SCHEME_Y_BACKEND = str(SCHEME_Y_PATH)
V22_FOURIER_NORM = 1.0 / (2.0 * math.pi)


def wpert_cs_for_row_v22(
    row,
    b_grid: np.ndarray,
    pdf,
    cfg,
    *,
    organization: str = "multiplicative",
) -> np.ndarray:
    """Return the v22 perturbative W integrand on a supplied b grid.

    organization:
      multiplicative:
          H_NLO times the product of two complete NLO OPE legs.
      strict:
          Born + one-loop hard + one-loop OPE, with no cross terms.
      born:
          exact legacy Born kernel, useful only for closure diagnostics.
    """

    Q = float(row["QM"])
    x1 = float(row["x1"])
    x2 = float(row["x2"])

    if Q <= 0.0 or x1 <= 0.0 or x2 <= 0.0:
        return np.zeros_like(
            np.asarray(b_grid, dtype=float),
            dtype=float,
        )

    mode = str(organization).strip().lower().replace("-", "_")

    if mode not in {
        "multiplicative",
        "multiplicative_nlo",
        "strict",
        "strict_nlo",
        "born",
        "lo",
    }:
        raise ValueError(
            "organization must be multiplicative, strict, or born"
        )

    b_values = np.asarray(b_grid, dtype=float)

    prefactor = float(
        _base.fixed_target_prefactor_cs(
            row,
            cfg,
        )
    )

    hard_factor = float(
        dy_hard_nlo_at_Q(
            Q_GeV=Q,
            alpha_s_at_Q=float(pdf.alphas(Q)),
        )
    )

    hard_fraction = hard_factor - 1.0

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

    out = np.empty_like(
        b_values,
        dtype=float,
    )

    for index, bT in enumerate(b_values):
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

        luminosity = _scheme._v22_luminosity(
            row=row,
            pdf=pdf,
            cfg=cfg,
            b_pert=b_pert,
            mu=mu,
            zeta=zeta,
            alpha_s_mu=alpha_s_mu,
        )

        if mode in {"born", "lo"}:
            luminosity_value = luminosity.born

        elif mode in {"strict", "strict_nlo"}:
            luminosity_value = (
                luminosity.born
                + luminosity.a_s
                * (
                    luminosity.delta_qq_coefficient
                    + luminosity.delta_qg_coefficient
                )
                + hard_fraction
                * luminosity.born
            )

        else:
            luminosity_value = (
                hard_factor
                * luminosity.naive_product
            )

        sudakov = float(
            _base.sudakov_s(
                float(bT),
                Q,
                pdf,
                cfg,
            )
        )

        out[index] = (
            prefactor
            * V22_FOURIER_NORM
            * x1
            * x2
            * math.exp(-sudakov)
            * luminosity_value
        )

    out[~np.isfinite(out)] = 0.0

    return out


def wpert_cs_for_row(
    row,
    b_grid: np.ndarray,
    pdf,
    cfg,
) -> np.ndarray:
    """Backend entry point: multiplicative v22 W by default."""

    organization = str(
        getattr(
            cfg,
            "v22_w_organization",
            "multiplicative",
        )
    )

    return wpert_cs_for_row_v22(
        row,
        b_grid,
        pdf,
        cfg,
        organization=organization,
    )


# Patch the loaded v19 globals used by compute_backend_grids.
_base.wpert_cs_for_row_v22 = wpert_cs_for_row_v22
_base.wpert_cs_for_row = wpert_cs_for_row

# Keep the intermediate wrapper consistent as well.
_scheme.wpert_cs_for_row_v22 = wpert_cs_for_row_v22
_scheme.wpert_cs_for_row = wpert_cs_for_row

globals()["wpert_cs_for_row_v22"] = wpert_cs_for_row_v22
globals()["wpert_cs_for_row"] = wpert_cs_for_row


def __getattr__(name: str):
    """Delegate dynamic diagnostics through the scheme-Y wrapper."""

    return getattr(_scheme, name)
