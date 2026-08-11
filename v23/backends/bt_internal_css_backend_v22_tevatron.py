"""v22 full backend wrapper with pbar-p and pp collider luminosity support.

The frozen v22 backend treats the first hadron as a proton beam and the second
as a fixed target.  Tevatron rows are pbar-p.  This wrapper leaves fixed-target
rows unchanged and patches the parton luminosity plumbing for rows whose
`target` or `beam_config` field is `pbar_p` or `pp`.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

import numpy as np
import pandas as pd

from v22.src.dy_w_nlo_reference import (
    QuarkLegNLO,
    build_dy_luminosity_nlo,
)
from v22.src.convolution import (
    convolve_plus,
    convolve_regular,
)
from v22.src.css2_ope_nlo import (
    a_s_from_alpha_s,
)
from v22.src.css2_ope_nlo_general import (
    CF,
    TR,
    c_qq_1_delta_general,
    integral_p_qq_zero_to_x,
    matching_logs,
    p_qg,
    p_qq_singular,
)


ROOT = Path(__file__).resolve().parents[2]
FULL_BACKEND_PATH = ROOT / "v22" / "backends" / "bt_internal_css_backend_v22_full.py"

if not FULL_BACKEND_PATH.exists():
    raise FileNotFoundError(FULL_BACKEND_PATH)

_spec = importlib.util.spec_from_file_location(
    "_v22_full_backend_for_tevatron",
    str(FULL_BACKEND_PATH),
)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import {FULL_BACKEND_PATH}")

_full = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _full
_spec.loader.exec_module(_full)

_base = _full._base
_scheme = _full._scheme

_ORIGINAL_FIXED_TARGET_PREFACTOR_CS = _base.fixed_target_prefactor_cs
_ORIGINAL_FULL_WPERT_CS_FOR_ROW = _full.wpert_cs_for_row
_ORIGINAL_CHARGE2 = dict(_base.CHARGE2)

_EW_ALPHA_INV_MZ = 127.955
_EW_MZ = 91.1876
_EW_GAMMA_Z = 2.4952
_EW_SIN2_THETA_W = 0.23122
_EW_COS2_THETA_W = 1.0 - _EW_SIN2_THETA_W
_TEVATRON_RAPIDITY_AVG_OVER_CENTRAL = 0.59


def _ew_params(charge: float, t3: float) -> tuple[float, float]:
    """Return arTeMiDe-style ZZ and gamma-Z coupling factors."""
    zz = (
        (1.0 - 2.0 * abs(charge) * _EW_SIN2_THETA_W) ** 2
        + 4.0 * charge * charge * _EW_SIN2_THETA_W ** 2
    ) / (8.0 * _EW_SIN2_THETA_W * _EW_COS2_THETA_W)
    mix = charge * (t3 - 2.0 * charge * _EW_SIN2_THETA_W) / (
        2.0 * math.sqrt(_EW_SIN2_THETA_W * _EW_COS2_THETA_W)
    )
    return float(zz), float(mix)


_EW_LEPTON_ZZ, _EW_LEPTON_MIX = _ew_params(-1.0, -0.5)
_EW_UP_ZZ, _EW_UP_MIX = _ew_params(2.0 / 3.0, 0.5)
_EW_DOWN_ZZ, _EW_DOWN_MIX = _ew_params(-1.0 / 3.0, -0.5)
_TEVATRON_CHARGE_CACHE: dict[tuple[float, float, float], dict[int, float]] = {}


def _tevatron_q_window(row) -> tuple[float, float, float]:
    q = max(float(row["QM"]), 1.0e-12)
    qlo = float(row.get("QM_Low", np.nan))
    qhi = float(row.get("QM_High", np.nan))
    if math.isfinite(qlo) and math.isfinite(qhi) and qlo > 0.0 and qhi > qlo:
        return qlo, qhi, q
    return q, q, q


def _float_row_value(row, key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
    except AttributeError:
        try:
            value = row[key]
        except (KeyError, TypeError):
            value = default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _tevatron_pt_jacobian(row) -> float:
    """Convert the radial Fourier kernel to Tevatron pb/GeV pT tables."""
    qt = max(_float_row_value(row, "qT", 0.0), 0.0)
    qlo = _float_row_value(row, "qT_low")
    qhi = _float_row_value(row, "qT_high")
    if math.isfinite(qlo) and math.isfinite(qhi) and qlo >= 0.0 and qhi > qlo:
        return float(math.pi * (qhi * qhi - qlo * qlo) / (qhi - qlo))
    return float(2.0 * math.pi * qt)


def _tevatron_rapidity_factor(row) -> float:
    """Approximate the rapidity-inclusive CDF/D0 pT tables from a y=0 row.

    A direct NNPDF40 Born-luminosity check for the accepted CDF/D0 mass windows
    gives int dy L(y) / L(0) = 3.48-3.63, i.e. about 0.59 times the full
    kinematic width 2 log(sqrt(s)/Q).  Full y quadrature inside every W(b)
    kernel would multiply the cache-build cost, so this keeps the fit-ready
    convention explicit while preserving the existing backend structure.
    """
    q = max(_float_row_value(row, "QM", 0.0), 1.0e-12)
    sqrts = _float_row_value(row, "SqrtS", 0.0)
    if not math.isfinite(sqrts) or sqrts <= q:
        return 1.0
    return float(_TEVATRON_RAPIDITY_AVG_OVER_CENTRAL * 2.0 * math.log(sqrts / q))


def _neutral_current_weight(pid: int, q: float) -> float:
    """Photon + gamma/Z + Z weights normalized to the photon prefactor."""
    pid = abs(int(pid))
    if pid in {2, 4, 6}:
        e2 = 4.0 / 9.0
        zz_q = _EW_UP_ZZ
        mix_q = _EW_UP_MIX
    else:
        e2 = 1.0 / 9.0
        zz_q = _EW_DOWN_ZZ
        mix_q = _EW_DOWN_MIX
    q2 = float(q) ** 2
    mz2 = _EW_MZ ** 2
    den = (q2 - mz2) ** 2 + (_EW_GAMMA_Z * _EW_MZ) ** 2
    interference = (
        _EW_LEPTON_MIX
        * mix_q
        * 2.0
        * q2
        * (q2 - mz2)
        / den
    )
    z_boson = _EW_LEPTON_ZZ * zz_q * q2 * q2 / den
    return float(e2 + interference + z_boson)


def _tevatron_charge_weights(row) -> dict[int, float]:
    qlo, qhi, q = _tevatron_q_window(row)
    key = (round(qlo, 8), round(qhi, 8), round(q, 8))
    cached = _TEVATRON_CHARGE_CACHE.get(key)
    if cached is not None:
        return cached

    if qhi <= qlo:
        weights = {
            pid: _neutral_current_weight(pid, q)
            for pid in _ORIGINAL_CHARGE2
        }
    else:
        n = 257
        grid = np.linspace(qlo, qhi, n, dtype=float)
        kernel = np.power(np.maximum(grid, 1.0e-12), -3.0)
        denom = float(np.trapezoid(kernel, grid))
        weights = {}
        for pid in _ORIGINAL_CHARGE2:
            values = kernel * np.asarray(
                [_neutral_current_weight(pid, qq) for qq in grid],
                dtype=float,
            )
            weights[pid] = float(np.trapezoid(values, grid) / denom)

    _TEVATRON_CHARGE_CACHE[key] = weights
    return weights


def _charge_weights_for_row(row) -> dict[int, float]:
    if _is_pbar_p(row) or _is_pp(row):
        return _tevatron_charge_weights(row)
    return _ORIGINAL_CHARGE2


def fixed_target_prefactor_cs(row, cfg) -> float:
    prefactor = float(_ORIGINAL_FIXED_TARGET_PREFACTOR_CS(row, cfg))
    if not (_is_pbar_p(row) or _is_pp(row)):
        return prefactor
    fiducial_factor = _float_row_value(row, "theory_fiducial_factor", 1.0)
    if not math.isfinite(fiducial_factor) or fiducial_factor <= 0.0:
        fiducial_factor = 1.0
    alpha_mz = 1.0 / _EW_ALPHA_INV_MZ
    alpha_ref = max(float(getattr(cfg, "alpha_em", alpha_mz)), 1.0e-300)
    out = (
        prefactor
        * (alpha_mz / alpha_ref) ** 2
        * _tevatron_pt_jacobian(row)
        * fiducial_factor
    )
    if _is_pbar_p(row):
        out *= _tevatron_rapidity_factor(row)
    return float(out)


def _row_value(row, key: str, default: str = "") -> str:
    if isinstance(row, dict):
        return str(row.get(key, default))
    if hasattr(row, "index") and key not in row.index:
        return str(default)
    try:
        return str(row[key])
    except (KeyError, TypeError):
        return str(default)


def _is_pbar_p(row) -> bool:
    for key in ("target", "beam_config"):
        value = _row_value(row, key).strip().lower().replace("-", "_")
        if value in {"pbar_p", "pbarp", "anti_p_p", "antiproton_proton"}:
            return True
    return False


def _is_pp(row) -> bool:
    for key in ("target", "beam_config"):
        value = _row_value(row, key).strip().lower().replace("-", "_")
        if value in {"pp", "p_p", "proton_proton"}:
            return True
    return False


def _xf_hadron(row, pdf, pid: int, x: float, mu: float, cfg, *, side: str, is_pbar_p: bool | None = None) -> float:
    """Return x*f_i(x,mu) for incoming side a or b.

    For pbar-p rows, side a is the anti-proton and side b is the proton.
    For all other rows, preserve the original fixed-target convention.
    """
    x = float(x)
    if not 0.0 < x < 1.0:
        return 0.0
    pid = int(pid)
    side = str(side).lower()
    if _is_pbar_p(row) if is_pbar_p is None else is_pbar_p:
        if side == "a" and abs(pid) <= 6:
            return float(pdf.xf_proton(-pid, x, mu))
        return float(pdf.xf_proton(pid, x, mu))
    if _is_pp(row):
        return float(pdf.xf_proton(pid, x, mu))
    if side == "a":
        return float(pdf.xf_proton(pid, x, mu))
    return float(
        pdf.xf_target(
            pid,
            x,
            mu,
            dataset=str(row["dataset"]),
            target_mode=cfg.target_mode,
        )
    )


def _density(row, pdf, pid: int, mu: float, cfg, *, side: str):
    is_pbar = _is_pbar_p(row)

    def evaluate(x: float) -> float:
        x = float(x)
        if not 0.0 < x < 1.0:
            return 0.0
        return _xf_hadron(row, pdf, int(pid), x, mu, cfg, side=side, is_pbar_p=is_pbar) / x

    return evaluate


def _rapidity_bin(row) -> tuple[float, float] | None:
    y = _float_row_value(row, "y", 0.0)
    ylo = _float_row_value(row, "y_Low")
    yhi = _float_row_value(row, "y_High")
    if math.isfinite(ylo) and math.isfinite(yhi) and yhi > ylo:
        return float(ylo), float(yhi)
    if math.isfinite(y):
        return float(y), float(y)
    return None


def _rapidity_nodes_weights(row, cfg) -> tuple[np.ndarray, np.ndarray]:
    ybin = _rapidity_bin(row)
    if ybin is None:
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)
    ylo, yhi = ybin
    if yhi <= ylo:
        return np.asarray([ylo], dtype=float), np.asarray([1.0], dtype=float)
    n = int(getattr(cfg, "collider_y_quad", 9))
    n = max(3, n)
    if n % 2 == 0:
        n += 1
    nodes, weights = np.polynomial.legendre.leggauss(n)
    center = 0.5 * (ylo + yhi)
    half_width = 0.5 * (yhi - ylo)
    return center + half_width * nodes, half_width * weights


def _row_at_rapidity(row, y: float):
    out = row.copy()
    q = max(_float_row_value(row, "QM", 0.0), 1.0e-12)
    sqrts = _float_row_value(row, "SqrtS", 0.0)
    tau = q / max(sqrts, 1.0e-12)
    out["y"] = float(y)
    out["x1"] = float(tau * math.exp(float(y)))
    out["x2"] = float(tau * math.exp(-float(y)))
    return out


def wpert_cs_for_row(row, b_grid: np.ndarray, pdf, cfg) -> np.ndarray:
    if not _is_pp(row):
        return _ORIGINAL_FULL_WPERT_CS_FOR_ROW(row, b_grid, pdf, cfg)
    nodes, weights = _rapidity_nodes_weights(row, cfg)
    total = np.zeros_like(np.asarray(b_grid, dtype=float), dtype=float)
    for y, weight in zip(nodes, weights):
        yrow = _row_at_rapidity(row, float(y))
        total += float(weight) * _ORIGINAL_FULL_WPERT_CS_FOR_ROW(yrow, b_grid, pdf, cfg)
    return total


def _checked_density(pdf_callable, x: float) -> float:
    value = float(pdf_callable(float(x)))
    if not math.isfinite(value):
        raise FloatingPointError(f"collinear density returned nonfinite value at x={x}")
    return value


def _qg_total_for_side(*, x: float, alpha_s_mu: float, b_pert: float, mu: float, zeta: float, gluon_pdf, epsabs: float, epsrel: float) -> tuple[float, float]:
    L_b, _ = matching_logs(
        b_pert_GeV_inv=float(b_pert),
        mu_GeV=float(mu),
        zeta_GeV2=float(zeta),
    )
    qg_log = (
        -2.0
        * L_b
        * TR
        * convolve_regular(
            gluon_pdf,
            x=float(x),
            kernel=p_qg,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
        )
    )
    qg_regular = (
        TR
        * convolve_regular(
            gluon_pdf,
            x=float(x),
            kernel=lambda z: 4.0 * z * (1.0 - z),
            epsabs=float(epsabs),
            epsrel=float(epsrel),
        )
    )
    return float(qg_log + qg_regular), float(a_s_from_alpha_s(alpha_s_mu))


def _build_quark_leg_nlo_shared_qg(
    *,
    pid: int,
    x: float,
    alpha_s_mu: float,
    b_pert: float,
    mu: float,
    zeta: float,
    quark_pdf,
    qg_total: float,
    a_s: float,
    epsabs: float,
    epsrel: float,
) -> QuarkLegNLO:
    L_b, l_zeta = matching_logs(
        b_pert_GeV_inv=float(b_pert),
        mu_GeV=float(mu),
        zeta_GeV2=float(zeta),
    )
    x = float(x)
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
            epsabs=float(epsabs),
            epsrel=float(epsrel),
        )
    )
    qq_regular = convolve_regular(
        quark_pdf,
        x=x,
        kernel=lambda z: 2.0 * CF * (1.0 - z),
        epsabs=float(epsabs),
        epsrel=float(epsrel),
    )
    qq_delta = c_qq_1_delta_general(L_b=L_b, l_zeta=l_zeta) * born
    qq_total = float(qq_plus_log + qq_regular + qq_delta)
    matched = float(born + float(a_s) * (qq_total + float(qg_total)))
    return QuarkLegNLO(
        pid=int(pid),
        born=float(born),
        delta_qq_coefficient=qq_total,
        delta_qg_coefficient=float(qg_total),
        a_s=float(a_s),
        matched=matched,
        L_b=float(L_b),
        l_zeta=float(l_zeta),
    )


def charge_weighted_lumi(row, mu, pdf, cfg) -> float:
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    charge_weights = _charge_weights_for_row(row)
    total = 0.0
    for flav in cfg.flavors:
        pid = int(abs(flav))
        e2 = charge_weights.get(pid, 0.0)
        if e2 == 0.0:
            continue
        q1 = _xf_hadron(row, pdf, pid, x1, mu, cfg, side="a")
        qb1 = _xf_hadron(row, pdf, -pid, x1, mu, cfg, side="a")
        q2 = _xf_hadron(row, pdf, pid, x2, mu, cfg, side="b")
        qb2 = _xf_hadron(row, pdf, -pid, x2, mu, cfg, side="b")
        total += e2 * (q1 * qb2 + qb1 * q2)
    return max(float(total), 0.0)


def charge_weighted_qg_lumi(row, mu, pdf, cfg) -> float:
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    g1 = _xf_hadron(row, pdf, 21, x1, mu, cfg, side="a")
    g2 = _xf_hadron(row, pdf, 21, x2, mu, cfg, side="b")
    charge_weights = _charge_weights_for_row(row)
    total = 0.0
    for flav in cfg.flavors:
        pid = int(abs(flav))
        e2 = charge_weights.get(pid, 0.0)
        if e2 == 0.0:
            continue
        q1 = (
            _xf_hadron(row, pdf, pid, x1, mu, cfg, side="a")
            + _xf_hadron(row, pdf, -pid, x1, mu, cfg, side="a")
        )
        q2 = (
            _xf_hadron(row, pdf, pid, x2, mu, cfg, side="b")
            + _xf_hadron(row, pdf, -pid, x2, mu, cfg, side="b")
        )
        total += e2 * (q1 * g2 + g1 * q2)
    return max(float(total), 0.0)


def channel_lumis_exact(row, xa, xb, mu, pdf, cfg):
    g1 = _xf_hadron(row, pdf, 21, xa, mu, cfg, side="a")
    g2 = _xf_hadron(row, pdf, 21, xb, mu, cfg, side="b")
    charge_weights = _charge_weights_for_row(row)
    qq = 0.0
    qg = 0.0
    gq = 0.0
    for flav in cfg.flavors:
        pid = int(abs(flav))
        e2 = charge_weights.get(pid, 0.0)
        if e2 == 0.0:
            continue
        q1 = _xf_hadron(row, pdf, pid, xa, mu, cfg, side="a")
        qb1 = _xf_hadron(row, pdf, -pid, xa, mu, cfg, side="a")
        q2 = _xf_hadron(row, pdf, pid, xb, mu, cfg, side="b")
        qb2 = _xf_hadron(row, pdf, -pid, xb, mu, cfg, side="b")
        qq += e2 * (q1 * qb2 + qb1 * q2)
        qg += e2 * ((q1 + qb1) * g2)
        gq += e2 * (g1 * (q2 + qb2))
    return max(float(qq), 0.0), max(float(qg), 0.0), max(float(gq), 0.0)


def _v22_luminosity(*, row, pdf, cfg, b_pert: float, mu: float, zeta: float, alpha_s_mu: float):
    x1 = float(row["x1"])
    x2 = float(row["x2"])

    gluon_a = _density(row, pdf, 21, mu, cfg, side="a")
    gluon_b = _density(row, pdf, 21, mu, cfg, side="b")
    epsabs = float(getattr(cfg, "v22_ope_epsabs", 1.0e-8))
    epsrel = float(getattr(cfg, "v22_ope_epsrel", 1.0e-7))
    qg_a, a_s = _qg_total_for_side(
        x=x1,
        alpha_s_mu=alpha_s_mu,
        b_pert=b_pert,
        mu=mu,
        zeta=zeta,
        gluon_pdf=gluon_a,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    qg_b, _ = _qg_total_for_side(
        x=x2,
        alpha_s_mu=alpha_s_mu,
        b_pert=b_pert,
        mu=mu,
        zeta=zeta,
        gluon_pdf=gluon_b,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    legs_a = {}
    legs_b = {}
    flavors = tuple(sorted({abs(int(pid)) for pid in cfg.flavors if int(pid) != 0}))

    for pid in flavors:
        for signed_pid in (pid, -pid):
            legs_a[signed_pid] = _build_quark_leg_nlo_shared_qg(
                pid=signed_pid,
                x=x1,
                alpha_s_mu=alpha_s_mu,
                b_pert=b_pert,
                mu=mu,
                zeta=zeta,
                quark_pdf=_density(row, pdf, signed_pid, mu, cfg, side="a"),
                qg_total=qg_a,
                a_s=a_s,
                epsabs=epsabs,
                epsrel=epsrel,
            )
            legs_b[signed_pid] = _build_quark_leg_nlo_shared_qg(
                pid=signed_pid,
                x=x2,
                alpha_s_mu=alpha_s_mu,
                b_pert=b_pert,
                mu=mu,
                zeta=zeta,
                quark_pdf=_density(row, pdf, signed_pid, mu, cfg, side="b"),
                qg_total=qg_b,
                a_s=a_s,
                epsabs=epsabs,
                epsrel=epsrel,
            )

    return build_dy_luminosity_nlo(
        legs_a=legs_a,
        legs_b=legs_b,
        charge_squared=_charge_weights_for_row(row),
        flavors=flavors,
    )


_TEVATRON_CHARGE_WEIGHTED_LUMI = charge_weighted_lumi
_TEVATRON_CHARGE_WEIGHTED_QG_LUMI = charge_weighted_qg_lumi
_TEVATRON_CHANNEL_LUMIS_EXACT = channel_lumis_exact
_TEVATRON_V22_LUMINOSITY = _v22_luminosity
_TEVATRON_FIXED_TARGET_PREFACTOR_CS = fixed_target_prefactor_cs
_TEVATRON_WPERT_CS_FOR_ROW = wpert_cs_for_row
_TEVATRON_ORIGINAL_FIXED_TARGET_PREFACTOR_CS = _ORIGINAL_FIXED_TARGET_PREFACTOR_CS

_WORKER_PDF = None
_WORKER_CFG = None
_WORKER_B_GRID = None


def _init_w_worker(pdf_set: str, member: int, use_toy_pdf: bool, cfg, b_grid) -> None:
    global _WORKER_PDF, _WORKER_CFG, _WORKER_B_GRID
    _WORKER_PDF = _base.LHAPDFProvider(pdf_set, int(member), use_toy_pdf=bool(use_toy_pdf))
    _WORKER_CFG = cfg
    _WORKER_B_GRID = np.asarray(b_grid, dtype=float)


def _compute_w_worker(item):
    index, row_dict = item
    if _WORKER_PDF is None or _WORKER_CFG is None or _WORKER_B_GRID is None:
        raise RuntimeError("W-grid worker was not initialized")
    row = pd.Series(row_dict)
    return int(index), _full.wpert_cs_for_row(row, _WORKER_B_GRID, _WORKER_PDF, _WORKER_CFG)


def _compute_y_worker(item):
    index, row_dict, w_base = item
    if _WORKER_PDF is None or _WORKER_CFG is None:
        raise RuntimeError("Y worker was not initialized")
    row = pd.Series(row_dict)
    cfg = _WORKER_CFG
    i = int(index)
    w_base = float(w_base)
    q = max(float(row["QM"]), 1.0e-12)
    qt = max(float(row["qT"]), 0.0)
    r = qt / q
    fo_raw = _base.fo_nlo_real_dev_for_row(row, _WORKER_PDF, cfg)
    real_tail_repair_factor = _base.nlo_real_tail_repair_factor(r, cfg)
    fo = fo_raw * real_tail_repair_factor
    sing = _base.singular_nlo_dev_for_row(row, _WORKER_PDF, cfg)
    raw_y = fo - sing
    comp = str(getattr(cfg, "nlo_y_component", "raw")).lower().replace("-", "_")
    if comp in {"raw", "fo_minus_sing"}:
        component_y = raw_y
    elif comp in {"positive", "positive_part", "pos"}:
        component_y = max(raw_y, 0.0)
    elif comp in {"fo", "fo_only", "real", "real_only"}:
        component_y = fo
    elif comp in {"minus_sing", "negative_singular"}:
        component_y = -sing
    elif comp in {"sing", "singular", "singular_only"}:
        component_y = sing
    elif comp in {"zero", "none"}:
        component_y = 0.0
    else:
        raise ValueError(f"Unsupported nlo_y_component={getattr(cfg, 'nlo_y_component', None)!r}")
    sw = _base.smooth_tail_switch(r, cfg.nlo_y_transition, cfg.nlo_y_transition_width) if cfg.nlo_dev_use_switch else 1.0
    y_unclipped = sw * component_y
    clip_mult = float(getattr(cfg, "nlo_y_clip_multiple", 5.0))
    if clip_mult > 0.0:
        max_abs = clip_mult * max(abs(w_base), abs(float(row.get("CS", 0.0))), 1.0e-30)
        y = float(np.clip(y_unclipped, -max_abs, max_abs))
    else:
        y = float(y_unclipped)
    y_value = y if np.isfinite(y) else 0.0
    diag = {
        "row_id": str(row.get("row_id", i)),
        "dataset": str(row.get("dataset", "")),
        "qT": qt,
        "QM": q,
        "qT_over_Q": r,
        "W_CS_baseline": w_base if np.isfinite(w_base) else np.nan,
        "FO_NLO_real_dev_raw_CS": float(fo_raw),
        "nlo_real_tail_repair": str(getattr(cfg, "nlo_real_tail_repair", "none")),
        "nlo_real_tail_repair_factor": float(real_tail_repair_factor),
        "FO_NLO_real_dev_CS": float(fo),
        "singular_NLO_dev_CS": float(sing),
        "raw_Y_NLO_dev_CS": float(raw_y) if np.isfinite(raw_y) else np.nan,
        "raw_Y_positive": bool(raw_y > 0.0),
        "nlo_y_component": comp,
        "component_Y_CS": float(component_y) if np.isfinite(component_y) else np.nan,
        "tail_switch": float(sw),
        "Y_unclipped_CS": float(y_unclipped) if np.isfinite(y_unclipped) else np.nan,
        "Y_was_clipped": bool(np.isfinite(y_unclipped) and abs(float(y_unclipped) - float(y)) > 1e-12),
        "Y_CS": float(y_value),
        "Y_over_W": float(y_value / w_base) if np.isfinite(w_base) and abs(w_base) > 0 else np.nan,
        "FO_over_singular": float(fo / sing) if abs(float(sing)) > 1.0e-30 else np.nan,
        "nlo_singular_mode": str(getattr(cfg, "nlo_singular_mode", "analytic")),
        "nlo_singular_rsub": float(getattr(cfg, "nlo_singular_rsub", np.nan)),
        "nlo_singular_power": float(getattr(cfg, "nlo_singular_power", np.nan)),
        "nlo_singular_damp_kind": str(getattr(cfg, "nlo_singular_damp_kind", "")),
        "nlo_singular_damping": float(_base.nlo_singular_damping_factor(row, cfg)) if str(getattr(cfg, "nlo_singular_mode", "")).lower().replace("-", "_") in {"asymptotic_damped", "analytic_damped", "stable", "stable_asymptotic", "damped"} else np.nan,
        "nlo_real_convention": str(getattr(cfg, "nlo_real_convention", "base")),
        "nlo_singular_convention": str(getattr(cfg, "nlo_singular_convention", "base")),
        "nlo_alpha_convention": str(getattr(cfg, "nlo_alpha_convention", "alpha_over_pi")),
    }
    return i, float(y_value), diag


def y_nlo_dev_for_rows(df, w_baseline, pdf, cfg):
    global _WORKER_PDF, _WORKER_CFG, _WORKER_B_GRID
    yvals = np.zeros(len(df), dtype=float)
    diagnostics: list[dict] = [None] * len(df)  # type: ignore[list-item]
    requested_workers = int(os.environ.get("TEVATRON_Y_WORKERS", os.environ.get("TEVATRON_WGRID_WORKERS", "1")))
    workers = max(1, min(requested_workers, len(df)))
    t0 = time.time()
    if workers == 1:
        _WORKER_PDF = pdf
        _WORKER_CFG = cfg
        _WORKER_B_GRID = np.asarray([], dtype=float)
        for done, (_, row) in enumerate(df.iterrows(), start=1):
            i, y, diag = _compute_y_worker((done - 1, row.to_dict(), float(w_baseline[done - 1])))
            yvals[i] = y
            diagnostics[i] = diag
            if done % max(1, len(df) // 20) == 0 or done == len(df):
                print(f"  Y rows: {done}/{len(df)} ({time.time() - t0:.1f}s)", flush=True)
    else:
        items = [
            (i, row.to_dict(), float(w_baseline[i]))
            for i, (_, row) in enumerate(df.iterrows())
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_w_worker,
            initargs=(
                str(getattr(pdf, "pdf_set", "NNPDF40_nnlo_as_01180")),
                int(getattr(pdf, "member", 0)),
                bool(getattr(pdf, "use_toy_pdf", False)),
                cfg,
                np.asarray([], dtype=float),
            ),
        ) as pool:
            futures = [pool.submit(_compute_y_worker, item) for item in items]
            for done, future in enumerate(as_completed(futures), start=1):
                i, y, diag = future.result()
                yvals[i] = y
                diagnostics[i] = diag
                if done % max(1, len(df) // 20) == 0 or done == len(df):
                    print(f"  Y rows: {done}/{len(df)} ({time.time() - t0:.1f}s)", flush=True)
    _base.LAST_BACKEND_ROW_DIAGNOSTICS = pd.DataFrame(diagnostics)
    return yvals


_TEVATRON_Y_NLO_DEV_FOR_ROWS = y_nlo_dev_for_rows


def _float_key(row, key: str) -> str:
    value = _row_value(row, key)
    try:
        number = float(value)
        if math.isfinite(number):
            return f"{number:.12g}"
    except ValueError:
        pass
    return value.strip()


def _w_cache_key(row) -> tuple[str, ...]:
    qt_key = (
        _float_key(row, "qT"),
        _float_key(row, "qT_low"),
        _float_key(row, "qT_high"),
    ) if (_is_pbar_p(row) or _is_pp(row)) else ("fixed_target_qt_independent", "", "")
    return tuple(
        _float_key(row, key)
        for key in (
            "dataset",
            "QM",
            "QM_Low",
            "QM_High",
            "x1",
            "x2",
            "SqrtS",
            "y",
            "y_Low",
            "y_High",
            "target",
            "beam_config",
            "PreFactor",
            "unit",
            "theory_fiducial_factor",
        )
    ) + qt_key


def compute_backend_grids(df, pdf, cfg, *, progress: bool = True):
    b_grid = _base.make_b_grid(cfg)
    w_matrix = np.empty((len(df), len(b_grid)), dtype=float)
    cache: dict[tuple[str, ...], np.ndarray] = {}
    key_to_unique_index: dict[tuple[str, ...], int] = {}
    unique_items: list[tuple[int, object]] = []
    row_unique_index: list[int] = []
    for _, row in df.iterrows():
        key = _w_cache_key(row)
        unique_index = key_to_unique_index.get(key)
        if unique_index is None:
            unique_index = len(unique_items)
            key_to_unique_index[key] = unique_index
            unique_items.append((unique_index, row.copy()))
        row_unique_index.append(unique_index)

    t0 = time.time()
    requested_workers = int(os.environ.get("TEVATRON_WGRID_WORKERS", "1"))
    workers = max(1, min(requested_workers, len(unique_items)))
    unique_values: list[np.ndarray | None] = [None] * len(unique_items)

    if progress:
        print(
            f"  W grid: {len(df)} rows collapse to {len(unique_items)} unique kernels; workers={workers}",
            flush=True,
        )

    if workers == 1:
        for done, (unique_index, row) in enumerate(unique_items, start=1):
            unique_values[unique_index] = _full.wpert_cs_for_row(row, b_grid, pdf, cfg)
            if progress and (done % max(1, len(unique_items) // 20) == 0 or done == len(unique_items)):
                elapsed = time.time() - t0
                print(
                    f"  W unique kernels: {done}/{len(unique_items)} ({elapsed:.1f}s)",
                    flush=True,
                )
    else:
        worker_items = [
            (unique_index, row.to_dict() if hasattr(row, "to_dict") else dict(row))
            for unique_index, row in unique_items
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_w_worker,
            initargs=(
                str(getattr(pdf, "pdf_set", "NNPDF40_nnlo_as_01180")),
                int(getattr(pdf, "member", 0)),
                bool(getattr(pdf, "use_toy_pdf", False)),
                cfg,
                b_grid,
            ),
        ) as pool:
            futures = [pool.submit(_compute_w_worker, item) for item in worker_items]
            for done, future in enumerate(as_completed(futures), start=1):
                unique_index, values = future.result()
                unique_values[unique_index] = np.asarray(values, dtype=float)
                if progress and (done % max(1, len(unique_items) // 20) == 0 or done == len(unique_items)):
                    elapsed = time.time() - t0
                    print(
                        f"  W unique kernels: {done}/{len(unique_items)} ({elapsed:.1f}s)",
                        flush=True,
                    )

    for row_index, unique_index in enumerate(row_unique_index):
        values = unique_values[unique_index]
        if values is None:
            raise RuntimeError(f"missing W grid for unique kernel {unique_index}")
        w_matrix[row_index, :] = values

    if progress:
        elapsed = time.time() - t0
        print(
            f"  W grid complete: {len(df)} rows from {len(unique_items)} unique kernels ({elapsed:.1f}s)",
            flush=True,
            )

    w_baseline = _base.torch_bessel_integral(df["qT"].to_numpy(float), b_grid, w_matrix)
    y_mode = cfg.y_mode.lower().replace("-", "_")
    match_order = cfg.match_order.lower().replace("-", "_")
    if match_order == "nlo":
        match_order = "nlo_dev"
    if y_mode == "data_minus_w_debug":
        y = df["CS"].to_numpy(float) - w_baseline
    elif y_mode == "nlo_pilot" or match_order == "nlo_pilot":
        y = _base.y_nlo_pilot_for_rows(df, w_baseline, pdf, cfg)
        if progress:
            print("WARNING: using nlo_pilot finite-tail scaffold, not audited production Y_NLO.", flush=True)
    elif match_order == "nlo_dev":
        y = y_nlo_dev_for_rows(df, w_baseline, pdf, cfg)
        if progress:
            print("WARNING: using v15 NLO finite-tail development path. Localized asymptotic singular mode is available/default; independent benchmarking is still required before production Y_NLO claims.", flush=True)
    elif y_mode == "zero" and match_order in {"none", "zero"}:
        y = np.zeros(len(df), dtype=float)
    else:
        raise ValueError("Unsupported y/match setting: y_mode must be zero, nlo_pilot, or data_minus_w_debug; match_order must be none, nlo_pilot, nlo_dev, or nlo; for nlo_dev audits use --nlo-y-component")
    return b_grid, w_matrix, y


_TEVATRON_COMPUTE_BACKEND_GRIDS = compute_backend_grids


def load_fixed_target_data(
    data_dir: str | Path,
    datasets: Sequence[str] = _full.DEFAULT_DATASETS,
    cuts=_full.CutConfig(),
) -> pd.DataFrame:
    """Load data while preserving cache-aligned row_id values when present."""
    data_dir = Path(data_dir).expanduser().resolve()
    frames = []
    for ds in datasets:
        path = data_dir / _full._dataset_to_filename(ds)
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        df = pd.read_csv(path)
        df["dataset"] = path.stem
        df["source_file"] = str(path)
        if "local_index" not in df.columns:
            df["local_index"] = np.arange(len(df), dtype=int)
        if "row_id" in df.columns:
            df["row_id"] = df["row_id"].astype(str)
        else:
            df["row_id"] = df["dataset"].astype(str) + ":" + df["local_index"].astype(str)
        frames.append(df)
    if not frames:
        raise ValueError("No datasets requested")
    df = pd.concat(frames, ignore_index=True)
    required = ["qT", "QM", "x1", "x2", "SqrtS", "y", "A", "PreFactor", "CS", "error"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required data columns: {missing}")
    for col in ("sysNorm", "sysP2P"):
        if col in df.columns:
            df[col + "_rel"] = df[col].map(_full.parse_percent).astype(float)
        else:
            df[col + "_rel"] = 0.0

    mask = np.ones(len(df), dtype=bool)
    mode = cuts.mode.lower()
    if mode == "tmd_only":
        mask &= df["qT"].to_numpy(float) < cuts.tmd_qT_max_over_Q * df["QM"].to_numpy(float)
    elif mode == "matched":
        mask &= df["qT"].to_numpy(float) < cuts.qT_max_over_Q * df["QM"].to_numpy(float)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown mode {cuts.mode!r}")

    if cuts.apply_upsilon_veto:
        q = df["QM"].to_numpy(float)
        mask &= ~((q > cuts.upsilon_veto_low) & (q < cuts.upsilon_veto_high))
    return df.loc[mask].copy().reset_index(drop=True)


_TEVATRON_LOAD_FIXED_TARGET_DATA = load_fixed_target_data

# Patch the loaded backend modules. Functions such as compute_backend_grids and
# y_nlo_dev_for_rows resolve these names in the loaded base/scheme modules.
_base.charge_weighted_lumi = _TEVATRON_CHARGE_WEIGHTED_LUMI
_base.charge_weighted_qg_lumi = _TEVATRON_CHARGE_WEIGHTED_QG_LUMI
_base.channel_lumis_exact = _TEVATRON_CHANNEL_LUMIS_EXACT
_base.fixed_target_prefactor_cs = _TEVATRON_FIXED_TARGET_PREFACTOR_CS
_base.wpert_cs_for_row = _TEVATRON_WPERT_CS_FOR_ROW
_full.wpert_cs_for_row = _TEVATRON_WPERT_CS_FOR_ROW
_scheme.wpert_cs_for_row = _TEVATRON_WPERT_CS_FOR_ROW
_scheme._v22_luminosity = _TEVATRON_V22_LUMINOSITY
_base.compute_backend_grids = _TEVATRON_COMPUTE_BACKEND_GRIDS
_base.y_nlo_dev_for_rows = _TEVATRON_Y_NLO_DEV_FOR_ROWS
_full.load_fixed_target_data = _TEVATRON_LOAD_FIXED_TARGET_DATA

TEVATRON_PBARP_LUMINOSITY_ACTIVE = True
TEVATRON_WRAPPED_BACKEND = str(FULL_BACKEND_PATH)

for _name, _value in vars(_full).items():
    if _name.startswith("__") or _name.startswith("LAST_"):
        continue
    globals()[_name] = _value

globals()["_ORIGINAL_FIXED_TARGET_PREFACTOR_CS"] = _TEVATRON_ORIGINAL_FIXED_TARGET_PREFACTOR_CS
globals()["charge_weighted_lumi"] = _TEVATRON_CHARGE_WEIGHTED_LUMI
globals()["charge_weighted_qg_lumi"] = _TEVATRON_CHARGE_WEIGHTED_QG_LUMI
globals()["channel_lumis_exact"] = _TEVATRON_CHANNEL_LUMIS_EXACT
globals()["fixed_target_prefactor_cs"] = _TEVATRON_FIXED_TARGET_PREFACTOR_CS
globals()["wpert_cs_for_row"] = _TEVATRON_WPERT_CS_FOR_ROW
globals()["_v22_luminosity"] = _TEVATRON_V22_LUMINOSITY
globals()["compute_backend_grids"] = _TEVATRON_COMPUTE_BACKEND_GRIDS
globals()["y_nlo_dev_for_rows"] = _TEVATRON_Y_NLO_DEV_FOR_ROWS
globals()["load_fixed_target_data"] = _TEVATRON_LOAD_FIXED_TARGET_DATA
globals()["TEVATRON_PBARP_LUMINOSITY_ACTIVE"] = TEVATRON_PBARP_LUMINOSITY_ACTIVE
globals()["TEVATRON_WRAPPED_BACKEND"] = TEVATRON_WRAPPED_BACKEND


def __getattr__(name: str):
    return getattr(_full, name)
