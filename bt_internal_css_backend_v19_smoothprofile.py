#!/usr/bin/env python3
"""
bt_internal_css_backend_v18.py

Single-file internal CSS-style b_T-space backend for the DNN/TMD project.

V12 adds a first self-contained analytic-development NLO finite-qT tail
(``match_order=nlo_dev``) with a one-dimensional recoil-rapidity integral and an
asymptotic same-backend singular subtraction.  This is the first code path toward
Y_NLO = FO_NLO - singular_NLO.  It is still labelled DEVELOPMENT because the
normalization/convention must be audited against an independent fixed-order
calculation before production claims.

Primary use in the current workflow
-----------------------------------
Generate row-matched external-grid CSVs that can be passed directly to
scripts/train_realdata.py from the btfit_real_local package:

    python3 bt_internal_css_backend_v15.py grids \
      --data-dir ../Data \
      --pdf-set NNPDF40_nnlo_as_01180 \
      --mode matched \
      --qT-max-over-Q 1.0 \
      --out-dir internal_css_grids \
      --n-b 160 --b-max 8.0

Then train with:

    python3 scripts/train_realdata.py \
      --data-dir ../Data \
      --mode matched \
      --qT-max-over-Q 1.0 \
      --w-backend external \
      --w-grid internal_css_grids/wpert_internal_css_matched.csv \
      --y-grid internal_css_grids/y_internal_css_matched.csv \
      --epochs 500 --batch-size 128 --learn-gk \
      --out outputs/internal_css_replica000

What this script is and is not
------------------------------
This is a self-contained, auditable CSS-style backend scaffold.  It computes a
b*-regulated perturbative W-term integrand, with LHAPDF PDFs and alpha_s, in the
same CS convention used by the fixed-target CSV files.  The default Y-term is
zero.  V12 also includes a development-level analytic NLO real-emission finite-qT
path, ``match_order=nlo_dev`` (or ``nlo`` as an alias), which evaluates the
q qbar -> gamma* g and q g/g q -> gamma* q recoil-rapidity channels and subtracts
a same-backend CSS asymptotic singular term.  This is intended for backend
validation and matched-mode development; it is not yet an audited production
N3LL' + Y_NLO calculation.

Conventions for the output W grid
---------------------------------
The produced Wpert_CS satisfies the convention used by btfit/theory/wterm.py:

    CS_W(row) = int db b J0(qT*b) Wpert_CS(row,b)

before neural nonperturbative factors are applied.  The DNN then multiplies
Wpert_CS inside the b integral by F_NP(x1,b) F_NP(x2,b) exp[-gK(b) ln(Q/Q0)].

The default prefactor follows the legacy fixed-target A convention from the
existing k_T-space code and converts to the fitted CS column with

    A = PreFactor * CS.

So Wpert_CS includes the fixed-target prefactor, Q-bin integral, hbar*c unit
factor, charge-weighted x*f luminosity, Fourier-Bessel 1/(2*pi), and Sudakov.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# torch is only needed for optional baseline checks using J0.  The grid writer
# itself only requires numpy/pandas/LHAPDF.
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

DEFAULT_DATASETS = ("E288_200", "E288_300", "E288_400", "E605")
ALL_DATASETS = DEFAULT_DATASETS + ("E772",)


def parse_percent(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    if s.endswith("%"):
        s = s[:-1]
        try:
            return float(s) / 100.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else 0.0


@dataclass(frozen=True)
class CutConfig:
    mode: str = "matched"                 # matched, tmd_only, none
    qT_max_over_Q: float = 1.0
    tmd_qT_max_over_Q: float = 0.2
    upsilon_veto_low: float = 9.0
    upsilon_veto_high: float = 11.0
    apply_upsilon_veto: bool = True


def _dataset_to_filename(name: str) -> str:
    return name if name.endswith(".csv") else f"{name}.csv"


def load_fixed_target_data(
    data_dir: str | Path,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    cuts: CutConfig = CutConfig(),
) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser().resolve()
    frames = []
    for ds in datasets:
        path = data_dir / _dataset_to_filename(ds)
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        df = pd.read_csv(path)
        df["dataset"] = path.stem
        df["source_file"] = str(path)
        df["local_index"] = np.arange(len(df), dtype=int)
        # Match the row_id convention used by btfit_real_local.
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
            df[col + "_rel"] = df[col].map(parse_percent).astype(float)
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


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("dataset", dropna=False)
        .agg(
            n=("row_id", "size"),
            qT_min=("qT", "min"),
            qT_max=("qT", "max"),
            Q_min=("QM", "min"),
            Q_max=("QM", "max"),
            x1_min=("x1", "min"),
            x1_max=("x1", "max"),
            x2_min=("x2", "min"),
            x2_max=("x2", "max"),
        )
        .reset_index()
    )


# -----------------------------------------------------------------------------
# PDF providers and target modes
# -----------------------------------------------------------------------------

CHARGE2 = {1: 1.0 / 9.0, 2: 4.0 / 9.0, 3: 1.0 / 9.0, 4: 4.0 / 9.0, 5: 1.0 / 9.0}
DEFAULT_FLAVORS = (1, 2, 3)  # d,u,s by LHAPDF PID convention

# Filled by compute_backend_grids when a finite-tail backend is used.  Training
# scripts can save this to the run directory for row-level Y diagnostics.
LAST_BACKEND_ROW_DIAGNOSTICS: Optional[pd.DataFrame] = None

# Approximate target Z/A for simple isospin mixing.  These are only used for
# target_mode='nuclear_isospin'.
NUCLEAR_ZA = {
    "E288_200": (4.0, 9.0121831),   # Be target approximation
    "E288_300": (4.0, 9.0121831),
    "E288_400": (4.0, 9.0121831),
    "E605": (29.0, 63.546),         # Cu target approximation
    "E772": (0.5, 1.0),             # unknown in this CSV; use isoscalar fallback
}


def neutron_pid_from_proton(pid: int) -> int:
    """Isospin map: u <-> d for a neutron, sea included by sign."""
    sign = 1 if pid >= 0 else -1
    apid = abs(int(pid))
    if apid == 2:
        return sign * 1
    if apid == 1:
        return sign * 2
    return int(pid)


class ToyPDF:
    """Small deterministic PDF for offline smoke tests when LHAPDF is absent."""

    def __init__(self):
        pass

    def xfxQ(self, pid: int, x: float, q: float) -> float:
        x = float(np.clip(x, 1e-8, 1.0 - 1e-8))
        apid = abs(int(pid))
        sea = 0.05 * x ** (-0.15) * (1 - x) ** 7 * (1 + 0.02 * math.log(max(q, 1.0)))
        if pid < 0:
            return x * sea
        if apid == 2:
            val = 2.5 * x ** (-0.25) * (1 - x) ** 3 + sea
        elif apid == 1:
            val = 1.3 * x ** (-0.20) * (1 - x) ** 4 + sea
        elif apid == 3:
            val = 0.6 * sea
        else:
            val = 0.2 * sea
        return x * max(val, 0.0)

    def alphasQ(self, q: float) -> float:
        # One-loop-ish toy alpha_s, only for tests.
        q = max(float(q), 0.7)
        return min(0.6, 12 * math.pi / (25.0 * math.log((q * q + 0.5) / (0.2 * 0.2))))


class LHAPDFProvider:
    def __init__(self, pdf_set: str, member: int = 0, *, use_toy_pdf: bool = False):
        self.pdf_set = pdf_set
        self.member = int(member)
        self.use_toy_pdf = bool(use_toy_pdf)
        if self.use_toy_pdf:
            self.pdf = ToyPDF()
            return
        try:
            import lhapdf  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "LHAPDF Python module was not found. Either run on a machine with LHAPDF "
                "installed or pass --toy-pdf for nonphysical smoke tests."
            ) from exc
        self.lhapdf = lhapdf
        self.pdf = lhapdf.mkPDF(pdf_set, int(member))

    def xf_proton(self, pid: int, x: float, q: float) -> float:
        if not np.isfinite(x) or not np.isfinite(q) or x <= 0.0 or x >= 1.0 or q <= 0.0:
            return 0.0
        try:
            val = float(self.pdf.xfxQ(int(pid), float(x), float(q)))
        except Exception:
            return 0.0
        if not np.isfinite(val):
            return 0.0
        return val

    def xf_neutron(self, pid: int, x: float, q: float) -> float:
        return self.xf_proton(neutron_pid_from_proton(pid), x, q)

    def xf_target(self, pid: int, x: float, q: float, *, dataset: str, target_mode: str) -> float:
        mode = target_mode.lower()
        if mode == "proton_approx":
            return self.xf_proton(pid, x, q)
        if mode == "isoscalar":
            return 0.5 * self.xf_proton(pid, x, q) + 0.5 * self.xf_neutron(pid, x, q)
        if mode == "nuclear_isospin":
            z, a = NUCLEAR_ZA.get(str(dataset), (0.5, 1.0))
            zp = float(z) / float(a)
            return zp * self.xf_proton(pid, x, q) + (1.0 - zp) * self.xf_neutron(pid, x, q)
        if mode == "nuclear_pdf":
            raise NotImplementedError(
                "target_mode='nuclear_pdf' needs a chosen nuclear PDF set and A/Z mapping. "
                "Use proton_approx, isoscalar, or nuclear_isospin for now."
            )
        raise ValueError(f"Unknown target_mode {target_mode!r}")

    def alphas(self, q: float) -> float:
        try:
            val = float(self.pdf.alphasQ(float(q)))
        except Exception:
            val = 0.118
        if not np.isfinite(val) or val <= 0.0:
            return 0.118
        return val


# -----------------------------------------------------------------------------
# CSS-style perturbative ingredients
# -----------------------------------------------------------------------------

CF = 4.0 / 3.0
CA = 3.0
TR = 0.5
ZETA3 = 1.2020569031595942854
C0 = 2.0 * math.exp(-0.5772156649015328606)


@dataclass(frozen=True)
class CSSConfig:
    # b-space grid used for the exported W table.  Must be compatible with the
    # training code, which currently trapezoid-integrates over the listed bT.
    b_min: float = 1.0e-4
    b_max: float = 8.0
    n_b: int = 160

    # b* and scales
    bstar_bmax: float = 1.5
    mu_min: float = 1.3

    # Smooth only the lower mu_b floor.  width=0 restores the original
    # hard max(mu_raw, mu_min) prescription.
    mu_floor_smooth_width: float = 0.12

    # Smooth the upper mu_b=Q cap. width=0 restores hard min(mu,Q).
    mu_cap_smooth_width: float = 0.12

    cap_mub_at_Q: bool = True
    q0: float = 2.0

    # perturbative content.  n3ll_pilot is NOT a full N3LL' calculation; it just
    # uses the highest coefficients included below.
    resum_order: str = "nnll"  # ll, nll, nnll, n3llp/n3ll_pilot
    # Matching order for the finite Y term.
    #   none      : Y=0, appropriate for TMD-only tests.
    #   nlo_pilot : data-independent finite-tail scaffold for matched-mode plumbing.
    #   nlo_dev   : first analytic-development recoil-rapidity NLO finite tail.
    #   nlo       : accepted as an alias of nlo_dev in this development script, with metadata warnings.
    # The production audited Y_NLO formula should replace/validate nlo_dev in this file.
    match_order: str = "none"
    nlo_y_pilot_strength: float = 1.0
    nlo_y_transition: float = 0.20
    nlo_y_transition_width: float = 0.15
    # Development-level analytic NLO real-emission finite-tail controls.
    # nlo_dev computes FO_NLO_real_dev - singular_NLO_dev and smoothly turns it on
    # above nlo_y_transition to avoid contaminating the strict TMD window before
    # the normalization audit is complete.
    nlo_real_quad: int = 48
    nlo_real_norm: float = 1.0
    nlo_singular_norm: float = 1.0
    # Audit knob: which development finite-tail component to export before switch/clip.
    # raw=FO-singular, positive=max(FO-singular,0), fo_only=FO, minus_sing=-singular,
    # singular_only=singular, zero=0.
    nlo_y_component: str = "raw"
    # Clip |Y| <= nlo_y_clip_multiple * max(|W|, |data CS|). Use <=0 to disable.
    nlo_y_clip_multiple: float = 5.0
    nlo_dev_use_switch: bool = True
    nlo_dev_min_qt_over_q: float = 1.0e-4

    # V16 audit controls for the finite tail.  These isolate convention and
    # same-scheme subtraction issues before production claims.
    nlo_singular_mode: str = "asymptotic_damped"  # analytic, asymptotic_damped, wexp_numeric, wexp_positive, none
    nlo_singular_rsub: float = 0.20
    nlo_singular_power: float = 4.0
    nlo_singular_damp_kind: str = "exp"  # exp or rational
    nlo_real_convention: str = "base"
    nlo_singular_convention: str = "base"
    nlo_alpha_convention: str = "alpha_over_pi"  # alpha_over_pi or alpha_over_2pi

    # V16 external-code calibrated high-qT/Q real-tail repair.
    # Motivation: MCFM and DYTurbo agree with the v15 real tail near qT/Q~0.32,
    # but both external codes are ~0.18--0.22 of v15 by qT/Q~0.55--0.60.
    # This optional factor multiplies only the finite-qT FO real term before
    # singular subtraction.  Keep mode=none for pure v15 behavior.
    nlo_real_tail_repair: str = "none"  # none, mcfm_logistic
    nlo_real_tail_r0: float = 0.520
    nlo_real_tail_width: float = 0.010
    nlo_real_tail_rinf: float = 0.180

    nf: int = 5
    n_sudakov_quad: int = 32

    # electromagnetic/fixed-target convention
    alpha_em: float = 1.0 / 137.035999084
    # hbar*c conversion used by the old code, effectively pb*GeV^2.
    hc_factor: float = 3.893793656e8
    prefactor_scheme: str = "oldA_to_CS"  # oldA_to_CS or unit
    global_norm: float = 1.0

    # flavor/target
    flavors: Tuple[int, ...] = DEFAULT_FLAVORS
    target_mode: str = "proton_approx"  # proton_approx, isoscalar, nuclear_isospin, nuclear_pdf

    # fixed-order tail override.  Normally leave y_mode="zero" and use match_order.
    # data_minus_w_debug forces W+Y through the data and must never be used physically.
    y_mode: str = "zero"  # zero, nlo_pilot, or data_minus_w_debug

    # numerical guards
    min_positive: float = 1.0e-300


def css_coefficients(order: str, nf: int) -> Tuple[List[float], List[float]]:
    """Return A_n and B_n in expansions sum_n (alpha_s/pi)^n coeff_n.

    Included coefficients are a standard CSS-style quark Sudakov set sufficient
    for a transparent pilot backend.  The n3ll_pilot mode is not a complete
    N3LL' implementation because hard/OPE matching constants and finite-tail
    matching are not yet included in this single-file backend.
    """
    nf = float(nf)
    pi = math.pi

    A1 = CF
    A2 = 0.5 * CF * (CA * (67.0 / 18.0 - pi * pi / 6.0) - 5.0 * nf / 9.0)
    # Common A3 expression in alpha_s/pi convention.
    A3 = CF * (
        CA * CA * (245.0 / 96.0 - 67.0 * pi * pi / 216.0 + 11.0 * pi ** 4 / 720.0 + 11.0 * ZETA3 / 24.0)
        + CA * nf * (-209.0 / 432.0 + 5.0 * pi * pi / 108.0 - 7.0 * ZETA3 / 12.0)
        + CF * nf * (-55.0 / 96.0 + ZETA3 / 2.0)
        - nf * nf / 108.0
    )

    B1 = -1.5 * CF
    B2 = (
        CF * CF * (pi * pi / 4.0 - 3.0 / 16.0 - 3.0 * ZETA3)
        + CF * CA * (11.0 * pi * pi / 36.0 - 193.0 / 48.0 + 1.5 * ZETA3)
        + CF * nf * (17.0 / 24.0 - pi * pi / 18.0)
    )

    order = order.lower().replace("-", "_")
    if order in {"n3llp", "n3ll_prime", "n3llprime", "n3llp_pilot"}:
        order = "n3ll_pilot"
    if order == "ll":
        return [A1], []
    if order == "nll":
        return [A1, A2], [B1]
    if order == "nnll":
        return [A1, A2, A3], [B1, B2]
    if order == "n3ll_pilot":
        # Deliberately same as nnll until the N3LL' hard/OPE/matching audit is done.
        return [A1, A2, A3], [B1, B2]
    raise ValueError("resum_order must be ll, nll, nnll, n3llp, or n3ll_pilot")


def bstar(b: np.ndarray | float, bmax: float) -> np.ndarray | float:
    return np.asarray(b) / np.sqrt(1.0 + (np.asarray(b) / float(bmax)) ** 2)


def _softplus_scalar(z: float) -> float:
    # Numerically stable scalar softplus.
    z = float(z)
    if z > 35.0:
        return z
    if z < -35.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def mu_b_of_b(b: float, q: float, cfg: CSSConfig) -> float:
    bs = float(bstar(max(float(b), 1e-12), cfg.bstar_bmax))
    mu_raw = C0 / max(bs, 1e-12)

    width = max(float(cfg.mu_floor_smooth_width), 0.0)

    if width > 0.0:
        # Smooth approximation to max(mu_raw, mu_min):
        #
        #   mu_min + width * softplus((mu_raw-mu_min)/width).
        #
        # Away from the transition this approaches the original profile,
        # but its first derivative is continuous.
        mu = float(cfg.mu_min) + width * _softplus_scalar(
            (mu_raw - float(cfg.mu_min)) / width
        )
    else:
        mu = max(mu_raw, float(cfg.mu_min))

    # Leave the upper Q cap unchanged in this first isolation test.
    if cfg.cap_mub_at_Q:
        q_cap = max(float(q), float(cfg.mu_min))
        cap_width = max(float(cfg.mu_cap_smooth_width), 0.0)

        if cap_width > 0.0:
            # Smooth approximation to min(mu, q_cap):
            #
            #   q_cap - width * softplus((q_cap-mu)/width).
            #
            # It approaches mu below the transition and q_cap above it,
            # while keeping the first derivative continuous.
            mu = q_cap - cap_width * _softplus_scalar(
                (q_cap - mu) / cap_width
            )
        else:
            mu = min(mu, q_cap)

    return float(mu)


def eval_series(coeffs: Sequence[float], a: float) -> float:
    # coeffs are 1-indexed conceptually: coeffs[0] * a^1 + coeffs[1] * a^2 + ...
    out = 0.0
    power = a
    for c in coeffs:
        out += float(c) * power
        power *= a
    return out


def sudakov_s(b: float, q: float, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    """Numerically integrate S(b,Q) = int_{mu_b}^{Q} dmu/mu [2 A ln(Q/mu)+B]."""
    q = float(q)
    if q <= cfg.mu_min:
        return 0.0
    mu0 = mu_b_of_b(b, q, cfg)
    if mu0 >= q * (1.0 - 1e-12):
        return 0.0

    xs, ws = np.polynomial.legendre.leggauss(int(cfg.n_sudakov_quad))
    t0, t1 = math.log(mu0), math.log(q)
    ts = 0.5 * (t1 - t0) * xs + 0.5 * (t1 + t0)
    weights = 0.5 * (t1 - t0) * ws
    A_coeffs, B_coeffs = css_coefficients(cfg.resum_order, cfg.nf)
    total = 0.0
    for t, w in zip(ts, weights):
        mu = math.exp(float(t))
        a = pdf.alphas(mu) / math.pi
        A = eval_series(A_coeffs, a)
        B = eval_series(B_coeffs, a) if B_coeffs else 0.0
        total += float(w) * (2.0 * A * math.log(q / mu) + B)
    # Sudakov exponent should generally damp.  Let negative values pass, but
    # avoid numerical explosion from extreme coefficient/profile combinations.
    return float(np.clip(total, -50.0, 200.0))


def qbin_integral(row: pd.Series) -> float:
    if "QM_Low" in row and "QM_High" in row and np.isfinite(row["QM_Low"]) and np.isfinite(row["QM_High"]):
        qlo = float(row["QM_Low"])
        qhi = float(row["QM_High"])
        if qlo > 0.0 and qhi > qlo:
            return 0.5 * (qlo ** -2 - qhi ** -2)
    q = float(row["QM"])
    return 1.0 / max(q, 1e-12) ** 3


def fixed_target_prefactor_cs(row: pd.Series, cfg: CSSConfig) -> float:
    """Prefactor that converts the b-space structure kernel to the CSV CS convention."""
    scheme = cfg.prefactor_scheme.lower()
    if scheme == "unit":
        return cfg.global_norm
    if scheme != "olda_to_cs":
        raise ValueError("prefactor_scheme must be oldA_to_CS or unit")
    factor = ((4.0 * math.pi * cfg.alpha_em) ** 2) / (9.0 * 2.0 * math.pi)
    a_pref = factor * cfg.hc_factor * qbin_integral(row)
    pre = float(row.get("PreFactor", 1.0))
    if not np.isfinite(pre) or abs(pre) < 1e-300:
        raise ValueError(f"Bad PreFactor for row_id={row.get('row_id')}: {pre}")
    return cfg.global_norm * a_pref / pre


def charge_weighted_lumi(row: pd.Series, mu: float, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    ds = str(row["dataset"])
    total = 0.0
    for flav in cfg.flavors:
        pid = int(abs(flav))
        e2 = CHARGE2.get(pid, 0.0)
        if e2 == 0.0:
            continue
        # beam is proton; target follows target_mode.
        q1 = pdf.xf_proton(pid, x1, mu)
        qb1 = pdf.xf_proton(-pid, x1, mu)
        q2 = pdf.xf_target(pid, x2, mu, dataset=ds, target_mode=cfg.target_mode)
        qb2 = pdf.xf_target(-pid, x2, mu, dataset=ds, target_mode=cfg.target_mode)
        total += e2 * (q1 * qb2 + qb1 * q2)
    if not np.isfinite(total):
        return 0.0
    return max(float(total), 0.0)


def charge_weighted_qg_lumi(row: pd.Series, mu: float, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    """Charge-weighted qg+gq luminosity used only by the nlo_pilot finite-tail scaffold."""
    x1 = float(row["x1"])
    x2 = float(row["x2"])
    ds = str(row["dataset"])
    g1 = pdf.xf_proton(21, x1, mu)
    g2 = pdf.xf_target(21, x2, mu, dataset=ds, target_mode=cfg.target_mode)
    total = 0.0
    for flav in cfg.flavors:
        pid = int(abs(flav))
        e2 = CHARGE2.get(pid, 0.0)
        if e2 == 0.0:
            continue
        q1 = pdf.xf_proton(pid, x1, mu) + pdf.xf_proton(-pid, x1, mu)
        q2 = pdf.xf_target(pid, x2, mu, dataset=ds, target_mode=cfg.target_mode) + pdf.xf_target(-pid, x2, mu, dataset=ds, target_mode=cfg.target_mode)
        total += e2 * (q1 * g2 + g1 * q2)
    if not np.isfinite(total):
        return 0.0
    return max(float(total), 0.0)


def smooth_tail_switch(r: float, r0: float, width: float) -> float:
    """Smoothly turns on for r=qT/Q above r0."""
    r = float(r)
    r0 = float(r0)
    width = max(float(width), 1e-6)
    if r <= r0:
        return 0.0
    z = (r - r0) / width
    return float(1.0 - math.exp(-z * z))


def nlo_real_tail_repair_factor(r: float, cfg: CSSConfig) -> float:
    """External-code calibrated multiplier for the finite-qT NLO real tail.

    mode=none returns 1 and reproduces v15.

    mode=mcfm_logistic implements a deliberately minimal empirical correction
    anchored to the MCFM/DYTurbo checks available during v18 development:
      * r=qT/Q≈0.318: external/v15 ≈ 1
      * r=qT/Q≈0.553: external/v15 ≈ 0.22
      * r=qT/Q≈0.600: external/v15 ≈ 0.18

    The correction is applied only to FO_NLO_real_dev, not to the singular
    subtraction.  The default parameters leave r<~0.49 nearly unchanged and
    approach rinf at high r.
    """
    mode = str(getattr(cfg, "nlo_real_tail_repair", "none")).lower().replace("-", "_")
    if mode in {"none", "off", "zero", "v15"}:
        return 1.0
    if mode not in {"mcfm_logistic", "external_logistic", "logistic"}:
        raise ValueError(f"Unsupported nlo_real_tail_repair={getattr(cfg, 'nlo_real_tail_repair', None)!r}")
    r = float(r)
    r0 = float(getattr(cfg, "nlo_real_tail_r0", 0.520))
    width = max(float(getattr(cfg, "nlo_real_tail_width", 0.010)), 1.0e-6)
    rinf = float(getattr(cfg, "nlo_real_tail_rinf", 0.180))
    rinf = float(np.clip(rinf, 0.0, 1.0))
    z = np.clip((r - r0) / width, -700.0, 700.0)
    logistic = 1.0 / (1.0 + math.exp(-float(z)))
    fac = 1.0 - (1.0 - rinf) * logistic
    return float(np.clip(fac, min(rinf, 1.0), 1.0))


def nlo_real_tail_repair_factor_for_row(row: pd.Series, cfg: CSSConfig) -> float:
    q = max(float(row["QM"]), 1.0e-12)
    qt = max(float(row["qT"]), 0.0)
    return nlo_real_tail_repair_factor(qt / q, cfg)


def _safe_pdf_xf(provider: LHAPDFProvider, pid: int, x: float, mu: float, *, dataset: str, target_mode: str, beam: bool) -> float:
    if beam:
        return provider.xf_proton(pid, x, mu)
    return provider.xf_target(pid, x, mu, dataset=dataset, target_mode=target_mode)


def channel_lumis_exact(row: pd.Series, xa: float, xb: float, mu: float, pdf: LHAPDFProvider, cfg: CSSConfig) -> Tuple[float, float, float]:
    """Charge-weighted luminosities for qqbar, qg(target), and gq(beam).

    The first incoming hadron is the proton beam, the second is the fixed target.
    qg means a quark/antiquark from the beam and a gluon from the target;
    gq means a gluon from the beam and a quark/antiquark from the target.
    """
    ds = str(row["dataset"])
    g1 = pdf.xf_proton(21, xa, mu)
    g2 = pdf.xf_target(21, xb, mu, dataset=ds, target_mode=cfg.target_mode)
    qq = 0.0
    qg = 0.0
    gq = 0.0
    for flav in cfg.flavors:
        pid = int(abs(flav))
        e2 = CHARGE2.get(pid, 0.0)
        if e2 == 0.0:
            continue
        q1 = pdf.xf_proton(pid, xa, mu)
        qb1 = pdf.xf_proton(-pid, xa, mu)
        q2 = pdf.xf_target(pid, xb, mu, dataset=ds, target_mode=cfg.target_mode)
        qb2 = pdf.xf_target(-pid, xb, mu, dataset=ds, target_mode=cfg.target_mode)
        qq += e2 * (q1 * qb2 + qb1 * q2)
        qg += e2 * ((q1 + qb1) * g2)
        gq += e2 * (g1 * (q2 + qb2))
    return max(float(qq), 0.0), max(float(qg), 0.0), max(float(gq), 0.0)


def _nlo_convention_multiplier(spec: str, row: pd.Series) -> float:
    """Development multiplier for auditing NLO-tail unit conventions."""
    if spec is None:
        return 1.0
    q = max(float(row.get("QM", 1.0)), 1.0e-30)
    qt = max(float(row.get("qT", 1.0)), 1.0e-30)
    out = 1.0
    clean = str(spec).strip().lower().replace("-", "_")
    if clean in {"", "base", "none", "1", "unit"}:
        return 1.0
    for tok in re.split(r"[,*\s]+", clean):
        if not tok or tok in {"base", "none", "1", "unit"}:
            continue
        if tok in {"times_qt", "qt", "q_t"}:
            out *= qt
        elif tok in {"times_2qt", "2qt", "2q_t"}:
            out *= 2.0 * qt
        elif tok in {"div_qt", "over_qt", "inv_qt"}:
            out /= qt
        elif tok in {"div_2qt", "over_2qt", "inv_2qt"}:
            out /= 2.0 * qt
        elif tok in {"times_q", "q", "times_qm", "qm"}:
            out *= q
        elif tok in {"times_2q", "2q", "times_2qm", "2qm"}:
            out *= 2.0 * q
        elif tok in {"div_q", "over_q", "inv_q", "div_qm", "over_qm", "inv_qm"}:
            out /= q
        elif tok in {"div_2q", "over_2q", "inv_2q", "div_2qm", "over_2qm", "inv_2qm"}:
            out /= 2.0 * q
        elif tok in {"times_2pi", "2pi"}:
            out *= 2.0 * math.pi
        elif tok in {"div_2pi", "over_2pi", "inv_2pi"}:
            out /= 2.0 * math.pi
        else:
            raise ValueError(f"Unsupported NLO convention multiplier token {tok!r} in {spec!r}")
    return float(out)


def _nlo_alpha_factor(pdf: LHAPDFProvider, mu: float, cfg: CSSConfig) -> float:
    conv = str(getattr(cfg, "nlo_alpha_convention", "alpha_over_pi")).lower().replace("-", "_")
    if conv in {"alpha_over_pi", "as_over_pi", "alphas_over_pi"}:
        return pdf.alphas(mu) / math.pi
    if conv in {"alpha_over_2pi", "as_over_2pi", "alphas_over_2pi"}:
        return pdf.alphas(mu) / (2.0 * math.pi)
    raise ValueError(f"Unsupported nlo_alpha_convention={cfg.nlo_alpha_convention!r}")


def sudakov_s_one_loop(b: float, q: float, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    """One-loop Sudakov exponent used for same-backend W-expansion subtraction."""
    q = float(q)
    if q <= cfg.mu_min:
        return 0.0
    mu0 = mu_b_of_b(b, q, cfg)
    if mu0 >= q * (1.0 - 1e-12):
        return 0.0
    xs, ws = np.polynomial.legendre.leggauss(int(cfg.n_sudakov_quad))
    t0, t1 = math.log(mu0), math.log(q)
    ts = 0.5 * (t1 - t0) * xs + 0.5 * (t1 + t0)
    weights = 0.5 * (t1 - t0) * ws
    A1 = CF
    B1 = -1.5 * CF
    total = 0.0
    for t, w in zip(ts, weights):
        mu = math.exp(float(t))
        a = _nlo_alpha_factor(pdf, mu, cfg)
        total += float(w) * (2.0 * A1 * a * math.log(q / mu) + B1 * a)
    return float(np.clip(total, -50.0, 200.0))


def singular_nlo_wexp_numeric_for_row(row: pd.Series, pdf: LHAPDFProvider, cfg: CSSConfig, *, positive: bool = False) -> float:
    """Numerical O(alpha_s) expansion of the same b-space W Sudakov.

    For qT>0, the Born delta contribution drops out.  This computes the Bessel
    transform of -W_LO*S1 using the same b grid, b* prescription, scale profile,
    PDF luminosity convention, and Fourier normalization as the resummed W term.
    It is a same-backend diagnostic and still requires independent validation.
    """
    if torch is None:
        return singular_nlo_analytic_for_row(row, pdf, cfg)
    q = float(row["QM"])
    qt = float(row["qT"])
    if q <= 0.0 or qt <= 0.0:
        return 0.0
    b = make_b_grid(cfg)
    pref = fixed_target_prefactor_cs(row, cfg)
    fourier_norm = 1.0 / (2.0 * math.pi)
    vals = np.empty_like(b, dtype=float)
    for j, bj in enumerate(b):
        mu = mu_b_of_b(float(bj), q, cfg)
        lum = charge_weighted_lumi(row, mu, pdf, cfg)
        s1 = sudakov_s_one_loop(float(bj), q, pdf, cfg)
        vals[j] = pref * fourier_norm * lum * s1
    tb = torch.tensor(b, dtype=torch.float64)
    tv = torch.tensor(vals, dtype=torch.float64)
    tq = torch.tensor(float(qt), dtype=torch.float64)
    integral = torch.trapezoid(tb * torch.special.bessel_j0(tq * tb) * tv, tb).item()
    val = -float(integral)
    if positive:
        val = max(val, 0.0)
    val *= float(getattr(cfg, "nlo_singular_norm", 1.0))
    val *= _nlo_convention_multiplier(getattr(cfg, "nlo_singular_convention", "base"), row)
    if not np.isfinite(val):
        return 0.0
    return float(val)

def _nlo_yj_bounds(qt: float, q: float, y: float, sqrts: float) -> Optional[Tuple[float, float]]:
    if qt <= 0.0 or q <= 0.0 or sqrts <= 0.0:
        return None
    mt = math.sqrt(q * q + qt * qt)
    upper_arg = (sqrts - mt * math.exp(y)) / qt
    lower_arg = (sqrts - mt * math.exp(-y)) / qt
    if upper_arg <= 0.0 or lower_arg <= 0.0:
        return None
    ymin = -math.log(lower_arg)
    ymax = math.log(upper_arg)
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin >= ymax:
        return None
    # Avoid excessively huge intervals from very small qT; the strict small-qT
    # region is protected by the smooth switch in the Y term anyway.
    return max(ymin, -25.0), min(ymax, 25.0)


def _nlo_matrix_shapes(q2: float, shat: float, that: float, uhat: float) -> Tuple[float, float, float]:
    """Positive dimensionless real-emission channel shapes.

    These are the standard soft/collinear pole structures for the q qbar and
    qg/gq channels expressed with positive variables Sbar=s-Q^2,
    Tbar=Q^2-t, Ubar=Q^2-u.  They are used by the nlo_dev backend as a
    transparent, auditable starting point; the absolute normalization must still
    be benchmarked against an independent NLO code before production use.
    """
    eps = 1.0e-30
    sbar = max(float(shat - q2), eps)
    tbar = max(float(q2 - that), eps)
    ubar = max(float(q2 - uhat), eps)
    mqq = CF * (tbar * tbar + ubar * ubar) / max(tbar * ubar, eps)
    mqg = TR * (sbar * sbar + ubar * ubar) / max(sbar * ubar, eps)
    mgq = TR * (sbar * sbar + tbar * tbar) / max(sbar * tbar, eps)
    return float(mqq), float(mqg), float(mgq)


def fo_nlo_real_dev_for_row(row: pd.Series, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    """Development-level finite-qT real-emission NLO DY cross section.

    The integral is over the recoil-parton rapidity y_j.  It includes the q qbar,
    qg, and gq real-emission pole structures and is returned in the same CS
    convention as the W baseline.  This is a first self-contained backend path,
    not yet a production-normalized NLO calculation.
    """
    q = float(row["QM"])
    qt = float(row["qT"])
    y = float(row["y"])
    sqrts = float(row["SqrtS"])
    if q <= 0.0 or qt <= 0.0 or sqrts <= 0.0:
        return 0.0
    bounds = _nlo_yj_bounds(qt, q, y, sqrts)
    if bounds is None:
        return 0.0
    ymin, ymax = bounds
    if ymin >= ymax:
        return 0.0
    mt = math.sqrt(q * q + qt * qt)
    q2 = q * q
    pref = fixed_target_prefactor_cs(row, cfg)
    mu = max(q, cfg.mu_min)
    alpha = _nlo_alpha_factor(pdf, mu, cfg)
    xs, ws = np.polynomial.legendre.leggauss(int(max(cfg.nlo_real_quad, 8)))
    yjs = 0.5 * (ymax - ymin) * xs + 0.5 * (ymax + ymin)
    weights = 0.5 * (ymax - ymin) * ws
    total = 0.0
    for yj, wt in zip(yjs, weights):
        xa = (mt * math.exp(y) + qt * math.exp(float(yj))) / sqrts
        xb = (mt * math.exp(-y) + qt * math.exp(-float(yj))) / sqrts
        if xa <= 0.0 or xa >= 1.0 or xb <= 0.0 or xb >= 1.0:
            continue
        shat = xa * xb * sqrts * sqrts
        that = q2 - xa * sqrts * mt * math.exp(-y)
        uhat = q2 - xb * sqrts * mt * math.exp(y)
        mqq, mqg, mgq = _nlo_matrix_shapes(q2, shat, that, uhat)
        lqq, lqg, lgq = channel_lumis_exact(row, xa, xb, mu, pdf, cfg)
        total += float(wt) * (lqq * mqq + lqg * mqg + lgq * mgq)
    # The recoil phase-space normalization below is chosen so this development
    # calculation has the right qualitative dimensions and remains comparable to
    # the W baseline.  It must be benchmarked before production claims.
    phase = qt / max(q2 + qt * qt, 1.0e-30)
    val = pref * cfg.nlo_real_norm * alpha * phase * total / (2.0 * math.pi)
    val *= _nlo_convention_multiplier(getattr(cfg, "nlo_real_convention", "base"), row)
    if not np.isfinite(val):
        return 0.0
    return max(float(val), 0.0)


def singular_nlo_analytic_for_row(row: pd.Series, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    """Same-backend CSS asymptotic singular approximation at O(alpha_s).

    Returned in the same CS convention as the W baseline.  This is intended to
    cancel the leading 1/qT logarithmic behavior of the real-emission dev term.
    It is regularized by cfg.nlo_dev_min_qt_over_q and smoothly turned off in
    the strict TMD region when nlo_dev_use_switch is active.
    """
    q = float(row["QM"])
    qt = float(row["qT"])
    if q <= 0.0 or qt <= 0.0:
        return 0.0
    qt_eff = max(qt, float(cfg.nlo_dev_min_qt_over_q) * q, 1.0e-6)
    mu = max(q, cfg.mu_min)
    pref = fixed_target_prefactor_cs(row, cfg)
    alpha = _nlo_alpha_factor(pdf, mu, cfg)
    lum = charge_weighted_lumi(row, mu, pdf, cfg)
    A1 = CF
    B1 = -1.5 * CF
    log_term = math.log(max(q * q / (qt_eff * qt_eff), 1.0))
    # d/dqT from the conventional 1/qT^2 singular distribution gives 2/qT.
    coeff = max(2.0 * A1 * log_term + 2.0 * B1, 0.0)
    val = pref * cfg.nlo_singular_norm * alpha * lum * coeff / max(qt_eff, 1.0e-30) / (2.0 * math.pi)
    val *= _nlo_convention_multiplier(getattr(cfg, "nlo_singular_convention", "base"), row)
    if not np.isfinite(val):
        return 0.0
    return max(float(val), 0.0)


def nlo_singular_damping_factor(row: pd.Series, cfg: CSSConfig) -> float:
    """Localized damping for the asymptotic singular subtraction.

    This is a v15 diagnostic/production-path stabilizer: the asymptotic
    subtraction should act only in the W/FO overlap region and should not be
    applied as a large finite correction throughout the moderate-qT tail.  The
    existing Y turn-on S_Y(r) is applied later in y_nlo_dev_for_rows; this factor
    is a separate subtraction-profile S_sub(r).
    """
    q = max(float(row["QM"]), 1.0e-12)
    qt = max(float(row["qT"]), 0.0)
    r = qt / q
    rsub = max(float(getattr(cfg, "nlo_singular_rsub", 0.20)), 1.0e-6)
    power = max(float(getattr(cfg, "nlo_singular_power", 4.0)), 0.25)
    x = (r / rsub) ** power
    kind = str(getattr(cfg, "nlo_singular_damp_kind", "exp")).lower().replace("-", "_")
    if kind in {"exp", "exponential", "gaussian_like"}:
        fac = math.exp(-min(x, 700.0))
    elif kind in {"rational", "lorentz", "lorentzian", "power"}:
        fac = 1.0 / (1.0 + x)
    else:
        raise ValueError(f"Unsupported nlo_singular_damp_kind={cfg.nlo_singular_damp_kind!r}")
    return float(np.clip(fac, 0.0, 1.0))


def singular_nlo_asymptotic_damped_for_row(row: pd.Series, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    """Stable qT-space asymptotic subtraction with localized damping.

    This starts from the analytic CSS small-qT asymptotic form used for the
    development subtraction, but multiplies it by an independent profile
    S_sub(qT/Q) so the subtraction is localized to the singular-overlap region.
    This avoids using the raw finite-grid Bessel transform of the expanded W
    term as a pointwise finite-qT subtraction.
    """
    return singular_nlo_analytic_for_row(row, pdf, cfg) * nlo_singular_damping_factor(row, cfg)


def singular_nlo_dev_for_row(row: pd.Series, pdf: LHAPDFProvider, cfg: CSSConfig) -> float:
    mode = str(getattr(cfg, "nlo_singular_mode", "analytic")).lower().replace("-", "_")
    if mode in {"none", "zero", "off"}:
        return 0.0
    if mode in {"analytic", "css_analytic", "v13"}:
        return singular_nlo_analytic_for_row(row, pdf, cfg)
    if mode in {"asymptotic_damped", "analytic_damped", "stable", "stable_asymptotic", "damped"}:
        return singular_nlo_asymptotic_damped_for_row(row, pdf, cfg)
    if mode in {"wexp", "wexp_numeric", "same_scheme", "same_scheme_numeric"}:
        return singular_nlo_wexp_numeric_for_row(row, pdf, cfg, positive=False)
    if mode in {"wexp_positive", "wexp_pos", "same_scheme_positive"}:
        return singular_nlo_wexp_numeric_for_row(row, pdf, cfg, positive=True)
    raise ValueError(f"Unsupported nlo_singular_mode={cfg.nlo_singular_mode!r}")

def y_nlo_dev_for_rows(df: pd.DataFrame, w_baseline: np.ndarray, pdf: LHAPDFProvider, cfg: CSSConfig) -> np.ndarray:
    """First self-contained development Y_NLO = FO_real_dev - singular_dev.

    The result is smoothly zeroed in the strict small-qT region by default.  This
    keeps the existing TMD-only behavior intact while exposing a real finite-tail
    path for matched-mode development.  Row-level diagnostics are stored in the
    module-level LAST_BACKEND_ROW_DIAGNOSTICS DataFrame.
    """
    global LAST_BACKEND_ROW_DIAGNOSTICS
    yvals = np.zeros(len(df), dtype=float)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        q = max(float(row["QM"]), 1.0e-12)
        qt = max(float(row["qT"]), 0.0)
        r = qt / q
        fo_raw = fo_nlo_real_dev_for_row(row, pdf, cfg)
        real_tail_repair_factor = nlo_real_tail_repair_factor(r, cfg)
        fo = fo_raw * real_tail_repair_factor
        sing = singular_nlo_dev_for_row(row, pdf, cfg)
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
            raise ValueError(f"Unsupported nlo_y_component={cfg.nlo_y_component!r}")
        sw = smooth_tail_switch(r, cfg.nlo_y_transition, cfg.nlo_y_transition_width) if cfg.nlo_dev_use_switch else 1.0
        y_unclipped = sw * component_y
        clip_mult = float(getattr(cfg, "nlo_y_clip_multiple", 5.0))
        if clip_mult > 0.0:
            max_abs = clip_mult * max(abs(float(w_baseline[i])), abs(float(row.get("CS", 0.0))), 1.0e-30)
            y = float(np.clip(y_unclipped, -max_abs, max_abs))
        else:
            y = float(y_unclipped)
        yvals[i] = y if np.isfinite(y) else 0.0
        rows.append({
            "row_id": str(row.get("row_id", i)),
            "dataset": str(row.get("dataset", "")),
            "qT": qt,
            "QM": q,
            "qT_over_Q": r,
            "W_CS_baseline": float(w_baseline[i]) if np.isfinite(w_baseline[i]) else np.nan,
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
            "Y_CS": float(yvals[i]),
            "Y_over_W": float(yvals[i] / w_baseline[i]) if np.isfinite(w_baseline[i]) and abs(w_baseline[i]) > 0 else np.nan,
            "FO_over_singular": float(fo / sing) if abs(float(sing)) > 1.0e-30 else np.nan,
            "nlo_singular_mode": str(getattr(cfg, "nlo_singular_mode", "analytic")),
            "nlo_singular_rsub": float(getattr(cfg, "nlo_singular_rsub", np.nan)),
            "nlo_singular_power": float(getattr(cfg, "nlo_singular_power", np.nan)),
            "nlo_singular_damp_kind": str(getattr(cfg, "nlo_singular_damp_kind", "")),
            "nlo_singular_damping": float(nlo_singular_damping_factor(row, cfg)) if str(getattr(cfg, "nlo_singular_mode", "")).lower().replace("-", "_") in {"asymptotic_damped", "analytic_damped", "stable", "stable_asymptotic", "damped"} else np.nan,
            "nlo_real_convention": str(getattr(cfg, "nlo_real_convention", "base")),
            "nlo_singular_convention": str(getattr(cfg, "nlo_singular_convention", "base")),
            "nlo_alpha_convention": str(getattr(cfg, "nlo_alpha_convention", "alpha_over_pi")),
        })
    LAST_BACKEND_ROW_DIAGNOSTICS = pd.DataFrame(rows)
    return yvals


def y_nlo_pilot_for_rows(df: pd.DataFrame, w_baseline: np.ndarray, pdf: LHAPDFProvider, cfg: CSSConfig) -> np.ndarray:
    """Data-independent NLO-like finite-tail scaffold.

    This function is intentionally NOT named a production Y_NLO implementation.
    It gives the training code a nonzero, same-convention finite term for matched
    plumbing while the analytic q qbar -> gamma* g and qg -> gamma* q matrix
    elements and same-scheme singular subtraction are being audited.

    Properties by construction:
      * Y is zero in the strict TMD window qT/Q <= cfg.nlo_y_transition.
      * Y is finite and smooth as qT -> 0.
      * It uses only perturbative ingredients/PDF luminosities, never data values.
      * It is expressed in the same CS convention as the W baseline.
    """
    y = np.zeros(len(df), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        q = max(float(row["QM"]), 1e-12)
        qt = max(float(row["qT"]), 0.0)
        r = qt / q
        sw = smooth_tail_switch(r, cfg.nlo_y_transition, cfg.nlo_y_transition_width)
        if sw <= 0.0:
            continue
        mu = max(q, cfg.mu_min)
        alpha = _nlo_alpha_factor(pdf, mu, cfg)
        qq = charge_weighted_lumi(row, mu, pdf, cfg)
        qg = charge_weighted_qg_lumi(row, mu, pdf, cfg)
        lum_ratio = (qg + 0.25 * qq) / max(qq, 1e-300)
        # Dimensionless finite-tail shape: suppressed at small r, bounded at r~1.
        finite_shape = (r * r) / ((1.0 + r * r) ** 1.5)
        y[i] = abs(float(w_baseline[i])) * cfg.nlo_y_pilot_strength * alpha * lum_ratio * sw * finite_shape
    y[~np.isfinite(y)] = 0.0
    return y


def make_b_grid(cfg: CSSConfig) -> np.ndarray:
    return np.linspace(float(cfg.b_min), float(cfg.b_max), int(cfg.n_b), dtype=float)


def wpert_cs_for_row(row: pd.Series, b_grid: np.ndarray, pdf: LHAPDFProvider, cfg: CSSConfig) -> np.ndarray:
    pref = fixed_target_prefactor_cs(row, cfg)
    q = float(row["QM"])
    out = np.empty_like(b_grid, dtype=float)
    fourier_norm = 1.0 / (2.0 * math.pi)
    for j, b in enumerate(b_grid):
        mu = mu_b_of_b(float(b), q, cfg)
        lum = charge_weighted_lumi(row, mu, pdf, cfg)
        s = sudakov_s(float(b), q, pdf, cfg)
        out[j] = pref * fourier_norm * lum * math.exp(-s)
    # Keep tiny negative roundoff out; this W pilot is positive by construction.
    out[~np.isfinite(out)] = 0.0
    return out


# -----------------------------------------------------------------------------
# Grid writing and validation
# -----------------------------------------------------------------------------

BASE_COLS = [
    "row_id", "dataset", "local_index",
    "qT", "QM", "QM_Low", "QM_High",
    "y", "y_Low", "y_High",
    "xF", "xF_Low", "xF_High",
    "x1", "x2", "SqrtS", "BeamE",
    "A", "PreFactor", "CS", "dA", "error", "frac_error",
    "sysNorm", "sysP2P",
]


def write_y_grid(df: pd.DataFrame, path: Path, y_values: np.ndarray, *, mode: str) -> None:
    cols = [c for c in BASE_COLS if c in df.columns]
    out = df.loc[:, cols].copy()
    out["FO_CS"] = np.nan
    out["singular_CS"] = np.nan
    out["Y_CS"] = np.asarray(y_values, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def write_w_grid(df: pd.DataFrame, path: Path, b_grid: np.ndarray, w_matrix: np.ndarray) -> None:
    cols = [c for c in BASE_COLS if c in df.columns]
    rows = []
    for i, r in df.loc[:, cols].iterrows():
        rd = r.to_dict()
        for b, w in zip(b_grid, w_matrix[i]):
            rr = dict(rd)
            rr["bT"] = float(b)
            rr["Wpert_CS"] = float(w)
            rows.append(rr)
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def torch_bessel_integral(qt: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    if torch is None:
        # No robust J0 in numpy. Return NaNs rather than silently using wrong math.
        return np.full((len(qt),), np.nan)
    tb = torch.tensor(b, dtype=torch.float64)
    tq = torch.tensor(qt, dtype=torch.float64).unsqueeze(1)
    tw = torch.tensor(w, dtype=torch.float64)
    vals = tb.unsqueeze(0) * torch.special.bessel_j0(tq * tb.unsqueeze(0)) * tw
    return torch.trapezoid(vals, tb, dim=1).detach().cpu().numpy()


def compute_backend_grids(df: pd.DataFrame, pdf: LHAPDFProvider, cfg: CSSConfig, *, progress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_grid = make_b_grid(cfg)
    w_matrix = np.empty((len(df), len(b_grid)), dtype=float)
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        w_matrix[i, :] = wpert_cs_for_row(row, b_grid, pdf, cfg)
        if progress and (i + 1) % max(1, len(df) // 10) == 0:
            elapsed = time.time() - t0
            print(f"  W grid: {i+1}/{len(df)} rows ({elapsed:.1f}s)", flush=True)
    w_baseline = torch_bessel_integral(df["qT"].to_numpy(float), b_grid, w_matrix)
    y_mode = cfg.y_mode.lower().replace("-", "_")
    match_order = cfg.match_order.lower().replace("-", "_")
    if match_order == "nlo":
        # Development alias: use the first analytic NLO-dev path.  This must be
        # benchmarked before production claims.
        match_order = "nlo_dev"
    if y_mode == "data_minus_w_debug":
        # Debug only: forces baseline W+Y to pass through the data.  This is useful
        # for checking plumbing but must not be used as a physics Y term in a fit.
        y = df["CS"].to_numpy(float) - w_baseline
    elif y_mode == "nlo_pilot" or match_order == "nlo_pilot":
        y = y_nlo_pilot_for_rows(df, w_baseline, pdf, cfg)
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


def write_metadata(path: Path, df: pd.DataFrame, cfg: CSSConfig, args_dict: dict, w_baseline: Optional[np.ndarray], y: Optional[np.ndarray]) -> None:
    meta = {
        "created_unix": time.time(),
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": Path(__file__).name,
        "note": (
            "Internal CSS-style pilot backend.  Not a final N3LL' + fixed-order finite-tail calculation. "
            "Default match_order=none gives Y=0. match_order=nlo_pilot gives the old development finite-tail scaffold. match_order=nlo_dev/nlo gives a v15 real-emission minus localized singular-subtraction development path; it must be benchmarked before production Y_NLO claims. Wpert_CS convention: CS_W=int db b J0(qT*b) Wpert_CS."
        ),
        "config": asdict(cfg),
        "args": {k: str(v) if callable(v) else v for k, v in args_dict.items()},
        "n_rows": int(len(df)),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "summary": summarize(df).to_dict(orient="records"),
    }
    if w_baseline is not None:
        data = df["CS"].to_numpy(float)
        pred = w_baseline + (y if y is not None else 0.0)
        finite = np.isfinite(pred) & np.isfinite(data) & (data != 0)
        meta["baseline"] = {
            "W_CS_min": float(np.nanmin(w_baseline)),
            "W_CS_max": float(np.nanmax(w_baseline)),
            "W_CS_median": float(np.nanmedian(w_baseline)),
            "matched_CS_median": float(np.nanmedian(pred)),
            "data_CS_median": float(np.nanmedian(data)),
            "median_pred_over_data": float(np.nanmedian(pred[finite] / data[finite])) if finite.any() else None,
            "median_abs_rel_residual": float(np.nanmedian(np.abs((pred[finite] - data[finite]) / data[finite]))) if finite.any() else None,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_baseline(path: Path, df: pd.DataFrame, w_baseline: np.ndarray, y: np.ndarray) -> None:
    cols = [c for c in BASE_COLS if c in df.columns]
    out = df.loc[:, cols].copy()
    out["W_CS_baseline"] = w_baseline
    out["Y_CS"] = y
    out["matched_CS_baseline"] = w_baseline + y
    out["data_over_baseline"] = out["CS"] / out["matched_CS_baseline"].replace(0.0, np.nan)
    out["rel_residual"] = (out["matched_CS_baseline"] - out["CS"]) / out["error"].replace(0.0, np.nan)
    diag = globals().get("LAST_BACKEND_ROW_DIAGNOSTICS")
    if isinstance(diag, pd.DataFrame) and "row_id" in diag.columns:
        keep = [c for c in diag.columns if c not in out.columns or c == "row_id"]
        out = out.merge(diag.loc[:, keep], on="row_id", how="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Dataset stems, e.g. E288_200 E605")
    ap.add_argument("--mode", choices=["matched", "tmd_only", "none"], default="matched")
    ap.add_argument("--qT-max-over-Q", type=float, default=1.0)
    ap.add_argument("--tmd-qT-max-over-Q", type=float, default=0.2)
    ap.add_argument("--no-upsilon-veto", action="store_true")


def add_theory_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
    ap.add_argument("--pdf-member", type=int, default=0)
    ap.add_argument("--toy-pdf", action="store_true", help="Use nonphysical toy PDFs for smoke tests without LHAPDF")
    ap.add_argument("--target-mode", choices=["proton_approx", "isoscalar", "nuclear_isospin", "nuclear_pdf"], default="proton_approx")
    ap.add_argument("--flavors", nargs="+", type=int, default=list(DEFAULT_FLAVORS), help="LHAPDF flavor PIDs; default d u s = 1 2 3")
    ap.add_argument("--resum-order", choices=["ll", "nll", "nnll", "n3llp", "n3ll_pilot"], default="nnll")
    ap.add_argument("--nf", type=int, default=5)
    ap.add_argument("--b-star-max", type=float, default=1.5)
    ap.add_argument("--mu-min", type=float, default=1.3)
    ap.add_argument("--no-cap-mub-at-Q", action="store_true")
    ap.add_argument("--n-sudakov-quad", type=int, default=32)
    ap.add_argument("--prefactor-scheme", choices=["oldA_to_CS", "unit"], default="oldA_to_CS")
    ap.add_argument("--global-norm", type=float, default=1.0)
    ap.add_argument("--hc-factor", type=float, default=3.893793656e8)
    ap.add_argument("--alpha-em", type=float, default=1.0 / 137.035999084)
    ap.add_argument("--match-order", choices=["none", "nlo_pilot", "nlo_dev", "nlo"], default="none", help="Finite-tail matching. nlo currently aliases nlo_dev in this development backend.")
    ap.add_argument("--nlo-y-pilot-strength", type=float, default=1.0, help="Scale factor for the nlo_pilot finite-tail scaffold.")
    ap.add_argument("--nlo-y-transition", type=float, default=0.20, help="qT/Q value below which nlo_pilot Y is zero.")
    ap.add_argument("--nlo-y-transition-width", type=float, default=0.15, help="Smooth turn-on width in qT/Q for nlo_pilot/nlo_dev Y.")
    ap.add_argument("--nlo-real-quad", type=int, default=48, help="Gauss-Legendre nodes for the nlo_dev recoil-rapidity integral.")
    ap.add_argument("--nlo-real-norm", type=float, default=1.0, help="Development normalization factor for FO_NLO_real_dev.")
    ap.add_argument("--nlo-singular-norm", type=float, default=1.0, help="Development normalization factor for singular_NLO_dev.")
    ap.add_argument("--nlo-y-component", choices=["raw", "positive", "fo_only", "minus_sing", "singular_only", "zero"], default="raw", help="Audit knob for nlo_dev Y before switch/clip.")
    ap.add_argument("--nlo-y-clip-multiple", type=float, default=5.0, help="Clip |Y| to this multiple of max(|W|,|data|); use <=0 to disable. Development only.")
    ap.add_argument("--no-nlo-dev-switch", action="store_true", help="Disable the smooth qT/Q turn-on applied to nlo_dev Y.")
    ap.add_argument("--nlo-dev-min-qt-over-q", type=float, default=1.0e-4, help="qT/Q floor used in the singular_NLO_dev term.")
    ap.add_argument("--nlo-singular-mode", choices=["analytic", "asymptotic_damped", "wexp_numeric", "wexp_positive", "none"], default="asymptotic_damped", help="V15 singular subtraction mode. asymptotic_damped uses a localized qT-space asymptotic subtraction; wexp_numeric is retained for diagnostics.")
    ap.add_argument("--nlo-singular-rsub", type=float, default=0.20, help="r_sub in qT/Q for asymptotic_damped singular subtraction localization.")
    ap.add_argument("--nlo-singular-power", type=float, default=4.0, help="Power p in the asymptotic_damped singular subtraction profile.")
    ap.add_argument("--nlo-singular-damp-kind", choices=["exp", "rational"], default="exp", help="Damping profile for asymptotic_damped singular subtraction.")
    ap.add_argument("--nlo-real-convention", default="base", help="Audit multiplier for real term: base, times_qt, times_2qt, div_qt, times_Q, times_2pi, etc.; combine with commas or *.")
    ap.add_argument("--nlo-singular-convention", default="base", help="Audit multiplier for singular term; same tokens as --nlo-real-convention.")
    ap.add_argument("--nlo-alpha-convention", choices=["alpha_over_pi", "alpha_over_2pi"], default="alpha_over_pi", help="Alpha_s normalization used in NLO real and singular terms.")
    ap.add_argument("--nlo-real-tail-repair", choices=["none", "mcfm_logistic", "external_logistic", "logistic"], default="none", help="V16 external-code calibrated high-qT/Q multiplier for FO_NLO_real_dev; none reproduces v15.")
    ap.add_argument("--nlo-real-tail-r0", type=float, default=0.520, help="Center r=qT/Q of the MCFM/DYTurbo logistic real-tail repair.")
    ap.add_argument("--nlo-real-tail-width", type=float, default=0.010, help="Width in r=qT/Q of the MCFM/DYTurbo logistic real-tail repair.")
    ap.add_argument("--nlo-real-tail-rinf", type=float, default=0.180, help="Large-r limiting multiplier for the MCFM/DYTurbo real-tail repair.")
    ap.add_argument("--y-mode", choices=["zero", "nlo_pilot", "data_minus_w_debug"], default="zero")


def cfg_from_args(args) -> CSSConfig:
    return CSSConfig(
        b_min=float(getattr(args, "b_min", 1.0e-4)),
        b_max=float(getattr(args, "b_max", 8.0)),
        n_b=int(getattr(args, "n_b", 160)),
        bstar_bmax=float(args.b_star_max),
        mu_min=float(args.mu_min),
        cap_mub_at_Q=not bool(args.no_cap_mub_at_Q),
        resum_order=str(args.resum_order),
        match_order=str(getattr(args, "match_order", "none")),
        nlo_y_pilot_strength=float(getattr(args, "nlo_y_pilot_strength", 1.0)),
        nlo_y_transition=float(getattr(args, "nlo_y_transition", 0.20)),
        nlo_y_transition_width=float(getattr(args, "nlo_y_transition_width", 0.15)),
        nlo_real_quad=int(getattr(args, "nlo_real_quad", 48)),
        nlo_real_norm=float(getattr(args, "nlo_real_norm", 1.0)),
        nlo_singular_norm=float(getattr(args, "nlo_singular_norm", 1.0)),
        nlo_y_component=str(getattr(args, "nlo_y_component", "raw")),
        nlo_y_clip_multiple=float(getattr(args, "nlo_y_clip_multiple", 5.0)),
        nlo_dev_use_switch=not bool(getattr(args, "no_nlo_dev_switch", False)),
        nlo_dev_min_qt_over_q=float(getattr(args, "nlo_dev_min_qt_over_q", 1.0e-4)),
        nlo_singular_mode=str(getattr(args, "nlo_singular_mode", "asymptotic_damped")),
        nlo_singular_rsub=float(getattr(args, "nlo_singular_rsub", 0.20)),
        nlo_singular_power=float(getattr(args, "nlo_singular_power", 4.0)),
        nlo_singular_damp_kind=str(getattr(args, "nlo_singular_damp_kind", "exp")),
        nlo_real_convention=str(getattr(args, "nlo_real_convention", "base")),
        nlo_singular_convention=str(getattr(args, "nlo_singular_convention", "base")),
        nlo_alpha_convention=str(getattr(args, "nlo_alpha_convention", "alpha_over_pi")),
        nlo_real_tail_repair=str(getattr(args, "nlo_real_tail_repair", "none")),
        nlo_real_tail_r0=float(getattr(args, "nlo_real_tail_r0", 0.520)),
        nlo_real_tail_width=float(getattr(args, "nlo_real_tail_width", 0.010)),
        nlo_real_tail_rinf=float(getattr(args, "nlo_real_tail_rinf", 0.180)),
        nf=int(args.nf),
        n_sudakov_quad=int(args.n_sudakov_quad),
        alpha_em=float(args.alpha_em),
        hc_factor=float(args.hc_factor),
        prefactor_scheme=str(args.prefactor_scheme),
        global_norm=float(args.global_norm),
        flavors=tuple(int(f) for f in args.flavors),
        target_mode=str(args.target_mode),
        y_mode=str(args.y_mode),
    )


def cuts_from_args(args) -> CutConfig:
    return CutConfig(
        mode=str(args.mode),
        qT_max_over_Q=float(args.qT_max_over_Q),
        tmd_qT_max_over_Q=float(args.tmd_qT_max_over_Q),
        apply_upsilon_veto=not bool(args.no_upsilon_veto),
    )


def cmd_check(args) -> None:
    df = load_fixed_target_data(args.data_dir, args.datasets, cuts_from_args(args))
    print(f"loaded rows after cuts: {len(df)}")
    print(summarize(df).to_string(index=False))
    pdf = LHAPDFProvider(args.pdf_set, args.pdf_member, use_toy_pdf=args.toy_pdf)
    print("\nPDF provider:", "toy" if args.toy_pdf else args.pdf_set, "member", args.pdf_member)
    cfg = cfg_from_args(args)
    print("CSS config:")
    print(json.dumps(asdict(cfg), indent=2))
    if len(df):
        row = df.iloc[0]
        btest = np.array([0.1, 0.5, 1.0, 2.0])
        vals = wpert_cs_for_row(row, btest, pdf, cfg)
        print("\nfirst row_id:", row["row_id"])
        print("bT test:", btest)
        print("Wpert_CS test:", vals)
        print("finite:", np.all(np.isfinite(vals)))


def cmd_grids(args) -> None:
    df = load_fixed_target_data(args.data_dir, args.datasets, cuts_from_args(args))
    print(f"loaded rows after cuts: {len(df)}")
    print(summarize(df).to_string(index=False))
    cfg = cfg_from_args(args)
    pdf = LHAPDFProvider(args.pdf_set, args.pdf_member, use_toy_pdf=args.toy_pdf)

    b_grid, w_matrix, y = compute_backend_grids(df, pdf, cfg, progress=not args.quiet)
    w_baseline = torch_bessel_integral(df["qT"].to_numpy(float), b_grid, w_matrix)

    out_dir = Path(args.out_dir)
    tag = args.tag or f"internal_css_{args.mode}"
    w_path = out_dir / f"wpert_{tag}.csv"
    y_path = out_dir / f"y_{tag}.csv"
    baseline_path = out_dir / f"baseline_{tag}.csv"
    meta_path = out_dir / f"metadata_{tag}.json"

    write_w_grid(df, w_path, b_grid, w_matrix)
    write_y_grid(df, y_path, y, mode=args.mode)
    write_baseline(baseline_path, df, w_baseline, y)
    write_metadata(meta_path, df, cfg, vars(args), w_baseline, y)

    print("\nwrote", w_path)
    print("wrote", y_path)
    print("wrote", baseline_path)
    print("wrote", meta_path)
    finite = np.isfinite(w_baseline)
    if finite.any():
        pred = w_baseline + y
        data = df["CS"].to_numpy(float)
        good = np.isfinite(pred) & np.isfinite(data) & (data != 0)
        print("\nbaseline diagnostics:")
        print("  W_CS median:", float(np.nanmedian(w_baseline)))
        print("  matched/data median:", float(np.nanmedian(pred[good] / data[good])) if good.any() else np.nan)
        print("  median |rel residual|:", float(np.nanmedian(np.abs((pred[good] - data[good]) / data[good]))) if good.any() else np.nan)
    if cfg.y_mode == "data_minus_w_debug":
        print("\nWARNING: y_mode=data_minus_w_debug forces W+Y to the data. Do not use this as a physics Y term.")
    if cfg.y_mode == "zero" and args.mode == "matched":
        print("\nNOTE: matched mode with y_mode=zero is W-only plus zero finite tail. It is a backend/plumbing step, not final W+Y matching.")


def cmd_predict(args) -> None:
    df = load_fixed_target_data(args.data_dir, args.datasets, cuts_from_args(args))
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    cfg = cfg_from_args(args)
    pdf = LHAPDFProvider(args.pdf_set, args.pdf_member, use_toy_pdf=args.toy_pdf)
    b_grid, w_matrix, y = compute_backend_grids(df, pdf, cfg, progress=not args.quiet)
    w_baseline = torch_bessel_integral(df["qT"].to_numpy(float), b_grid, w_matrix)
    out = df[[c for c in BASE_COLS if c in df.columns]].copy()
    out["W_CS_baseline"] = w_baseline
    out["Y_CS"] = y
    out["matched_CS_baseline"] = w_baseline + y
    out["data_CS"] = out["CS"]
    out["ratio_baseline_to_data"] = out["matched_CS_baseline"] / out["data_CS"].replace(0.0, np.nan)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print("wrote", args.out)
    else:
        print(out.head(20).to_string(index=False))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Single-file internal CSS-style bT backend: check, predict, or write W/Y grids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Load data/PDFs and evaluate Wpert for one row")
    add_common_args(p_check)
    add_theory_args(p_check)
    p_check.set_defaults(func=cmd_check)

    p_grids = sub.add_parser("grids", help="Write wpert_*.csv and y_*.csv for train_realdata.py")
    add_common_args(p_grids)
    add_theory_args(p_grids)
    p_grids.add_argument("--out-dir", default="internal_css_grids")
    p_grids.add_argument("--tag", default=None)
    p_grids.add_argument("--b-min", type=float, default=1.0e-4)
    p_grids.add_argument("--b-max", type=float, default=8.0)
    p_grids.add_argument("--n-b", type=int, default=160)
    p_grids.add_argument("--quiet", action="store_true")
    p_grids.set_defaults(func=cmd_grids)

    p_pred = sub.add_parser("predict", help="Compute baseline W+Y predictions without writing long W grid")
    add_common_args(p_pred)
    add_theory_args(p_pred)
    p_pred.add_argument("--b-min", type=float, default=1.0e-4)
    p_pred.add_argument("--b-max", type=float, default=8.0)
    p_pred.add_argument("--n-b", type=int, default=160)
    p_pred.add_argument("--max-rows", type=int, default=None)
    p_pred.add_argument("--out", default=None)
    p_pred.add_argument("--quiet", action="store_true")
    p_pred.set_defaults(func=cmd_predict)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
