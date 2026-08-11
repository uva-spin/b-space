#!/usr/bin/env python3
"""Isolated local-start control for the accepted production-only FiLM objective."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "systematics/high_qt_direct_production_benchmark/experimental_unitary_transition"
PRODUCTION = ROOT / "systematics/collins_factorization_validity/outputs/rowidfix_stageFT_all_qmax0p20_lam0p50_central_s303"
TRAINER_PATH = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
W_GRID = ROOT / "outputs/v23a_tevatron_plus_lhcb7_fidacc_merged_cache/backend_cache/wpert_v23a_tevatron_plus_lhcb7_fidacc_b160.csv"
FNP_GRID_X = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7]


class C1MatchedLogTailNP(torch.nn.Module):
    """FiLM core below b_match, with a positive-curvature C1 log-F tail.

    The tail value and slope are inherited from the core at the match point,
    so there is no independently floating tail amplitude.  Only the positive
    quadratic damping curvature is learned on a sparse log-x knot grid.
    """

    def __init__(self, base, b_match: float, x_knots: torch.Tensor):
        super().__init__()
        if b_match <= 0.0:
            raise ValueError("C1 tail matching requires b_match > 0")
        self.base, self.b_match = base, float(b_match)
        x_knots = torch.unique(x_knots.sort().values)
        self.register_buffer("tail_log_x_knots", torch.log(x_knots))
        with torch.no_grad():
            b_fit = torch.linspace(0.0001, 8.0, 641, dtype=x_knots.dtype, device=x_knots.device)
            desired = torch.log(base(x_knots, b_fit).clamp_min(1.0e-30))
            match_index = int(torch.argmin(torch.abs(b_fit - self.b_match)))
            b0 = b_fit[match_index]
            logf0 = desired[:, match_index]
            slope0 = -2.0 * b0 * base.A(x_knots, b0.reshape(1)).squeeze(1)
            delta = b_fit[match_index:] - b0
            residual = desired[:, match_index:] - logf0.reshape(-1, 1) - slope0.reshape(-1, 1) * delta
            design = delta.square()
            curvature = -(residual * design.reshape(1, -1)).sum(dim=1) / design.pow(2).sum().clamp_min(1.0e-12)
            curvature = curvature.clamp(1.0e-5, 10.0)
        self.tail_log_curvature = torch.nn.Parameter(torch.log(curvature))

    def curvature(self, x):
        log_x = torch.log(x.clamp(torch.exp(self.tail_log_x_knots[0]), torch.exp(self.tail_log_x_knots[-1])))
        upper = torch.bucketize(log_x, self.tail_log_x_knots).clamp(1, len(self.tail_log_x_knots) - 1)
        lower = upper - 1
        x0, x1 = self.tail_log_x_knots[lower], self.tail_log_x_knots[upper]
        weight = (log_x - x0) / (x1 - x0).clamp_min(1.0e-12)
        log_c = (1.0 - weight) * self.tail_log_curvature[lower] + weight * self.tail_log_curvature[upper]
        return torch.exp(torch.clamp(log_c, -12.0, 3.0))

    def A(self, x, b):
        # Effective A representation, used only by optional diagnostics.
        base_a = self.base.A(x, b)
        bm = torch.tensor([self.b_match], dtype=b.dtype, device=b.device)
        slope0 = -2.0 * self.b_match * self.base.A(x, bm).squeeze(1)
        delta = (b - self.b_match).clamp_min(0.0)
        tail_rate = -slope0.reshape(-1, 1) + 2.0 * self.curvature(x).reshape(-1, 1) * delta.reshape(1, -1)
        tail_a = tail_rate / (2.0 * b.clamp_min(1.0e-4)).reshape(1, -1)
        return torch.where((b <= self.b_match).reshape(1, -1), base_a, tail_a)

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        base_f = self.base(x, b)
        below = torch.nonzero(b <= self.b_match).flatten()
        above = torch.nonzero(b > self.b_match).flatten()
        if len(above) == 0:
            return base_f
        if len(below) == 0:
            raise ValueError("b grid must include points below the C1 tail match")
        lo = int(below[-1])
        hi = int(above[0])
        weight = (self.b_match - b[lo]) / (b[hi] - b[lo]).clamp_min(1.0e-12)
        log_base = torch.log(base_f.clamp_min(1.0e-30))
        logf0 = (1.0 - weight) * log_base[:, lo] + weight * log_base[:, hi]
        bm = torch.tensor([self.b_match], dtype=b.dtype, device=b.device)
        slope0 = -2.0 * self.b_match * self.base.A(x, bm).squeeze(1)
        delta = (b - self.b_match).clamp_min(0.0)
        log_tail = (logf0.reshape(-1, 1) + slope0.reshape(-1, 1) * delta.reshape(1, -1)
                    - self.curvature(x).reshape(-1, 1) * delta.square().reshape(1, -1))
        tail_f = torch.exp(torch.clamp(log_tail, -self.base.exponent_clip, self.base.exponent_clip))
        return torch.where((b <= self.b_match).reshape(1, -1), base_f, tail_f)


class LocalizedLogFShift(torch.nn.Module):
    """Add an explicit nonsaturated local coordinate in log FNP."""

    def __init__(self, base, x0: float, b0: float, sigma_logx: float,
                 sigma_b: float, initial_shift: float):
        super().__init__()
        self.base = base
        self.x0 = float(x0)
        self.b0 = float(b0)
        self.sigma_logx = float(sigma_logx)
        self.sigma_b = float(sigma_b)
        self.profile_shift = torch.nn.Parameter(torch.tensor(
            float(initial_shift), dtype=next(base.parameters()).dtype,
            device=next(base.parameters()).device))

    def _window(self, x, b):
        wx = torch.exp(-0.5 * (
            (torch.log(x.clamp_min(1.0e-12)) - np.log(self.x0))
            / self.sigma_logx).square())
        wb = torch.exp(-0.5 * ((b - self.b0) / self.sigma_b).square())
        return wx.reshape(-1, 1) * wb.reshape(1, -1)

    def A(self, x, b):
        return self.base.A(x, b)

    def forward(self, x, b):
        return self.base(x, b) * torch.exp(
            torch.clamp(self.profile_shift * self._window(x, b), -20.0, 20.0))


class ClosureTailCoordinateNP(torch.nn.Module):
    """Add a positive C2 attenuation coordinate outside the fitted core.

    The coordinate is identically zero through ``b_start`` and saturates at
    one after ``b_end``.  It changes only the normalization of the already
    damped remote tail and leaves the base model's asymptotic rate intact.
    """

    def __init__(self, base, b_start: float, b_end: float,
                 x_knots: torch.Tensor, initial_max_fnp: float):
        super().__init__()
        if not 0.0 < b_start < b_end:
            raise ValueError("closure coordinate requires 0 < b_start < b_end")
        self.base = base
        self.b_start, self.b_end = float(b_start), float(b_end)
        self.exponent_clip = float(base.exponent_clip)
        knots = torch.unique(x_knots.sort().values)
        self.register_buffer("closure_log_x_knots", torch.log(knots))
        if not 0.0 < initial_max_fnp < 1.0:
            raise ValueError("closure-coordinate initial maximum must be in (0,1)")
        # Initialize directly on the numerical-closure boundary.  This avoids
        # asking a high-dimensional optimizer to discover a remote-tail-only
        # direction while leaving all amplitudes free in the subsequent fit.
        self.initial_max_fnp = float(initial_max_fnp)
        self.closure_raw_amplitudes = torch.nn.Parameter(torch.empty(
            len(knots), dtype=knots.dtype, device=knots.device))
        self.initialize_closure_from_base()

    def initialize_closure_from_base(self):
        """Reset amplitudes to the closure boundary for the loaded base."""
        with torch.no_grad():
            knots = torch.exp(self.closure_log_x_knots)
            b_fit = torch.linspace(
                0.0001, self.b_end, 321,
                dtype=knots.dtype, device=knots.device)
            base_endpoint = self.base(
                knots, b_fit)[:, -1].clamp_min(1.0e-30)
            amplitudes = torch.relu(
                torch.log(base_endpoint / self.initial_max_fnp)
            ).clamp_min(1.0e-4)
            self.closure_raw_amplitudes.copy_(
                torch.log(torch.expm1(amplitudes)))

    def amplitudes(self, x):
        log_x = torch.log(x.clamp(
            torch.exp(self.closure_log_x_knots[0]),
            torch.exp(self.closure_log_x_knots[-1])))
        upper = torch.bucketize(
            log_x, self.closure_log_x_knots).clamp(
                1, len(self.closure_log_x_knots) - 1)
        lower = upper - 1
        x0 = self.closure_log_x_knots[lower]
        x1 = self.closure_log_x_knots[upper]
        weight = (log_x - x0) / (x1 - x0).clamp_min(1.0e-12)
        raw = (
            (1.0 - weight) * self.closure_raw_amplitudes[lower]
            + weight * self.closure_raw_amplitudes[upper])
        return torch.nn.functional.softplus(raw)

    def window(self, b):
        t = ((b - self.b_start) / (
            self.b_end - self.b_start)).clamp(0.0, 1.0)
        return t.pow(3) * (10.0 - 15.0 * t + 6.0 * t.square())

    def window_derivative(self, b):
        t = ((b - self.b_start) / (
            self.b_end - self.b_start)).clamp(0.0, 1.0)
        inside = (b > self.b_start) & (b < self.b_end)
        derivative = 30.0 * t.square() * (1.0 - t).square() / (
            self.b_end - self.b_start)
        return torch.where(inside, derivative, torch.zeros_like(derivative))

    def A(self, x, b):
        base_a = self.base.A(x, b)
        extra_rate = (
            self.amplitudes(x).reshape(-1, 1)
            * self.window_derivative(b).reshape(1, -1))
        return base_a + extra_rate / (
            2.0 * b.clamp_min(1.0e-4)).reshape(1, -1)

    def forward(self, x, b):
        log_attenuation = (
            self.amplitudes(x).reshape(-1, 1)
            * self.window(b).reshape(1, -1))
        return self.base(x, b) * torch.exp(-log_attenuation)


class EndpointConstrainedFNP(torch.nn.Module):
    """Impose endpoint values without replacing the fitted interior.

    The constrained path is the analytic shortest path between the declared
    endpoints (linear in F or in log-F).  Inside the interval we retain the
    base model's residual around that path, multiplied by a C1 bump that is
    exactly zero, with zero derivative, at both endpoints.  Outside the
    interval the original base model is retained.  Thus the endpoint values
    are hard constraints, while the fit is not silently replaced by an
    arbitrary extrapolation of the endpoint correction through the remote
    transform tail.

    This is opt-in diagnostic machinery.  The historical objective is
    unchanged unless an endpoint reference CSV is supplied.
    """

    def __init__(self, base, b_min: float, b_max: float,
                 x_knots: torch.Tensor, endpoint_values: torch.Tensor,
                 metric: str = "logF"):
        super().__init__()
        if not 0.0 <= b_min < b_max:
            raise ValueError("endpoint constraint requires 0 <= b_min < b_max")
        self.base = base
        self.b_min, self.b_max = float(b_min), float(b_max)
        if metric not in ("F", "logF"):
            raise ValueError("endpoint metric must be F or logF")
        self.metric = metric
        knots = torch.unique(x_knots.sort().values)
        self.register_buffer("endpoint_log_x_knots", torch.log(knots))
        if endpoint_values.shape != (len(knots), 2):
            raise ValueError("endpoint values must have shape [n_x, 2]")
        self.register_buffer(
            "endpoint_log_values", torch.log(endpoint_values.clamp_min(1.0e-30)))

    def _endpoints_at_x(self, x):
        log_x = torch.log(x.clamp(
            torch.exp(self.endpoint_log_x_knots[0]),
            torch.exp(self.endpoint_log_x_knots[-1])))
        upper = torch.bucketize(log_x, self.endpoint_log_x_knots).clamp(
            1, len(self.endpoint_log_x_knots) - 1)
        lower = upper - 1
        x0 = self.endpoint_log_x_knots[lower]
        x1 = self.endpoint_log_x_knots[upper]
        weight = (log_x - x0) / (x1 - x0).clamp_min(1.0e-12)
        return ((1.0 - weight).reshape(-1, 1)
                * self.endpoint_log_values[lower]
                + weight.reshape(-1, 1)
                * self.endpoint_log_values[upper])

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        base_f = self.base(x, b).clamp_min(1.0e-30)
        log_base = torch.log(base_f)
        target_log = self._endpoints_at_x(x)
        t = ((b - self.b_min) / (self.b_max - self.b_min)).clamp(0.0, 1.0)
        if self.metric == "F":
            f0, f1 = target_log.exp()[:, 0], target_log.exp()[:, 1]
            shortest = f0.reshape(-1, 1) + (f1 - f0).reshape(-1, 1) * t.reshape(1, -1)
            log_shortest = torch.log(shortest.clamp_min(1.0e-30))
        else:
            log_shortest = (
                target_log[:, 0].reshape(-1, 1)
                + (target_log[:, 1] - target_log[:, 0]).reshape(-1, 1)
                * t.reshape(1, -1))
        # A quartic bump is one in the unconstrained center and has value and
        # first derivative zero at both endpoints.  This is the key correction
        # to the earlier pilot, which propagated an endpoint offset unchanged
        # through every b node beyond b_max and badly altered the transform.
        bump = (16.0 * t.square() * (1.0 - t).square()).reshape(1, -1)
        inside = ((b >= self.b_min) & (b <= self.b_max)).reshape(1, -1)
        constrained = log_shortest + bump * (log_base - log_shortest)
        result = torch.where(inside, constrained, log_base)
        return torch.exp(torch.clamp(
            result, -self.base.exponent_clip, self.base.exponent_clip))

    def A(self, x, b):
        # Endpoint correction is used only with forward-based constraints in
        # the pilot.  Delegate diagnostics to the positive base representation.
        return self.base.A(x, b)

class ReducedTailNP(torch.nn.Module):
    """FiLM core smoothly matched to a positive knot-interpolated tail A(x)."""

    def __init__(self, base, b_start: float, b_end: float, x_knots: torch.Tensor):
        super().__init__()
        if not 0.0 <= b_start < b_end:
            raise ValueError("Reduced-tail matching requires 0 <= b_start < b_end")
        self.base = base
        self.b_start, self.b_end = float(b_start), float(b_end)
        x_knots = torch.unique(x_knots.sort().values)
        self.register_buffer("tail_log_x_knots", torch.log(x_knots))
        with torch.no_grad():
            b_fit = torch.linspace(0.0001, 8.0, 321, dtype=x_knots.dtype, device=x_knots.device)
            t = ((b_fit - self.b_start) / (self.b_end - self.b_start)).clamp(0.0, 1.0)
            weight = t.pow(3) * (10.0 - 15.0 * t + 6.0 * t.square())
            base_a = base.A(x_knots, b_fit)
            floor = torch.zeros_like(b_fit)
            if base.a_tail_amp > 0.0:
                floor = base.a_tail_amp * torch.sigmoid((b_fit - base.a_tail_b0) / base.a_tail_width)
            fixed_a = (1.0 - weight.reshape(1, -1)) * base_a + weight.reshape(1, -1) * floor.reshape(1, -1)
            fixed_logf = -base._cumulative_trapezoid(2.0 * b_fit.reshape(1, -1) * fixed_a, b_fit)
            q = base._cumulative_trapezoid(2.0 * b_fit.reshape(1, -1) * weight.reshape(1, -1), b_fit).squeeze(0)
            desired_logf = torch.log(base(x_knots, b_fit).clamp_min(1.0e-12))
            fit_mask = (b_fit >= self.b_start) & (desired_logf.mean(dim=0) > -20.0)
            q_fit = q[fit_mask]
            amplitudes = torch.sum(
                q_fit.reshape(1, -1) * (fixed_logf[:, fit_mask] - desired_logf[:, fit_mask]), dim=1
            ) / torch.sum(q_fit.square()).clamp_min(1.0e-12)
            target = torch.log(amplitudes.clamp_min(1.0e-8))
        self.tail_log_a_values = torch.nn.Parameter(target)

    def tail_A(self, x):
        log_x = torch.log(x.clamp(torch.exp(self.tail_log_x_knots[0]), torch.exp(self.tail_log_x_knots[-1])))
        upper = torch.bucketize(log_x, self.tail_log_x_knots).clamp(1, len(self.tail_log_x_knots) - 1)
        lower = upper - 1
        x0, x1 = self.tail_log_x_knots[lower], self.tail_log_x_knots[upper]
        weight = (log_x - x0) / (x1 - x0).clamp_min(1.0e-12)
        log_a = (1.0 - weight) * self.tail_log_a_values[lower] + weight * self.tail_log_a_values[upper]
        return torch.exp(torch.clamp(log_a, -8.0, 2.0))

    def A(self, x, b):
        base_a = self.base.A(x, b)
        t = ((b - self.b_start) / (self.b_end - self.b_start)).clamp(0.0, 1.0)
        weight = t.pow(3) * (10.0 - 15.0 * t + 6.0 * t.square())
        tail = self.tail_A(x).reshape(-1, 1)
        if self.base.a_tail_amp > 0.0:
            fixed_floor = self.base.a_tail_amp * torch.sigmoid(
                (b - self.base.a_tail_b0) / self.base.a_tail_width)
            tail = tail + fixed_floor.reshape(1, -1)
        return (1.0 - weight.reshape(1, -1)) * base_a + weight.reshape(1, -1) * tail

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        integrand = 2.0 * b.reshape(1, -1) * self.A(x, b)
        exponent = -self.base._cumulative_trapezoid(integrand, b)
        if self.base.exponent_clip > 0:
            exponent = torch.clamp(exponent, -self.base.exponent_clip, self.base.exponent_clip)
        return torch.exp(exponent)


class GlobalReducedNP(torch.nn.Module):
    """Positive low-rank A(x,b) with log-x knot amplitudes and fixed tail floor."""

    def __init__(self, accepted_model, x_knots: torch.Tensor, components: int, b_scale: float = 1.0):
        super().__init__()
        if components not in (1, 2):
            raise ValueError("GlobalReducedNP supports one or two A components")
        self.components, self.b_scale = int(components), float(b_scale)
        self.exponent_clip = float(accepted_model.exponent_clip)
        self.a_tail_amp = float(accepted_model.a_tail_amp)
        self.a_tail_b0 = float(accepted_model.a_tail_b0)
        self.a_tail_width = float(accepted_model.a_tail_width)
        x_knots = torch.unique(x_knots.sort().values)
        self.register_buffer("log_x_knots", torch.log(x_knots))
        with torch.no_grad():
            b_fit = torch.linspace(0.0001, 8.0, 321, dtype=x_knots.dtype, device=x_knots.device)
            bases = self.bases(b_fit)
            q = torch.stack([
                self._cumulative_trapezoid(2.0 * b_fit.reshape(1, -1) * basis.reshape(1, -1), b_fit).squeeze(0)
                for basis in bases
            ], dim=1)
            floor = self.fixed_floor(b_fit)
            floor_log = -self._cumulative_trapezoid(2.0 * b_fit.reshape(1, -1) * floor.reshape(1, -1), b_fit).squeeze(0)
            desired = torch.log(accepted_model(x_knots, b_fit).clamp_min(1.0e-12))
            mask = desired.mean(dim=0) > -20.0
            design = q[mask]
            targets = -desired[:, mask] + floor_log[mask].reshape(1, -1)
            coefficients = []
            for target in targets:
                solution = torch.linalg.lstsq(design, target).solution.clamp_min(1.0e-5)
                coefficients.append(solution)
            initial = torch.stack(coefficients)
        self.log_amplitudes = torch.nn.Parameter(torch.log(initial))

    @staticmethod
    def _cumulative_trapezoid(y, x):
        dx = x[1:] - x[:-1]
        area = 0.5 * (y[:, 1:] + y[:, :-1]) * dx.reshape(1, -1)
        return torch.cat((torch.zeros((y.shape[0], 1), dtype=y.dtype, device=y.device),
                          torch.cumsum(area, dim=1)), dim=1)

    def fixed_floor(self, b):
        if self.a_tail_amp <= 0.0:
            return torch.zeros_like(b)
        return self.a_tail_amp * torch.sigmoid((b - self.a_tail_b0) / self.a_tail_width)

    def bases(self, b):
        result = [torch.ones_like(b)]
        if self.components == 2:
            result.append(b.square() / (b.square() + self.b_scale**2))
        return result

    def amplitudes(self, x):
        log_x = torch.log(x.clamp(torch.exp(self.log_x_knots[0]), torch.exp(self.log_x_knots[-1])))
        upper = torch.bucketize(log_x, self.log_x_knots).clamp(1, len(self.log_x_knots) - 1)
        lower = upper - 1
        x0, x1 = self.log_x_knots[lower], self.log_x_knots[upper]
        weight = ((log_x - x0) / (x1 - x0).clamp_min(1.0e-12)).reshape(-1, 1)
        logs = (1.0 - weight) * self.log_amplitudes[lower] + weight * self.log_amplitudes[upper]
        return torch.exp(torch.clamp(logs, -12.0, 3.0))

    def A(self, x, b):
        amplitudes = self.amplitudes(x)
        basis_matrix = torch.stack(self.bases(b), dim=0)
        return amplitudes @ basis_matrix + self.fixed_floor(b).reshape(1, -1)

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        exponent = -self._cumulative_trapezoid(2.0 * b.reshape(1, -1) * self.A(x, b), b)
        if self.exponent_clip > 0:
            exponent = torch.clamp(exponent, -self.exponent_clip, self.exponent_clip)
        return torch.exp(exponent)


class GlobalSplineNP(torch.nn.Module):
    """Positive bilinear spline for A(log x,b), enforcing monotone FNP."""

    def __init__(self, accepted_model, x_knots: torch.Tensor, b_knots: torch.Tensor):
        super().__init__()
        self.exponent_clip = float(accepted_model.exponent_clip)
        self.register_buffer("log_x_knots", torch.log(torch.unique(x_knots.sort().values)))
        self.register_buffer("b_knots", torch.unique(b_knots.sort().values))
        with torch.no_grad():
            values = accepted_model.A(torch.exp(self.log_x_knots), self.b_knots).clamp_min(1.0e-6)
        self.log_a_knots = torch.nn.Parameter(torch.log(values))

    @staticmethod
    def _interp_indices(values, knots):
        upper = torch.bucketize(values, knots).clamp(1, len(knots) - 1)
        lower = upper - 1
        weight = (values - knots[lower]) / (knots[upper] - knots[lower]).clamp_min(1.0e-12)
        return lower, upper, weight

    def A(self, x, b):
        log_x = torch.log(x.clamp(torch.exp(self.log_x_knots[0]), torch.exp(self.log_x_knots[-1])))
        b_value = b.clamp(self.b_knots[0], self.b_knots[-1])
        xl, xu, xw = self._interp_indices(log_x, self.log_x_knots)
        bl, bu, bw = self._interp_indices(b_value, self.b_knots)
        row_logs = (1.0 - xw).reshape(-1, 1) * self.log_a_knots[xl] + xw.reshape(-1, 1) * self.log_a_knots[xu]
        logs = (1.0 - bw).reshape(1, -1) * row_logs[:, bl] + bw.reshape(1, -1) * row_logs[:, bu]
        return torch.exp(torch.clamp(logs, -12.0, 3.0))

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        exponent = -GlobalReducedNP._cumulative_trapezoid(
            2.0 * b.reshape(1, -1) * self.A(x, b), b)
        if self.exponent_clip > 0:
            exponent = torch.clamp(exponent, -self.exponent_clip, self.exponent_clip)
        return torch.exp(exponent)


class GlobalMonotoneLogFSplineNP(torch.nn.Module):
    """Bilinear log-F spline with positive decrements along b."""

    def __init__(self, accepted_model, x_knots: torch.Tensor, b_knots: torch.Tensor):
        super().__init__()
        self.exponent_clip = float(accepted_model.exponent_clip)
        self.register_buffer("log_x_knots", torch.log(torch.unique(x_knots.sort().values)))
        self.register_buffer("b_knots", torch.unique(b_knots.sort().values))
        with torch.no_grad():
            desired = torch.log(
                accepted_model(torch.exp(self.log_x_knots), self.b_knots)
                .clamp_min(1.0e-30))
            decrements = (desired[:, :-1] - desired[:, 1:]).clamp_min(1.0e-7)
        self.log_decrements = torch.nn.Parameter(torch.log(decrements))

    @staticmethod
    def _interp_indices(values, knots):
        upper = torch.bucketize(values, knots).clamp(1, len(knots) - 1)
        lower = upper - 1
        weight = (values - knots[lower]) / (
            knots[upper] - knots[lower]).clamp_min(1.0e-12)
        return lower, upper, weight

    def knot_logf(self):
        decrements = torch.exp(torch.clamp(self.log_decrements, -18.0, 4.0))
        return torch.cat((
            torch.zeros((len(self.log_x_knots), 1),
                        dtype=decrements.dtype, device=decrements.device),
            -torch.cumsum(decrements, dim=1)), dim=1)

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        log_x = torch.log(x.clamp(
            torch.exp(self.log_x_knots[0]), torch.exp(self.log_x_knots[-1])))
        b_value = b.clamp(self.b_knots[0], self.b_knots[-1])
        xl, xu, xw = self._interp_indices(log_x, self.log_x_knots)
        bl, bu, bw = self._interp_indices(b_value, self.b_knots)
        knots = self.knot_logf()
        row_logs = (
            (1.0 - xw).reshape(-1, 1) * knots[xl]
            + xw.reshape(-1, 1) * knots[xu])
        logs = (
            (1.0 - bw).reshape(1, -1) * row_logs[:, bl]
            + bw.reshape(1, -1) * row_logs[:, bu])
        return torch.exp(torch.clamp(
            logs, -self.exponent_clip, self.exponent_clip))

    def A(self, x, b):
        # Diagnostic-only effective exponent coefficient.
        f = self.forward(x, b)
        return -torch.log(f.clamp_min(1.0e-30)) / b.square().clamp_min(1.0e-8)


class EmpiricalLogFPCANP(torch.nn.Module):
    """Low-rank log-FNP manifold learned from independent admissible fits.

    The mean and principal directions are fixed read-only arrays.  Only the
    declared number of PCA coordinates is fitted, making this a controlled
    identifiability test rather than another high-capacity neural refit.
    """

    def __init__(self, basis_path: Path, rank: int, dtype, device):
        super().__init__()
        arrays = np.load(basis_path)
        x_knots = np.asarray(arrays["x"], dtype=float)
        b_knots = np.asarray(arrays["bT"], dtype=float)
        mean = np.asarray(arrays["mean_logf"], dtype=float)
        components = np.asarray(arrays["components"], dtype=float)
        scales = np.asarray(arrays["score_std"], dtype=float)
        if mean.shape != (len(x_knots), len(b_knots)):
            raise ValueError("empirical mean shape does not match knot grids")
        if components.ndim != 3 or components.shape[1:] != mean.shape:
            raise ValueError("empirical component shape does not match mean")
        if not 1 <= rank <= len(components):
            raise ValueError(
                f"empirical PCA rank must lie in [1,{len(components)}]")
        self.exponent_clip = 80.0
        self.rank = int(rank)
        self.basis_path = str(Path(basis_path).resolve())
        self.register_buffer(
            "log_x_knots", torch.tensor(
                np.log(x_knots), dtype=dtype, device=device))
        self.register_buffer(
            "b_knots", torch.tensor(b_knots, dtype=dtype, device=device))
        self.register_buffer(
            "mean_logf", torch.tensor(mean, dtype=dtype, device=device))
        self.register_buffer(
            "components", torch.tensor(
                components[:rank], dtype=dtype, device=device))
        self.register_buffer(
            "score_std", torch.tensor(
                scales[:rank], dtype=dtype, device=device))
        # Dimensionless coordinates: one unit is one empirical score standard
        # deviation.  Zero is the ensemble mean and is seed-independent.
        self.coordinates = torch.nn.Parameter(torch.zeros(
            rank, dtype=dtype, device=device))

    @staticmethod
    def _interp_indices(values, knots):
        upper = torch.bucketize(values, knots).clamp(1, len(knots) - 1)
        lower = upper - 1
        weight = (values - knots[lower]) / (
            knots[upper] - knots[lower]).clamp_min(1.0e-12)
        return lower, upper, weight

    def knot_logf(self):
        displacement = torch.sum(
            (self.coordinates * self.score_std).reshape(-1, 1, 1)
            * self.components, dim=0)
        return self.mean_logf + displacement

    def forward(self, x, b):
        b = b.to(dtype=x.dtype, device=x.device)
        log_x = torch.log(x.clamp(
            torch.exp(self.log_x_knots[0]),
            torch.exp(self.log_x_knots[-1])))
        b_value = b.clamp(self.b_knots[0], self.b_knots[-1])
        xl, xu, xw = self._interp_indices(log_x, self.log_x_knots)
        bl, bu, bw = self._interp_indices(b_value, self.b_knots)
        knots = self.knot_logf()
        row_logs = (
            (1.0 - xw).reshape(-1, 1) * knots[xl]
            + xw.reshape(-1, 1) * knots[xu])
        logs = (
            (1.0 - bw).reshape(1, -1) * row_logs[:, bl]
            + bw.reshape(1, -1) * row_logs[:, bu])
        return torch.exp(torch.clamp(
            logs, -self.exponent_clip, self.exponent_clip))

    def A(self, x, b):
        f = self.forward(x, b)
        return -torch.log(f.clamp_min(1.0e-30)) / b.square().clamp_min(1.0e-8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-production", type=Path, default=PRODUCTION,
                        help="Read-only central output supplying metrics and accepted rows.")
    parser.add_argument("--w-grid", type=Path, default=W_GRID,
                        help="Read-only perturbative b-space kernel cache.")
    parser.add_argument("--output-root", type=Path, default=BASE / "outputs",
                        help="Root for the newly tagged isolated output.")
    parser.add_argument("--initial-perturbation", type=float, default=0.0)
    parser.add_argument("--initial-state", type=Path, default=None,
                        help="Optional isolated model_state.pt to continue from.")
    parser.add_argument("--initial-norms", type=Path, default=None,
                        help="Optional isolated dataset_norms.csv paired with --initial-state.")
    parser.add_argument("--allow-initial-state-perturbation", action="store_true",
                        help="Permit controlled local starts around an explicitly supplied state.")
    parser.add_argument("--replica-seed", type=int, default=None,
                        help="Fit an isolated experimental pseudo-data replica.")
    parser.add_argument("--max-epochs", type=int, default=20000)
    parser.add_argument("--min-epochs", type=int, default=5000)
    parser.add_argument("--plateau-patience", type=int, default=1500)
    parser.add_argument("--min-delta", type=float, default=1.0e-7)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--np-width", type=int, default=None)
    parser.add_argument("--np-cond-width", type=int, default=None)
    parser.add_argument("--np-blocks", type=int, default=None)
    parser.add_argument("--distill-accepted-steps", type=int, default=0,
                        help="Pretrain a complexity-override model to the frozen accepted FNP.")
    parser.add_argument("--distill-b-scale", type=float, default=2.0,
                        help="Scale in the low-b weighted FNP distillation metric.")
    parser.add_argument("--distill-b-power", type=float, default=4.0,
                        help="Power in weight 1/(1+(b/distill_b_scale)^power).")
    parser.add_argument("--distill-logx-nodes", type=int, default=64,
                        help="Dense log-x nodes used with the diagnostic x knots.")
    parser.add_argument("--distill-prediction-steps", type=int, default=0,
                        help="Pretrain a complexity override to selected-source predictions.")
    parser.add_argument("--lbfgs-max-iter", type=int, default=0)
    parser.add_argument("--float64", action="store_true",
                        help="Use double precision, primarily for stationary-point polishing.")
    parser.add_argument("--lambda-fnp-ratecurv", type=float, default=0.0)
    parser.add_argument("--fnp-ratecurv-bmin", type=float, default=0.25)
    parser.add_argument("--fnp-ratecurv-bmax", type=float, default=4.0)
    parser.add_argument("--lambda-fnp-a-slope", type=float, default=0.0)
    parser.add_argument("--fnp-a-slope-bmin", type=float, default=1.0)
    parser.add_argument("--fnp-a-slope-bmax", type=float, default=4.0)
    parser.add_argument("--lambda-fnp-logcurv", type=float, default=0.0)
    parser.add_argument("--fnp-logcurv-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-logcurv-bmax", type=float, default=3.0)
    parser.add_argument("--lambda-fnp-loglength", type=float, default=0.0)
    parser.add_argument(
        "--fnp-length-space", choices=("logF", "F"), default="logF")
    parser.add_argument("--fnp-loglength-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-loglength-bmax", type=float, default=3.0)
    parser.add_argument("--lambda-fnp-f-slope", type=float, default=0.0)
    parser.add_argument("--fnp-f-slope-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-f-slope-bmax", type=float, default=3.0)
    parser.add_argument("--lambda-fnp-logx-curv", type=float, default=0.0)
    parser.add_argument("--fnp-logx-curv-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-logx-curv-bmax", type=float, default=3.0)
    parser.add_argument("--lambda-fnp-x-slope", type=float, default=0.0)
    parser.add_argument("--fnp-x-slope-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-x-slope-bmax", type=float, default=3.0)
    parser.add_argument("--lambda-fnp-moment-anchor", type=float, default=0.0)
    parser.add_argument("--fnp-moment-anchor-csv", type=Path, default=None)
    parser.add_argument("--fnp-moment-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-moment-bmax", type=float, default=2.0)
    parser.add_argument("--lambda-fnp-reference-distance", type=float, default=0.0)
    parser.add_argument("--fnp-reference-distance-csv", type=Path, default=None)
    parser.add_argument("--fnp-reference-distance-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-reference-distance-bmax", type=float, default=2.0)
    parser.add_argument(
        "--lambda-fnp-shortest-path", type=float, default=0.0,
        help=("Soft quadrature-weighted distance to an analytic fixed-endpoint "
              "shortest path. Requires --fnp-shortest-path-csv."))
    parser.add_argument("--fnp-shortest-path-csv", type=Path, default=None)
    parser.add_argument(
        "--fnp-shortest-path-metric", choices=("F", "logF"), default="logF")
    parser.add_argument("--fnp-shortest-path-bmin", type=float, default=0.10)
    parser.add_argument("--fnp-shortest-path-bmax", type=float, default=2.0)
    parser.add_argument(
        "--endpoint-constrained-reference-csv", type=Path, default=None,
        help=("Opt-in isolated mode: impose exact FNP values at the declared "
              "reference-distance endpoints for every optimizer start."))
    parser.add_argument(
        "--endpoint-constrained-metric", choices=("F", "logF"), default="logF",
        help="Metric for the unique shortest endpoint path in the opt-in mode.")
    parser.add_argument("--endpoint-constrained-bmin", type=float, default=0.10)
    parser.add_argument("--endpoint-constrained-bmax", type=float, default=2.0)
    parser.add_argument(
        "--likelihood-weight", type=float, default=1.0,
        help=(
            "Positive Lagrange weight multiplying data chi2 plus floated-"
            "normalization penalty. Unweighted fit statistics are recorded "
            "separately."))
    parser.add_argument("--fit-quality-ceiling-total-chi2", type=float, default=None)
    parser.add_argument("--lambda-fit-quality-barrier", type=float, default=0.0)
    parser.add_argument(
        "--fit-quality-barrier-power", type=int, choices=(1, 2), default=2,
        help="Use an exact linear hinge (1) or the legacy quadratic hinge (2).")
    parser.add_argument("--lambda-fnp-transform-closure", type=float, default=0.0)
    parser.add_argument("--fnp-transform-closure-bmin", type=float, default=6.0)
    parser.add_argument("--fnp-transform-closure-max", type=float, default=1.0e-4)
    parser.add_argument("--closure-tail-coordinate", action="store_true")
    parser.add_argument("--closure-tail-b-start", type=float, default=6.0)
    parser.add_argument("--closure-tail-b-end", type=float, default=8.0)
    parser.add_argument("--reduced-tail", action="store_true")
    parser.add_argument("--reduced-tail-b-start", type=float, default=0.5)
    parser.add_argument("--reduced-tail-b-end", type=float, default=1.0)
    parser.add_argument("--c1-log-tail", action="store_true")
    parser.add_argument("--c1-log-tail-b-match", type=float, default=2.0)
    parser.add_argument("--global-reduced-components", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--global-reduced-b-scale", type=float, default=1.0)
    parser.add_argument("--global-spline", action="store_true")
    parser.add_argument("--global-spline-nx", type=int, default=9)
    parser.add_argument("--global-spline-nb", type=int, default=6)
    parser.add_argument("--global-monotone-logf-spline", action="store_true")
    parser.add_argument("--empirical-logf-pca", type=Path, default=None,
                        help="Read-only NPZ containing an empirical log-FNP PCA basis.")
    parser.add_argument("--empirical-logf-pca-rank", type=int, default=0)
    parser.add_argument("--empirical-logf-pca-initial-scale", type=float, default=0.0,
                        help="Seeded Gaussian initial PCA-coordinate scale in empirical SD units.")
    parser.add_argument("--empirical-logf-pca-initial-member", type=int, default=None,
                        help="Initialize from a source seed stored in the empirical basis.")
    parser.add_argument("--profile-x", type=float, default=None)
    parser.add_argument("--profile-b", type=float, default=None)
    parser.add_argument("--profile-logf-target", type=float, default=None)
    parser.add_argument("--profile-lambda", type=float, default=0.0,
                        help="Per-row quadratic penalty enforcing a pointwise log-FNP profile target.")
    parser.add_argument("--profile-local-shift", action="store_true",
                        help="Add a localized nonsaturated log-FNP coordinate for the profile.")
    parser.add_argument("--profile-sigma-logx", type=float, default=0.45)
    parser.add_argument("--profile-sigma-b", type=float, default=0.20)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    if args.likelihood_weight <= 0.0:
        raise ValueError("--likelihood-weight must be positive")
    source_production = args.source_production.resolve()
    w_grid_path = args.w_grid.resolve()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dtype = torch.float64 if args.float64 else torch.float32
    trainer = load_module("production_control_trainer", TRAINER_PATH)
    refit = load_module("production_control_refit", BASE / "scripts/run_differentiable_fnp_refit.py")
    metrics = json.loads((source_production / "metrics.json").read_text())
    config = metrics["config"]
    accepted = pd.read_csv(source_production / "predictions.csv")
    b_np, w_matrix = trainer.load_external_w_grid(accepted.row_id.astype(str), w_grid_path)
    kernel_np = trainer.precompute_kernel_matrix(
        accepted.qT.to_numpy(), b_np, w_matrix, dtype=work_dtype).astype(
            np.float64 if args.float64 else np.float32)

    def tensor(values, dtype=None):
        if dtype is None:
            dtype = work_dtype
        return torch.tensor(np.array(values, copy=True), dtype=dtype, device=device)

    b, kernel = tensor(b_np), tensor(kernel_np)
    x1, x2 = tensor(accepted.x1), tensor(accepted.x2)
    y_term = tensor(accepted.Y_CS_used)
    target_values = accepted.target_used.to_numpy(float)
    if args.replica_seed is not None:
        rng = np.random.default_rng(args.replica_seed)
        target_values = target_values.copy()
        for _, sub in accepted.groupby("dataset", sort=False):
            idx = sub.index.to_numpy()
            norm_rel = float(sub["norm_rel_used"].iloc[0])
            target_values[idx] *= 1.0 + rng.normal() * norm_rel
        target_values += rng.normal(size=len(accepted)) * accepted.sigma_uncorr.to_numpy(float)
    data, error = tensor(target_values), tensor(accepted.sigma_used)
    datasets = list(dict.fromkeys(accepted.dataset.astype(str)))
    dataset_to_i = {name: i for i, name in enumerate(datasets)}
    dataset_index = torch.tensor(accepted.dataset.map(dataset_to_i).to_numpy(), dtype=torch.long, device=device)
    norm_width = accepted.groupby("dataset").norm_rel_used.first().reindex(datasets).to_numpy(float)
    norm_start = accepted.groupby("dataset").dataset_norm_factor.first().reindex(datasets).to_numpy(float)
    free_norm = norm_width > 0.0
    free_norm_t = tensor(free_norm.astype(float))
    norm_width_t = tensor(np.where(free_norm, norm_width, 1.0))

    complexity_override = any(
        value is not None for value in (args.np_width, args.np_cond_width, args.np_blocks)
    )
    if complexity_override:
        model = trainer.FilmNPFactor(
            width=int(args.np_width if args.np_width is not None else config["np_width"]),
            cond_width=int(args.np_cond_width if args.np_cond_width is not None else config["np_cond_width"]),
            n_blocks=int(args.np_blocks if args.np_blocks is not None else config["np_blocks"]),
            a0=float(config["np_a0"]), min_a=float(config["np_min_a"]),
            a_mode=str(config["np_a_mode"]), exponent_clip=float(config["fnp_exponent_clip"]),
            shape_mode=str(config["np_shape_mode"]),
            a_smooth_sigma=float(config["np_a_smooth_sigma"]),
            a_tail_amp=float(config["np_a_tail_amp"]),
            a_tail_b0=float(config["np_a_tail_b0"]),
            a_tail_width=float(config["np_a_tail_width"]),
            dtype=work_dtype,
        ).to(device)
    else:
        model = refit.make_model(trainer, config, device).to(dtype=work_dtype)
    if args.distill_accepted_steps:
        if not complexity_override:
            raise ValueError("--distill-accepted-steps requires a complexity override")
        accepted_model = refit.make_model(trainer, config, device).to(dtype=work_dtype)
        # ``make_model`` carries the legacy production state for compatibility
        # with the original refit script.  Capacity studies must instead
        # distill the explicitly selected, read-only dataset candidate.
        accepted_saved = torch.load(
            source_production / "model_state.pt",
            map_location=device, weights_only=True)
        accepted_state = (
            {
                key[len("np_factor."):]: value
                for key, value in accepted_saved.items()
                if key.startswith("np_factor.")
            }
            if any(key.startswith("np_factor.") for key in accepted_saved)
            else accepted_saved
        )
        accepted_model.load_state_dict(accepted_state, strict=True)
        accepted_model.eval()
        if args.distill_logx_nodes < 2:
            raise ValueError("distillation requires at least two log-x nodes")
        dense_logx = torch.logspace(
            np.log10(0.0005), np.log10(0.7),
            args.distill_logx_nodes, dtype=work_dtype, device=device)
        distill_x = torch.unique(torch.cat((
            dense_logx, tensor([0.0005] + FNP_GRID_X + [0.7])))).sort().values
        distill_b = torch.linspace(0.0001, 8.0, 321, dtype=work_dtype, device=device)
        with torch.no_grad():
            distill_target = torch.log(accepted_model(distill_x, distill_b).clamp_min(1.0e-20))
        distill_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        if args.distill_b_scale <= 0.0 or args.distill_b_power <= 0.0:
            raise ValueError("distillation scale and power must be positive")
        distill_weight = 1.0 / (
            1.0 + (distill_b / args.distill_b_scale).pow(
                args.distill_b_power))
        distill_weight = distill_weight / distill_weight.sum()
        for _ in range(args.distill_accepted_steps):
            distill_optimizer.zero_grad(set_to_none=True)
            prediction = torch.log(model(distill_x, distill_b).clamp_min(1.0e-20))
            distill_loss = torch.mean(torch.sum(
                (prediction - distill_target).square()
                * distill_weight.reshape(1, -1), dim=1))
            distill_loss.backward()
            distill_optimizer.step()
        if args.distill_prediction_steps:
            with torch.no_grad():
                accepted_factors = (
                    accepted_model(x1, b) * accepted_model(x2, b))
                accepted_raw = (
                    torch.sum(kernel * accepted_factors, dim=1) + y_term)
                initial_norm = tensor(norm_start)[dataset_index]
                accepted_prediction = initial_norm * accepted_raw
            for _ in range(args.distill_prediction_steps):
                distill_optimizer.zero_grad(set_to_none=True)
                student_factors = model(x1, b) * model(x2, b)
                student_raw = (
                    torch.sum(kernel * student_factors, dim=1) + y_term)
                student_prediction = initial_norm * student_raw
                prediction_distill_loss = torch.mean(
                    ((student_prediction - accepted_prediction)
                     / error).square())
                prediction_distill_loss.backward()
                distill_optimizer.step()
    regularizer_x = tensor(FNP_GRID_X)
    profile_values = (args.profile_x, args.profile_b, args.profile_logf_target)
    profile_enabled = any(value is not None for value in profile_values) or args.profile_lambda > 0.0
    if profile_enabled and (
            any(value is None for value in profile_values) or args.profile_lambda <= 0.0):
        raise ValueError(
            "A profile requires --profile-x, --profile-b, "
            "--profile-logf-target, and positive --profile-lambda")
    profile_x = tensor([args.profile_x]) if profile_enabled else None
    profile_b = tensor([args.profile_b]) if profile_enabled else None
    if sum((bool(args.reduced_tail), bool(args.c1_log_tail),
            bool(args.closure_tail_coordinate),
            bool(args.global_reduced_components), bool(args.global_spline),
            bool(args.global_monotone_logf_spline),
            args.empirical_logf_pca is not None)) > 1:
        raise ValueError("Choose only one reduced FNP model")
    if args.reduced_tail:
        tail_knots = tensor([0.0005] + FNP_GRID_X + [0.7])
        model = ReducedTailNP(model, args.reduced_tail_b_start, args.reduced_tail_b_end, tail_knots)
    elif args.c1_log_tail:
        tail_knots = tensor([0.0005] + FNP_GRID_X + [0.7])
        model = C1MatchedLogTailNP(model, args.c1_log_tail_b_match, tail_knots)
    elif args.closure_tail_coordinate:
        tail_knots = tensor([0.0005] + FNP_GRID_X + [0.7])
        model = ClosureTailCoordinateNP(
            model, args.closure_tail_b_start,
            args.closure_tail_b_end, tail_knots,
            args.fnp_transform_closure_max)
    elif args.global_reduced_components:
        tail_knots = tensor([0.0005] + FNP_GRID_X + [0.7])
        model = GlobalReducedNP(model, tail_knots, args.global_reduced_components,
                                args.global_reduced_b_scale)
    elif args.global_spline:
        if args.global_spline_nx < 3:
            raise ValueError("--global-spline-nx must be at least 3")
        spline_x = torch.logspace(
            np.log10(0.0005), np.log10(0.7), args.global_spline_nx,
            dtype=work_dtype, device=device)
        if args.global_spline_nb < 3:
            raise ValueError("--global-spline-nb must be at least 3")
        spline_b = torch.linspace(0.0001, 8.0, args.global_spline_nb, dtype=work_dtype, device=device)
        model = GlobalSplineNP(model, spline_x, spline_b)
    elif args.global_monotone_logf_spline:
        if args.global_spline_nx < 3 or args.global_spline_nb < 3:
            raise ValueError("monotone spline dimensions must be at least 3")
        spline_x = torch.logspace(
            np.log10(0.0005), np.log10(0.7), args.global_spline_nx,
            dtype=work_dtype, device=device)
        spline_t = torch.linspace(
            0.0, 1.0, args.global_spline_nb,
            dtype=work_dtype, device=device)
        spline_b = 0.0001 + (8.0 - 0.0001) * spline_t.pow(1.5)
        model = GlobalMonotoneLogFSplineNP(model, spline_x, spline_b)
    elif args.empirical_logf_pca is not None:
        if args.empirical_logf_pca_rank < 1:
            raise ValueError("--empirical-logf-pca requires a positive rank")
        if args.empirical_logf_pca_initial_scale < 0.0:
            raise ValueError("empirical PCA initial scale cannot be negative")
        model = EmpiricalLogFPCANP(
            args.empirical_logf_pca.resolve(),
            args.empirical_logf_pca_rank, work_dtype, device)
        if args.empirical_logf_pca_initial_member is not None:
            arrays = np.load(args.empirical_logf_pca.resolve())
            member_seeds = np.asarray(arrays["source_seeds"], dtype=int)
            matches = np.flatnonzero(
                member_seeds == args.empirical_logf_pca_initial_member)
            if len(matches) != 1:
                raise ValueError(
                    "requested empirical PCA initial member is absent")
            member_scores = np.asarray(arrays["member_scores"], dtype=float)
            with torch.no_grad():
                model.coordinates.copy_(tensor(
                    member_scores[matches[0], :args.empirical_logf_pca_rank]))
        if args.empirical_logf_pca_initial_scale:
            if args.empirical_logf_pca_initial_member is not None:
                raise ValueError(
                    "choose either empirical initial member or random scale")
            with torch.no_grad():
                model.coordinates.normal_(
                    mean=0.0, std=args.empirical_logf_pca_initial_scale)
    if args.initial_state is not None:
        saved = torch.load(args.initial_state, map_location=device, weights_only=True)
        if any(key.startswith("np_factor.") for key in saved):
            state = {
                key[len("np_factor."):]: value
                for key, value in saved.items()
                if key.startswith("np_factor.")
            }
        else:
            state = saved
        if (args.closure_tail_coordinate
                and not any(key.startswith("base.") for key in state)):
            model.base.load_state_dict(state, strict=True)
            model.initialize_closure_from_base()
        else:
            model.load_state_dict(state, strict=True)
    if args.profile_local_shift:
        if not profile_enabled:
            raise ValueError("--profile-local-shift requires a complete point profile")
        with torch.no_grad():
            current_logf = torch.log(
                model(profile_x, profile_b).clamp_min(1.0e-30)).squeeze()
        model = LocalizedLogFShift(
            model, args.profile_x, args.profile_b,
            args.profile_sigma_logx, args.profile_sigma_b,
            args.profile_logf_target - float(current_logf))
    if args.endpoint_constrained_reference_csv is not None:
        if not 0.0 <= args.endpoint_constrained_bmin < args.endpoint_constrained_bmax:
            raise ValueError("endpoint-constrained b range is invalid")
        endpoint_table = pd.read_csv(args.endpoint_constrained_reference_csv)
        if not {"x", "bT", "F_NP"}.issubset(endpoint_table.columns):
            raise ValueError(
                "endpoint-constrained reference CSV requires x, bT, F_NP")
        endpoint_values = []
        for x_value in FNP_GRID_X:
            group = endpoint_table[
                np.isclose(endpoint_table["x"], x_value)].sort_values("bT")
            if len(group) < 3:
                raise ValueError(f"endpoint reference lacks x={x_value}")
            endpoint_values.append([
                float(np.interp(args.endpoint_constrained_bmin,
                                group["bT"].to_numpy(float),
                                group["F_NP"].to_numpy(float))),
                float(np.interp(args.endpoint_constrained_bmax,
                                group["bT"].to_numpy(float),
                                group["F_NP"].to_numpy(float))),
            ])
        model = EndpointConstrainedFNP(
            model, args.endpoint_constrained_bmin,
            args.endpoint_constrained_bmax, regularizer_x,
            tensor(np.asarray(endpoint_values, dtype=float)),
            metric=args.endpoint_constrained_metric)
    rate_mask = (b >= args.fnp_ratecurv_bmin) & (b <= args.fnp_ratecurv_bmax)
    a_slope_mask = (b >= args.fnp_a_slope_bmin) & (b <= args.fnp_a_slope_bmax)
    # Dense, origin-anchored evaluation is essential for every regularizer
    # applied to a cumulative FNP model.  Evaluating only a masked interval
    # resets its numerical integral at the first selected node and can hide
    # between-node structure.
    regularizer_b_dense = torch.linspace(
        float(b[0]), float(b[-1]), 641, dtype=b.dtype, device=b.device)
    logcurv_mask = (
        (regularizer_b_dense >= args.fnp_logcurv_bmin)
        & (regularizer_b_dense <= args.fnp_logcurv_bmax))
    loglength_mask = (
        (regularizer_b_dense >= args.fnp_loglength_bmin)
        & (regularizer_b_dense <= args.fnp_loglength_bmax))
    f_slope_mask = (
        (regularizer_b_dense >= args.fnp_f_slope_bmin)
        & (regularizer_b_dense <= args.fnp_f_slope_bmax))
    logx_curv_mask = (
        (regularizer_b_dense >= args.fnp_logx_curv_bmin)
        & (regularizer_b_dense <= args.fnp_logx_curv_bmax))
    x_slope_mask = (
        (regularizer_b_dense >= args.fnp_x_slope_bmin)
        & (regularizer_b_dense <= args.fnp_x_slope_bmax))
    moment_mask = (
        (regularizer_b_dense >= args.fnp_moment_bmin)
        & (regularizer_b_dense <= args.fnp_moment_bmax))
    reference_distance_mask = (
        (regularizer_b_dense >= args.fnp_reference_distance_bmin)
        & (regularizer_b_dense <= args.fnp_reference_distance_bmax))
    moment_target = None
    if args.lambda_fnp_moment_anchor > 0.0:
        if args.fnp_moment_anchor_csv is None:
            raise ValueError(
                "--lambda-fnp-moment-anchor requires "
                "--fnp-moment-anchor-csv")
        moment_table = pd.read_csv(args.fnp_moment_anchor_csv)
        if not {"x", "moment"}.issubset(moment_table.columns):
            raise ValueError("moment-anchor CSV requires x and moment columns")
        moment_values = np.interp(
            np.asarray(FNP_GRID_X, dtype=float),
            moment_table["x"].to_numpy(float),
            moment_table["moment"].to_numpy(float))
        if np.any(moment_values <= 0.0):
            raise ValueError("moment-anchor targets must be positive")
        moment_target = tensor(moment_values)
    reference_distance_target = None
    if args.lambda_fnp_reference_distance > 0.0:
        if args.fnp_reference_distance_csv is None:
            raise ValueError(
                "--lambda-fnp-reference-distance requires a target CSV")
        reference_table = pd.read_csv(args.fnp_reference_distance_csv)
        if not {"x", "bT", "F_NP"}.issubset(reference_table.columns):
            raise ValueError("reference-distance CSV requires x, bT, F_NP")
        targets = []
        dense_b_np = regularizer_b_dense.detach().cpu().numpy()
        for x_value in FNP_GRID_X:
            group = reference_table[
                np.isclose(reference_table["x"], x_value)].sort_values("bT")
            if len(group) < 3:
                raise ValueError(f"reference target lacks x={x_value}")
            targets.append(np.interp(
                dense_b_np, group["bT"].to_numpy(float),
                group["F_NP"].to_numpy(float)))
        reference_distance_target = tensor(np.asarray(targets))
    shortest_path_target = None
    shortest_path_scale = None
    if args.lambda_fnp_shortest_path > 0.0:
        if args.fnp_shortest_path_csv is None:
            raise ValueError(
                "--lambda-fnp-shortest-path requires --fnp-shortest-path-csv")
        if not 0.0 <= args.fnp_shortest_path_bmin < args.fnp_shortest_path_bmax:
            raise ValueError("shortest-path b interval is invalid")
        path_table = pd.read_csv(args.fnp_shortest_path_csv)
        if not {"x", "bT", "F_NP"}.issubset(path_table.columns):
            raise ValueError("shortest-path CSV requires x, bT, F_NP")
        targets = []
        dense_b_np = regularizer_b_dense.detach().cpu().numpy()
        for x_value in FNP_GRID_X:
            group = path_table[np.isclose(path_table["x"], x_value)].sort_values("bT")
            if len(group) < 3:
                raise ValueError(f"shortest-path target lacks x={x_value}")
            targets.append(np.interp(
                dense_b_np, group["bT"].to_numpy(float),
                group["F_NP"].to_numpy(float)))
        shortest_path_target = tensor(np.asarray(targets))
        b0 = np.asarray(targets)[:, np.argmin(np.abs(dense_b_np - args.fnp_shortest_path_bmin))]
        b1 = np.asarray(targets)[:, np.argmin(np.abs(dense_b_np - args.fnp_shortest_path_bmax))]
        if args.fnp_shortest_path_metric == "logF":
            shortest_path_scale = tensor(np.maximum(np.abs(np.log(b1) - np.log(b0)), 1.0))
        else:
            shortest_path_scale = tensor(np.maximum(np.abs(b1 - b0), 0.10))
    closure_mask = b >= args.fnp_transform_closure_bmin
    if args.lambda_fnp_ratecurv > 0.0 and int(torch.count_nonzero(rate_mask)) < 3:
        raise ValueError("The damping-rate curvature interval requires at least three b nodes")
    if args.lambda_fnp_a_slope > 0.0 and int(torch.count_nonzero(a_slope_mask)) < 2:
        raise ValueError("The A-slope interval requires at least two b nodes")
    if args.lambda_fnp_logcurv > 0.0 and int(torch.count_nonzero(logcurv_mask)) < 3:
        raise ValueError("The log-FNP curvature interval requires at least three b nodes")
    if (args.lambda_fnp_loglength > 0.0
            and int(torch.count_nonzero(loglength_mask)) < 2):
        raise ValueError("The log-FNP length interval requires at least two b nodes")
    if (args.lambda_fnp_f_slope > 0.0
            and int(torch.count_nonzero(f_slope_mask)) < 2):
        raise ValueError("The FNP slope-energy interval requires at least two b nodes")
    if (args.lambda_fnp_logx_curv > 0.0
            and (len(regularizer_x) < 3
                 or int(torch.count_nonzero(logx_curv_mask)) < 1)):
        raise ValueError(
            "The log-x curvature penalty requires at least three x nodes "
            "and one b node")
    if (args.lambda_fnp_x_slope > 0.0
            and (len(regularizer_x) < 2
                 or int(torch.count_nonzero(x_slope_mask)) < 1)):
        raise ValueError(
            "The x-slope penalty requires at least two x nodes and one b node")
    if (args.lambda_fnp_moment_anchor > 0.0
            and int(torch.count_nonzero(moment_mask)) < 2):
        raise ValueError("The FNP moment interval requires at least two b nodes")
    if (args.lambda_fnp_reference_distance > 0.0
            and int(torch.count_nonzero(reference_distance_mask)) < 2):
        raise ValueError("The FNP reference-distance interval needs two b nodes")
    if args.lambda_fnp_transform_closure > 0.0:
        if int(torch.count_nonzero(closure_mask)) < 1:
            raise ValueError("The transform-closure interval has no b nodes")
        if not 0.0 < args.fnp_transform_closure_max < 1.0:
            raise ValueError("--fnp-transform-closure-max must lie between zero and one")
    if (args.initial_state is not None and args.initial_perturbation
            and not args.allow_initial_state_perturbation):
        raise ValueError(
            "Combining --initial-state with --initial-perturbation requires "
            "--allow-initial-state-perturbation")
    if args.initial_perturbation:
        with torch.no_grad():
            for parameter in model.parameters():
                scale = torch.sqrt(torch.mean(parameter.square())).clamp_min(1.0e-6)
                parameter.add_(args.initial_perturbation * scale * torch.randn_like(parameter))
    if args.initial_norms is not None:
        norm_table = pd.read_csv(args.initial_norms).set_index("dataset")
        norm_column = "control_norm" if "control_norm" in norm_table else "norm_scale"
        continued_norms = norm_table[norm_column].reindex(datasets).to_numpy(float)
        if not np.all(np.isfinite(continued_norms)):
            raise ValueError("initial normalization table does not cover every dataset")
        norm_start = continued_norms
    log_norms = torch.nn.Parameter(tensor(np.where(free_norm, np.log(norm_start), 0.0)))
    parameters = list(model.parameters()) + [log_norms]
    learning_rate = float(args.learning_rate if args.learning_rate is not None else config["lr"])
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=375, threshold=args.min_delta, threshold_mode="abs", min_lr=2.0e-7)

    def evaluate():
        factors = model(x1, b) * model(x2, b)
        raw = torch.sum(kernel * factors, dim=1) + y_term
        norms = torch.exp(log_norms * free_norm_t)
        prediction = norms[dataset_index] * raw
        data_chi2 = torch.sum(((prediction - data) / error).square())
        norm_penalty = torch.sum((free_norm_t * (norms - 1.0) / norm_width_t).square())
        ratecurv_penalty = torch.zeros((), dtype=work_dtype, device=device)
        a_slope_penalty = torch.zeros((), dtype=work_dtype, device=device)
        logcurv_penalty = torch.zeros((), dtype=work_dtype, device=device)
        loglength_penalty = torch.zeros((), dtype=work_dtype, device=device)
        f_slope_penalty = torch.zeros((), dtype=work_dtype, device=device)
        logx_curv_penalty = torch.zeros(
            (), dtype=work_dtype, device=device)
        x_slope_penalty = torch.zeros((), dtype=work_dtype, device=device)
        moment_anchor_penalty = torch.zeros(
            (), dtype=work_dtype, device=device)
        reference_distance_penalty = torch.zeros(
            (), dtype=work_dtype, device=device)
        shortest_path_penalty = torch.zeros(
            (), dtype=work_dtype, device=device)
        closure_penalty = torch.zeros((), dtype=work_dtype, device=device)
        fit_quality_barrier = torch.zeros((), dtype=work_dtype, device=device)
        if args.lambda_fnp_ratecurv > 0.0:
            b_rate = b[rate_mask]
            h_rate = 2.0 * b_rate.reshape(1, -1) * model.A(regularizer_x, b_rate)
            d2h = trainer._second_diff_nonuniform(h_rate, b_rate, dim=1)
            brange = torch.clamp(b_rate[-1] - b_rate[0], min=1.0)
            hscale = torch.clamp(torch.sqrt(torch.mean(h_rate.detach().square())), min=0.25)
            ratecurv_penalty = args.lambda_fnp_ratecurv * torch.mean(
                (d2h * brange.square() / hscale).square())
        if args.lambda_fnp_a_slope > 0.0:
            b_a = b[a_slope_mask]
            a_values = model.A(regularizer_x, b_a)
            da_db = (a_values[:, 1:] - a_values[:, :-1]) / (b_a[1:] - b_a[:-1]).reshape(1, -1)
            brange_a = torch.clamp(b_a[-1] - b_a[0], min=1.0)
            ascale = torch.clamp(torch.sqrt(torch.mean(a_values.detach().square())), min=0.05)
            a_slope_penalty = args.lambda_fnp_a_slope * torch.mean(
                (da_db * brange_a / ascale).square())
        if args.lambda_fnp_logcurv > 0.0:
            b_log = regularizer_b_dense[logcurv_mask]
            logf = torch.log(
                model(regularizer_x, regularizer_b_dense).clamp_min(1.0e-30)
            )[:, logcurv_mask]
            d2logf = trainer._second_diff_nonuniform(logf, b_log, dim=1)
            brange_log = torch.clamp(b_log[-1] - b_log[0], min=1.0)
            logscale = torch.clamp(
                torch.sqrt(torch.mean(logf.detach().square())), min=0.25)
            logcurv_penalty = args.lambda_fnp_logcurv * torch.mean(
                (d2logf * brange_log.square() / logscale).square())
        if args.lambda_fnp_loglength > 0.0:
            b_length = regularizer_b_dense[loglength_mask]
            # Evaluate the cumulative model from the origin, then select the
            # requested interval.  Calling it directly on ``b_length`` would
            # reset the numerical integral at b_min and regularize a different
            # function than the FNP used in the cross-section likelihood.
            fnp_length = model(
                regularizer_x, regularizer_b_dense
            )[:, loglength_mask].clamp_min(1.0e-30)
            length_values = (
                torch.log(fnp_length)
                if args.fnp_length_space == "logF" else fnp_length)
            dlength = (
                (length_values[:, 1:] - length_values[:, :-1])
                / (b_length[1:] - b_length[:-1]).reshape(1, -1))
            brange_length = torch.clamp(
                b_length[-1] - b_length[0], min=1.0)
            scale_floor = 0.25 if args.fnp_length_space == "logF" else 0.10
            length_scale = torch.clamp(
                torch.sqrt(torch.mean(length_values.detach().square())),
                min=scale_floor)
            dimensionless_slope = (
                dlength * brange_length / length_scale)
            loglength_penalty = args.lambda_fnp_loglength * torch.mean(
                torch.sqrt(1.0 + dimensionless_slope.square()) - 1.0)
        if args.lambda_fnp_f_slope > 0.0:
            b_fslope = regularizer_b_dense[f_slope_mask]
            f_fslope = model(
                regularizer_x, regularizer_b_dense)[:, f_slope_mask]
            df_db = (
                (f_fslope[:, 1:] - f_fslope[:, :-1])
                / (b_fslope[1:] - b_fslope[:-1]).reshape(1, -1))
            brange_fslope = torch.clamp(
                b_fslope[-1] - b_fslope[0], min=1.0)
            f_slope_penalty = args.lambda_fnp_f_slope * torch.mean(
                (df_db * brange_fslope).square())
        if args.lambda_fnp_logx_curv > 0.0:
            log_x = torch.log(regularizer_x)
            logf_x = torch.log(
                model(regularizer_x, regularizer_b_dense).clamp_min(1.0e-30)
            )[:, logx_curv_mask]
            d2logf_dlogx2 = trainer._second_diff_nonuniform(
                logf_x, log_x, dim=0)
            logx_range = torch.clamp(log_x[-1] - log_x[0], min=1.0)
            logf_x_scale = torch.clamp(
                torch.sqrt(torch.mean(logf_x.detach().square())), min=0.25)
            logx_curv_penalty = (
                args.lambda_fnp_logx_curv
                * torch.mean(
                    (d2logf_dlogx2 * logx_range.square()
                     / logf_x_scale).square()))
        if args.lambda_fnp_x_slope > 0.0:
            log_x = torch.log(regularizer_x)
            fnp_x = model(
                regularizer_x, regularizer_b_dense)[:, x_slope_mask]
            df_dlogx = (
                (fnp_x[1:, :] - fnp_x[:-1, :])
                / (log_x[1:] - log_x[:-1]).reshape(-1, 1))
            logx_range = torch.clamp(log_x[-1] - log_x[0], min=1.0)
            x_slope_penalty = (
                args.lambda_fnp_x_slope
                * torch.mean((df_dlogx * logx_range).square()))
        if args.lambda_fnp_moment_anchor > 0.0:
            b_moment = regularizer_b_dense[moment_mask]
            fnp_moment = model(
                regularizer_x, regularizer_b_dense)[:, moment_mask]
            moment = torch.trapezoid(
                fnp_moment, b_moment, dim=1
            ) / (b_moment[-1] - b_moment[0])
            moment_anchor_penalty = (
                args.lambda_fnp_moment_anchor
                * torch.mean(
                    ((moment - moment_target)
                     / moment_target.clamp_min(0.10)).square()))
        if args.lambda_fnp_reference_distance > 0.0:
            current = model(regularizer_x, regularizer_b_dense)[
                :, reference_distance_mask]
            target_distance = reference_distance_target[
                :, reference_distance_mask]
            reference_distance_penalty = (
                args.lambda_fnp_reference_distance
                * torch.mean(((current - target_distance)
                              / target_distance.clamp_min(0.10)).square()))
        if args.lambda_fnp_shortest_path > 0.0:
            current = model(regularizer_x, regularizer_b_dense).clamp_min(1.0e-30)
            target_path = shortest_path_target
            if args.fnp_shortest_path_metric == "logF":
                residual = torch.log(current) - torch.log(target_path.clamp_min(1.0e-30))
            else:
                residual = current - target_path
            residual = residual / shortest_path_scale.reshape(-1, 1).clamp_min(1.0e-12)
            path_mask = (
                (regularizer_b_dense >= args.fnp_shortest_path_bmin)
                & (regularizer_b_dense <= args.fnp_shortest_path_bmax))
            path_b = regularizer_b_dense[path_mask]
            path_residual = residual[:, path_mask]
            shortest_path_penalty = args.lambda_fnp_shortest_path * torch.mean(
                torch.trapezoid(path_residual.square(), path_b, dim=1)
                / (path_b[-1] - path_b[0]))
        if args.lambda_fnp_transform_closure > 0.0:
            # A one-sided numerical-closure condition: it is indifferent to
            # the tail shape and to any value already small enough that the
            # finite-b transform is closed.
            # Evaluate cumulative models from b=0 before selecting the remote
            # tail. Calling such a model on the masked endpoint grid would
            # incorrectly reset its integral at the first selected point.
            full_fnp_for_closure = model(regularizer_x, b)
            logf_closure = torch.log(
                full_fnp_for_closure[:, closure_mask].clamp_min(1.0e-30))
            excess = torch.relu(
                logf_closure - np.log(args.fnp_transform_closure_max))
            closure_penalty = (
                args.lambda_fnp_transform_closure * torch.mean(excess.square()))
        profile_penalty = torch.zeros((), dtype=work_dtype, device=device)
        if profile_enabled:
            profile_logf = torch.log(model(profile_x, profile_b).clamp_min(1.0e-30)).squeeze()
            profile_penalty = args.profile_lambda * (
                profile_logf - args.profile_logf_target).square()
        if args.lambda_fit_quality_barrier > 0.0:
            if args.fit_quality_ceiling_total_chi2 is None:
                raise ValueError(
                    "fit-quality barrier requires a total-chi2 ceiling")
            excess_chi2 = torch.relu(
                data_chi2 + norm_penalty
                - args.fit_quality_ceiling_total_chi2)
            fit_quality_barrier = (
                args.lambda_fit_quality_barrier
                * excess_chi2.pow(args.fit_quality_barrier_power)
                / len(accepted))
        total = args.likelihood_weight * (
            data_chi2 + norm_penalty) + len(accepted) * (
            ratecurv_penalty + a_slope_penalty + logcurv_penalty
            + loglength_penalty + f_slope_penalty
            + logx_curv_penalty
            + x_slope_penalty
            + moment_anchor_penalty
            + reference_distance_penalty
            + shortest_path_penalty
            + closure_penalty + profile_penalty
            + fit_quality_barrier)
        return (total, data_chi2, norm_penalty, ratecurv_penalty,
                a_slope_penalty, prediction, norms, profile_penalty,
                logcurv_penalty, loglength_penalty, closure_penalty,
                fit_quality_barrier, f_slope_penalty,
                logx_curv_penalty, x_slope_penalty,
                moment_anchor_penalty, reference_distance_penalty,
                shortest_path_penalty)

    n_rows = len(accepted)
    best_loss, best, last_improvement = float("inf"), None, 0
    stopped = False
    history = []
    for epoch in range(args.max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        values = evaluate()
        loss = values[0] / n_rows
        if epoch:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config["grad_clip"]))
            optimizer.step()
        current = float(loss.detach())
        scheduler.step(current)
        if current < best_loss - args.min_delta:
            best_loss, last_improvement = current, epoch
            best = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "log_norms": log_norms.detach().cpu().clone(), "epoch": epoch,
            }
        if epoch % 25 == 0 or epoch == args.max_epochs:
            history.append({"epoch": epoch, "objective_per_row": current,
                            "data_chi2": float(values[1].detach()),
                            "norm_penalty": float(values[2].detach()),
                            "ratecurv_penalty_per_row_objective": float(values[3].detach()),
                            "a_slope_penalty_per_row_objective": float(values[4].detach()),
                            "logcurv_penalty_per_row_objective": float(values[8].detach()),
                            "loglength_penalty_per_row_objective": float(values[9].detach()),
                            "transform_closure_penalty_per_row_objective": float(values[10].detach()),
                            "fit_quality_barrier_per_row_objective": float(values[11].detach()),
                            "f_slope_penalty_per_row_objective": float(values[12].detach()),
                            "logx_curv_penalty_per_row_objective": float(values[13].detach()),
                            "x_slope_penalty_per_row_objective": float(values[14].detach()),
                            "shortest_path_penalty_per_row_objective": float(values[17].detach()),
                            "learning_rate": optimizer.param_groups[0]["lr"]})
        if epoch % 500 == 0:
            print(f"epoch={epoch} objective/N={current:.8f} "
                  f"data_chi2={float(values[1].detach()):.5f}", flush=True)
        if epoch >= args.min_epochs and epoch - last_improvement >= args.plateau_patience:
            stopped = True
            break

    assert best is not None
    model.load_state_dict(best["model"])
    with torch.no_grad():
        log_norms.copy_(best["log_norms"].to(device))
    lbfgs_start = float(evaluate()[0].detach()) / n_rows
    lbfgs_closures = 0
    if args.lbfgs_max_iter > 0:
        polish = torch.optim.LBFGS(
            parameters, lr=1.0, max_iter=args.lbfgs_max_iter,
            tolerance_grad=1.0e-7, tolerance_change=1.0e-10,
            history_size=50, line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal lbfgs_closures
            polish.zero_grad(set_to_none=True)
            value = evaluate()[0] / n_rows
            value.backward()
            lbfgs_closures += 1
            if lbfgs_closures == 1 or lbfgs_closures % 500 == 0:
                print(
                    f"lbfgs_closure={lbfgs_closures} "
                    f"objective/N={float(value.detach()):.8f}",
                    flush=True)
            return value

        polish.step(closure)
    objective = evaluate()[0] / n_rows
    gradients = torch.autograd.grad(objective, parameters)
    fnp_grad = float(torch.sqrt(sum(torch.sum(g.square()) for g in gradients[:-1])).detach())
    norm_grad = float(torch.sqrt(torch.sum(gradients[-1].square())).detach())
    with torch.no_grad():
        (total, data_chi2, norm_penalty, ratecurv_penalty,
         a_slope_penalty, prediction, norms, profile_penalty,
         logcurv_penalty, loglength_penalty, closure_penalty,
         fit_quality_barrier, f_slope_penalty,
         logx_curv_penalty, x_slope_penalty,
         moment_anchor_penalty, reference_distance_penalty,
         shortest_path_penalty) = evaluate()

    output = accepted[["dataset", "row_id", "qT", "target_used", "sigma_used"]].copy()
    output["fit_target"] = target_values
    output["production_prediction"] = accepted.pred_match_CS.to_numpy()
    output["control_prediction"] = prediction.cpu().numpy()
    output["control_pull"] = (output.control_prediction - output.target_used) / output.sigma_used
    target = args.output_root.resolve() / args.tag
    target.mkdir(parents=True, exist_ok=True)
    output.to_csv(target / "accepted_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(target / "loss_history.csv", index=False)
    pd.DataFrame({"dataset": datasets, "norm_width": norm_width,
                  "production_norm": norm_start, "control_norm": norms.cpu().numpy()}).to_csv(
                      target / "dataset_norms.csv", index=False)
    torch.save({f"np_factor.{k}": v.detach().cpu() for k, v in model.state_dict().items()}, target / "model_state.pt")
    grid_b = torch.linspace(0.0001, 8.0, 321, dtype=work_dtype, device=device)
    with torch.no_grad():
        grid_values = model(regularizer_x, grid_b).cpu().numpy()
    fnp_grid = pd.DataFrame({"x": np.repeat(FNP_GRID_X, len(grid_b)),
                             "bT": np.tile(grid_b.cpu().numpy(), len(FNP_GRID_X)),
                             "F_NP": grid_values.ravel()})
    fnp_grid.to_csv(target / "fnp_grid.csv", index=False)
    status = {
        "status": "experimental_production_objective_stability_control_not_production",
        "source_production": str(source_production),
        "w_grid": str(w_grid_path),
        "production_state_modified": False,
        "seed": args.seed, "initial_relative_parameter_perturbation": args.initial_perturbation,
        "initial_state": str(args.initial_state) if args.initial_state is not None else None,
        "replica_seed": args.replica_seed,
        "learning_rate": learning_rate,
        "lbfgs": {
            "max_iter": args.lbfgs_max_iter,
            "closure_evaluations": lbfgs_closures,
            "start_objective_per_row": lbfgs_start,
        },
        "row_count": n_rows, "epochs_run": epoch, "max_epochs": args.max_epochs,
        "best_epoch": int(best["epoch"]), "stopped_on_plateau": stopped,
        "convergence_gate_pass": stopped,
        "objective": (
            "weighted_accepted_data_chi2_plus_dataset_normalization_penalty_"
            "plus_explicitly_declared_regularizers"),
        "regularization": {
            "likelihood_weight": {
                "value": args.likelihood_weight,
                "definition": (
                    "Lagrange weight multiplying data chi2 plus floated-"
                    "normalization penalty; unweighted fit statistics are "
                    "recorded separately")
            },
            "ratecurv": {"lambda": args.lambda_fnp_ratecurv,
                          "b_min": args.fnp_ratecurv_bmin, "b_max": args.fnp_ratecurv_bmax},
            "a_slope": {"lambda": args.lambda_fnp_a_slope,
                         "b_min": args.fnp_a_slope_bmin, "b_max": args.fnp_a_slope_bmax},
            "logf_curvature": {
                "lambda": args.lambda_fnp_logcurv,
                "b_min": args.fnp_logcurv_bmin,
                "b_max": args.fnp_logcurv_bmax,
                "definition": "mean squared second derivative of log(FNP), dimensionlessly scaled"
            },
            "logf_arc_length": {
                "lambda": args.lambda_fnp_loglength,
                "b_min": args.fnp_loglength_bmin,
                "b_max": args.fnp_loglength_bmax,
                "space": args.fnp_length_space,
                "definition": (
                    "mean excess dimensionless arc length of "
                    + ("log(FNP)" if args.fnp_length_space == "logF" else "FNP"))
            },
            "fnp_slope_energy": {
                "lambda": args.lambda_fnp_f_slope,
                "b_min": args.fnp_f_slope_bmin,
                "b_max": args.fnp_f_slope_bmax,
                "definition": (
                    "mean squared first derivative of FNP, scaled by the "
                    "declared b interval without a fitted-function amplitude scale")
            },
            "logx_curvature": {
                "lambda": args.lambda_fnp_logx_curv,
                "b_min": args.fnp_logx_curv_bmin,
                "b_max": args.fnp_logx_curv_bmax,
                "definition": (
                    "mean squared second derivative of log(FNP) with "
                    "respect to log(x), dimensionlessly scaled")
            },
            "fnp_x_slope_energy": {
                "lambda": args.lambda_fnp_x_slope,
                "b_min": args.fnp_x_slope_bmin,
                "b_max": args.fnp_x_slope_bmax,
                "definition": (
                    "mean squared first derivative of FNP with respect to "
                    "log(x), scaled by the declared log-x interval")
            },
            "fnp_moment_anchor": {
                "lambda": args.lambda_fnp_moment_anchor,
                "target_csv": (
                    str(args.fnp_moment_anchor_csv)
                    if args.fnp_moment_anchor_csv is not None else None),
                "b_min": args.fnp_moment_bmin,
                "b_max": args.fnp_moment_bmax,
                "definition": (
                    "mean squared relative displacement of the normalized "
                    "FNP b-integral from the empirical independent-start "
                    "median")
            },
            "fnp_reference_distance": {
                "lambda": args.lambda_fnp_reference_distance,
                "target_csv": (
                    str(args.fnp_reference_distance_csv)
                    if args.fnp_reference_distance_csv is not None else None),
                "b_min": args.fnp_reference_distance_bmin,
                "b_max": args.fnp_reference_distance_bmax,
                "definition": (
                    "mean squared relative distance in direct FNP from a "
                    "declared reference curve")
            },
            "fnp_shortest_path": {
                "lambda": args.lambda_fnp_shortest_path,
                "target_csv": (
                    str(args.fnp_shortest_path_csv)
                    if args.fnp_shortest_path_csv is not None else None),
                "metric": args.fnp_shortest_path_metric,
                "b_min": args.fnp_shortest_path_bmin,
                "b_max": args.fnp_shortest_path_bmax,
                "definition": (
                    "quadrature-weighted mean squared residual from the analytic "
                    "fixed-endpoint shortest path, normalized by endpoint span")
            },
            "endpoint_constrained_reference": {
                "enabled": args.endpoint_constrained_reference_csv is not None,
                "target_csv": (
                    str(args.endpoint_constrained_reference_csv)
                    if args.endpoint_constrained_reference_csv is not None else None),
                "metric": args.endpoint_constrained_metric,
                "b_min": args.endpoint_constrained_bmin,
                "b_max": args.endpoint_constrained_bmax,
                "definition": (
                    "hard endpoint values with a C1 quartic-bump residual around "
                    "the unique direct-F or log-F shortest path"),
            },
            "transform_closure": {
                "lambda": args.lambda_fnp_transform_closure,
                "b_min": args.fnp_transform_closure_bmin,
                "maximum_fnp": args.fnp_transform_closure_max,
                "definition": (
                    "mean squared positive excess of log(FNP) above the "
                    "declared numerical-closure threshold")
            },
            "fit_quality_barrier": {
                "lambda": args.lambda_fit_quality_barrier,
                "ceiling_total_chi2": args.fit_quality_ceiling_total_chi2,
                "power": args.fit_quality_barrier_power,
                "definition": (
                    f"power-{args.fit_quality_barrier_power} one-sided "
                    "barrier above the declared "
                    "unpenalized total-chi2 ceiling")
            },
            "x_grid": FNP_GRID_X},
        "point_profile": {
            "enabled": profile_enabled,
            "x": args.profile_x,
            "bT": args.profile_b,
            "target_log_fnp": args.profile_logf_target,
            "target_fnp": float(np.exp(args.profile_logf_target)) if profile_enabled else None,
            "lambda_per_row": args.profile_lambda,
            "localized_shift_coordinate": args.profile_local_shift,
            "sigma_logx": args.profile_sigma_logx if args.profile_local_shift else None,
            "sigma_bT": args.profile_sigma_b if args.profile_local_shift else None,
            "achieved_fnp": (
                float(model(profile_x, profile_b).detach().cpu().squeeze())
                if profile_enabled else None),
        },
        "model_constraint": {
            "kind": (
                "endpoint_constrained_logF_C1_reference"
                if args.endpoint_constrained_reference_csv is not None else
                "reduced_logx_knot_constant_A_tail" if args.reduced_tail else
                "c1_matched_logF_positive_quadratic_tail" if args.c1_log_tail else
                "c2_remote_tail_closure_coordinate"
                if args.closure_tail_coordinate else
                f"global_positive_A_{args.global_reduced_components}_component_logx_knots"
                if args.global_reduced_components else
                "global_positive_A_logx_b_bilinear_spline" if args.global_spline else
                "global_monotone_logF_logx_b_bilinear_spline"
                if args.global_monotone_logf_spline else
                "empirical_admissible_logF_PCA"
                if args.empirical_logf_pca is not None else "none"),
            "b_start": args.endpoint_constrained_bmin
                       if args.endpoint_constrained_reference_csv is not None else
                       args.reduced_tail_b_start if args.reduced_tail else
                       args.c1_log_tail_b_match if args.c1_log_tail else
                       args.closure_tail_b_start
                       if args.closure_tail_coordinate else None,
            "b_end": args.endpoint_constrained_bmax
                     if args.endpoint_constrained_reference_csv is not None else
                     args.reduced_tail_b_end if args.reduced_tail else
                     args.closure_tail_b_end
                     if args.closure_tail_coordinate else None,
            "endpoint_reference_csv": (
                str(args.endpoint_constrained_reference_csv)
                if args.endpoint_constrained_reference_csv is not None else None),
            "tail_parameter_count": (
                9 if (args.reduced_tail or args.c1_log_tail
                      or args.closure_tail_coordinate)
                else 9 * args.global_reduced_components
                if args.global_reduced_components
                else args.global_spline_nx * args.global_spline_nb
                if (args.global_spline or args.global_monotone_logf_spline)
                else args.empirical_logf_pca_rank
                if args.empirical_logf_pca is not None
                else None),
        },
        "model_complexity": {
            "np_width": int(args.np_width if args.np_width is not None else config["np_width"]),
            "np_cond_width": int(args.np_cond_width if args.np_cond_width is not None else config["np_cond_width"]),
            "np_blocks": int(args.np_blocks if args.np_blocks is not None else config["np_blocks"]),
            "global_spline_nx": (
                args.global_spline_nx
                if (args.global_spline or args.global_monotone_logf_spline)
                else None),
            "global_spline_nb": (
                args.global_spline_nb
                if (args.global_spline or args.global_monotone_logf_spline)
                else None),
            "initialized_from_accepted_production_state": not complexity_override,
            "distill_accepted_steps": args.distill_accepted_steps,
            "distill_target": (
                str(source_production / "model_state.pt")
                if args.distill_accepted_steps else None),
            "distill_b_weight": (
                f"1/(1+(b/{args.distill_b_scale})^{args.distill_b_power})"
                if args.distill_accepted_steps else None),
            "distill_logx_nodes": (
                args.distill_logx_nodes
                if args.distill_accepted_steps else None),
            "distill_prediction_steps": args.distill_prediction_steps,
            "prediction_distill_target": (
                "selected-source model predictions in experimental-sigma metric"
                if args.distill_prediction_steps else None),
        },
        "final": {"total_chi2": float(total), "data_chi2": float(data_chi2),
                  "norm_penalty": float(norm_penalty), "objective_per_row": float(total) / n_rows,
                  "unpenalized_total_chi2": float(data_chi2 + norm_penalty),
                  "weighted_likelihood_per_row_objective": float(
                      args.likelihood_weight * (data_chi2 + norm_penalty)
                      / n_rows),
                  "profile_penalty_per_row_objective": float(profile_penalty),
                  "ratecurv_penalty_per_row_objective": float(ratecurv_penalty),
                  "a_slope_penalty_per_row_objective": float(a_slope_penalty),
                  "logcurv_penalty_per_row_objective": float(logcurv_penalty),
                  "loglength_penalty_per_row_objective": float(loglength_penalty),
                  "f_slope_penalty_per_row_objective": float(f_slope_penalty),
                  "logx_curv_penalty_per_row_objective": float(logx_curv_penalty),
                  "x_slope_penalty_per_row_objective": float(x_slope_penalty),
                  "moment_anchor_penalty_per_row_objective": float(
                      moment_anchor_penalty),
                  "reference_distance_penalty_per_row_objective": float(
                      reference_distance_penalty),
                  "shortest_path_penalty_per_row_objective": float(
                      shortest_path_penalty),
                  "transform_closure_penalty_per_row_objective": float(closure_penalty),
                  "fit_quality_barrier_per_row_objective": float(fit_quality_barrier),
                  "fnp_gradient_l2_per_row_objective": fnp_grad,
                  "normalization_gradient_l2_per_row_objective": norm_grad,
                  "max_prediction_shift_over_experimental_sigma": float(np.max(np.abs(
                      output.control_prediction - output.production_prediction) / output.sigma_used))},
        "promotion_authorized": False,
    }
    (target / "fit_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
