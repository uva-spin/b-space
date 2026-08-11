#!/usr/bin/env python3
"""Render standardized PRD Figs. 2, 6, 7, and 8 for the active 96-start package.

This is a read-only rendering step.  It uses the active 96-start x 50-replica
crossed bands for Figs. 2 and 6 and reconstructs the corresponding x-dependent
surfaces for Figs. 7 and 8 from the same 96 terminal model states and the
frozen 50 experimental replica ensemble.  Outputs are written to a new
Downloads directory; no production source is modified.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import j0
import torch


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "systematics/dataset_identifiability_campaign_2026"
PACKAGE = CAMPAIGN / "production_lambda1_empirical_reference_full96x50"
CROSSED = CAMPAIGN / (
    "summaries/matched_baseline_reference_distance_lam1e00_"
    "full96_crossed_experimental"
)
ENSEMBLE = CAMPAIGN / "summaries/lambda1_start_expansion96_bspace/bspace_tmd_ensemble_long.csv"
REFERENCE = ROOT / (
    "systematics/collins_factorization_validity/plots/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_central_exactx/"
    "v22_scheme_tmd_bspace_long.csv"
)
REPLICAS = ROOT / (
    "systematics/collins_factorization_validity/replicas/"
    "rowidfix_stageFT_all_qmax0p20_lam0p50_lambda3_50rep/"
    "tmd_bspace_bands_exactx_50rep/v22_tmd_replica_bspace_long.csv"
)
TRAINER = ROOT / "v21_smoothedA_tail_candidate/train_bt_dnn_v21_smoothedA_tail.py"
TRANSFORM = ROOT / "construct_v23a_regularized_kspace_tmd_v2.py"
X_EXACT = np.array([0.1, 0.2, 0.3, 0.5])
FLAVORS = ("u", "d", "s", "ubar", "dbar", "sbar")
COLORS = {
    "u": "#1f77b4", "d": "#ff7f0e", "s": "#2ca02c",
    "ubar": "#d62728", "dbar": "#9467bd", "sbar": "#8c564b",
}
STYLES = {"u": "-", "d": "--", "s": "--", "ubar": "-.",
          "dbar": ":", "sbar": ":"}
LABELS = {"u": r"$u$", "d": r"$d$", "s": r"$s$",
          "ubar": r"$\bar u$", "dbar": r"$\bar d$", "sbar": r"$\bar s$"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_for_state(trainer, state_path: Path):
    model = trainer.FilmNPFactor(
        width=48, cond_width=32, n_blocks=3, a0=.05, min_a=0.,
        a_mode="positive", exponent_clip=40., shape_mode="monotone",
        a_smooth_sigma=.45, a_tail_amp=.08, a_tail_b0=3.5,
        a_tail_width=.25, dtype=torch.float32)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("np_factor."): value for key, value in state.items()
             if key.startswith("np_factor.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def transform_curves(curves: np.ndarray, b_in: np.ndarray, transform):
    b = np.linspace(0., 24., 6001)
    k = np.linspace(0., 3., 181)
    window = transform.taper_window(b, .92)
    weights = b * window * transform.trapezoid_weights_uniform(b) / (2 * np.pi)
    J = j0(np.outer(k, b))
    result = []
    for curve in curves:
        extended = transform.extend_curve(b_in, curve, b, "expb2", None, 1e-300)
        result.append(J @ (extended * weights))
    return k, np.asarray(result)


def configure_plot(font_size=13):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": font_size,
        "axes.linewidth": 1.15, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })


def render_fig2(out: Path, reference: pd.DataFrame, bands: pd.DataFrame):
    crossed = bands[bands.flavor.astype(str).eq("u")].sort_values("bT")
    q10 = reference[
        (reference.flavor.astype(str).eq("u")) & np.isclose(reference.x, .1)
        & np.isclose(reference.Q, 10.)
    ].sort_values("bT")
    q10_map = {round(float(row.bT), 10): float(row.ftilde_no_np)
               for row in q10.itertuples(index=False)}
    fnp = {
        round(float(row.bT), 10): tuple(float(getattr(row, name)) / q10_map[round(float(row.bT), 10)]
                                        for name in ("q16", "central", "q84"))
        for row in crossed.itertuples(index=False)
    }
    rows = []
    for flavor in FLAVORS:
        view = reference[(reference.flavor.astype(str).eq(flavor))
                         & np.isclose(reference.Q, 7.5)
                         & np.isclose(reference.x, .1)].sort_values("bT")
        for row in view.itertuples(index=False):
            lo, med, hi = fnp[round(float(row.bT), 10)]
            p = float(row.ftilde_no_np)
            rows.append({"flavor": flavor, "bT": float(row.bT),
                         "q16": p * lo, "median": p * med, "q84": p * hi})
    table = pd.DataFrame(rows)
    table.to_csv(out / "fig2_lambda1_full96_combined_six_flavor_bands.csv", index=False)
    configure_plot(13)
    fig, ax = plt.subplots(figsize=(7.1, 4.35), constrained_layout=True)
    for flavor in FLAVORS:
        g = table[(table.flavor == flavor) & (table.bT <= 4.)]
        ax.fill_between(g.bT, g["q16"], g["q84"], color=COLORS[flavor], alpha=.18,
                        linewidth=0, zorder=1)
        ax.plot(g.bT, g["median"], color=COLORS[flavor], ls=STYLES[flavor], lw=2.35,
                zorder=2)
    handles = [Line2D([0], [0], color=COLORS[f], ls=STYLES[f], lw=2.35,
                       label=LABELS[f]) for f in FLAVORS]
    handles.append(Patch(facecolor=".45", edgecolor="none", alpha=.18,
                          label=r"combined central 68% interval"))
    ax.legend(handles=handles, ncol=4, frameon=False, loc="upper right",
              fontsize=11.5, columnspacing=1.15, handlelength=2.35,
              handletextpad=.55)
    ax.set_title(r"$x=0.1,\quad Q=7.5\ \mathrm{GeV}$", loc="left", fontsize=15, pad=7)
    ax.set_xlim(0., 4.); ax.set_ylim(bottom=0.)
    ax.set_xlabel(r"$b_T\ [\mathrm{GeV}^{-1}]$", fontsize=17)
    ax.set_ylabel(r"$\widetilde f_1^{q}(x,b_T;Q)$", fontsize=17)
    ax.tick_params(which="major", labelsize=14, length=6, width=1.15)
    ax.tick_params(which="minor", length=3.5, width=.9); ax.minorticks_on()
    fig.savefig(out / "fig2_lambda1_full96_combined_six_flavor.pdf")
    fig.savefig(out / "fig2_lambda1_full96_combined_six_flavor.png", dpi=300)
    plt.close(fig)


def render_fig6(out: Path, bands: pd.DataFrame):
    bands = bands[bands.kT <= 2.25].copy()
    bands.to_csv(out / "fig6_lambda1_full96_combined_ud_bands.csv", index=False)
    configure_plot(14)
    fig, ax = plt.subplots(figsize=(7.1, 4.45), constrained_layout=True)
    for flavor in ("u", "d"):
        g = bands[bands.flavor.astype(str).eq(flavor)].sort_values("kT")
        ax.fill_between(g.kT, g.q16, g.q84, color=COLORS[flavor], alpha=.22, linewidth=0)
        ax.plot(g.kT, g.central, color=COLORS[flavor], lw=2.5)
    handles = [Line2D([0], [0], color=COLORS["u"], lw=2.5, label=r"$u$ quark"),
               Line2D([0], [0], color=COLORS["d"], lw=2.5, label=r"$d$ quark"),
               Line2D([0], [0], color=".45", lw=8, alpha=.22,
                      label="combined central 68% interval")]
    ax.legend(handles=handles, frameon=False, fontsize=12.5, loc="upper right",
              ncol=1, handlelength=2.4)
    ax.set_title(r"$x=0.1,\quad Q=10\ \mathrm{GeV}$", loc="left", fontsize=15.5, pad=7)
    ax.set_xlim(0., 2.25); ax.set_ylim(bottom=0.)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$", fontsize=17)
    ax.set_ylabel(r"$f_1^q(x,k_T;Q)$", fontsize=17)
    ax.minorticks_on(); ax.tick_params(which="major", labelsize=14, length=6, width=1.15)
    ax.tick_params(which="minor", length=3.5, width=.9)
    fig.savefig(out / "fig6_lambda1_full96_combined_ud.pdf")
    fig.savefig(out / "fig6_lambda1_full96_combined_ud.png", dpi=300)
    plt.close(fig)


def render_surface(out: Path, flavor: str, reference: pd.DataFrame, replicas: pd.DataFrame,
                   tags: list[str], trainer, transform):
    pid = {"u": 2, "d": 1}[flavor]
    ref = reference[(reference.flavor.astype(str).eq(flavor)) & np.isclose(reference.Q, 7.5)
                    & reference.x.isin(X_EXACT)].copy()
    first = ref[np.isclose(ref.x, X_EXACT[0])].sort_values("bT")
    b = first.bT.to_numpy(float)
    perturbative = np.asarray([
        ref[np.isclose(ref.x, x)].sort_values("bT").ftilde_no_np.to_numpy(float)
        for x in X_EXACT
    ])
    x_tensor = torch.tensor(X_EXACT, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    starts = []
    for i, tag in enumerate(tags, 1):
        model = model_for_state(trainer, CAMPAIGN / "outputs" / tag / "model_state.pt")
        with torch.no_grad():
            fnp = model(x_tensor, b_tensor).numpy()
        starts.append(perturbative * fnp)
        if i % 16 == 0 or i == len(tags):
            print(f"{flavor}: evaluated {i}/{len(tags)} starts", flush=True)
    starts = np.asarray(starts)
    rep = replicas[(replicas.flavor.astype(str).eq(flavor)) & np.isclose(replicas.Q, 7.5)
                   & replicas.x.isin(X_EXACT)]
    replica_curves = np.asarray([
        [rep[(rep.seed == seed) & np.isclose(rep.x, x)].sort_values("bT").ftilde.to_numpy(float)
         for x in X_EXACT]
        for seed in sorted(rep.seed.unique())
    ])
    if starts.shape != (len(tags), 4, len(b)) or replica_curves.shape != (50, 4, len(b)):
        raise RuntimeError(f"unexpected {flavor} surface shapes: {starts.shape}, {replica_curves.shape}")
    exact = []
    for ix in range(4):
        k, start_k = transform_curves(starts[:, ix, :], b, transform)
        _, rep_k = transform_curves(replica_curves[:, ix, :], b, transform)
        crossed = (start_k[:, None, :] + (rep_k - np.median(rep_k, axis=0))[None, :, :]).reshape(
            len(tags) * 50, len(k))
        exact.append(np.quantile(crossed, [.16, .5, .84], axis=0))
    exact = np.asarray(exact)
    log_exact = np.log10(X_EXACT); log_grid = np.linspace(log_exact.min(), log_exact.max(), 41)
    x_grid = 10 ** log_grid
    surfaces = np.empty((3, len(x_grid), len(k)))
    for iq in range(3):
        for ik in range(len(k)):
            surfaces[iq, :, ik] = PchipInterpolator(log_exact, exact[:, iq, ik])(log_grid)
    q16, med, q84 = surfaces
    qlo = np.minimum(np.minimum(q16, q84), med); qhi = np.maximum(np.maximum(q16, q84), med)
    peak = np.max(np.abs(med), axis=1, keepdims=True); floor = np.maximum(.05 * peak, 1e-300)
    rel = 100 * .5 * (qhi - qlo) / np.maximum(np.abs(med), floor)
    active = np.abs(med) >= floor
    records = []
    for ix, x in enumerate(x_grid):
        for ik, kv in enumerate(k):
            records.append({"quantity": "x_ftilde", "flavor": flavor, "pid": pid, "Q": 7.5,
                            "x": x, "kT": kv, "median": med[ix, ik], "q16": qlo[ix, ik],
                            "q84": qhi[ix, ik], "relative_68_halfwidth_percent": rel[ix, ik],
                            "uncertainty_active": bool(active[ix, ik])})
    stem = f"fig{7 if flavor == 'u' else 8}_lambda1_full96_{flavor}_combined_surface"
    pd.DataFrame(records).to_csv(out / f"{stem}.csv", index=False)
    vmax = max(5., float(np.ceil(np.quantile(rel[active], .99) / 5) * 5))
    cmap = matplotlib.colormaps.get_cmap("magma"); norm = colors.Normalize(0, vmax, clip=True)
    K, LX = np.meshgrid(k, log_grid); face = cmap(norm(rel)); face[..., -1] = .96
    configure_plot(13)
    fig = plt.figure(figsize=(7.5, 5.8))
    ax = fig.add_axes([.02, .06, .70, .86], projection="3d", computed_zorder=False)
    ax.plot_surface(K, LX, med, facecolors=face, linewidth=.08,
                    edgecolor=(1, 1, 1, .16), antialiased=True, shade=True,
                    rcount=len(x_grid), ccount=len(k))
    lift = .008 * max(float(med.max() - med.min()), 1e-12)
    for xv in X_EXACT:
        ix = int(np.argmin(abs(x_grid - xv)))
        ax.plot(k, np.full_like(k, log_grid[ix]), med[ix] + lift,
                color=".22", lw=1.9, alpha=.9, zorder=20)
    zmin, zmax = float(med.min()), float(med.max()); zspan = max(zmax - zmin, 1e-12)
    ax.set_xlim(0, 3); ax.set_ylim(log_grid.max(), log_grid.min())
    ax.set_zlim(min(0., zmin) - .03 * zspan, zmax + .08 * zspan)
    ax.set_xlabel(r"$k_T\ [\mathrm{GeV}]$", labelpad=11, fontsize=16)
    ax.set_ylabel(r"$x$", labelpad=14, fontsize=16)
    ax.set_zlabel(rf"$x f_1^{{{flavor}}}(x,k_T;Q)\ [\mathrm{{GeV}}^{{-2}}]$", labelpad=2, fontsize=13)
    ax.set_yticks(np.log10(X_EXACT)); ax.set_yticklabels([rf"${x:g}$" for x in X_EXACT], fontsize=12)
    ax.tick_params(axis="x", labelsize=12); ax.tick_params(axis="z", labelsize=11)
    ax.set_box_aspect((1.35, 1, .78)); ax.view_init(elev=24, azim=-54); ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor((.7, .7, .7, .75))
    fig.suptitle(rf"${flavor}$, $Q=7.5\ \mathrm{{GeV}}$", fontsize=21, y=.965)
    cax = fig.add_axes([.805, .24, .027, .50]); sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax); ticks = np.linspace(0, vmax, 5); cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:g}%" for v in ticks]); cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Combined relative 68% half-width", fontsize=13, labelpad=11)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"flavor": flavor, "start_count": len(tags), "experimental_replica_count": 50,
            "combined_members_per_x": len(tags) * 50, "color_vmax_percent": vmax,
            "active_p90_percent": float(np.quantile(rel[active], .9))}


def main():
    parser = argparse.ArgumentParser()
    default_out = Path.home() / "Downloads" / "prd_lambda1_full96x50_figures_2026-08-11"
    parser.add_argument("--out-dir", type=Path, default=default_out)
    args = parser.parse_args(); out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE)
    bspace_bands = pd.read_csv(CROSSED / "bspace_combined_bands.csv")
    kspace_bands = pd.read_csv(CROSSED / "kspace_combined_bands.csv")
    ensemble_tags = sorted(pd.read_csv(ENSEMBLE, usecols=["run_tag"]).run_tag.unique())
    if len(ensemble_tags) != 96:
        raise RuntimeError(f"expected 96 terminal tags, found {len(ensemble_tags)}")
    render_fig2(out, reference, bspace_bands)
    render_fig6(out, kspace_bands)
    trainer = load_module("prd96_trainer", TRAINER)
    transform = load_module("prd96_transform", TRANSFORM)
    replicas = pd.read_csv(REPLICAS)
    surfaces = [render_surface(out, flavor, reference, replicas, ensemble_tags, trainer, transform)
                for flavor in ("u", "d")]
    manifest = {
        "status": "complete", "production_id": "lambda1_empirical_reference_full96x50",
        "start_count": 96, "experimental_replica_count": 50,
        "combined_member_count_per_flavor": 4800,
        "figures": {"fig2": "fig2_lambda1_full96_combined_six_flavor",
                    "fig6": "fig6_lambda1_full96_combined_ud",
                    "fig7": "fig7_lambda1_full96_u_combined_surface",
                    "fig8": "fig8_lambda1_full96_d_combined_surface"},
        "interval": "pointwise operational q16--q84 ensemble band; not a calibrated confidence interval",
        "sources": {"production_package": str(PACKAGE), "crossed_bands": str(CROSSED),
                    "96_start_ensemble": str(ENSEMBLE), "reference": str(REFERENCE),
                    "experimental_replicas": str(REPLICAS)},
        "surface_diagnostics": surfaces, "production_sources_modified": False,
    }
    (out / "PRD_FIGURES_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "figures": manifest["figures"],
                      "start_count": 96, "replicas": 50}, indent=2))


if __name__ == "__main__":
    main()
