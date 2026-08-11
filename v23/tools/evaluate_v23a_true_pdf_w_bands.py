#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from v23.tools import plot_v23a_smooth_cross_section_panels as smooth


def parse_members(text: str) -> list[int]:
    members: list[int] = []
    for piece in str(text).replace(",", " ").split():
        m = re.fullmatch(r"(\d+)-(\d+)", piece)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise SystemExit(f"bad PDF member range: {piece}")
            members.extend(range(a, b + 1))
        else:
            members.append(int(piece))
    members = sorted(dict.fromkeys(members))
    if not members:
        raise SystemExit("no PDF members requested")
    return members


def select_dense_rows(dense: pd.DataFrame, datasets: list[str], panels: list[str]) -> pd.DataFrame:
    keep = dense[dense["dataset"].astype(str).isin(datasets)].copy()
    if panels:
        masks = []
        for panel in panels:
            if ":" not in panel:
                raise SystemExit(f"panel selector must be DATASET:Q, got {panel!r}")
            ds, q_text = panel.split(":", 1)
            masks.append(
                keep["dataset"].astype(str).eq(ds)
                & np.isclose(pd.to_numeric(keep["QM"], errors="coerce").to_numpy(float), float(q_text))
            )
        if masks:
            mask = np.logical_or.reduce(masks)
            keep = keep[mask].copy()
    if keep.empty:
        raise SystemExit("selected dense row set is empty")
    keep = keep.sort_values(["dataset", "QM", "qT", "row_id"]).reset_index(drop=True)
    keep["true_pdf_row_index"] = np.arange(len(keep), dtype=int)
    return keep


def evaluate_pdf_member(
    *,
    trainer,
    backend,
    config: dict,
    cfg,
    dense: pd.DataFrame,
    state_path: Path,
    run: Path,
    member: int,
    device: torch.device,
    dtype: torch.dtype,
    progress: bool,
) -> np.ndarray:
    pdf = backend.LHAPDFProvider(str(config.get("pdf_set", "NNPDF40_nnlo_as_01180")), int(member), use_toy_pdf=False)
    b_grid, w_matrix, _ = backend.compute_backend_grids(dense, pdf, cfg, progress=progress)
    kernel = trainer.precompute_kernel_matrix(dense["qT"].to_numpy(float), b_grid, w_matrix, dtype=dtype)
    raw = smooth.state_prediction(trainer, config, dense, b_grid, kernel, state_path, device, dtype)
    norm = smooth.norm_factors_for_run(run, dense)
    prefactor_scale = pd.to_numeric(dense.get("smooth_prefactor_scale", 1.0), errors="coerce").fillna(1.0).to_numpy(float)
    return raw * norm * prefactor_scale


def attach_true_pdf_bands(
    *,
    trainer,
    backend,
    config: dict,
    cfg,
    dense: pd.DataFrame,
    run: Path,
    members: list[int],
    device: torch.device,
    dtype: torch.dtype,
    progress: bool,
    member_cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = []
    records = []
    state_path = run / "model_state.pt"
    member_cache_dir.mkdir(parents=True, exist_ok=True)
    for i, member in enumerate(members, start=1):
        cache_path = member_cache_dir / f"member_{int(member):04d}.npz"
        if cache_path.exists():
            loaded = np.load(cache_path, allow_pickle=False)
            pred = np.asarray(loaded["pred"], dtype=float)
            if pred.shape != (len(dense),):
                raise RuntimeError(f"cached prediction shape mismatch for {cache_path}: {pred.shape} vs {(len(dense),)}")
            print(f"  true W(b) PDF member {i}/{len(members)}: {member} (cached)", flush=True)
        else:
            pred = evaluate_pdf_member(
                trainer=trainer,
                backend=backend,
                config=config,
                cfg=cfg,
                dense=dense,
                state_path=state_path,
                run=run,
                member=member,
                device=device,
                dtype=dtype,
                progress=progress,
            )
            np.savez_compressed(cache_path, pred=pred)
            print(f"  true W(b) PDF member {i}/{len(members)}: {member}", flush=True)
        values.append(pred)
        rel = pred / np.maximum(np.abs(dense["pred_smooth_CS"].to_numpy(float)), 1.0e-300) - 1.0
        records.append(
            {
                "pdf_member": int(member),
                "n_points": int(len(pred)),
                "all_finite": bool(np.isfinite(pred).all()),
                "median_rel_shift_vs_central_curve": float(np.nanmedian(rel)),
                "p16_rel_shift_vs_central_curve": float(np.nanquantile(rel, 0.16)),
                "p84_rel_shift_vs_central_curve": float(np.nanquantile(rel, 0.84)),
            }
        )
    arr = np.vstack(values)
    dense = dense.copy()
    dense["pred_pdf_truew_q16"] = np.nanquantile(arr, 0.16, axis=0)
    dense["pred_pdf_truew_q50"] = np.nanquantile(arr, 0.50, axis=0)
    dense["pred_pdf_truew_q84"] = np.nanquantile(arr, 0.84, axis=0)

    c = dense["pred_smooth_CS"].to_numpy(float)
    tmd_lo = dense["pred_smooth_replica_q16"].to_numpy(float)
    tmd_hi = dense["pred_smooth_replica_q84"].to_numpy(float)
    pdf_lo = dense["pred_pdf_truew_q16"].to_numpy(float)
    pdf_hi = dense["pred_pdf_truew_q84"].to_numpy(float)
    down = np.sqrt(np.maximum(c - tmd_lo, 0.0) ** 2 + np.maximum(c - pdf_lo, 0.0) ** 2)
    up = np.sqrt(np.maximum(tmd_hi - c, 0.0) ** 2 + np.maximum(pdf_hi - c, 0.0) ** 2)
    dense["pred_total_truew_q16"] = c - down
    dense["pred_total_truew_q84"] = c + up
    return dense, pd.DataFrame(records)


def label_for_group(g: pd.DataFrame) -> str:
    row = g.iloc[0]
    bits = [str(row["dataset"])]
    if np.isfinite(float(row.get("QM", np.nan))):
        bits.append(rf"$Q={float(row['QM']):g}$")
    if np.isfinite(float(row.get("xF", np.nan))):
        bits.append(rf"$x_F={float(row['xF']):g}$")
    elif np.isfinite(float(row.get("y_Low", np.nan))) and np.isfinite(float(row.get("y_High", np.nan))):
        bits.append(rf"$y\in[{float(row['y_Low']):g},{float(row['y_High']):g}]$")
    return "\n".join(bits)


def plot_dataset(central: pd.DataFrame, dense: pd.DataFrame, dataset: str, out_dir: Path) -> dict:
    groups = smooth.make_groups(central, [dataset])
    n = len(groups)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(4.8, 3.3 * ncols), max(3.4, 2.65 * nrows)), dpi=190, squeeze=False)
    data_vals = pd.to_numeric(central.loc[central["dataset"].astype(str).eq(dataset), "CS"], errors="coerce")
    data_vals = data_vals[np.isfinite(data_vals) & (data_vals > 0)]
    yscale = "log" if len(data_vals) and data_vals.max() / max(data_vals.min(), 1.0e-300) > 80 else "linear"

    for ax, (_, _, g) in zip(axes.ravel(), groups):
        g = g.sort_values("qT")
        dsub = dense[dense["dataset"].astype(str).eq(dataset)].copy()
        if dataset.startswith(("E288", "E605", "E772")):
            dsub = dsub[np.isclose(pd.to_numeric(dsub["QM"], errors="coerce").to_numpy(float), float(g["QM"].iloc[0]))]
        dsub = dsub.sort_values("qT")
        if dsub.empty:
            ax.axis("off")
            continue
        ax.fill_between(
            dsub["qT"].to_numpy(float),
            dsub["pred_total_truew_q16"].to_numpy(float),
            dsub["pred_total_truew_q84"].to_numpy(float),
            color="#2b7bbb",
            alpha=0.20,
            linewidth=0,
        )
        ax.fill_between(
            dsub["qT"].to_numpy(float),
            dsub["pred_pdf_truew_q16"].to_numpy(float),
            dsub["pred_pdf_truew_q84"].to_numpy(float),
            color="#d95f02",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(dsub["qT"], dsub["pred_smooth_CS"], color="#0f6aa8", lw=1.8)
        ax.errorbar(
            g["qT"],
            g["CS"],
            yerr=g["sigma_used"],
            fmt="o",
            ms=3.0,
            mfc="white",
            mec="black",
            mew=0.75,
            ecolor="black",
            elinewidth=0.75,
            capsize=1.4,
        )
        ax.text(0.06, 0.91, label_for_group(g), transform=ax.transAxes, va="top", ha="left", fontsize=8.0)
        ax.set_yscale(yscale)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if yscale == "log":
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in", top=True, right=True, length=4.0, width=0.85, labelsize=8.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.3, width=0.65)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(r"$q_T$ [GeV]", fontsize=10)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(r"$d\sigma$ (table units)", fontsize=10)
    handles = [
        Line2D([0], [0], color="#0f6aa8", lw=1.9, label="central fit"),
        Line2D([0], [0], color="#d95f02", lw=7, alpha=0.16, label="68% PDF through W"),
        Line2D([0], [0], color="#2b7bbb", lw=7, alpha=0.20, label="68% TMD+PDF"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", mfc="white", label="data"),
    ]
    fig.suptitle(dataset, x=0.02, y=0.99, ha="left", va="top", fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.985), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{dataset}_true_pdf_w_bands.png"
    pdf = out_dir / f"{dataset}_true_pdf_w_bands.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"dataset": dataset, "png": str(png), "pdf": str(pdf), "n_panels": int(n)}


def summarize(dense: pd.DataFrame) -> dict:
    out = {}
    for ds, g in dense.groupby("dataset", sort=True):
        c = g["pred_smooth_CS"].to_numpy(float)
        pdf = (g["pred_pdf_truew_q84"].to_numpy(float) - g["pred_pdf_truew_q16"].to_numpy(float)) / (2 * np.maximum(np.abs(c), 1.0e-300))
        tmd = (g["pred_smooth_replica_q84"].to_numpy(float) - g["pred_smooth_replica_q16"].to_numpy(float)) / (2 * np.maximum(np.abs(c), 1.0e-300))
        total = (g["pred_total_truew_q84"].to_numpy(float) - g["pred_total_truew_q16"].to_numpy(float)) / (2 * np.maximum(np.abs(c), 1.0e-300))
        out[str(ds)] = {
            "n_dense_points": int(len(g)),
            "median_pdf_truew_rel_halfwidth": float(np.nanmedian(pdf)),
            "max_pdf_truew_rel_halfwidth": float(np.nanmax(pdf)),
            "median_tmd_rel_halfwidth": float(np.nanmedian(tmd)),
            "max_tmd_rel_halfwidth": float(np.nanmax(tmd)),
            "median_total_rel_halfwidth": float(np.nanmedian(total)),
            "max_total_rel_halfwidth": float(np.nanmax(total)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--trainer-script", default="v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_replica_stable.py")
    ap.add_argument("--central-predictions", required=True)
    ap.add_argument("--smooth-predictions", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--panels", nargs="*", default=[], help="Optional fixed-target panel selectors like E605:15.75")
    ap.add_argument("--pdf-members", default="1-50")
    ap.add_argument("--pdf-set", default=None, help="Override the fit PDF set for uncertainty diagnostics.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    })
    run = Path(args.run)
    metrics = json.loads((run / "metrics.json").read_text())
    config = metrics["config"]
    if args.pdf_set:
        config = dict(config)
        config["pdf_set"] = str(args.pdf_set)
    dtype = torch.float32 if str(config.get("dtype", "float32")) == "float32" else torch.float64
    device = torch.device(args.device)
    trainer = smooth.import_from_path(Path(args.trainer_script), "v23_true_pdf_w_trainer")
    backend = smooth.import_from_path(Path(config["backend_script"]), "v23_true_pdf_w_backend")
    cfg = smooth.backend_cfg_from_config(backend, config)
    central = pd.read_csv(args.central_predictions)
    smooth_dense = pd.read_csv(args.smooth_predictions)
    dense = select_dense_rows(smooth_dense, [str(d) for d in args.datasets], [str(p) for p in args.panels])
    members = parse_members(args.pdf_members)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dense, member_summary = attach_true_pdf_bands(
        trainer=trainer,
        backend=backend,
        config=config,
        cfg=cfg,
        dense=dense,
        run=run,
        members=members,
        device=device,
        dtype=dtype,
        progress=not bool(args.no_progress),
        member_cache_dir=out / "member_predictions",
    )

    dense.to_csv(out / "v23a_true_pdf_w_predictions.csv", index=False)
    member_summary.to_csv(out / "v23a_true_pdf_w_member_summary.csv", index=False)
    plot_summaries = [plot_dataset(central, dense, str(ds), out) for ds in args.datasets if str(ds) in set(dense["dataset"].astype(str))]
    manifest = {
        "run": str(run),
        "central_predictions": str(args.central_predictions),
        "smooth_predictions_input": str(args.smooth_predictions),
        "pdf_set": str(config.get("pdf_set", "NNPDF40_nnlo_as_01180")),
        "pdf_members": members,
        "datasets": [str(d) for d in args.datasets],
        "panels": [str(p) for p in args.panels],
        "plots": plot_summaries,
        "relative_band_summary": summarize(dense),
        "definition": (
            "PDF uncertainty is computed by recomputing the dense W(b) backend grids for each PDF member, "
            "then evaluating the fixed central fitted F_NP model and central profiled dataset normalizations. "
            "This propagates PDFs through W(b), but it is not a PDF-member refit."
        ),
        "total_band_definition": "TMD-shape replica q16/q84 and true-W PDF q16/q84 are combined in quadrature around the central smooth curve.",
    }
    (out / "v23a_true_pdf_w_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
