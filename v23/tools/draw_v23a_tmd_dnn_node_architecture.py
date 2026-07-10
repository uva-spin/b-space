#!/usr/bin/env python3
"""Clean publication-style node diagram for the v23a FiLM TMD DNN.

This is a deliberately uncluttered schematic for notes/papers. It shows the
actual neural architecture at the node/layer level while leaving detailed
hyperparameters for the caption.

Outputs PDF, PNG, and SVG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D


@dataclass
class Layer:
    name: str
    sub: str
    x: float
    y: float
    n: int
    color: str
    r: float = 0.065
    spacing: float = 0.19
    ellipsis: bool = False
    label: str = "top"

    def pos(self):
        return [(self.x, self.y + (i - (self.n - 1) / 2) * self.spacing) for i in range(self.n)]

    def top(self):
        return max(y for _, y in self.pos()) + self.r

    def bottom(self):
        return min(y for _, y in self.pos()) - self.r


def draw_layer(ax, L: Layer):
    for x, y in L.pos():
        ax.add_patch(Circle((x, y), L.r, facecolor=L.color, edgecolor="#1b1b1b", lw=0.75, zorder=3))
    if L.ellipsis:
        ax.text(L.x, L.y, r"$\vdots$", ha="center", va="center", fontsize=15, zorder=4)
    if L.label == "top":
        y = L.top() + 0.24
        ax.text(L.x, y + 0.18, L.name, ha="center", va="bottom", fontsize=9.6, weight="bold")
        ax.text(L.x, y, L.sub, ha="center", va="bottom", fontsize=8.2)
    elif L.label == "bottom":
        y = L.bottom() - 0.23
        ax.text(L.x, y - 0.01, L.name, ha="center", va="top", fontsize=9.6, weight="bold")
        ax.text(L.x, y - 0.20, L.sub, ha="center", va="top", fontsize=8.2)


def connect(ax, A: Layer, B: Layer, *, color="#bdbdbd", alpha=0.42, lw=0.45, fan=1):
    pa, pb = A.pos(), B.pos()
    nb = len(pb)
    for i, (x1, y1) in enumerate(pa):
        j0 = int(round(i * (nb - 1) / max(len(pa) - 1, 1)))
        js = sorted(set(j for j in range(j0 - fan, j0 + fan + 1) if 0 <= j < nb))
        for j in js:
            x2, y2 = pb[j]
            ax.add_line(Line2D([x1 + A.r, x2 - B.r], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1))


def arrow(ax, a, b, *, color="#222", lw=1.1, alpha=1.0, rad=0.0, ms=9, style="-|>", z=5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                 alpha=alpha, connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4, zorder=z))


def box(ax, xy, wh, text, *, fc="white", ec="#222", lw=1.0, fontsize=8.8, radius=0.035):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.016,rounding_size={radius}",
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/v23a_tmd_dnn_node_architecture_clean2.pdf")
    ap.add_argument("--title", default=r"FiLM-conditioned neural network for $F_{\rm NP}$")
    ap.add_argument("--width", type=float, default=15.2)
    ap.add_argument("--height", type=float, default=7.2)
    ap.add_argument("--dpi", type=int, default=260)
    ap.add_argument("--no-scaffold", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    blue = "#a8cff8"
    green = "#bdecc7"
    amber = "#ffd291"
    red = "#f3a2a7"
    red_box = "#fde9ea"
    green_box = "#e8f7ec"
    gray_box = "#f2f2f2"
    dark_green = "#2f7a3c"
    dark_red = "#9b3338"

    ax.text(7.6, 6.93, args.title, ha="center", va="top", fontsize=17.2, weight="bold")
    ax.text(7.6, 6.61,
            r"node schematic of the learned nonperturbative factor; the perturbative $N^3LL$ $W$ kernel is fixed outside the DNN",
            ha="center", va="top", fontsize=9.6)

    # Top branch: b -> trunk
    y_top = 4.65
    layers = [
        Layer("b inputs", "4 features", 0.90, y_top, 4, blue, spacing=0.30),
        Layer("radial lift", r"$4\to48$", 2.45, y_top, 8, amber, ellipsis=True),
        Layer("FiLM 1", "48 nodes", 4.25, y_top, 8, amber, ellipsis=True),
        Layer("FiLM 2", "48 nodes", 6.05, y_top, 8, amber, ellipsis=True),
        Layer("FiLM 3", "48 nodes", 7.85, y_top, 8, amber, ellipsis=True),
        Layer("Softplus head", r"$48\to1$", 9.45, y_top, 1, red, r=0.095),
    ]
    for L in layers:
        draw_layer(ax, L)
    for A, B in zip(layers[:-1], layers[1:]):
        connect(ax, A, B, fan=1 if B.n > 1 else 3)

    # Feature vector label, not over the edges.
    ax.text(0.90, 5.95, r"$[\,b,\ b^2,\ \sqrt{b+\epsilon},\ \ln(1+b)\,]$",
            ha="center", va="bottom", fontsize=8.8, color="#333333")

    # Bottom branch: x conditioning
    y_cond = 1.75
    x_layers = [
        Layer("x inputs", "2 features", 0.90, y_cond, 2, green, spacing=0.42, label="bottom"),
        Layer("condition 1", r"$2\to32$", 2.45, y_cond, 7, green, ellipsis=True, label="bottom"),
        Layer("condition 2", r"$32\to32$", 4.25, y_cond, 7, green, ellipsis=True, label="bottom"),
    ]
    for L in x_layers:
        draw_layer(ax, L)
    connect(ax, x_layers[0], x_layers[1], color="#8bd196", alpha=0.50, fan=2)
    connect(ax, x_layers[1], x_layers[2], color="#8bd196", alpha=0.45, fan=1)

    ax.text(0.90, 0.92, r"$[\,x,\ \mathrm{logit}(x)\,]$",
            ha="center", va="top", fontsize=8.8, color="#333333")

    # FiLM parameter generators: small, clean, no long text.
    film_y = 3.04
    for i, target in enumerate(layers[2:5], start=1):
        bx = target.x - 0.28
        box(ax, (bx, film_y), (0.56, 0.34), rf"$\gamma_{i},\beta_{i}$",
            fc="#effaf0", ec=dark_green, fontsize=8.5, lw=0.9)
        arrow(ax, (x_layers[-1].x + 0.10, x_layers[-1].y + 0.18),
              (bx + 0.28, film_y - 0.02),
              color=dark_green, lw=0.95, alpha=0.76, rad=0.12, ms=8)
        arrow(ax, (bx + 0.28, film_y + 0.34),
              (target.x, target.bottom() - 0.06),
              color=dark_green, lw=0.95, alpha=0.76, rad=0.00, ms=8)

    ax.text(6.05, 3.48, r"FiLM parameters from the $x$ branch",
            ha="center", va="center", fontsize=8.8, color=dark_green)

    # Output/scaffold right side.
    box(ax, (10.35, 4.29), (1.18, 0.72), r"$A_\theta(x,b)\geq0$" "\n" "rate",
        fc=red_box, ec=dark_red, fontsize=8.8, lw=1.0)
    arrow(ax, (layers[-1].x + 0.15, y_top), (10.35, 4.65), color=dark_red, lw=1.15, ms=10)

    if not args.no_scaffold:
        box(ax, (12.00, 4.27), (1.58, 0.76), r"$I_\theta(x,b)$" "\n" r"$=\int_0^b2b'A_\theta\,db'$",
            fc=gray_box, ec="#555555", fontsize=8.3, lw=1.0)
        box(ax, (14.00, 4.27), (0.95, 0.76), r"$F_{\rm NP}$" "\n" r"$=e^{-I_\theta}$",
            fc=green_box, ec=dark_green, fontsize=8.8, lw=1.0)
        arrow(ax, (11.53, 4.65), (12.00, 4.65), color="#333333", lw=1.1, ms=10)
        arrow(ax, (13.58, 4.65), (14.00, 4.65), color="#333333", lw=1.1, ms=10)

    # Subtle label inside trunk for residuals.
    for L in layers[2:5]:
        ax.text(L.x + 0.30, L.top() + 0.04, "+", ha="center", va="center",
                fontsize=10.5, color="#8a5a00")

    # Clean legend.
    leg_items = [
        (blue, "b features"),
        (green, "x conditioning"),
        (amber, "FiLM trunk"),
        (red, "positive head"),
        (green_box, r"$F_{\rm NP}$ scaffold"),
    ]
    x0, y0 = 0.75, 0.28
    for col, label in leg_items:
        ax.add_patch(Circle((x0, y0), 0.048, facecolor=col, edgecolor="#222", lw=0.65))
        ax.text(x0 + 0.10, y0, label, ha="left", va="center", fontsize=8.0)
        x0 += 2.25

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote: {out}")
    print(f"wrote: {out.with_suffix('.png')}")
    print(f"wrote: {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
