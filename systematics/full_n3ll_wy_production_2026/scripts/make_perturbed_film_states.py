#!/usr/bin/env python3
"""Create isolated seeded FiLM warm-start states for start-sensitivity tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--perturbation", type=float, default=0.01)
    args = ap.parse_args()
    state = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("source must be a state dictionary")
    args.target.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(int(seed))
        out = {}
        for key, value in state.items():
            tensor = value.detach().clone()
            if key in {"b", "kernel_matrix"} or not torch.is_floating_point(tensor):
                out[key] = tensor
                continue
            scale = torch.sqrt(torch.mean(tensor.square())).clamp_min(1.0e-6)
            out[key] = tensor + float(args.perturbation) * scale * torch.randn_like(tensor)
        path = args.target / f"state_s{seed}.pt"
        torch.save(out, path)
        print(path)
    manifest = args.target / "manifest.json"
    manifest.write_text(
        "{\n"
        f"  \"source\": \"{args.source}\",\n"
        f"  \"perturbation\": {float(args.perturbation)},\n"
        f"  \"seeds\": {list(map(int, args.seeds))},\n"
        "  \"frozen_production_modified\": false\n"
        "}\n"
    )


if __name__ == "__main__":
    main()
