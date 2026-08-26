"""Metadata-preserving reader and schema profiler for public SIDIS tables."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
from pathlib import Path
import re
from typing import Iterable

import numpy as np

CAMPAIGN_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SidisTable:
    path: Path
    metadata: dict[str, str]
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def read_covariance_matrix(path: str | Path, expected_size: int | None = None) -> np.ndarray:
    """Read a numeric square covariance matrix from plain text or ``.gz``."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        rows: list[list[float]] = []
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            try:
                row = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(f"non-numeric covariance token at {source}:{line_number}") from exc
            rows.append(row)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError(f"covariance matrix must be non-empty and square: {source}")
    matrix = np.asarray(rows, dtype=float)
    if expected_size is not None and matrix.shape != (expected_size, expected_size):
        raise ValueError(f"covariance shape {matrix.shape} does not match expected size {expected_size}")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"covariance matrix contains non-finite values: {source}")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError(f"covariance matrix is not symmetric: {source}")
    return matrix


def _unique_columns(columns: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for raw in columns:
        name = raw.strip() or "unnamed"
        count = seen.get(name, 0)
        result.append(name if count == 0 else f"{name}__{count + 1}")
        seen[name] = count + 1
    return tuple(result)


def read_hepdata_csv(path: str | Path) -> SidisTable:
    path = Path(path)
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#:"):
            body = line[2:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
            continue
        if not line.strip():
            continue
        candidate = next(csv.reader([line]))
        if header is None:
            # A few records contain an unprefixed prose continuation before the
            # actual CSV header.
            if len(candidate) < 2:
                continue
            header = candidate
        else:
            data_lines.append(line)
    if header is None:
        raise ValueError(f"HEPData table has no header: {path}")
    columns = _unique_columns(header)
    rows: list[dict[str, str]] = []
    for values in csv.reader(data_lines):
        if not any(value.strip() for value in values):
            continue
        padded = values + [""] * (len(columns) - len(values))
        rows.append(dict(zip(columns, padded[: len(columns)])))
    return SidisTable(path, metadata, columns, tuple(rows))


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _axis_columns(columns: Iterable[str]) -> dict[str, str | None]:
    axes: dict[str, str | None] = {"x": None, "q2": None, "y": None, "z": None, "pht": None, "pht2": None}
    for column in columns:
        token = _normalized(column)
        if axes["x"] is None and token in {"x", "xb", "xbj"}:
            axes["x"] = column
        elif axes["q2"] is None and (token == "q2" or token.startswith("q2")):
            axes["q2"] = column
        elif axes["y"] is None and token == "y":
            axes["y"] = column
        elif axes["z"] is None and token == "z":
            axes["z"] = column
        elif axes["pht2"] is None and ("pht2" in token or "phperp2" in token):
            axes["pht2"] = column
        elif axes["pht"] is None and (token in {"pht", "phperp"} or "pht" in token or "phperp" in token or token.startswith("pt")):
            axes["pht"] = column
    return axes


def profile_table(table: SidisTable) -> dict:
    axes = _axis_columns(table.columns)
    try:
        display_path = str(table.path.resolve().relative_to(CAMPAIGN_ROOT.resolve()))
    except ValueError:
        display_path = str(table.path)
    profile = {
        "path": display_path,
        "metadata": table.metadata,
        "columns": list(table.columns),
        "row_count": len(table.rows),
        "axes": axes,
        "has_statistical_columns": any("stat" in _normalized(c) for c in table.columns),
        "has_systematic_columns": any("sys" in _normalized(c) for c in table.columns),
        "has_transverse_momentum": axes["pht"] is not None or axes["pht2"] is not None,
        "warnings": [],
    }
    if axes["x"] is None:
        profile["warnings"].append("no x axis")
    if axes["z"] is None:
        profile["warnings"].append("no z axis")
    if axes["q2"] is None:
        profile["warnings"].append("no Q2 axis")
    if not profile["has_transverse_momentum"]:
        profile["warnings"].append("no transverse-momentum axis; collinear complement")
    if not profile["has_statistical_columns"]:
        profile["warnings"].append("no explicit statistical uncertainty column")
    if not profile["has_systematic_columns"]:
        profile["warnings"].append("no explicit systematic uncertainty column")
    return profile
