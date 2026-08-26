"""Small, dependency-light reader and schema profiler for public SIDIS tables.

HEPData CSV files carry table metadata in ``#: key: value`` comment lines and
then a deliberately human-readable header. This module preserves those
metadata lines, handles duplicate column labels and repeated target/charge
blocks, and identifies common SIDIS axes without asserting that a table is
fit-ready. It is an ingestion boundary, not a physics prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
from collections import Counter
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
    row_metadata: tuple[dict[str, str], ...] = ()


def read_covariance_matrix(path: str | Path, expected_size: int | None = None) -> np.ndarray:
    """Read a numeric square covariance matrix from plain text or ``.gz``.

    The parser accepts comma- or whitespace-separated numeric rows, ignores
    blank and ``#`` comment lines, and rejects ragged, non-finite, non-square,
    or materially non-symmetric input. It does not repair a matrix or combine
    statistical and systematic components.
    """

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
    active_header: list[str] | None = None
    active_columns: tuple[str, ...] = ()
    all_columns: list[str] = []
    active_block = "primary"
    rows: list[dict[str, str]] = []
    row_metadata: list[dict[str, str]] = []
    current_block: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#:"):
            body = line[2:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
            else:
                fields = [field.strip() for field in body.split(",")]
                key = fields[0].lower()
                value = next((field for field in reversed(fields[1:]) if field), "")
                if key == "re":
                    current_block["reaction"] = value
                elif _metadata_axis(key) == "z":
                    _store_axis_metadata(current_block, "z", value)
                elif key.startswith("e("):
                    current_block["beam_energy"] = body
                elif key.startswith("plab"):
                    current_block["beam_energy"] = value
                elif _metadata_axis(key) == "x":
                    _store_axis_metadata(current_block, "x", value)
                elif _metadata_axis(key) == "q2":
                    _store_axis_metadata(current_block, "q2", value)
                elif _metadata_axis(key) == "y":
                    _store_axis_metadata(current_block, "y", value)
            continue
        if not line.strip():
            continue
        if header is None:
            candidate = next(csv.reader([line]))
            # A few records contain an unprefixed prose continuation before the
            # actual CSV header.
            if len(candidate) < 2:
                continue
            header = candidate
            active_header = candidate
            active_columns = _unique_columns(candidate)
            all_columns.extend(active_columns)
        else:
            candidate = next(csv.reader([line]))
            if len(candidate) == len(header) and all(_normalized(left) == _normalized(right) for left, right in zip(candidate, header)):
                active_header = header
                active_columns = _unique_columns(header)
                active_block = "primary"
                continue
            if _looks_like_data_header(candidate):
                active_header = candidate
                active_columns = _unique_columns(candidate)
                active_block = "primary" if _same_header(candidate, header) else "auxiliary"
                for column in active_columns:
                    if column not in all_columns:
                        all_columns.append(column)
                continue
            if not any(value.strip() for value in candidate):
                continue
            if active_header is None:
                active_header = header
                active_columns = _unique_columns(header)
            padded = candidate + [""] * (len(active_header) - len(candidate))
            rows.append(dict(zip(active_columns, padded[: len(active_header)])))
            block = dict(current_block)
            block["data_block"] = active_block
            row_metadata.append(block)
    if header is None:
        raise ValueError(f"HEPData table has no header: {path}")
    columns = tuple(all_columns or _unique_columns(header))
    return SidisTable(path, metadata, columns, tuple(rows), tuple(row_metadata))


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _same_header(left: Iterable[str], right: Iterable[str] | None) -> bool:
    if right is None:
        return False
    left_values, right_values = tuple(left), tuple(right)
    return len(left_values) == len(right_values) and all(
        _normalized(a) == _normalized(b) for a, b in zip(left_values, right_values)
    )


def _looks_like_data_header(candidate: list[str]) -> bool:
    if len(candidate) < 2:
        return False
    try:
        float(candidate[0].strip())
    except ValueError:
        pass
    else:
        return False
    tokens = [_normalized(item) for item in candidate]
    return any("low" in token or "high" in token for token in tokens)


def _metadata_axis(key: str) -> str | None:
    token = _normalized(key)
    if token in {"x", "xb", "xbj"} or token.startswith("xbj"):
        return "x"
    if token in {"q2", "q2gev2"} or token.startswith("q2"):
        return "q2"
    if token == "z" or token.startswith("zgev"):
        return "z"
    if token == "y":
        return "y"
    return None


def _store_axis_metadata(block: dict[str, str], axis: str, value: str) -> None:
    text = value.strip()
    block[f"{axis}_value"] = text
    match = re.search(
        r"\(\s*BIN\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s+TO\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        block[f"{axis}_bin"] = f"{match.group(1)}-{match.group(2)}"
    elif text:
        block[f"{axis}_bin"] = text


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
        elif axes["pht2"] is None and not any(bound in token for bound in ("low", "high")) and (
            "pht2" in token or "phperp2" in token or token.startswith("pt2")
        ):
            axes["pht2"] = column
        elif axes["pht"] is None and not any(bound in token for bound in ("low", "high")) and (token in {"pht", "phperp"} or "pht" in token or "phperp" in token or token.startswith("pt")):
            axes["pht"] = column
    return axes


def _float_values(table: SidisTable, column: str | None) -> list[float]:
    if column is None:
        return []
    values: list[float] = []
    for row in table.rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


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
        "block_count": len({tuple(sorted(item.items())) for item in table.row_metadata}) if table.row_metadata else 0,
        "data_block_counts": dict(Counter(
            item.get("data_block", "primary") for item in table.row_metadata
        )) if table.row_metadata else {"primary": len(table.rows)},
        "axes": axes,
        "axis_ranges": {},
        "has_statistical_columns": any("stat" in _normalized(c) for c in table.columns),
        "has_systematic_columns": any("sys" in _normalized(c) for c in table.columns),
        "has_uncertainty_columns": any(any(token in _normalized(c) for token in ("stat", "sys", "error", "uncert")) for c in table.columns),
        "has_transverse_momentum": axes["pht"] is not None or axes["pht2"] is not None,
        "warnings": [],
    }
    profile["primary_row_count"] = profile["data_block_counts"].get("primary", 0)
    profile["auxiliary_row_count"] = profile["data_block_counts"].get("auxiliary", 0)
    for axis, column in axes.items():
        values = _float_values(table, column)
        if values:
            profile["axis_ranges"][axis] = {"min": min(values), "max": max(values)}
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
    if not profile["has_uncertainty_columns"]:
        profile["warnings"].append("no explicit uncertainty column")
    return profile
