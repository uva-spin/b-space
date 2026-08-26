"""Explicit conversion of a public SIDIS table to canonical observations.

The HEPData submissions have several layouts, so this adapter requires a
caller-supplied mapping and preserves row/block provenance instead of guessing
how a table should enter a likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping

try:
    from .sidis_data import SidisTable
except ImportError:
    from sidis_data import SidisTable


def _number(value: str | float | int | None, *, label: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def parse_interval(value: str) -> tuple[float, float]:
    """Parse one number or an unambiguous positive interval."""
    text = value.strip().replace("−", "-")
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    single = re.fullmatch(number, text)
    if single is not None:
        point = float(single.group(0))
        return point, point
    interval = re.fullmatch(rf"({number})\s*(?:-|–)\s*(\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", text)
    if interval is not None:
        low, high = map(float, interval.groups())
        if math.isfinite(low) and math.isfinite(high) and low <= high:
            return low, high
    raise ValueError(f"interval is not unambiguous: {value!r}")


@dataclass(frozen=True)
class SidisColumnMap:
    value: str
    axis_columns: Mapping[str, str] = field(default_factory=dict)
    metadata_axes: Mapping[str, str] = field(default_factory=dict)
    bin_columns: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    stat_columns: tuple[str, str] | None = None
    sys_columns: tuple[str, str] | None = None
    total_columns: tuple[str, str] | None = None
    required_axes: tuple[str, ...] = ()
    observable: str = "MULT"
    target: str | None = None
    hadron: str | None = None


@dataclass(frozen=True)
class SidisObservation:
    source: str
    table: str
    row_index: int
    observable: str
    value: float
    axes: Mapping[str, float]
    bins: Mapping[str, tuple[float, float]]
    uncertainties: Mapping[str, tuple[float, float]]
    block_metadata: Mapping[str, str]
    target: str | None = None
    hadron: str | None = None


def _uncertainty_pair(row: Mapping[str, str], columns: tuple[str, str] | None, *, label: str):
    if columns is None:
        return None
    plus = _number(row.get(columns[0]), label=f"{label}+")
    minus = _number(row.get(columns[1]), label=f"{label}-")
    if plus is None or minus is None:
        raise ValueError(f"both {label} uncertainty columns are required")
    return abs(plus), abs(minus)


def canonicalize_table(table: SidisTable, mapping: SidisColumnMap, *, source: str, require_uncertainty: bool = True) -> tuple[SidisObservation, ...]:
    """Convert rows using an explicit mapping; reject unresolved ambiguity."""
    columns = set(table.columns)
    if mapping.value not in columns:
        raise KeyError(f"value column {mapping.value!r} is not in {table.path}")
    for axis, column in mapping.axis_columns.items():
        if axis in mapping.metadata_axes:
            raise ValueError(f"axis {axis!r} has both column and metadata mappings")
        if column not in columns:
            raise KeyError(f"axis column {column!r} is not in {table.path}")
    for axis in mapping.required_axes:
        if axis not in mapping.axis_columns and axis not in mapping.metadata_axes:
            raise ValueError(f"required axis {axis!r} has no explicit mapping")
    for axis, (low, high) in mapping.bin_columns.items():
        if low not in columns or high not in columns:
            raise KeyError(f"bin columns for {axis!r} are not in {table.path}")
    if len(table.row_metadata) not in (0, len(table.rows)):
        raise ValueError("row metadata and row counts disagree")
    table_name = table.metadata.get("name", table.path.name)
    observations = []
    for row_index, row in enumerate(table.rows):
        value = _number(row.get(mapping.value), label="value")
        if value is None:
            raise ValueError(f"missing value at row {row_index} in {table.path}")
        block = dict(table.row_metadata[row_index]) if table.row_metadata else {}
        axes = {}
        for axis, column in mapping.axis_columns.items():
            parsed = _number(row.get(column), label=f"{axis} axis")
            if parsed is None:
                raise ValueError(f"missing {axis} axis at row {row_index} in {table.path}")
            axes[axis] = parsed
        for axis, metadata_key in mapping.metadata_axes.items():
            raw = block.get(metadata_key)
            if raw is None:
                raise ValueError(f"missing block metadata {metadata_key!r} at row {row_index}")
            low, high = parse_interval(raw)
            axes[axis] = 0.5 * (low + high)
        if axes.get("pht2", 0.0) < 0.0:
            raise ValueError(f"negative pht2 at row {row_index} in {table.path}")
        bins = {}
        for axis, (low_column, high_column) in mapping.bin_columns.items():
            low = _number(row.get(low_column), label=f"{axis} low edge")
            high = _number(row.get(high_column), label=f"{axis} high edge")
            if low is None or high is None or low > high:
                raise ValueError(f"invalid {axis} bin at row {row_index} in {table.path}")
            bins[axis] = (low, high)
        uncertainties = {}
        for label, pair in (("stat", mapping.stat_columns), ("sys", mapping.sys_columns), ("total", mapping.total_columns)):
            parsed = _uncertainty_pair(row, pair, label=label)
            if parsed is not None:
                uncertainties[label] = parsed
        if require_uncertainty and not uncertainties:
            raise ValueError(f"no explicit uncertainty mapping at row {row_index} in {table.path}")
        observations.append(SidisObservation(source, table_name, row_index, mapping.observable, value, axes, bins, uncertainties, block, mapping.target, mapping.hadron))
    return tuple(observations)
