"""LHAPDF-backed collinear fragmentation-function interface.

This module deliberately supplies only the collinear FF boundary.  The
perturbative TMD matching/evolution and the FiLM nonperturbative transverse
factor are separate callables owned by the eventual SIDIS fit.  LHAPDF's
``xfxQ`` convention is converted explicitly from ``z D(z,Q)`` to ``D(z,Q)``
for fragmentation-function sets; no positivity clamp or error combination is
performed silently.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

import numpy as np


class LHAPDFFError(RuntimeError):
    """Raised when an LHAPDF FF set cannot be used safely."""


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LHAPDFFError(f"LHAPDF metadata {name!r} is not numeric: {value!r}") from exc
    if not np.isfinite(result):
        raise LHAPDFFError(f"LHAPDF metadata {name!r} is not finite: {value!r}")
    return result


class LHAPDFFMember:
    """One member of a validated LHAPDF fragmentation-function set.

    Parameters
    ----------
    set_name:
        LHAPDF set name, for example ``NNFF10_PIp_nnlo``.
    member:
        Set member index.  Member zero is the central/average member for the
        configured sets; the remaining members are kept available for the
        external FF uncertainty propagation.
    backend, pdfset:
        Private test hooks allowing unit tests to use a fake LHAPDF backend
        without requiring a globally installed grid.
    """

    def __init__(self, set_name: str, member: int = 0, *, backend=None, pdfset=None):
        if not isinstance(set_name, str) or not set_name:
            raise ValueError("set_name must be a non-empty string")
        if int(member) != member or int(member) < 0:
            raise ValueError("member must be a non-negative integer")
        self.set_name = set_name
        self.member = int(member)
        if backend is None or pdfset is None:
            try:
                import lhapdf
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise LHAPDFFError(
                    "LHAPDF Python bindings are required; install LHAPDF or provide a test backend"
                ) from exc
            try:
                pdfset = lhapdf.getPDFSet(set_name)
                backend = pdfset.mkPDF(self.member)
            except Exception as exc:  # pragma: no cover - LHAPDF error text varies
                raise LHAPDFFError(f"could not load LHAPDF FF set {set_name!r}, member {self.member}") from exc
        size = int(getattr(pdfset, "size", 0))
        if size <= 0 or self.member >= size:
            raise LHAPDFFError(f"member {self.member} is outside {set_name!r} with size {size}")
        set_type = str(pdfset.get_entry("SetType")).lower()
        if set_type != "fragfn":
            raise LHAPDFFError(f"LHAPDF set {set_name!r} has SetType={set_type!r}, not fragfn")
        self._pdf = backend
        self._pdfset = pdfset

    def metadata(self) -> dict[str, Any]:
        """Return portable metadata needed for a provenance manifest."""

        keys = (
            "SetType", "SetIndex", "DataVersion", "OrderQCD", "ErrorType",
            "NumMembers", "NumFlavors", "FlavorScheme", "XMin", "XMax",
            "QMin", "QMax", "Particle", "Flavors", "Reference", "SetDesc",
        )
        result: dict[str, Any] = {"set_name": self.set_name, "member": self.member}
        for key in keys:
            try:
                value = self._pdfset.get_entry(key)
            except Exception:
                continue
            if key in {"SetIndex", "DataVersion", "OrderQCD", "NumMembers", "NumFlavors"}:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            elif key in {"XMin", "XMax", "QMin", "QMax"}:
                value = _as_float(value, key)
            elif key == "Flavors":
                if isinstance(value, str):
                    try:
                        value = ast.literal_eval(value)
                    except (SyntaxError, ValueError) as exc:
                        raise LHAPDFFError(
                            f"LHAPDF metadata Flavors is not a list: {value!r}"
                        ) from exc
                value = [int(v) for v in value]
            result[key] = value
        result["lhapdf_error_type"] = str(getattr(self._pdfset, "errorType", "unknown"))
        return result

    def _density_scalar(self, pid: int, z: float, q: float) -> float:
        if int(pid) != pid:
            raise ValueError("pid must be an integer PDG parton id")
        if not np.isfinite(z) or not 0.0 < z < 1.0:
            raise ValueError("z must be finite and lie strictly between 0 and 1")
        if not np.isfinite(q) or q <= 0.0:
            raise ValueError("Q must be finite and positive")
        try:
            # LHAPDF's xfxQ returns x*f(x,Q), including for SetType=fragfn.
            value = float(self._pdf.xfxQ(int(pid), float(z), float(q))) / float(z)
        except Exception as exc:  # pragma: no cover - backend-specific errors
            raise LHAPDFFError(
                f"LHAPDF evaluation failed for {self.set_name!r}, pid={pid}, z={z}, Q={q}"
            ) from exc
        if not np.isfinite(value):
            raise LHAPDFFError(
                f"non-finite FF value for {self.set_name!r}, pid={pid}, z={z}, Q={q}"
            )
        return value

    def density(self, pid: int, z, q: float):
        """Evaluate ``D_pid^h(z,Q)`` after the explicit ``xfxQ/z`` conversion.

        Arrays are evaluated element-by-element because the LHAPDF Python
        binding is scalar.  Signed values are returned unchanged: fixed-order
        FFs are scheme-dependent and a positivity clamp would be an untracked
        theory modification.
        """

        values = np.asarray(z, dtype=float)
        if values.ndim == 0:
            return self._density_scalar(pid, float(values), q)
        if values.ndim != 1:
            raise ValueError("z must be a scalar or one-dimensional array")
        return np.asarray([self._density_scalar(pid, float(v), q) for v in values], dtype=float)

    def raw_xfxQ(self, pid: int, z: float, q: float) -> float:
        """Return the unconverted LHAPDF ``xfxQ`` value for diagnostics."""

        if not np.isfinite(z) or not 0.0 < z < 1.0:
            raise ValueError("z must be finite and lie strictly between 0 and 1")
        return float(self._pdf.xfxQ(int(pid), float(z), float(q)))


class LHAPDFFamily:
    """Map hadron labels to LHAPDF FF sets with one shared member index."""

    def __init__(self, hadron_sets: Mapping[str, str], member: int = 0):
        if not hadron_sets:
            raise ValueError("hadron_sets must not be empty")
        self.hadron_sets = dict(hadron_sets)
        self.member = int(member)
        self._members = {
            hadron: LHAPDFFMember(name, self.member)
            for hadron, name in self.hadron_sets.items()
        }

    def member_for(self, hadron: str) -> LHAPDFFMember:
        try:
            return self._members[hadron]
        except KeyError as exc:
            raise KeyError(f"no LHAPDF FF set configured for hadron {hadron!r}") from exc

    def density(self, hadron: str, pid: int, z, q: float):
        return self.member_for(hadron).density(pid, z, q)

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {hadron: member.metadata() for hadron, member in self._members.items()}


__all__ = ["LHAPDFFError", "LHAPDFFMember", "LHAPDFFFamily"]
