from __future__ import annotations

from typing import Any

import pandas as pd

from molecule_gen.constraints.lookup import describe_for_scoring
from molecule_gen.constraints.spec import ConstraintSpec, DescriptorRange, SpecError
from molecule_gen.data.clean import canonicalize_smiles, connectivity_from_inchikey


def range_penalty(value: float | None, spec: DescriptorRange) -> float:
    """0 if inside [min, max]. Scaled squared distance outside. 1.0 if missing."""
    if value is None:
        return 1.0
    if spec.minimum <= value <= spec.maximum:
        return 0.0
    width = spec.maximum - spec.minimum
    scale = width if width > 0 else 1.0
    if value < spec.minimum:
        delta = spec.minimum - value
    else:
        delta = value - spec.maximum
    return float((delta / scale) ** 2)


def score_vector(
    values: dict[str, float | None],
    spec: ConstraintSpec,
) -> dict[str, Any]:
    """Score a descriptor vector against the spec. This is the source of truth."""
    details: dict[str, Any] = {}
    total = 0.0
    all_passed = True
    for item in spec.ranges:
        value = values.get(item.name)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
        penalty = range_penalty(value, item)
        passed = penalty == 0.0 and value is not None
        all_passed = all_passed and passed
        total += penalty
        details[item.name] = {
            "min": item.minimum,
            "max": item.maximum,
            "value": value,
            "passed": passed,
            "penalty": penalty,
        }
    return {
        "in_range": all_passed,
        "penalty": total,
        "n_requested": spec.n,
        "n_descriptors": len(spec.ranges),
        "n_passed": sum(1 for row in details.values() if row["passed"]),
        "descriptors": details,
    }


def score_frame(frame: pd.DataFrame, spec: ConstraintSpec) -> pd.DataFrame:
    """Vectorized penalty / in-range flags for a table of descriptor columns."""
    missing = [item.name for item in spec.ranges if item.name not in frame.columns]
    if missing:
        raise SpecError(f"Table missing descriptor columns: {missing}")
    out = frame.copy()
    total = None
    in_range = pd.Series(True, index=frame.index)
    n_passed = pd.Series(0, index=frame.index, dtype=int)
    for item in spec.ranges:
        col = pd.to_numeric(frame[item.name], errors="coerce")
        width = item.maximum - item.minimum
        scale = width if width > 0 else 1.0
        below = ((item.minimum - col) / scale).clip(lower=0)
        above = ((col - item.maximum) / scale).clip(lower=0)
        part = below.pow(2) + above.pow(2)
        part = part.where(col.notna(), 1.0)
        ok = col.notna() & (col >= item.minimum) & (col <= item.maximum)
        total = part if total is None else total + part
        in_range &= ok
        n_passed += ok.astype(int)
    out["penalty"] = total.astype(float)
    out["in_range"] = in_range
    out["n_passed"] = n_passed
    return out


def score_smiles(smiles: str, spec: ConstraintSpec) -> dict[str, Any]:
    parsed = canonicalize_smiles(smiles)
    if parsed is None:
        raise SpecError(f"Invalid SMILES: {smiles}")
    canon, inchikey = parsed
    connectivity = connectivity_from_inchikey(inchikey)
    values, source = describe_for_scoring(
        names=spec.names(),
        smiles=canon,
        inchikey=inchikey,
        connectivity=connectivity,
    )
    result = score_vector(values, spec)
    result.update(
        {
            "canonical_smiles": canon,
            "inchikey": inchikey,
            "connectivity": connectivity,
            "source": source,
        }
    )
    return result


def score_connectivity(connectivity: str, spec: ConstraintSpec) -> dict[str, Any]:
    key = str(connectivity).strip()
    if not key:
        raise SpecError("connectivity is empty")
    values, source = describe_for_scoring(
        names=spec.names(),
        smiles=None,
        inchikey=None,
        connectivity=key,
    )
    result = score_vector(values, spec)
    result.update(
        {
            "canonical_smiles": None,
            "inchikey": None,
            "connectivity": key,
            "source": source,
        }
    )
    return result
