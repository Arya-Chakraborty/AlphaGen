from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from molecule_gen.descriptors.names import (
    WARN_NAMES,
    is_detour_descriptor,
    resolve_descriptor_name,
)


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class DescriptorRange:
    name: str
    minimum: float
    maximum: float
    source_name: str

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise SpecError(
                f"{self.name}: min ({self.minimum}) is greater than max ({self.maximum})"
            )


@dataclass(frozen=True)
class ConstraintSpec:
    n: int
    ranges: tuple[DescriptorRange, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def names(self) -> list[str]:
        return [item.name for item in self.ranges]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "descriptors": {
                item.name: {"min": item.minimum, "max": item.maximum}
                for item in self.ranges
            },
        }


def load_spec(path: str | Path) -> ConstraintSpec:
    spec_path = Path(path)
    if not spec_path.exists():
        raise SpecError(f"Spec file not found: {spec_path}")
    try:
        payload = json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        raise SpecError(f"Invalid JSON in {spec_path}: {exc}") from exc
    return parse_spec(payload)


def parse_spec(payload: dict[str, Any]) -> ConstraintSpec:
    if not isinstance(payload, dict):
        raise SpecError("Spec must be a JSON object")
    n = payload.get("n", 10)
    try:
        n_int = int(n)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"n must be an integer, got {n!r}") from exc
    if n_int < 1:
        raise SpecError("n must be >= 1")

    raw = payload.get("descriptors")
    if not isinstance(raw, dict) or not raw:
        raise SpecError("Spec must have a non-empty 'descriptors' object")

    ranges: list[DescriptorRange] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for source_name, bounds in raw.items():
        local = _resolve_allowed(source_name)
        if local in seen:
            raise SpecError(f"Duplicate descriptor after alias mapping: {local}")
        seen.add(local)
        minimum, maximum = _parse_bounds(source_name, bounds)
        ranges.append(
            DescriptorRange(
                name=local,
                minimum=minimum,
                maximum=maximum,
                source_name=str(source_name),
            )
        )
        if local in WARN_NAMES:
            warnings.append(
                f"{local} disagrees slightly between Papyrus and local Mordred; "
                "treat ranges as approximate"
            )
    return ConstraintSpec(n=n_int, ranges=tuple(ranges), warnings=tuple(warnings))


def _resolve_allowed(name: str) -> str:
    try:
        local = resolve_descriptor_name(name)
    except KeyError as exc:
        raise SpecError(str(exc)) from exc
    if is_detour_descriptor(local):
        raise SpecError(
            f"{name} is a detour-matrix descriptor; Papyrus and local Mordred "
            "do not agree on these. Pick a different descriptor."
        )
    return local


def _parse_bounds(name: str, bounds: Any) -> tuple[float, float]:
    if not isinstance(bounds, dict):
        raise SpecError(f"{name}: range must be an object with min and max")
    if "min" not in bounds or "max" not in bounds:
        raise SpecError(f"{name}: range needs both 'min' and 'max'")
    try:
        minimum = float(bounds["min"])
        maximum = float(bounds["max"])
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{name}: min/max must be numbers") from exc
    if minimum > maximum:
        raise SpecError(f"{name}: min ({minimum}) is greater than max ({maximum})")
    return minimum, maximum
