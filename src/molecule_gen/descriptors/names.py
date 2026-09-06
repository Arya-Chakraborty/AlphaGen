from __future__ import annotations

from functools import lru_cache

# Papyrus (older Mordred) used aliph/arom; mordredcommunity uses A/a.
RING_RENAMES = (
    ("FAHRing", "FaliphHRing"),
    ("FaHRing", "FaromHRing"),
    ("FARing", "FaliphRing"),
    ("FaRing", "FaromRing"),
    ("AHRing", "aliphHRing"),
    ("aHRing", "aromHRing"),
    ("ARing", "aliphRing"),
    ("aRing", "aromRing"),
)
DETOUR_EXACT = frozenset({"DetourIndex"})
DETOUR_SUFFIXES = ("_Dt",)
WARN_NAMES = frozenset({"nHBAcc"})
FEATURED_DESCRIPTORS = (
    "MW",
    "SLogP",
    "TopoPSA",
    "nRot",
    "nRing",
    "nHBDon",
    "nHBAcc",
    "nAtom",
)


def papyrus_name_to_local(name: str) -> str:
    """Map a Papyrus Mordred column name onto the current mordredcommunity name."""
    for local_suffix, papyrus_suffix in RING_RENAMES:
        if name.endswith(papyrus_suffix):
            return name[: -len(papyrus_suffix)] + local_suffix
    return name


def local_name_to_papyrus(name: str) -> str:
    """Map a local Mordred name onto the Papyrus table column name."""
    for local_suffix, papyrus_suffix in RING_RENAMES:
        if name.endswith(local_suffix):
            return name[: -len(local_suffix)] + papyrus_suffix
    return name


def is_detour_descriptor(name: str) -> bool:
    return name in DETOUR_EXACT or any(name.endswith(sfx) for sfx in DETOUR_SUFFIXES)


@lru_cache(maxsize=1)
def local_descriptor_names() -> frozenset[str]:
    from molecule_gen.descriptors.mordred import descriptor_names

    return frozenset(descriptor_names())


def resolve_descriptor_name(name: str) -> str:
    """Accept local or Papyrus names; return the local canonical name."""
    text = str(name).strip()
    if not text:
        raise ValueError("Descriptor name is empty")
    local = papyrus_name_to_local(text)
    known = local_descriptor_names()
    if local in known:
        return local
    if text in known:
        return text
    raise KeyError(f"Unknown Mordred 2D descriptor: {name}")


def allowed_descriptor_names() -> list[str]:
    """Mordred 2D names that may appear in a spec (detour family excluded)."""
    return sorted(name for name in local_descriptor_names() if not is_detour_descriptor(name))
