from __future__ import annotations

from typing import Any

from molecule_gen.constraints.spec import ConstraintSpec
from molecule_gen.data.clean import connectivity_from_inchikey


def scaffold_key(candidate: dict[str, Any]) -> str:
    """Murcko scaffold if the molecule has a ring system, else connectivity."""
    smiles = candidate.get("canonical_smiles")
    conn = candidate.get("connectivity") or connectivity_from_inchikey(
        candidate.get("inchikey")
    )
    if not smiles:
        return f"conn:{conn}" if conn else "unknown"
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"conn:{conn}" if conn else str(smiles)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    if scaffold:
        return f"murcko:{scaffold}"
    if conn:
        return f"conn:{conn}"
    return str(smiles)


def diverse_elites(
    population: list[dict[str, Any]],
    n_elite: int,
    *,
    max_per_scaffold: int = 1,
) -> list[dict[str, Any]]:
    """Best individuals with a cap on copies of the same Murcko scaffold."""
    if n_elite <= 0 or not population:
        return []
    ranked = sorted(
        population,
        key=lambda row: (0 if row.get("in_range") else 1, float(row.get("penalty", 1e9))),
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    cap = max(1, max_per_scaffold)
    while len(selected) < min(n_elite, len(ranked)):
        progressed = False
        for cand in ranked:
            if cand in selected:
                continue
            key = scaffold_key(cand)
            if counts.get(key, 0) >= cap:
                continue
            selected.append(cand)
            counts[key] = counts.get(key, 0) + 1
            progressed = True
            if len(selected) >= n_elite:
                break
        if not progressed:
            cap += 1
            if cap > n_elite:
                for cand in ranked:
                    if cand not in selected:
                        selected.append(cand)
                    if len(selected) >= n_elite:
                        break
                break
    return selected[:n_elite]


def failed_descriptor_names(candidate: dict[str, Any] | None, spec: ConstraintSpec) -> set[str]:
    if not candidate:
        return set()
    values = candidate.get("values") or {}
    failed: set[str] = set()
    for item in spec.ranges:
        value = values.get(item.name)
        if value is None:
            failed.add(item.name)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            failed.add(item.name)
            continue
        if not (item.minimum <= number <= item.maximum):
            failed.add(item.name)
    return failed
