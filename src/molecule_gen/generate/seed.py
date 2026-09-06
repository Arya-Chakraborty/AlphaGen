from __future__ import annotations

import sys

import pandas as pd
from tqdm import tqdm

from molecule_gen.constraints.lookup import load_papyrus_descriptor_table
from molecule_gen.constraints.score import score_frame
from molecule_gen.constraints.spec import ConstraintSpec
from molecule_gen.data.papyrus import MORDRED_ID_COL, molecules_parquet_path
from molecule_gen.generate.selfies_util import encode_roundtrip


def seed_from_papyrus(
    spec: ConstraintSpec,
    *,
    n_seeds: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Pick Papyrus molecules closest to the spec (in-range first, then lowest penalty)."""
    mol_path = molecules_parquet_path()
    if not mol_path.exists():
        raise FileNotFoundError(
            f"Molecule table not found at {mol_path}. Run fetch-papyrus first."
        )
    print(f"scoring Papyrus molecules against the spec ({', '.join(spec.names())})", file=sys.stderr)
    table = load_papyrus_descriptor_table(spec.names())
    scored = score_frame(table, spec)
    molecules = pd.read_parquet(
        mol_path, columns=[MORDRED_ID_COL, "canonical_smiles", "inchikey"]
    )
    merged = scored.merge(molecules, on=MORDRED_ID_COL, how="inner")
    in_range = merged[merged["in_range"]]
    print(
        f"{len(in_range):,} / {len(merged):,} Papyrus molecules already in range",
        file=sys.stderr,
    )
    n_take = min(n_seeds, len(merged))
    if len(in_range) >= n_take:
        picked = in_range.sample(n=n_take, random_state=seed).reset_index(drop=True)
    else:
        misses = merged[~merged["in_range"]].sort_values("penalty", kind="mergesort")
        need = n_take - len(in_range)
        picked = pd.concat(
            [in_range, misses.head(need)], ignore_index=True
        )
    return picked


def pick_near_misses(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Lowest-penalty out-of-range rows, preferring those that already pass most constraints."""
    if n <= 0 or frame.empty or "in_range" not in frame.columns:
        return frame.iloc[0:0].copy()
    misses = frame.loc[~frame["in_range"]].copy()
    if misses.empty:
        return misses
    misses = misses.sort_values(
        ["n_passed", "penalty"],
        ascending=[False, True],
        kind="mergesort",
    )
    return misses.head(n).reset_index(drop=True)


def seed_papyrus_near_misses(
    spec: ConstraintSpec,
    *,
    n_seeds: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Papyrus molecules closest to the spec that are not already in range."""
    mol_path = molecules_parquet_path()
    if not mol_path.exists():
        raise FileNotFoundError(
            f"Molecule table not found at {mol_path}. Run fetch-papyrus first."
        )
    print(
        f"picking Papyrus near-misses for ({', '.join(spec.names())})",
        file=sys.stderr,
        flush=True,
    )
    table = load_papyrus_descriptor_table(spec.names())
    scored = score_frame(table, spec)
    molecules = pd.read_parquet(
        mol_path, columns=[MORDRED_ID_COL, "canonical_smiles", "inchikey"]
    )
    merged = scored.merge(molecules, on=MORDRED_ID_COL, how="inner")
    in_range = merged[merged["in_range"]]
    print(
        f"{len(in_range):,} Papyrus already in range; using out-of-range near-misses as seeds",
        file=sys.stderr,
        flush=True,
    )
    picked = pick_near_misses(merged, max(n_seeds * 3, n_seeds))
    return picked.head(n_seeds * 3).reset_index(drop=True)


def candidates_from_seed_table(
    frame: pd.DataFrame,
    *,
    origin: str = "seed",
    require_reasonable: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Encode seed SMILES to SELFIES; drop rows that do not round-trip."""
    from molecule_gen.generate.chemistry import is_reasonable_molecule

    rows: list[dict] = []
    for rec in tqdm(
        frame.itertuples(index=False),
        total=len(frame),
        desc="SELFIES",
        unit="mol",
        file=sys.stderr,
        mininterval=0.5,
    ):
        smiles = str(getattr(rec, "canonical_smiles"))
        if require_reasonable and not is_reasonable_molecule(smiles):
            continue
        encoded = encode_roundtrip(smiles)
        if encoded is None:
            continue
        canon, selfies = encoded
        values = {
            col: getattr(rec, col)
            for col in frame.columns
            if col not in {MORDRED_ID_COL, "canonical_smiles", "inchikey", "penalty", "in_range", "n_passed"}
        }
        rows.append(
            {
                "canonical_smiles": canon,
                "selfies": selfies,
                "inchikey": str(getattr(rec, "inchikey")),
                "connectivity": str(getattr(rec, MORDRED_ID_COL)),
                "values": values,
                "penalty": float(getattr(rec, "penalty")),
                "in_range": bool(getattr(rec, "in_range")),
                "n_passed": int(getattr(rec, "n_passed")),
                "source": "papyrus",
                "origin": origin,
                "generation": 0,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
