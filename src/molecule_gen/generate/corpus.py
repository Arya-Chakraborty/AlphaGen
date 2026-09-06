from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from molecule_gen.data.papyrus import MORDRED_ID_COL, molecules_parquet_path
from molecule_gen.generate.selfies_util import (
    MAX_SELFIES_TOKENS,
    smiles_to_selfies,
    split_selfies,
)
from molecule_gen.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_processed_dir

SELFIES_PARQUET = "papyrus_selfies.parquet"
SELFIES_MANIFEST = "papyrus_selfies_manifest.json"


class CorpusError(RuntimeError):
    pass


def selfies_parquet_path() -> Path:
    return PROCESSED_DIR / SELFIES_PARQUET


def selfies_manifest_path() -> Path:
    return PROCESSED_DIR / SELFIES_MANIFEST


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_selfies_corpus(
    *,
    limit: int | None = None,
    force: bool = False,
    max_tokens: int = MAX_SELFIES_TOKENS,
) -> dict[str, Any]:
    """Encode Papyrus canonical SMILES to SELFIES. Skip failures and over-long strings."""
    ensure_processed_dir()
    mol_path = molecules_parquet_path()
    if not mol_path.exists():
        raise CorpusError(
            f"Molecule table not found at {mol_path}. Run fetch-papyrus first."
        )
    dest = selfies_parquet_path()
    if dest.exists() and not force:
        print(f"SELFIES corpus already exists at {dest} (pass --force to rebuild)", file=sys.stderr)
        return json.loads(selfies_manifest_path().read_text())

    import pyarrow.parquet as pq

    print(f"encoding Papyrus SMILES to SELFIES from {mol_path}", file=sys.stderr)
    reader = pq.ParquetFile(mol_path)
    chunks: list[pd.DataFrame] = []
    n_seen = 0
    n_ok = 0
    n_fail = 0
    n_long = 0
    for batch in tqdm(
        reader.iter_batches(batch_size=20_000, columns=[MORDRED_ID_COL, "canonical_smiles"]),
        total=max(1, reader.metadata.num_rows // 20_000),
        desc="SELFIES",
        unit="batch",
        file=sys.stderr,
    ):
        frame = batch.to_pandas()
        records: list[dict[str, Any]] = []
        for rec in frame.itertuples(index=False):
            if limit is not None and n_seen >= limit:
                break
            n_seen += 1
            smiles = str(rec.canonical_smiles)
            encoded = smiles_to_selfies(smiles)
            if encoded is None:
                n_fail += 1
                continue
            n_tokens = len(split_selfies(encoded))
            if n_tokens > max_tokens:
                n_long += 1
                continue
            n_ok += 1
            records.append(
                {
                    MORDRED_ID_COL: str(getattr(rec, MORDRED_ID_COL)),
                    "canonical_smiles": smiles,
                    "selfies": encoded,
                    "n_tokens": n_tokens,
                }
            )
        if records:
            chunks.append(pd.DataFrame.from_records(records))
        if limit is not None and n_seen >= limit:
            break

    if not chunks:
        raise CorpusError("No SELFIES strings could be encoded")
    out = pd.concat(chunks, ignore_index=True)
    out.to_parquet(dest, index=False)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_seen": n_seen,
        "n_encoded": int(len(out)),
        "n_fail": n_fail,
        "n_too_long": n_long,
        "max_tokens": max_tokens,
        "limit": limit,
        "output_parquet": _relpath(dest),
    }
    selfies_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(out):,} SELFIES rows to {dest}", file=sys.stderr)
    return manifest


def smiles_to_selfies_list(smiles_list: list[str], *, max_tokens: int = MAX_SELFIES_TOKENS) -> list[str]:
    encoded: list[str] = []
    for smiles in smiles_list:
        selfies = smiles_to_selfies(smiles)
        if selfies is None:
            continue
        if len(split_selfies(selfies)) > max_tokens:
            continue
        encoded.append(selfies)
    return encoded


def load_training_selfies(
    *,
    limit: int | None = None,
    max_tokens: int = MAX_SELFIES_TOKENS,
    smiles_list: list[str] | None = None,
) -> list[str]:
    """Load SELFIES for training: explicit SMILES, else corpus parquet, else Papyrus SMILES."""
    if smiles_list:
        rows = smiles_to_selfies_list(smiles_list, max_tokens=max_tokens)
        if not rows:
            raise CorpusError("None of the provided SMILES encoded to SELFIES")
        if limit is not None:
            rows = rows[:limit]
        return rows

    dest = selfies_parquet_path()
    if dest.exists():
        frame = pd.read_parquet(dest, columns=["selfies", "n_tokens"])
        frame = frame[frame["n_tokens"] <= max_tokens]
        if limit is not None:
            frame = frame.head(limit)
        rows = [str(x) for x in frame["selfies"].tolist() if str(x)]
        if not rows:
            raise CorpusError(f"No usable SELFIES in {dest}")
        return rows

    mol_path = molecules_parquet_path()
    if not mol_path.exists():
        raise CorpusError(
            "No SELFIES corpus or Papyrus molecule table. Run build-selfies or fetch-papyrus."
        )
    print(
        f"SELFIES parquet missing; encoding from {mol_path} (run build-selfies to cache)",
        file=sys.stderr,
    )
    frame = pd.read_parquet(mol_path, columns=["canonical_smiles"])
    if limit is not None:
        frame = frame.head(limit)
    return smiles_to_selfies_list(frame["canonical_smiles"].astype(str).tolist(), max_tokens=max_tokens)
