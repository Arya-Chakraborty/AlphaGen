from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from molecule_gen.data.clean import connectivity_from_inchikey
from molecule_gen.data.papyrus import EXPECTED_N_MORDRED_2D, molecules_parquet_path
from molecule_gen.descriptors.cache import DescriptorCache
from molecule_gen.paths import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    ensure_processed_dir,
    mordred_cache_path,
)

EXPECTED_N_DESCRIPTORS = EXPECTED_N_MORDRED_2D
SAMPLE_PARQUET_NAME = "mordred_sample.parquet"
SAMPLE_MANIFEST_NAME = "mordred_sample_manifest.json"
ID_COLUMNS = ("connectivity", "canonical_smiles", "inchikey")
DEFAULT_BATCH_SIZE = 64


class MordredError(RuntimeError):
    pass


ComputeFn = Callable[[list[str]], list[dict[str, float | None] | None]]


def sample_parquet_path() -> Path:
    return PROCESSED_DIR / SAMPLE_PARQUET_NAME


def sample_manifest_path() -> Path:
    return PROCESSED_DIR / SAMPLE_MANIFEST_NAME


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "na", "none", "null", "inf", "-inf"}:
        return None
    try:
        from mordred.error import Error, Missing

        if isinstance(value, (Missing, Error)):
            return None
    except ImportError:
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_descriptors(raw: dict[str, Any]) -> dict[str, float | None]:
    skip = {"Name", "name", "SMILES", "smiles", "connectivity", "InChIKey", "inchikey"}
    return {key: _to_float(val) for key, val in raw.items() if key not in skip}


def _calculator():
    from mordred import Calculator, descriptors

    return Calculator(descriptors, ignore_3D=True)


@lru_cache(maxsize=32)
def _subset_calculator(names: tuple[str, ...]):
    from mordred import Calculator

    wanted = set(names)
    calc = Calculator(ignore_3D=True)
    calc.register([desc for desc in _all_descriptors() if str(desc) in wanted])
    return calc


@lru_cache(maxsize=1)
def _all_descriptors():
    return tuple(_calculator().descriptors)


def descriptor_names() -> list[str]:
    return [str(desc) for desc in _calculator().descriptors]


def mordred_from_smiles_batch(
    smiles_list: list[str], names: list[str] | None = None
) -> list[dict[str, float | None] | None]:
    """Compute Mordred 2D descriptors for each SMILES. None if the molecule is invalid.

    `names` restricts the calculation to those descriptors (default: all).
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    calc = _subset_calculator(tuple(sorted(set(names)))) if names else _calculator()
    rows: list[dict[str, float | None] | None] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append(None)
            continue
        result = calc(mol)
        payload = {
            str(desc): _to_float(value)
            for desc, value in zip(calc.descriptors, result)
        }
        rows.append(payload)
    return rows


def describe_molecules(
    frame: pd.DataFrame,
    cache: DescriptorCache,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_fn: ComputeFn | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return Mordred 2D descriptors for each row, using the InChIKey cache."""
    needed = {"canonical_smiles", "inchikey"}
    missing = needed - set(frame.columns)
    if missing:
        raise MordredError(f"Input table missing columns: {sorted(missing)}")

    compute = compute_fn or mordred_from_smiles_batch
    records: list[dict[str, Any]] = []
    n_cached = 0
    n_computed = 0
    n_failed = 0

    rows = list(frame.itertuples(index=False))
    bar = tqdm(
        total=len(rows),
        desc="Mordred",
        unit="mol",
        file=sys.stderr,
        disable=not show_progress,
    )
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            keys = [str(row.inchikey) for row in batch]
            cached = cache.get_many(keys)
            to_compute_idx: list[int] = []
            to_compute_smiles: list[str] = []
            batch_desc: list[dict[str, float | None] | None] = [None] * len(batch)

            for i, row in enumerate(batch):
                hit = cached.get(str(row.inchikey))
                if hit is not None:
                    batch_desc[i] = hit
                    n_cached += 1
                else:
                    to_compute_idx.append(i)
                    to_compute_smiles.append(str(row.canonical_smiles))

            if to_compute_smiles:
                computed = compute(to_compute_smiles)
                store: list[tuple[str, str, dict[str, float | None]]] = []
                for idx, smiles, desc in zip(
                    to_compute_idx, to_compute_smiles, computed
                ):
                    if desc is None:
                        n_failed += 1
                        continue
                    n_computed += 1
                    batch_desc[idx] = desc
                    store.append((str(batch[idx].inchikey), smiles, desc))
                if store:
                    cache.put_many(store)

            for row, desc in zip(batch, batch_desc):
                if desc is None:
                    continue
                inchikey = str(row.inchikey)
                connectivity = str(getattr(row, "connectivity", "") or "") or (
                    connectivity_from_inchikey(inchikey)
                )
                records.append(
                    {
                        "connectivity": connectivity,
                        "canonical_smiles": str(row.canonical_smiles),
                        "inchikey": inchikey,
                        **desc,
                    }
                )
            bar.update(len(batch))
            bar.set_postfix(cached=n_cached, computed=n_computed, failed=n_failed)
    finally:
        bar.close()

    if not records:
        raise MordredError("Mordred produced no descriptor rows")

    stats = {
        "n_input": len(rows),
        "n_cached": n_cached,
        "n_computed": n_computed,
        "n_failed": n_failed,
        "n_output": len(records),
        "n_descriptors": EXPECTED_N_DESCRIPTORS,
        "cache_size": len(cache),
    }
    return pd.DataFrame.from_records(records), stats


def compute_mordred_sample(
    *,
    n: int = 2000,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Compute Mordred 2D for a random sample of the Papyrus molecule table."""
    ensure_processed_dir()
    source = molecules_parquet_path()
    if not source.exists():
        raise MordredError(
            f"Molecule table not found at {source}. Run fetch-papyrus first."
        )

    out_path = sample_parquet_path()
    if not force and out_path.exists() and sample_manifest_path().exists():
        existing = json.loads(sample_manifest_path().read_text())
        if existing.get("n") == n and existing.get("seed") == seed:
            _log(
                f"already have Mordred sample ({existing.get('n_output', '?')} rows); "
                "pass --force to refresh"
            )
            return existing

    _log(f"reading {source}")
    frame = pd.read_parquet(source)
    missing_cols = [c for c in ID_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise MordredError(f"Molecule table missing columns: {missing_cols}")
    frame = frame[list(ID_COLUMNS)]
    n_take = min(n, len(frame))
    sample = frame.sample(n=n_take, random_state=seed).reset_index(drop=True)
    _log(f"sampled {len(sample):,} molecules (seed={seed})")

    with DescriptorCache() as cache:
        table, stats = describe_molecules(
            sample,
            cache=cache,
            batch_size=batch_size,
        )

    n_desc = len([c for c in table.columns if c not in ID_COLUMNS])
    if n_desc != EXPECTED_N_DESCRIPTORS:
        _log(f"warning: expected {EXPECTED_N_DESCRIPTORS} descriptors, got {n_desc}")

    table.to_parquet(out_path, index=False)
    _log(f"wrote {len(table):,} rows × {n_desc} descriptors to {out_path}")

    manifest = {
        "source_parquet": str(source.relative_to(PROJECT_ROOT)),
        "output_parquet": str(out_path.relative_to(PROJECT_ROOT)),
        "cache": str(mordred_cache_path().relative_to(PROJECT_ROOT)),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "seed": seed,
        "batch_size": batch_size,
        "n_descriptor_columns": n_desc,
        **stats,
    }
    sample_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
