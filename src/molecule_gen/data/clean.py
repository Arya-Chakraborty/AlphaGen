from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
from tqdm import tqdm


class CleanError(RuntimeError):
    pass


def connectivity_from_inchikey(inchikey: str | None) -> str:
    """Papyrus 2D join key: first block of the InChIKey (14 characters)."""
    if not inchikey:
        return ""
    return str(inchikey).split("-", 1)[0]


def _disable_rdkit_logs() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smiles: str | None) -> tuple[str, str] | None:
    """Parse SMILES with RDKit. Return (canonical_smiles, inchikey) or None if invalid."""
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text or text.lower() == "nan":
        return None

    from rdkit import Chem

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True)
    key = Chem.MolToInchiKey(mol)
    if not canon or not key:
        return None
    return canon, key


def _canonicalize_batch(smiles_list: list[str | None]) -> list[tuple[str, str] | None]:
    _disable_rdkit_logs()
    return [canonicalize_smiles(s) for s in smiles_list]


def _split_batches(items: list[str | None], batch_size: int) -> list[list[str | None]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _smiles_column(frame: pd.DataFrame) -> str:
    for name in ("canonical_smiles", "SMILES", "smiles"):
        if name in frame.columns:
            return name
    raise CleanError("Input table must have a SMILES column")


def canonicalize_table(
    frame: pd.DataFrame,
    *,
    workers: int | None = None,
    batch_size: int = 5_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sanitize, canonicalize, recompute InChIKey, drop invalid, dedupe by connectivity."""
    smiles_col = _smiles_column(frame)
    n_input = len(frame)
    smiles_list = frame[smiles_col].tolist()
    if "connectivity" in frame.columns:
        source_ids = [str(v) if pd.notna(v) else "" for v in frame["connectivity"]]
    else:
        source_ids = [""] * n_input
    source_keys = (
        frame["inchikey"].tolist()
        if "inchikey" in frame.columns
        else (
            frame["InChIKey"].tolist()
            if "InChIKey" in frame.columns
            else [None] * n_input
        )
    )

    if workers is None:
        cpu = os.cpu_count() or 1
        workers = max(1, cpu - 1)

    batches = _split_batches(smiles_list, batch_size)
    parsed: list[tuple[str, str] | None] = []

    _log(f"canonicalizing {n_input:,} molecules with {workers} worker(s)")
    started = time.perf_counter()

    if workers <= 1:
        _disable_rdkit_logs()
        for smiles in tqdm(
            smiles_list, desc="RDKit", unit="mol", file=sys.stderr, mininterval=0.5
        ):
            parsed.append(canonicalize_smiles(smiles))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_disable_rdkit_logs
        ) as pool:
            with tqdm(
                total=n_input, desc="RDKit", unit="mol", file=sys.stderr, mininterval=0.5
            ) as bar:
                for batch_result in pool.map(_canonicalize_batch, batches, chunksize=1):
                    parsed.extend(batch_result)
                    bar.update(len(batch_result))

    elapsed = time.perf_counter() - started
    _log(f"rdkit pass finished in {elapsed:.1f}s")

    records: list[dict[str, str]] = []
    n_invalid = 0
    n_inchikey_mismatch = 0
    for source_id, source_key, result in zip(source_ids, source_keys, parsed):
        if result is None:
            n_invalid += 1
            continue
        canon, key = result
        if source_key and str(source_key) != key:
            n_inchikey_mismatch += 1
        connectivity = source_id or connectivity_from_inchikey(key)
        records.append(
            {
                "connectivity": connectivity,
                "canonical_smiles": canon,
                "inchikey": key,
            }
        )

    valid = pd.DataFrame.from_records(records)
    n_valid = len(valid)
    if n_valid == 0:
        raise CleanError("No valid molecules after RDKit sanitization")

    unique = valid.drop_duplicates(subset=["connectivity"], keep="first").reset_index(
        drop=True
    )
    stats = {
        "n_input": n_input,
        "n_invalid": n_invalid,
        "n_valid": n_valid,
        "n_unique": int(len(unique)),
        "n_duplicate_structures": int(n_valid - len(unique)),
        "n_inchikey_mismatch": n_inchikey_mismatch,
        "workers": workers,
        "elapsed_sec": round(elapsed, 2),
    }
    return unique, stats
