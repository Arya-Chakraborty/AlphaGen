from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from molecule_gen.data.papyrus import MORDRED_ID_COL, mordred_parquet_path
from molecule_gen.descriptors.cache import DescriptorCache
from molecule_gen.descriptors.mordred import mordred_from_smiles_batch
from molecule_gen.descriptors.names import local_name_to_papyrus, papyrus_name_to_local


class LookupError(RuntimeError):
    pass


def describe_for_scoring(
    *,
    names: list[str],
    smiles: str | None,
    inchikey: str | None,
    connectivity: str | None,
    cache: DescriptorCache | None = None,
    allow_papyrus: bool = True,
) -> tuple[dict[str, float | None], str]:
    """Resolve requested descriptors from cache, Papyrus, or local Mordred."""
    owned = False
    store = cache
    if store is None and inchikey:
        store = DescriptorCache()
        owned = True
    try:
        if inchikey and store is not None:
            hit = store.get(inchikey)
            if hit is not None:
                return {name: _as_float(hit.get(name)) for name in names}, "cache"

        if allow_papyrus and connectivity:
            try:
                frame = lookup_papyrus_descriptors([connectivity], names)
            except LookupError:
                frame = pd.DataFrame()
            if not frame.empty:
                row = frame.iloc[0]
                return {name: _as_float(row.get(name)) for name in names}, "papyrus"

        if smiles:
            computed = mordred_from_smiles_batch([smiles])[0]
            if computed is None:
                raise LookupError(f"Local Mordred failed on SMILES: {smiles}")
            if inchikey and store is not None:
                store.put(inchikey, smiles, computed)
            return {name: _as_float(computed.get(name)) for name in names}, "local"

        raise LookupError(
            "Need a Papyrus connectivity match or a SMILES to compute descriptors"
        )
    finally:
        if owned and store is not None:
            store.close()


def _papyrus_read_plan(names: list[str]) -> tuple[Path, list[str], dict[str, str]]:
    import pyarrow.parquet as pq

    path = mordred_parquet_path()
    if not path.exists():
        raise LookupError(
            f"Papyrus Mordred table not found at {path}. Run import-mordred first."
        )
    if not names:
        raise LookupError("No descriptor names to look up")
    schema_names = set(pq.read_schema(path).names)
    if MORDRED_ID_COL not in schema_names:
        raise LookupError(f"{path.name} has no {MORDRED_ID_COL} column")
    read_cols = [MORDRED_ID_COL]
    rename: dict[str, str] = {}
    for name in names:
        if name in schema_names:
            read_cols.append(name)
            continue
        papyrus = local_name_to_papyrus(name)
        if papyrus in schema_names:
            read_cols.append(papyrus)
            rename[papyrus] = name
            continue
        raise LookupError(f"Column {name} not in Papyrus Mordred table")
    return path, list(dict.fromkeys(read_cols)), rename


def _align_papyrus_frame(frame: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    if rename:
        frame = frame.rename(columns=rename)
    extra = {
        col: papyrus_name_to_local(col)
        for col in frame.columns
        if col != MORDRED_ID_COL
    }
    return frame.rename(columns=extra)


def lookup_papyrus_descriptors(
    connectivities: list[str],
    names: list[str],
) -> pd.DataFrame:
    """Load only the requested descriptor columns for the given connectivities."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    keys = list(dict.fromkeys(str(k) for k in connectivities if k))
    if not keys:
        raise LookupError("No connectivities to look up")
    path, read_cols, rename = _papyrus_read_plan(names)
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(
        columns=read_cols,
        filter=pc.field(MORDRED_ID_COL).isin(keys),
    )
    return _align_papyrus_frame(table.to_pandas(), rename)


def load_papyrus_descriptor_table(names: list[str]) -> pd.DataFrame:
    """Load requested descriptor columns for every Papyrus molecule."""
    import pyarrow.parquet as pq

    path, read_cols, rename = _papyrus_read_plan(names)
    return _align_papyrus_frame(pq.read_table(path, columns=read_cols).to_pandas(), rename)


def papyrus_percentiles(
    names: list[str],
    *,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95),
) -> dict[str, dict[str, float]]:
    """Percentiles of selected Papyrus Mordred columns (full table, projected)."""
    frame = load_papyrus_descriptor_table(names)
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        series = pd.to_numeric(frame[name], errors="coerce").dropna()
        if series.empty:
            raise LookupError(f"No finite values for {name}")
        stats[name] = {
            f"p{int(q * 100):02d}": float(series.quantile(q)) for q in quantiles
        }
        stats[name]["min"] = float(series.min())
        stats[name]["max"] = float(series.max())
        stats[name]["n"] = float(len(series))
    return stats


def papyrus_corpus_stats(names: list[str]) -> dict[str, dict[str, float]]:
    """Corpus min/max (and p10/p90) for UI reference. Not a hard limit on user ranges."""
    return papyrus_percentiles(names, quantiles=(0.10, 0.90))


def draft_spec_from_percentiles(
    names: list[str],
    *,
    n: int = 10,
    low: float = 0.10,
    high: float = 0.90,
) -> dict[str, Any]:
    stats = papyrus_percentiles(names, quantiles=(low, high))
    low_key = f"p{int(low * 100):02d}"
    high_key = f"p{int(high * 100):02d}"
    return {
        "n": n,
        "descriptors": {
            name: {
                "min": stats[name][low_key],
                "max": stats[name][high_key],
            }
            for name in names
        },
        "source": "Papyrus 05.7 Mordred 2D percentiles",
        "percentiles": stats,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
