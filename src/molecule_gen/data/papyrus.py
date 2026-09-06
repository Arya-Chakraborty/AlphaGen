from __future__ import annotations

import json
import lzma
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from molecule_gen.data.clean import canonicalize_smiles, connectivity_from_inchikey
from molecule_gen.data.download import (
    DownloadError,
    download_with_resume,
    find_local_file,
    format_bytes,
    log,
    md5_file,
)
from molecule_gen.paths import (
    DATA_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    ensure_processed_dir,
    ensure_raw_dir,
)

PAPYRUS_VERSION = "05.7"
PAPYRUS_RELEASE = "2024.2"
ZENODO_RECORD = "13987985"
ZENODO_DOI = "10.5281/zenodo.13987985"
ZENODO_FILES = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

STRUCTURES_NAME = "05.7_combined_2D_set_without_stereochemistry.sd.xz"
STRUCTURES_MD5 = "0cb3c70501d45b8a4ec5fafbfb8dc0e1"
STRUCTURES_SIZE = 453_721_500

MORDRED_NAME = "05.7_combined_2D_moldescs_mordred2D.tsv.xz"
MORDRED_MD5 = "63cf3d52855a8e216794d5b91e98478c"
MORDRED_SIZE = 3_149_391_660
MORDRED_ID_COL = "connectivity"
EXPECTED_N_MORDRED_2D = 1613

MOLECULES_PARQUET = "papyrus_molecules.parquet"
MOLECULES_MANIFEST = "papyrus_molecules_manifest.json"
MORDRED_PARQUET = "papyrus_mordred2d.parquet"
MORDRED_MANIFEST = "papyrus_mordred2d_manifest.json"

KEEP_COLUMNS = ("connectivity", "canonical_smiles", "inchikey")
ATTRIBUTION = (
    "Papyrus 05.7 / 2024.2 (https://doi.org/10.5281/zenodo.13987985), "
    "CC BY-SA 4.0. Described in Béquignon et al., J. Cheminform. 15, 3 (2023)."
)


class PapyrusError(DownloadError):
    pass


def zenodo_url(filename: str) -> str:
    return f"{ZENODO_FILES}/{filename}/content"


def structures_path() -> Path:
    return RAW_DIR / STRUCTURES_NAME


def mordred_raw_path() -> Path:
    return RAW_DIR / MORDRED_NAME


def molecules_parquet_path() -> Path:
    return PROCESSED_DIR / MOLECULES_PARQUET


def molecules_manifest_path() -> Path:
    return PROCESSED_DIR / MOLECULES_MANIFEST


def mordred_parquet_path() -> Path:
    return PROCESSED_DIR / MORDRED_PARQUET


def mordred_manifest_path() -> Path:
    return PROCESSED_DIR / MORDRED_MANIFEST


def file_help(filename: str, size: int) -> str:
    dest = RAW_DIR / filename
    return (
        f"{filename} ({format_bytes(size)})\n"
        f"  URL:  {zenodo_url(filename)}\n"
        f"  save: {dest}"
    )


def _locate(name: str) -> Path | None:
    ensure_raw_dir()
    return find_local_file(name, RAW_DIR, DATA_DIR, PROJECT_ROOT)


def _ensure_file(
    name: str,
    *,
    md5: str,
    size: int,
    download: bool,
    force: bool,
) -> Path:
    dest = RAW_DIR / name
    local = _locate(name)
    if local is not None and local.resolve() != dest.resolve():
        log(f"moving {local} → {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        local.rename(dest)
        local = dest
    elif local is not None:
        dest = local

    if dest.exists() and not force:
        log(f"using local {dest} ({format_bytes(dest.stat().st_size)})")
        _check_md5(dest, md5)
        return dest

    if not download:
        raise PapyrusError(
            f"Missing {name}. Download it yourself or drop it in data/raw/:\n"
            f"{file_help(name, size)}"
        )

    ensure_raw_dir()
    log(f"fetching {name} from Zenodo record {ZENODO_RECORD}")
    log(file_help(name, size))
    download_with_resume(zenodo_url(name), dest)
    _check_md5(dest, md5)
    return dest


def _check_md5(path: Path, expected: str) -> None:
    digest = md5_file(path)
    if digest != expected:
        raise PapyrusError(
            f"MD5 mismatch for {path.name}: got {digest}, expected {expected}"
        )
    log(f"checksum ok ({digest[:12]}…)")


def _mol_prop(mol: Any, *names: str) -> str:
    for name in names:
        if mol.HasProp(name):
            value = str(mol.GetProp(name)).strip()
            if value:
                return value
    return ""


def parse_papyrus_sd(
    path: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read the unique 2D SD(XZ) and return connectivity / SMILES / InChIKey."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    opener = lzma.open if path.name.endswith(".xz") else open
    records: list[dict[str, str]] = []
    n_empty = 0
    log(f"parsing {path.name}")
    with opener(path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False)
        for mol in tqdm(
            supplier, desc="Papyrus SD", unit="mol", file=__import__("sys").stderr
        ):
            if limit is not None and len(records) >= limit:
                break
            if mol is None:
                n_empty += 1
                continue
            smiles = _mol_prop(mol, "SMILES", "smiles")
            connectivity = _mol_prop(mol, "connectivity")
            inchikey = _mol_prop(mol, "InChIKey", "inchikey")
            if not smiles:
                try:
                    Chem.SanitizeMol(mol)
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    n_empty += 1
                    continue
            if not connectivity and inchikey:
                connectivity = connectivity_from_inchikey(inchikey)
            if not smiles or not connectivity:
                n_empty += 1
                continue
            records.append(
                {
                    "connectivity": connectivity,
                    "canonical_smiles": smiles,
                    "inchikey": inchikey,
                }
            )

    if not records:
        raise PapyrusError(f"No molecules parsed from {path}")
    if n_empty:
        log(f"skipped {n_empty:,} empty/unreadable SD records")
    frame = pd.DataFrame.from_records(records)
    frame = frame.drop_duplicates(subset=["connectivity"], keep="first").reset_index(
        drop=True
    )
    return frame


def canonicalize_molecules(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """RDKit-sanitize Papyrus SMILES and fill missing InChIKeys."""
    n_input = len(frame)
    n_invalid = 0
    n_inchikey_filled = 0
    rows: list[dict[str, str]] = []
    for rec in tqdm(
        frame.itertuples(index=False),
        total=n_input,
        desc="RDKit",
        unit="mol",
        file=__import__("sys").stderr,
        mininterval=0.5,
    ):
        smiles = str(getattr(rec, "canonical_smiles", "") or "")
        parsed = canonicalize_smiles(smiles)
        if parsed is None:
            n_invalid += 1
            continue
        canon, key = parsed
        connectivity = str(getattr(rec, "connectivity", "") or "") or (
            connectivity_from_inchikey(key)
        )
        source_key = str(getattr(rec, "inchikey", "") or "")
        if not source_key:
            n_inchikey_filled += 1
        rows.append(
            {
                "connectivity": connectivity,
                "canonical_smiles": canon,
                "inchikey": key or source_key,
            }
        )

    if not rows:
        raise PapyrusError("No valid molecules after RDKit sanitization")
    cleaned = pd.DataFrame.from_records(rows)
    unique = cleaned.drop_duplicates(subset=["connectivity"], keep="first").reset_index(
        drop=True
    )
    stats = {
        "n_input": n_input,
        "n_invalid": n_invalid,
        "n_valid": len(cleaned),
        "n_unique": int(len(unique)),
        "n_duplicate_structures": int(len(cleaned) - len(unique)),
        "n_inchikey_filled": n_inchikey_filled,
    }
    return unique, stats


def fetch_papyrus(
    *,
    force: bool = False,
    limit: int | None = None,
    download: bool = True,
) -> dict[str, Any]:
    """Download the unique 2D SD (or use a local copy) and write a molecule parquet."""
    ensure_processed_dir()
    out_path = molecules_parquet_path()
    existing = _load_json(molecules_manifest_path())
    if (
        not force
        and existing
        and existing.get("limit") == limit
        and existing.get("version") == PAPYRUS_VERSION
        and out_path.exists()
    ):
        log(
            f"already have Papyrus molecules ({existing.get('n_unique', '?')} unique); "
            "pass --force to refresh"
        )
        return existing

    sd_path = _ensure_file(
        STRUCTURES_NAME,
        md5=STRUCTURES_MD5,
        size=STRUCTURES_SIZE,
        download=download,
        force=force,
    )
    parsed = parse_papyrus_sd(sd_path, limit=limit)
    cleaned, stats = canonicalize_molecules(parsed)
    cleaned.to_parquet(out_path, index=False)
    log(f"wrote {len(cleaned):,} unique molecules to {out_path}")

    manifest = {
        "source": "Papyrus",
        "version": PAPYRUS_VERSION,
        "release": PAPYRUS_RELEASE,
        "doi": ZENODO_DOI,
        "url": zenodo_url(STRUCTURES_NAME),
        "filename": STRUCTURES_NAME,
        "md5": STRUCTURES_MD5,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "columns": list(KEEP_COLUMNS),
        "parquet": str(out_path.relative_to(PROJECT_ROOT)),
        "attribution": ATTRIBUTION,
        **stats,
    }
    molecules_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _to_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "na", "none", "null", "inf", "-inf"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if pd.isna(number):
        return None
    return number


def import_mordred(
    *,
    force: bool = False,
    limit: int | None = None,
    download: bool = True,
    chunksize: int = 25_000,
) -> dict[str, Any]:
    """Convert the Papyrus Mordred 2D TSV into a parquet keyed by connectivity."""
    ensure_processed_dir()
    out_path = mordred_parquet_path()
    existing = _load_json(mordred_manifest_path())
    if (
        not force
        and existing
        and existing.get("limit") == limit
        and existing.get("version") == PAPYRUS_VERSION
        and out_path.exists()
    ):
        log(
            f"already have Papyrus Mordred 2D ({existing.get('n_rows', '?')} rows); "
            "pass --force to refresh"
        )
        return existing

    tsv_path = _ensure_file(
        MORDRED_NAME,
        md5=MORDRED_MD5,
        size=MORDRED_SIZE,
        download=download,
        force=force,
    )
    n_rows, n_desc = mordred_tsv_to_parquet(
        tsv_path, out_path, limit=limit, chunksize=chunksize
    )
    if n_desc != EXPECTED_N_MORDRED_2D:
        log(f"warning: expected {EXPECTED_N_MORDRED_2D} Mordred 2D columns, got {n_desc}")

    manifest = {
        "source": "Papyrus",
        "version": PAPYRUS_VERSION,
        "release": PAPYRUS_RELEASE,
        "doi": ZENODO_DOI,
        "url": zenodo_url(MORDRED_NAME),
        "filename": MORDRED_NAME,
        "md5": MORDRED_MD5,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "id_column": MORDRED_ID_COL,
        "n_rows": n_rows,
        "n_descriptors": n_desc,
        "parquet": str(out_path.relative_to(PROJECT_ROOT)),
        "attribution": ATTRIBUTION,
    }
    mordred_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def mordred_tsv_to_parquet(
    tsv_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    chunksize: int = 25_000,
) -> tuple[int, int]:
    """Stream a Mordred TSV (plain or xz) to parquet. First column is connectivity."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    compression = "xz" if tsv_path.name.endswith(".xz") else "infer"
    log(f"converting {tsv_path.name} → {out_path.name}")
    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        compression=compression,
        chunksize=chunksize,
        low_memory=False,
        na_values=["", "nan", "NaN", "NA", "None", "null", "inf", "-inf", "Inf", "-Inf"],
    )
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    n_desc = 0
    try:
        for chunk in reader:
            if MORDRED_ID_COL not in chunk.columns:
                raise PapyrusError(
                    f"{tsv_path.name} has no '{MORDRED_ID_COL}' column. "
                    f"Found: {list(chunk.columns[:8])}"
                )
            if limit is not None:
                remaining = limit - n_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            desc_cols = [c for c in chunk.columns if c != MORDRED_ID_COL]
            n_desc = len(desc_cols)
            converted = pd.DataFrame(
                {MORDRED_ID_COL: chunk[MORDRED_ID_COL].astype("string")}
            )
            numeric = chunk[desc_cols].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            converted = pd.concat([converted, numeric], axis=1)
            table = pa.Table.from_pandas(converted, preserve_index=False)
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += len(converted)
            log(f"wrote {n_rows:,} Mordred rows ({n_desc} descriptors)")
    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        raise PapyrusError(f"No Mordred rows parsed from {tsv_path}")
    log(f"wrote {n_rows:,} rows × {n_desc} descriptors to {out_path}")
    return n_rows, n_desc


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def iter_descriptor_help() -> Iterable[str]:
    yield file_help(STRUCTURES_NAME, STRUCTURES_SIZE)
    yield file_help(MORDRED_NAME, MORDRED_SIZE)
