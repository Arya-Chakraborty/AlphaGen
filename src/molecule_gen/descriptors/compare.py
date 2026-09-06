from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molecule_gen.data.papyrus import (
    MORDRED_ID_COL,
    molecules_parquet_path,
    mordred_parquet_path,
)
from molecule_gen.descriptors.cache import DescriptorCache
from molecule_gen.descriptors.mordred import (
    DEFAULT_BATCH_SIZE,
    ID_COLUMNS,
    MordredError,
    describe_molecules,
)
from molecule_gen.descriptors.names import is_detour_descriptor, papyrus_name_to_local
from molecule_gen.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_processed_dir

COMPARE_MANIFEST_NAME = "mordred_compare_manifest.json"
ABS_TOLERANCES = (1e-6, 1e-4, 1e-2)

LANDMARKS = (
    "SLogP",
    "MW",
    "nHBAcc",
    "nHBDon",
    "TopoPSA",
    "nRot",
    "nRing",
    "nAromAtom",
    "nAcid",
    "nBase",
    "nAtom",
    "nBond",
    "nHeavyAtom",
)


def compare_manifest_path() -> Path:
    return PROCESSED_DIR / COMPARE_MANIFEST_NAME


def _log(msg: str) -> None:
    import sys

    print(msg, file=sys.stderr, flush=True)


def load_papyrus_mordred(connectivities: list[str]) -> pd.DataFrame:
    """Load Papyrus Mordred rows for the given connectivity keys."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    path = mordred_parquet_path()
    if not path.exists():
        raise MordredError(
            f"Papyrus Mordred table not found at {path}. Run import-mordred first."
        )
    keys = list(dict.fromkeys(str(k) for k in connectivities if k))
    _log(f"looking up {len(keys):,} connectivities in {path.name}")
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(filter=pc.field(MORDRED_ID_COL).isin(keys))
    frame = table.to_pandas()
    if frame.empty:
        raise MordredError("No matching rows in the Papyrus Mordred table")
    return frame


def _descriptor_columns(frame: pd.DataFrame) -> list[str]:
    skip = set(ID_COLUMNS) | {MORDRED_ID_COL, "SMILES", "smiles", "InChIKey"}
    return [c for c in frame.columns if c not in skip]


def align_papyrus_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: papyrus_name_to_local(col) for col in frame.columns}
    return frame.rename(columns=renamed)


def compare_descriptor_tables(
    local: pd.DataFrame,
    papyrus: pd.DataFrame,
    *,
    top_n: int = 15,
) -> dict[str, Any]:
    """Compare local vs Papyrus descriptor tables joined on connectivity."""
    local_ids = local[[MORDRED_ID_COL]].copy()
    papyrus = align_papyrus_columns(papyrus)
    paper_ids = papyrus[[MORDRED_ID_COL]].copy()
    local_cols = _descriptor_columns(local)
    paper_cols = _descriptor_columns(papyrus)
    shared = sorted(set(local_cols) & set(paper_cols))
    only_local = sorted(set(local_cols) - set(paper_cols))
    only_papyrus = sorted(set(paper_cols) - set(local_cols))

    merged = local[[MORDRED_ID_COL] + shared].merge(
        papyrus[[MORDRED_ID_COL] + shared],
        on=MORDRED_ID_COL,
        how="inner",
        suffixes=("_local", "_papyrus"),
    )
    n_molecules = int(len(merged))
    if n_molecules == 0:
        raise MordredError("No overlapping connectivities to compare")

    n_both_finite = 0
    n_both_missing = 0
    n_missing_mismatch = 0
    abs_errors: list[np.ndarray] = []
    per_desc: list[dict[str, Any]] = []

    for name in shared:
        left = pd.to_numeric(merged[f"{name}_local"], errors="coerce")
        right = pd.to_numeric(merged[f"{name}_papyrus"], errors="coerce")
        left_na = left.isna()
        right_na = right.isna()
        both_na = left_na & right_na
        one_na = left_na ^ right_na
        both_num = ~(left_na | right_na)
        n_both_finite += int(both_num.sum())
        n_both_missing += int(both_na.sum())
        n_missing_mismatch += int(one_na.sum())

        if not both_num.any():
            per_desc.append(
                {
                    "descriptor": name,
                    "n_finite": 0,
                    "mae": None,
                    "max_abs": None,
                }
            )
            continue
        err = (left[both_num] - right[both_num]).abs().to_numpy(dtype=np.float64)
        abs_errors.append(err)
        per_desc.append(
            {
                "descriptor": name,
                "n_finite": int(both_num.sum()),
                "mae": float(err.mean()),
                "max_abs": float(err.max()),
            }
        )

    if abs_errors:
        all_err = np.concatenate(abs_errors)
    else:
        all_err = np.array([], dtype=np.float64)

    within = {
        f"within_{tol:.0e}": (
            float((all_err <= tol).mean()) if len(all_err) else None
        )
        for tol in ABS_TOLERANCES
    }
    worst = sorted(
        [row for row in per_desc if row["mae"] is not None],
        key=lambda row: row["mae"],
        reverse=True,
    )[:top_n]
    core_err = np.concatenate(
        [
            (pd.to_numeric(merged[f"{row['descriptor']}_local"], errors="coerce")
             - pd.to_numeric(merged[f"{row['descriptor']}_papyrus"], errors="coerce"))
            .abs()
            .dropna()
            .to_numpy(dtype=np.float64)
            for row in per_desc
            if row["mae"] is not None and not is_detour_descriptor(row["descriptor"])
        ]
        or [np.array([], dtype=np.float64)]
    )
    landmarks = [row for row in per_desc if row["descriptor"] in LANDMARKS]

    return {
        "n_molecules_local": int(len(local_ids)),
        "n_molecules_papyrus": int(len(paper_ids)),
        "n_molecules_compared": n_molecules,
        "n_local_descriptors": len(local_cols),
        "n_papyrus_descriptors": len(paper_cols),
        "n_shared_descriptors": len(shared),
        "n_only_local": len(only_local),
        "n_only_papyrus": len(only_papyrus),
        "only_local": only_local[:50],
        "only_papyrus": only_papyrus[:50],
        "n_both_finite": n_both_finite,
        "n_both_missing": n_both_missing,
        "n_missing_mismatch": n_missing_mismatch,
        "mae": float(all_err.mean()) if len(all_err) else None,
        "median_abs": float(np.median(all_err)) if len(all_err) else None,
        "p99_abs": float(np.quantile(all_err, 0.99)) if len(all_err) else None,
        "max_abs": float(all_err.max()) if len(all_err) else None,
        **within,
        "core_n_finite": int(len(core_err)),
        "core_mae": float(core_err.mean()) if len(core_err) else None,
        "core_median_abs": float(np.median(core_err)) if len(core_err) else None,
        "core_p99_abs": float(np.quantile(core_err, 0.99)) if len(core_err) else None,
        "core_max_abs": float(core_err.max()) if len(core_err) else None,
        "core_within_1e-02": (
            float((core_err <= 1e-2).mean()) if len(core_err) else None
        ),
        "landmarks": landmarks,
        "worst_descriptors": worst,
    }


def compare_local_to_papyrus(
    *,
    n: int = 100,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Compute local Mordred 2D on a Papyrus sample and compare to the lookup table."""
    ensure_processed_dir()
    mol_path = molecules_parquet_path()
    if not mol_path.exists():
        raise MordredError(
            f"Molecule table not found at {mol_path}. Run fetch-papyrus first."
        )

    _log(f"reading {mol_path}")
    molecules = pd.read_parquet(mol_path, columns=list(ID_COLUMNS))
    n_take = min(n, len(molecules))
    sample = molecules.sample(n=n_take, random_state=seed).reset_index(drop=True)
    _log(f"sampled {len(sample):,} molecules (seed={seed})")

    with DescriptorCache(cache_path) as cache:
        local, compute_stats = describe_molecules(
            sample, cache=cache, batch_size=batch_size
        )

    papyrus = load_papyrus_mordred(local[MORDRED_ID_COL].astype(str).tolist())
    missing = sorted(set(local[MORDRED_ID_COL]) - set(papyrus[MORDRED_ID_COL]))
    if missing:
        _log(f"warning: {len(missing)} sampled molecules missing from Papyrus Mordred")

    stats = compare_descriptor_tables(local, papyrus)
    report = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "seed": seed,
        "molecules_parquet": str(mol_path.relative_to(PROJECT_ROOT)),
        "papyrus_mordred_parquet": str(mordred_parquet_path().relative_to(PROJECT_ROOT)),
        "n_missing_from_papyrus": len(missing),
        **compute_stats,
        **stats,
    }
    compare_manifest_path().write_text(json.dumps(report, indent=2) + "\n")
    _log(f"wrote compare report to {compare_manifest_path()}")
    return report
