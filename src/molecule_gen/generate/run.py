from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

from molecule_gen.constraints.score import score_vector
from molecule_gen.constraints.spec import ConstraintSpec
from molecule_gen.data.clean import connectivity_from_inchikey
from molecule_gen.generate.ga import evolve
from molecule_gen.generate.seed import candidates_from_seed_table, seed_from_papyrus
from molecule_gen.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_processed_dir

DEFAULT_POPULATION = 64
DEFAULT_GENERATIONS = 20
HITS_PARQUET = "generate_hits.parquet"
HITS_MANIFEST = "generate_manifest.json"


def hits_parquet_path() -> Path:
    return PROCESSED_DIR / HITS_PARQUET


def hits_manifest_path() -> Path:
    return PROCESSED_DIR / HITS_MANIFEST


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def candidates_from_smiles_list(
    smiles_list: list[str],
    spec: ConstraintSpec,
    *,
    evaluate_fn=None,
    origin: str = "seed",
) -> list[dict[str, Any]]:
    from molecule_gen.generate.ga import evaluate_smiles

    rows: list[dict[str, Any]] = []
    for smiles in smiles_list:
        if evaluate_fn is not None:
            result = evaluate_fn(smiles)
            if result is None:
                continue
            result = dict(result)
            result.setdefault("origin", origin)
            result.setdefault("generation", 0)
            rows.append(result)
            continue
        result = evaluate_smiles(
            smiles, spec, allow_papyrus=True, generation=0, origin=origin
        )
        if result is not None:
            rows.append(result)
    return rows


def hits_to_frame(hits: list[dict[str, Any]], spec: ConstraintSpec) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for hit in hits:
        values = hit.get("values") or {}
        if "descriptors" in hit:
            values = {
                name: hit["descriptors"][name]["value"]
                for name in spec.names()
                if name in hit["descriptors"]
            }
        scored = score_vector(values, spec)
        record = {
            "canonical_smiles": hit.get("canonical_smiles"),
            "inchikey": hit.get("inchikey"),
            "connectivity": hit.get("connectivity")
            or connectivity_from_inchikey(hit.get("inchikey")),
            "selfies": hit.get("selfies"),
            "penalty": scored["penalty"],
            "in_range": scored["in_range"],
            "n_passed": scored["n_passed"],
            "source": hit.get("source"),
            "origin": hit.get("origin"),
            "generation": hit.get("generation", 0),
        }
        for name in spec.names():
            record[f"value_{name}"] = scored["descriptors"][name]["value"]
            record[f"passed_{name}"] = scored["descriptors"][name]["passed"]
        records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    return frame.drop_duplicates(subset=["inchikey"], keep="first").reset_index(drop=True)


def hits_to_records(frame: pd.DataFrame, spec: ConstraintSpec) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for rec in frame.to_dict(orient="records"):
        values = {name: rec.get(f"value_{name}") for name in spec.names()}
        passed = {name: bool(rec.get(f"passed_{name}")) for name in spec.names()}
        rows.append(
            {
                "canonical_smiles": rec.get("canonical_smiles"),
                "inchikey": rec.get("inchikey"),
                "connectivity": rec.get("connectivity"),
                "penalty": rec.get("penalty"),
                "in_range": bool(rec.get("in_range")),
                "n_passed": rec.get("n_passed"),
                "source": rec.get("source"),
                "origin": rec.get("origin"),
                "generation": rec.get("generation", 0),
                "values": values,
                "passed": passed,
            }
        )
    return rows


def generate_molecules(
    spec: ConstraintSpec,
    *,
    population_size: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    seed: int = 0,
    smiles_seeds: list[str] | None = None,
    prior: str = "transformer",
    checkpoint: Path | None = None,
    out_path: Path | None = None,
    evaluate_fn=None,
    sample_fn=None,
    on_progress=None,
) -> dict[str, Any]:
    """Seed from the SELFIES GPT (default), Papyrus, or given SMILES; evolve if needed."""
    ensure_processed_dir()
    rng = Random(seed)
    n_seeds = max(population_size, spec.n)
    prior_name = (prior or "transformer").strip().lower()
    if smiles_seeds:
        population = candidates_from_smiles_list(
            smiles_seeds, spec, evaluate_fn=evaluate_fn, origin="seed"
        )
        prior_name = "smiles"
    elif prior_name == "transformer":
        from molecule_gen.generate.sample import sample_valid_smiles

        message = f"sampling {n_seeds} seeds from the SELFIES GPT"
        print(message, file=sys.stderr, flush=True)
        if on_progress is not None:
            on_progress({"type": "sampling", "message": message, "n_seeds": n_seeds})
        sampled = sample_valid_smiles(
            n_seeds, checkpoint=checkpoint, seed=seed, sample_fn=sample_fn
        )
        if on_progress is not None:
            on_progress(
                {
                    "type": "sampled",
                    "message": f"scored {len(sampled)} GPT seeds against the spec",
                    "n_seeds": len(sampled),
                }
            )
        population = candidates_from_smiles_list(
            sampled, spec, evaluate_fn=evaluate_fn, origin="transformer"
        )
    elif prior_name == "papyrus":
        table = seed_from_papyrus(spec, n_seeds=n_seeds, seed=seed)
        population = candidates_from_seed_table(table)
    else:
        raise ValueError(f"Unknown prior {prior!r}. Use 'papyrus' or 'transformer'.")
    if not population:
        raise RuntimeError("No valid seed molecules")

    hits = evolve(
        spec,
        population,
        population_size=population_size,
        generations=generations,
        rng=rng,
        evaluate_fn=evaluate_fn,
        on_progress=on_progress,
    )
    frame = hits_to_frame(hits, spec)
    dest = out_path or hits_parquet_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_parquet(dest, index=False)
    else:
        if dest.exists():
            dest.unlink()

    print(
        f"kept {len(frame)} unique in-range molecules (requested {spec.n})",
        file=sys.stderr,
    )
    if not frame.empty:
        print(f"wrote {dest}", file=sys.stderr)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_requested": spec.n,
        "n_hits": int(len(frame)),
        "n_seeds": len(population),
        "n_seed_in_range": sum(1 for row in population if row["in_range"]),
        "population_size": population_size,
        "generations": generations,
        "seed": seed,
        "prior": prior_name,
        "spec": spec.as_dict(),
        "output_parquet": _relpath(dest) if dest.exists() else None,
    }
    hits_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    if on_progress is not None:
        on_progress(
            {
                "type": "done",
                "message": (
                    f"kept {len(frame)} unique in-range molecules (requested {spec.n})"
                ),
                "manifest": manifest,
                "hits": hits_to_records(frame, spec),
            }
        )
    return manifest


def parse_smiles_seeds(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]
