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
from molecule_gen.generate.seed import (
    candidates_from_seed_table,
    seed_from_papyrus,
    seed_papyrus_near_misses,
)
from molecule_gen.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_processed_dir

DEFAULT_POPULATION = 64
DEFAULT_GENERATIONS = 20
DEFAULT_NEAR_MISS_FRAC = 0.5
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


def _dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("inchikey") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _population_from_transformer(
    spec: ConstraintSpec,
    *,
    n_seeds: int,
    seed: int,
    checkpoint: Path | None,
    evaluate_fn,
    sample_fn,
    on_progress,
    mix_near_misses: bool,
    near_miss_frac: float,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    from molecule_gen.generate.chemistry import is_reasonable_molecule
    from molecule_gen.generate.sample import sample_valid_smiles

    use_near = (
        mix_near_misses
        and evaluate_fn is None
        and sample_fn is None
        and 0 < near_miss_frac < 1
    )
    n_near = int(round(n_seeds * near_miss_frac)) if use_near else 0
    n_gpt = max(1, n_seeds - n_near) if n_near else n_seeds

    message = f"sampling {n_gpt} seeds from the SELFIES GPT"
    print(message, file=sys.stderr, flush=True)
    if on_progress is not None:
        on_progress({"type": "sampling", "message": message, "n_seeds": n_gpt})
    sampled = sample_valid_smiles(
        n_gpt * 2 if sample_fn is None else n_gpt,
        checkpoint=checkpoint,
        seed=seed,
        sample_fn=sample_fn,
    )
    if sample_fn is None:
        sampled = [smi for smi in sampled if is_reasonable_molecule(smi)][:n_gpt]
    else:
        sampled = sampled[:n_gpt]
    if on_progress is not None:
        on_progress(
            {
                "type": "sampled",
                "message": f"kept {len(sampled)} chemically reasonable GPT seeds",
                "n_seeds": len(sampled),
            }
        )
    gpt_rows = candidates_from_smiles_list(
        sampled, spec, evaluate_fn=evaluate_fn, origin="transformer"
    )

    near_rows: list[dict[str, Any]] = []
    near_pool: list[dict[str, Any]] = []
    if n_near:
        try:
            table = seed_papyrus_near_misses(spec, n_seeds=n_near, seed=seed)
            encoded = candidates_from_seed_table(
                table,
                origin="near_miss",
                require_reasonable=True,
                limit=n_near + 64,
            )
            near_rows = encoded[:n_near]
            near_pool = encoded[n_near:]
            msg = f"mixed in {len(near_rows)} Papyrus near-miss seeds"
            print(msg, file=sys.stderr, flush=True)
            if on_progress is not None:
                on_progress({"type": "near_miss", "message": msg, "n_seeds": len(near_rows)})
        except FileNotFoundError as exc:
            print(f"near-miss seeds skipped: {exc}", file=sys.stderr, flush=True)

    population = _dedupe_candidates(near_rows + gpt_rows)
    if len(population) < n_seeds and gpt_rows and not near_rows:
        population = gpt_rows
    return population, len(gpt_rows), len(near_rows), near_pool


def _make_immigrant_fn(
    spec: ConstraintSpec,
    *,
    checkpoint: Path | None,
    seed: int,
    evaluate_fn,
    sample_fn,
    pool: list[dict[str, Any]],
):
    from molecule_gen.generate.chemistry import is_reasonable_molecule
    from molecule_gen.generate.diversity import failed_descriptor_names

    counter = {"n": seed}

    def immigrant_fn(n: int, stuck, seen: set[str]) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        stuck_failed = failed_descriptor_names(stuck, spec)
        picked: list[dict[str, Any]] = []
        ranked = sorted(
            pool,
            key=lambda row: (
                failed_descriptor_names(row, spec) == stuck_failed,
                float(row.get("penalty", 1e9)),
            ),
        )
        for row in ranked:
            key = row.get("inchikey")
            if not key or key in seen:
                continue
            row_failed = failed_descriptor_names(row, spec)
            if stuck_failed and row_failed == stuck_failed:
                continue
            picked.append(dict(row))
            if len(picked) >= max(1, n // 2):
                break
        need = n - len(picked)
        if need <= 0:
            return picked[:n]
        counter["n"] += 17
        try:
            from molecule_gen.generate.sample import sample_valid_smiles
            from molecule_gen.generate.transformer import TransformerError

            sampled = sample_valid_smiles(
                need,
                checkpoint=checkpoint,
                seed=counter["n"],
                sample_fn=sample_fn,
                max_draws=max(need * 8, need),
            )
        except (TransformerError, FileNotFoundError, Exception):
            sampled = []
        if sample_fn is None:
            sampled = [smi for smi in sampled if is_reasonable_molecule(smi)]
        gpt_rows = candidates_from_smiles_list(
            sampled, spec, evaluate_fn=evaluate_fn, origin="immigrant"
        )
        have = {row.get("inchikey") for row in picked}
        have.update(seen)
        for row in gpt_rows:
            key = row.get("inchikey")
            if not key or key in have:
                continue
            picked.append(row)
            have.add(key)
            if len(picked) >= n:
                break
        return picked[:n]

    return immigrant_fn


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
    mix_near_misses: bool = True,
    near_miss_frac: float = DEFAULT_NEAR_MISS_FRAC,
) -> dict[str, Any]:
    """Seed from the SELFIES GPT mixed with Papyrus near-misses; evolve if needed."""
    ensure_processed_dir()
    rng = Random(seed)
    n_seeds = max(population_size, spec.n)
    prior_name = (prior or "transformer").strip().lower()
    n_gpt_seeds = 0
    n_near_miss_seeds = 0
    immigrant_pool: list[dict[str, Any]] = []
    if smiles_seeds:
        population = candidates_from_smiles_list(
            smiles_seeds, spec, evaluate_fn=evaluate_fn, origin="seed"
        )
        prior_name = "smiles"
    elif prior_name == "transformer":
        population, n_gpt_seeds, n_near_miss_seeds, immigrant_pool = (
            _population_from_transformer(
                spec,
                n_seeds=n_seeds,
                seed=seed,
                checkpoint=checkpoint,
                evaluate_fn=evaluate_fn,
                sample_fn=sample_fn,
                on_progress=on_progress,
                mix_near_misses=mix_near_misses,
                near_miss_frac=near_miss_frac,
            )
        )
        if n_near_miss_seeds:
            prior_name = "transformer+near_miss"
    elif prior_name == "papyrus":
        table = seed_from_papyrus(spec, n_seeds=n_seeds, seed=seed)
        population = candidates_from_seed_table(table)
        n_near_miss_seeds = sum(1 for row in population if not row["in_range"])
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
        immigrant_fn=_make_immigrant_fn(
            spec,
            checkpoint=checkpoint,
            seed=seed,
            evaluate_fn=evaluate_fn,
            sample_fn=sample_fn,
            pool=immigrant_pool,
        ),
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
        "n_gpt_seeds": n_gpt_seeds,
        "n_near_miss_seeds": n_near_miss_seeds,
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
