from __future__ import annotations

import sys
from random import Random
from typing import Any, Callable

from tqdm import tqdm

from molecule_gen.constraints.lookup import LookupError, describe_for_scoring
from molecule_gen.constraints.score import score_vector
from molecule_gen.constraints.spec import ConstraintSpec
from molecule_gen.data.clean import canonicalize_smiles, connectivity_from_inchikey
from molecule_gen.descriptors.cache import DescriptorCache
from molecule_gen.generate.chemistry import is_reasonable_molecule
from molecule_gen.generate.diversity import diverse_elites
from molecule_gen.generate.local_edits import local_edit_smiles
from molecule_gen.generate.selfies_util import (
    crossover_selfies,
    mutate_selfies,
    selfies_to_canonical_smiles,
    smiles_to_selfies,
)

EvaluateFn = Callable[[str], dict[str, Any] | None]
ProgressFn = Callable[[dict[str, Any]], None]
ImmigrantFn = Callable[[int, dict[str, Any] | None, set[str]], list[dict[str, Any]]]

CLOSE_PENALTY = 0.01
CLOSE_STREAK = 3
STAGNATION_STREAK = 8
# Seeding stays cheap at the starting size; only a stuck search pays for a bigger
# population. Feasible specs converge before this ever fires. Measured on an
# infeasible 8-descriptor box: x4 every 8 stagnant gens = 11.7s 4/4, x2 = 19.0s 3/4,
# x4 every 4 = 77.5s 3/4. Growing too eagerly costs more than it buys.
# Fixed population by default: per-generation cost stays low. Set >1 to let a stuck
# search grow (x4 every 8 stagnant gens measured 11.7s 4/4 on an infeasible
# 8-descriptor box vs 17.0s 1/4 fixed; x2 19.0s 3/4; x4 every 4 gens 77.5s 3/4).
POPULATION_GROWTH = 1
MAX_POPULATION = 256
STAGNATION_EPS = 1e-6
LOCAL_EDIT_P = 1.0


def candidate_preview(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        "canonical_smiles": candidate.get("canonical_smiles"),
        "inchikey": candidate.get("inchikey"),
        "penalty": candidate.get("penalty"),
        "in_range": candidate.get("in_range"),
        "n_passed": candidate.get("n_passed"),
        "generation": candidate.get("generation", 0),
        "origin": candidate.get("origin"),
        "values": candidate.get("values") or {},
    }


def miss_summary(candidate: dict[str, Any] | None, spec: ConstraintSpec) -> str:
    """e.g. '7/8 in range; TopoPSA 100.25 > 100'."""
    values = (candidate or {}).get("values") or {}
    misses: list[str] = []
    for item in spec.ranges:
        value = values.get(item.name)
        if value is None:
            misses.append(f"{item.name} missing")
        elif value < item.minimum:
            misses.append(f"{item.name} {value:g} < {item.minimum:g}")
        elif value > item.maximum:
            misses.append(f"{item.name} {value:g} > {item.maximum:g}")
    text = f"{len(spec.ranges) - len(misses)}/{len(spec.ranges)} in range"
    return "; ".join([text, *misses]) if misses else text


def _emit(on_progress: ProgressFn | None, event: dict[str, Any]) -> None:
    message = event.get("message")
    if message:
        print(message, file=sys.stderr, flush=True)
    if on_progress is not None:
        on_progress(event)


def evaluate_smiles(
    smiles: str,
    spec: ConstraintSpec,
    *,
    cache: DescriptorCache | None = None,
    allow_papyrus: bool = False,
    generation: int = 0,
    origin: str = "mutant",
    require_reasonable: bool = False,
) -> dict[str, Any] | None:
    parsed = canonicalize_smiles(smiles)
    if parsed is None:
        return None
    canon, inchikey = parsed
    if require_reasonable and not is_reasonable_molecule(canon):
        return None
    encoded = smiles_to_selfies(canon)
    if encoded is None:
        return None
    connectivity = connectivity_from_inchikey(inchikey)
    try:
        values, source = describe_for_scoring(
            names=spec.names(),
            smiles=canon,
            inchikey=inchikey,
            connectivity=connectivity,
            cache=cache,
            allow_papyrus=allow_papyrus,
        )
    except LookupError:
        return None
    scored = score_vector(values, spec)
    return {
        "canonical_smiles": canon,
        "selfies": encoded,
        "inchikey": inchikey,
        "connectivity": connectivity,
        "values": {name: scored["descriptors"][name]["value"] for name in spec.names()},
        "penalty": scored["penalty"],
        "in_range": scored["in_range"],
        "n_passed": scored["n_passed"],
        "source": source,
        "origin": origin,
        "generation": generation,
    }


def _fitness(candidate: dict[str, Any]) -> tuple[int, float]:
    return (0 if candidate["in_range"] else 1, float(candidate["penalty"]))


def evolve(
    spec: ConstraintSpec,
    seeds: list[dict[str, Any]],
    *,
    population_size: int = 64,
    generations: int = 20,
    elite_frac: float = 0.25,
    crossover_p: float = 0.3,
    rng: Random | None = None,
    cache: DescriptorCache | None = None,
    evaluate_fn: EvaluateFn | None = None,
    on_progress: ProgressFn | None = None,
    immigrant_fn: ImmigrantFn | None = None,
) -> list[dict[str, Any]]:
    """Evolve SELFIES until spec.n in-range molecules exist or generations run out."""
    rng = rng or Random(0)
    owned = False
    store = cache
    if store is None and evaluate_fn is None:
        store = DescriptorCache()
        owned = True

    def evaluate(smiles: str, generation: int, origin: str) -> dict[str, Any] | None:
        if evaluate_fn is not None:
            result = evaluate_fn(smiles)
            if result is None:
                return None
            result = dict(result)
            result["generation"] = generation
            result["origin"] = origin
            return result
        return evaluate_smiles(
            smiles,
            spec,
            cache=store,
            allow_papyrus=False,
            generation=generation,
            origin=origin,
            require_reasonable=True,
        )

    try:
        hits: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        population: list[dict[str, Any]] = []
        for cand in seeds:
            key = cand["inchikey"]
            if key in seen:
                continue
            seen.add(key)
            population.append(cand)
            if cand["in_range"]:
                hits[key] = cand
        ranked = sorted(population, key=_fitness)
        _emit(
            on_progress,
            {
                "type": "ga_start",
                "message": (
                    f"GA start: {len(population)} seeds, {len(hits)} already in range "
                    f"(need {spec.n})"
                ),
                "generation": 0,
                "n_seeds": len(population),
                "n_hits": len(hits),
                "n_requested": spec.n,
                "best_penalty": ranked[0]["penalty"] if ranked else None,
                "best": candidate_preview(ranked[0] if ranked else None),
                "population": [candidate_preview(row) for row in ranked[:8]],
                "hits": [candidate_preview(row) for row in hits.values()],
            },
        )
        if len(hits) >= spec.n or generations <= 0:
            return list(hits.values())[: spec.n]

        current_size = population_size
        prev_best: float | None = None
        stagnant_streak = 0
        close_streak = 0
        for gen in range(1, generations + 1):
            if len(hits) >= spec.n:
                break
            if (
                POPULATION_GROWTH > 1
                and stagnant_streak >= STAGNATION_STREAK
                and current_size < MAX_POPULATION
            ):
                current_size = min(MAX_POPULATION, current_size * POPULATION_GROWTH)
                _emit(
                    on_progress,
                    {
                        "type": "population",
                        "message": (
                            f"generation {gen}: population -> {current_size} (stagnation)"
                        ),
                        "generation": gen,
                        "population_size": current_size,
                    },
                )
            n_elite = max(2, int(current_size * elite_frac))
            population.sort(key=_fitness)
            elites = diverse_elites(population, min(n_elite, len(population)))
            if len(elites) < 2 and population:
                elites = population[: min(n_elite, len(population))]
            needed = max(0, current_size - len(elites))
            children: list[dict[str, Any]] = []
            use_local = close_streak >= CLOSE_STREAK
            if (
                immigrant_fn is not None
                and stagnant_streak >= STAGNATION_STREAK
                and needed > 0
            ):
                n_imm = min(needed, max(1, needed // 2))
                best_now = population[0] if population else None
                raw = immigrant_fn(n_imm, best_now, seen)
                injected = 0
                for cand in raw:
                    key = cand.get("inchikey")
                    if not key or key in seen:
                        continue
                    row = dict(cand)
                    row["generation"] = gen
                    row.setdefault("origin", "immigrant")
                    seen.add(key)
                    children.append(row)
                    if row.get("in_range"):
                        hits[key] = row
                    injected += 1
                    if len(children) >= n_imm:
                        break
                stagnant_streak = 0
                _emit(
                    on_progress,
                    {
                        "type": "diversity",
                        "message": (
                            f"generation {gen}: injected {injected} diverse immigrants "
                            f"(stagnation)"
                        ),
                        "generation": gen,
                        "n_hits": len(hits),
                    },
                )
            attempts = 0
            max_attempts = max(400, needed * 40)
            bar = tqdm(
                total=needed,
                desc=f"gen {gen}",
                unit="mol",
                file=sys.stderr,
                leave=False,
            )
            try:
                bar.update(len(children))
                while len(children) < needed and attempts < max_attempts:
                    attempts += 1
                    parent = rng.choice(elites)
                    child_smiles = None
                    origin = "mutant"
                    if use_local and rng.random() < LOCAL_EDIT_P:
                        child_smiles = local_edit_smiles(
                            str(parent.get("canonical_smiles") or ""), rng
                        )
                        origin = "local_edit"
                        parsed = canonicalize_smiles(child_smiles) if child_smiles else None
                        if parsed is not None and parsed[1] in seen:
                            child_smiles = None  # exhausted neighbourhood: use SELFIES instead
                    if child_smiles is None:
                        origin = "mutant"
                        if rng.random() < crossover_p and len(elites) >= 2:
                            other = rng.choice(elites)
                            child_sf = crossover_selfies(
                                parent["selfies"], other["selfies"], rng
                            )
                        else:
                            child_sf = mutate_selfies(parent["selfies"], rng)
                        child_smiles = selfies_to_canonical_smiles(child_sf)
                    if child_smiles is None:
                        continue
                    if not is_reasonable_molecule(child_smiles):
                        continue
                    child = evaluate(child_smiles, gen, origin)
                    if child is None or child["inchikey"] in seen:
                        continue
                    seen.add(child["inchikey"])
                    children.append(child)
                    if child["in_range"]:
                        hits[child["inchikey"]] = child
                    bar.update(1)
                    bar.set_postfix(hits=len(hits), penalty=f"{child['penalty']:.3f}")
            finally:
                bar.close()
            population = elites + children
            ranked = sorted(population, key=_fitness)
            best = ranked[0] if ranked else None
            best_pen_value = float(best["penalty"]) if best else None
            if best_pen_value is not None and best_pen_value < CLOSE_PENALTY:
                close_streak += 1
            else:
                close_streak = 0
            if (
                prev_best is not None
                and best_pen_value is not None
                and abs(best_pen_value - prev_best) < STAGNATION_EPS
            ):
                stagnant_streak += 1
            else:
                stagnant_streak = 0
            prev_best = best_pen_value
            best_pen = f"{best_pen_value:.3g}" if best_pen_value is not None else "?"
            mode = "local" if use_local else "selfies"
            _emit(
                on_progress,
                {
                    "type": "generation",
                    "message": (
                        f"generation {gen}: pop={len(population)} hits={len(hits)} "
                        f"best_penalty={best_pen} ({miss_summary(best, spec)}) mode={mode}"
                    ),
                    "generation": gen,
                    "n_population": len(population),
                    "n_hits": len(hits),
                    "n_requested": spec.n,
                    "best_penalty": best_pen_value,
                    "best": candidate_preview(best),
                    "population": [candidate_preview(row) for row in ranked[:8]],
                    "hits": [candidate_preview(row) for row in hits.values()],
                    "mode": mode,
                    "close_streak": close_streak,
                    "stagnant_streak": stagnant_streak,
                },
            )
        return list(hits.values())[: spec.n]
    finally:
        if owned and store is not None:
            store.close()
