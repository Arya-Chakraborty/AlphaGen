from random import Random

from molecule_gen.constraints.spec import parse_spec
from molecule_gen.data.clean import canonicalize_smiles, connectivity_from_inchikey
from molecule_gen.generate.diversity import diverse_elites, failed_descriptor_names, scaffold_key
from molecule_gen.generate.ga import STAGNATION_STREAK, evolve
from molecule_gen.generate.local_edits import (
    add_methyl,
    insert_ch2,
    local_edit_smiles,
    remove_methyl,
    swap_hydrogen_fluorine,
)
from molecule_gen.generate.selfies_util import smiles_to_selfies


ETHYL_BENZOATE = "CCOC(=O)c1ccccc1"


def test_add_and_remove_methyl_change_molecule():
    added = add_methyl(ETHYL_BENZOATE, Random(0))
    assert added is not None
    assert added != ETHYL_BENZOATE
    removed = remove_methyl(added, Random(1))
    assert removed is not None


def test_insert_ch2_lengthens_aliphatic_chain():
    longer = insert_ch2(ETHYL_BENZOATE, Random(0))
    assert longer is not None
    assert longer != ETHYL_BENZOATE


def test_swap_hydrogen_fluorine_adds_fluorine():
    swapped = swap_hydrogen_fluorine("c1ccccc1", Random(0))
    assert swapped is not None
    assert "F" in swapped


def test_local_edit_returns_different_reasonable_smiles():
    rng = Random(2)
    edited = None
    for _ in range(8):
        edited = local_edit_smiles(ETHYL_BENZOATE, rng)
        if edited:
            break
    assert edited
    assert edited != ETHYL_BENZOATE
    parsed = canonicalize_smiles(edited)
    assert parsed is not None


def test_diverse_elites_cap_same_murcko():
    naphthalene = "c1ccc2ccccc2c1"
    methyl_naphthalene = "Cc1ccc2ccccc2c1"
    benzene = "c1ccccc1CCO"
    population = [
        {
            "canonical_smiles": naphthalene,
            "inchikey": "n1",
            "connectivity": "NAPH1",
            "penalty": 0.01,
            "in_range": False,
        },
        {
            "canonical_smiles": methyl_naphthalene,
            "inchikey": "n2",
            "connectivity": "NAPH2",
            "penalty": 0.02,
            "in_range": False,
        },
        {
            "canonical_smiles": benzene,
            "inchikey": "b1",
            "connectivity": "BENZ1",
            "penalty": 0.05,
            "in_range": False,
        },
    ]
    elites = diverse_elites(population, 2)
    assert len(elites) == 2
    keys = {scaffold_key(row) for row in elites}
    assert len(keys) == 2
    smiles = {row["canonical_smiles"] for row in elites}
    assert naphthalene in smiles
    assert benzene in smiles


def test_failed_descriptor_names():
    spec = parse_spec(
        {
            "n": 1,
            "descriptors": {
                "SLogP": {"min": 2.8, "max": 3.2},
                "MW": {"min": 360, "max": 390},
            },
        }
    )
    failed = failed_descriptor_names(
        {"values": {"SLogP": 3.0, "MW": 400.0}},
        spec,
    )
    assert failed == {"MW"}


def _eval(smiles: str):
    parsed = canonicalize_smiles(smiles)
    if parsed is None:
        return None
    canon, key = parsed
    encoded = smiles_to_selfies(canon)
    if encoded is None:
        return None
    return {
        "canonical_smiles": canon,
        "selfies": encoded,
        "inchikey": key,
        "connectivity": connectivity_from_inchikey(key),
        "values": {"MW": 80.0},
        "penalty": 0.005,
        "in_range": False,
        "n_passed": 0,
        "source": "fake",
    }


def test_stagnation_calls_immigrant_fn():
    spec = parse_spec({"n": 5, "descriptors": {"MW": {"min": 40, "max": 50}}})
    seeds = []
    for smiles in (ETHYL_BENZOATE, "CC(=O)Nc1ccc(O)cc1", "c1ccc(CCO)cc1"):
        row = _eval(smiles)
        assert row is not None
        row["origin"] = "seed"
        row["generation"] = 0
        seeds.append(row)
    calls: list[int] = []

    def immigrant_fn(n, stuck, seen):
        calls.append(n)
        assert stuck is not None
        assert isinstance(seen, set)
        return []

    evolve(
        spec,
        seeds,
        population_size=8,
        generations=STAGNATION_STREAK + 3,
        evaluate_fn=_eval,
        immigrant_fn=immigrant_fn,
    )
    assert calls, "expected diversity injection after unchanged best penalty"


def test_miss_summary_names_failing_descriptors():
    from molecule_gen.generate.ga import miss_summary

    spec = parse_spec(
        {
            "n": 1,
            "descriptors": {
                "MW": {"min": 300, "max": 400},
                "TopoPSA": {"min": 40, "max": 100},
                "nRot": {"min": 10, "max": 20},
            },
        }
    )
    row = {"values": {"MW": 350.0, "TopoPSA": 100.25, "nRot": 9.0}}
    text = miss_summary(row, spec)
    assert text.startswith("1/3 in range")
    assert "TopoPSA 100.25 > 100" in text
    assert "nRot 9 < 10" in text
    assert "MW" not in text
    assert miss_summary({"values": {"MW": 350.0, "TopoPSA": 50.0, "nRot": 12.0}}, spec) == (
        "3/3 in range"
    )


def test_local_mode_falls_back_when_local_edits_are_duplicates(monkeypatch):
    from molecule_gen.generate import ga

    spec = parse_spec({"n": 5, "descriptors": {"MW": {"min": 40, "max": 50}}})
    seeds = []
    for smiles in (ETHYL_BENZOATE, "CC(=O)Nc1ccc(O)cc1", "c1ccc(CCO)cc1", "CCCCCCO"):
        row = _eval(smiles)
        row["origin"] = "seed"
        row["generation"] = 0
        seeds.append(row)
    # every local edit proposes a molecule that is already known
    monkeypatch.setattr(ga, "local_edit_smiles", lambda smiles, rng: ETHYL_BENZOATE)
    events: list[dict] = []
    evolve(
        spec,
        seeds,
        population_size=16,
        generations=8,
        evaluate_fn=_eval,
        on_progress=lambda ev: events.append(ev) if ev["type"] == "generation" else None,
    )
    local_gens = [ev for ev in events if ev["mode"] == "local"]
    assert local_gens, "test must reach local mode"
    n_elite = max(2, int(16 * 0.25))
    assert all(ev["n_population"] > n_elite for ev in local_gens)


def _stagnant_seeds():
    seeds = []
    for smiles in (ETHYL_BENZOATE, "CC(=O)Nc1ccc(O)cc1", "c1ccc(CCO)cc1", "CCCCCCO"):
        row = _eval(smiles)
        row["origin"] = "seed"
        row["generation"] = 0
        seeds.append(row)
    return seeds


def _run_sizes(generations: int, population_size: int = 8) -> list[int]:
    spec = parse_spec({"n": 5, "descriptors": {"MW": {"min": 40, "max": 50}}})
    events: list[dict] = []
    evolve(
        spec,
        _stagnant_seeds(),
        population_size=population_size,
        generations=generations,
        evaluate_fn=_eval,
        on_progress=events.append,
    )
    return [ev["n_population"] for ev in events if ev["type"] == "generation"]


def test_population_is_fixed_by_default():
    sizes = _run_sizes(STAGNATION_STREAK * 3 + 6)
    assert set(sizes) <= {8}, f"population should stay at 8, saw {sizes}"


def test_population_grows_when_growth_enabled(monkeypatch):
    from molecule_gen.generate import ga

    monkeypatch.setattr(ga, "POPULATION_GROWTH", 4)
    sizes = _run_sizes(STAGNATION_STREAK + 6)
    assert max(sizes) > 8, f"expected growth past 8, saw {sizes}"


def test_population_stable_before_stagnation_threshold():
    sizes = _run_sizes(STAGNATION_STREAK - 1)
    assert max(sizes) <= 8, f"grew too early: {sizes}"


def test_population_growth_respects_cap(monkeypatch):
    from molecule_gen.generate import ga

    monkeypatch.setattr(ga, "POPULATION_GROWTH", 4)
    monkeypatch.setattr(ga, "MAX_POPULATION", 16)
    sizes = _run_sizes(STAGNATION_STREAK * 4 + 6)
    assert max(sizes) <= 16, f"exceeded cap: {sizes}"
