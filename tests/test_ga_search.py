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
