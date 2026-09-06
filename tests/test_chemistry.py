from random import Random

import pandas as pd

from molecule_gen.generate.chemistry import (
    is_reasonable_molecule,
    mutation_alphabet,
)
from molecule_gen.generate.seed import pick_near_misses
from molecule_gen.generate.selfies_util import mutate_selfies, smiles_to_selfies


def test_mutation_alphabet_excludes_boron_and_charge():
    alphabet = mutation_alphabet()
    assert "[C]" in alphabet
    assert "[N]" in alphabet
    assert "[Branch1]" in alphabet
    assert "[B]" not in alphabet
    assert "[C+1]" not in alphabet
    assert "[O-1]" not in alphabet


def test_reasonable_molecule_accepts_ordinary_organics():
    assert is_reasonable_molecule("CCOC(=O)c1ccccc1")
    assert is_reasonable_molecule("c1ccccc1")
    assert is_reasonable_molecule("CC(=O)Nc1ccc(O)cc1")


def test_reasonable_molecule_rejects_boron_charge_allene_tiny():
    assert not is_reasonable_molecule("CCO")
    assert not is_reasonable_molecule("CCB(C)C")
    assert not is_reasonable_molecule("C[N+](C)(C)C")
    assert not is_reasonable_molecule("C=C=C")
    assert not is_reasonable_molecule(
        "C=CC(F)C=CC(=N)B(CNC=CC([S-])ONCCCCCCCCC1[C+]OC1)C1C=C=CC=C1"
    )


def test_mutate_stays_in_restricted_alphabet():
    encoded = smiles_to_selfies("c1ccccc1CCO")
    assert encoded is not None
    rng = Random(1)
    alphabet = set(mutation_alphabet())
    for _ in range(20):
        encoded = mutate_selfies(encoded, rng)
        from molecule_gen.generate.selfies_util import split_selfies

        for token in split_selfies(encoded):
            assert token in alphabet


def test_pick_near_misses_prefers_high_n_passed_then_low_penalty():
    frame = pd.DataFrame(
        {
            "in_range": [True, False, False, False],
            "n_passed": [5, 4, 4, 3],
            "penalty": [0.0, 0.5, 0.01, 0.0],
            "canonical_smiles": ["A", "B", "C", "D"],
        }
    )
    picked = pick_near_misses(frame, 2)
    assert list(picked["canonical_smiles"]) == ["C", "B"]
    assert bool(picked["in_range"].any()) is False
