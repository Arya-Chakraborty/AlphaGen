from random import Random

import pandas as pd
import pytest

from molecule_gen.constraints.spec import parse_spec
from molecule_gen.data.clean import canonicalize_smiles, connectivity_from_inchikey
from molecule_gen.generate.run import generate_molecules
from molecule_gen.generate.selfies_util import (
    encode_roundtrip,
    mutate_selfies,
    smiles_to_selfies,
)


def test_ethanol_selfies_roundtrip():
    result = encode_roundtrip("CCO")
    assert result is not None
    smiles, encoded = result
    assert smiles == "CCO"
    assert encoded
    assert "[" in encoded


def test_mutate_selfies_returns_string():
    encoded = smiles_to_selfies("CCO")
    assert encoded is not None
    rng = Random(0)
    mutant = mutate_selfies(encoded, rng)
    assert isinstance(mutant, str)
    assert mutant


def _fake_eval(smiles: str):
    parsed = canonicalize_smiles(smiles)
    if parsed is None:
        return None
    canon, key = parsed
    encoded = smiles_to_selfies(canon)
    if encoded is None:
        return None
    in_range = canon == "CCO"
    return {
        "canonical_smiles": canon,
        "selfies": encoded,
        "inchikey": key,
        "connectivity": connectivity_from_inchikey(key),
        "values": {"MW": 46.0 if in_range else 80.0},
        "penalty": 0.0 if in_range else 1.0,
        "in_range": in_range,
        "n_passed": 1 if in_range else 0,
        "source": "fake",
    }


def test_generate_keeps_in_range_seeds_only(tmp_path):
    spec = parse_spec({"n": 2, "descriptors": {"MW": {"min": 40, "max": 50}}})
    out = tmp_path / "hits.parquet"
    manifest = generate_molecules(
        spec,
        population_size=8,
        generations=0,
        seed=0,
        smiles_seeds=["CCO", "CCC", "CCN"],
        out_path=out,
        evaluate_fn=_fake_eval,
    )
    assert manifest["n_hits"] == 1
    hits = pd.read_parquet(out)
    assert list(hits["canonical_smiles"]) == ["CCO"]
    assert bool(hits.iloc[0]["in_range"]) is True
