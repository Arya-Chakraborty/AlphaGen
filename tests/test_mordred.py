from pathlib import Path

import pandas as pd
import pytest

from molecule_gen.descriptors.cache import DescriptorCache
from molecule_gen.descriptors.compare import (
    compare_descriptor_tables,
    papyrus_name_to_local,
)
from molecule_gen.descriptors.mordred import describe_molecules, normalize_descriptors


def test_normalize_descriptors():
    raw = {"Name": "AUTOGEN_1", "nAtom": "9", "SLogP": "0.5", "bad": "not-a-number"}
    out = normalize_descriptors(raw)
    assert "Name" not in out
    assert out["nAtom"] == 9.0
    assert out["SLogP"] == 0.5
    assert out["bad"] is None


def test_cache_roundtrip(tmp_path: Path):
    cache = DescriptorCache(tmp_path / "cache.sqlite")
    payload = {"nAtom": 9.0, "SLogP": 0.5}
    cache.put("KEY1", "CCO", payload)
    assert cache.get("KEY1") == payload
    assert cache.get("missing") is None
    assert cache.has("KEY1")
    assert not cache.has("missing")
    assert len(cache) == 1
    cache.close()


def test_describe_uses_cache_on_second_pass(tmp_path: Path):
    calls: list[list[str]] = []

    def fake_compute(smiles_list: list[str]):
        calls.append(list(smiles_list))
        return [{"nAtom": float(len(s)), "nBond": 1.0} for s in smiles_list]

    frame = pd.DataFrame(
        {
            "connectivity": ["AAA", "BBB"],
            "canonical_smiles": ["CCO", "CCN"],
            "inchikey": ["KEYA", "KEYB"],
        }
    )
    cache = DescriptorCache(tmp_path / "cache.sqlite")
    first, stats1 = describe_molecules(
        frame, cache=cache, batch_size=2, compute_fn=fake_compute, show_progress=False
    )
    second, stats2 = describe_molecules(
        frame, cache=cache, batch_size=2, compute_fn=fake_compute, show_progress=False
    )
    cache.close()

    assert stats1["n_computed"] == 2
    assert stats1["n_cached"] == 0
    assert stats2["n_computed"] == 0
    assert stats2["n_cached"] == 2
    assert len(calls) == 1
    assert list(first["inchikey"]) == list(second["inchikey"])
    assert list(first["connectivity"]) == ["AAA", "BBB"]


def test_compare_descriptor_tables_agreement():
    local = pd.DataFrame(
        {
            "connectivity": ["AAA", "BBB"],
            "canonical_smiles": ["CCO", "CCN"],
            "inchikey": ["KEYA", "KEYB"],
            "SLogP": [0.5, 1.25],
            "nAtom": [9.0, 10.0],
            "only_local": [1.0, 2.0],
        }
    )
    papyrus = pd.DataFrame(
        {
            "connectivity": ["AAA", "BBB"],
            "SLogP": [0.5, 1.25 + 1e-8],
            "nAtom": [9.0, 10.0],
            "n6aliphRing": [1.0, 0.0],
            "only_papyrus": [3.0, 4.0],
        }
    )
    local["n6ARing"] = [1.0, 0.0]
    stats = compare_descriptor_tables(local, papyrus)
    assert papyrus_name_to_local("n6aliphRing") == "n6ARing"
    assert stats["n_molecules_compared"] == 2
    assert stats["n_shared_descriptors"] == 3
    assert stats["n_only_local"] == 1
    assert stats["n_only_papyrus"] == 1
    assert stats["n_both_finite"] == 6
    assert stats["max_abs"] < 1e-7
    assert stats["within_1e-06"] == 1.0


@pytest.mark.integration
def test_mordred_ethanol_cache(tmp_path: Path):
    pytest.importorskip("mordred")
    from molecule_gen.data.clean import canonicalize_smiles
    from molecule_gen.descriptors.mordred import (
        EXPECTED_N_DESCRIPTORS,
        mordred_from_smiles_batch,
    )

    result = canonicalize_smiles("CCO")
    assert result is not None
    smiles, key = result
    frame = pd.DataFrame(
        {
            "connectivity": [key.split("-", 1)[0]],
            "canonical_smiles": [smiles],
            "inchikey": [key],
        }
    )
    cache = DescriptorCache(tmp_path / "cache.sqlite")
    first, stats1 = describe_molecules(
        frame,
        cache=cache,
        batch_size=1,
        compute_fn=mordred_from_smiles_batch,
        show_progress=False,
    )
    second, stats2 = describe_molecules(
        frame,
        cache=cache,
        batch_size=1,
        compute_fn=mordred_from_smiles_batch,
        show_progress=False,
    )
    cache.close()

    n_desc = len(
        [
            c
            for c in first.columns
            if c not in {"connectivity", "canonical_smiles", "inchikey"}
        ]
    )
    assert n_desc == EXPECTED_N_DESCRIPTORS
    assert stats1["n_computed"] == 1
    assert stats2["n_cached"] == 1
    assert stats2["n_computed"] == 0
    assert first["nAtom"].iloc[0] == second["nAtom"].iloc[0]


@pytest.mark.integration
def test_mordred_subset_computes_only_requested():
    pytest.importorskip("mordred")
    from molecule_gen.descriptors.mordred import mordred_from_smiles_batch

    names = ["nAtom", "SLogP"]
    subset = mordred_from_smiles_batch(["CCO", "not-a-smiles"], names=names)
    full = mordred_from_smiles_batch(["CCO"])
    assert subset[1] is None
    assert set(subset[0]) == set(names)
    assert subset[0] == {name: full[0][name] for name in names}


def test_describe_for_scoring_computes_only_missing_names(tmp_path: Path, monkeypatch):
    from molecule_gen.constraints import lookup

    calls: list[tuple[list[str], list[str] | None]] = []

    def fake_batch(smiles_list, names=None):
        calls.append((list(smiles_list), names))
        return [{name: 1.0 for name in names} for _ in smiles_list]

    monkeypatch.setattr(lookup, "mordred_from_smiles_batch", fake_batch)
    cache = DescriptorCache(tmp_path / "cache.sqlite")
    kwargs = dict(
        smiles="CCO", inchikey="KEY", connectivity=None, cache=cache, allow_papyrus=False
    )

    values, source = lookup.describe_for_scoring(names=["nAtom"], **kwargs)
    assert (values, source) == ({"nAtom": 1.0}, "local")
    assert calls == [(["CCO"], ["nAtom"])]

    # same names again: pure cache hit
    _, source = lookup.describe_for_scoring(names=["nAtom"], **kwargs)
    assert source == "cache"
    assert len(calls) == 1

    # new name: cached partial payload must not yield None; only the gap is computed
    values, source = lookup.describe_for_scoring(names=["nAtom", "SLogP"], **kwargs)
    assert values == {"nAtom": 1.0, "SLogP": 1.0}
    assert calls[-1] == (["CCO"], ["SLogP"])
    assert set(cache.get("KEY")) == {"nAtom", "SLogP"}
    cache.close()
