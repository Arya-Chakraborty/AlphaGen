from pathlib import Path

import pandas as pd
import pytest

from molecule_gen.generate.plausibility import (
    environment_set,
    plausibility,
    unseen_fraction,
    unstable_motifs,
)

# Molecules the GA produced for an 8-descriptor spec. All passed the old filter.
GA_OUTPUT = (
    "C=CCNCNONNC=CN(C)C1[SH]=NNN1NC",
    "OCCPCCCCCNCC(O)C=CC=C(O)C(CO)CCCl",
    "OCC=[SH]CCCCCCNCC(O)c1ccc(O)c(CO)c1",
)
REAL_DRUGS = (
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CCOC(=O)c1ccccc1",  # ethyl benzoate
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "OC(=O)c1ccccc1O",  # salicylic acid
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)NNC(=O)c1ccncc1",  # a hydrazide: legitimate, must not be rejected
)


@pytest.mark.parametrize("smiles", GA_OUTPUT)
def test_unstable_motifs_flags_ga_output(smiles: str):
    assert unstable_motifs(smiles)


@pytest.mark.parametrize("smiles", REAL_DRUGS)
def test_unstable_motifs_passes_real_drugs(smiles: str):
    assert unstable_motifs(smiles) == []


def test_unstable_motifs_names_the_motif():
    assert "n_o_n" in unstable_motifs("CNONC")
    assert "peroxide" in unstable_motifs("CCOOCC")
    assert unstable_motifs("not-a-smiles") == []


def test_is_reasonable_molecule_rejects_unstable_motifs():
    from molecule_gen.generate.chemistry import is_reasonable_molecule

    for smiles in GA_OUTPUT:
        assert not is_reasonable_molecule(smiles)
    for smiles in REAL_DRUGS:
        assert is_reasonable_molecule(smiles)


def _write_env_set(path: Path, smiles_list: list[str], min_count: int = 1) -> None:
    from collections import Counter

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as fg

    gen = fg.GetMorganGenerator(radius=2)
    counts: Counter = Counter()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        counts.update(gen.GetSparseCountFingerprint(mol).GetNonzeroElements())
    keep = [int(k) for k, v in counts.items() if v >= min_count]
    pd.DataFrame({"env": pd.Series(keep, dtype="uint64")}).to_parquet(path, index=False)


def test_unseen_fraction_zero_for_corpus_member(tmp_path: Path):
    path = tmp_path / "env.parquet"
    _write_env_set(path, ["CC(=O)Nc1ccc(O)cc1", "CCOC(=O)c1ccccc1"])
    assert unseen_fraction("CC(=O)Nc1ccc(O)cc1", path=path) == 0.0


def test_unseen_fraction_high_for_molecule_outside_corpus(tmp_path: Path):
    path = tmp_path / "env.parquet"
    _write_env_set(path, ["CCOC(=O)c1ccccc1"])
    assert unseen_fraction("C=CCNCNONNC=CN(C)C1[SH]=NNN1NC", path=path) > 0.5


def test_unseen_fraction_is_none_without_artifact(tmp_path: Path):
    assert unseen_fraction("CCO", path=tmp_path / "missing.parquet") is None


def test_unseen_fraction_none_for_invalid_smiles(tmp_path: Path):
    path = tmp_path / "env.parquet"
    _write_env_set(path, ["CCO"])
    assert unseen_fraction("not-a-smiles", path=path) is None


def test_environment_set_is_cached(tmp_path: Path):
    path = tmp_path / "env.parquet"
    _write_env_set(path, ["CCO"])
    first = environment_set(path)
    assert environment_set(path) is first


def test_plausibility_reports_both_signals(tmp_path: Path):
    path = tmp_path / "env.parquet"
    _write_env_set(path, ["CCOC(=O)c1ccccc1"])

    good = plausibility("CCOC(=O)c1ccccc1", path=path)
    assert good == {"plausible": True, "motifs": [], "unseen_frac": 0.0}

    bad = plausibility(GA_OUTPUT[0], path=path)
    assert bad["plausible"] is False
    assert bad["motifs"]
    assert bad["unseen_frac"] > 0.5


@pytest.mark.integration
def test_unstable_motifs_rare_in_papyrus():
    """Hard rules must not reject real medicinal chemistry."""
    from molecule_gen.data.papyrus import molecules_parquet_path

    path = molecules_parquet_path()
    if not path.exists():
        pytest.skip("Papyrus molecule table not built")
    sample = (
        pd.read_parquet(path, columns=["canonical_smiles"])
        .canonical_smiles.sample(4000, random_state=11)
        .tolist()
    )
    flagged = sum(1 for smiles in sample if unstable_motifs(smiles))
    assert flagged / len(sample) < 0.01


def test_unstable_motifs_in_mol_matches_smiles_version():
    from rdkit import Chem

    from molecule_gen.generate.plausibility import unstable_motifs_in_mol

    for smiles in (*GA_OUTPUT, *REAL_DRUGS):
        mol = Chem.MolFromSmiles(smiles)
        assert unstable_motifs_in_mol(mol) == unstable_motifs(smiles)
    assert unstable_motifs_in_mol(None) == []


def test_is_reasonable_molecule_parses_smiles_once(monkeypatch):
    """The motif check must reuse the parsed Mol, not re-parse the string."""
    from rdkit import Chem

    from molecule_gen.generate.chemistry import is_reasonable_molecule

    calls: list[str] = []
    real = Chem.MolFromSmiles

    def counting(smiles, *a, **k):
        calls.append(smiles)
        return real(smiles, *a, **k)

    monkeypatch.setattr(Chem, "MolFromSmiles", counting)
    assert is_reasonable_molecule("CC(=O)Nc1ccc(O)cc1")
    assert len(calls) == 1
