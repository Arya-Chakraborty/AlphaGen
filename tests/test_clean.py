import pandas as pd

from molecule_gen.data.clean import (
    canonicalize_smiles,
    canonicalize_table,
    connectivity_from_inchikey,
)

ETHANOL_KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_canonicalize_ethanol():
    result = canonicalize_smiles("CCO")
    assert result is not None
    smiles, key = result
    assert smiles == "CCO"
    assert key == ETHANOL_KEY


def test_canonicalize_equivalent_smiles():
    a = canonicalize_smiles("OCC")
    b = canonicalize_smiles("CCO")
    assert a is not None and b is not None
    assert a[0] == b[0] == "CCO"
    assert a[1] == b[1] == ETHANOL_KEY


def test_canonicalize_invalid():
    assert canonicalize_smiles("not-a-molecule") is None
    assert canonicalize_smiles("") is None
    assert canonicalize_smiles(None) is None


def test_connectivity_from_inchikey():
    assert connectivity_from_inchikey(ETHANOL_KEY) == "LFQSCWFLJHTTHZ"
    assert connectivity_from_inchikey("") == ""
    assert connectivity_from_inchikey(None) == ""


def test_canonicalize_table_drops_invalid_and_dedupes():
    frame = pd.DataFrame(
        {
            "connectivity": ["LFQSCWFLJHTTHZ", "LFQSCWFLJHTTHZ", "", "UHOVQNZJYSORNB"],
            "canonical_smiles": ["CCO", "OCC", "not-valid", "c1ccccc1"],
            "inchikey": ["", "", "", ""],
        }
    )
    cleaned, stats = canonicalize_table(frame, workers=1, batch_size=2)
    assert stats["n_input"] == 4
    assert stats["n_invalid"] == 1
    assert stats["n_valid"] == 3
    assert stats["n_unique"] == 2
    assert stats["n_duplicate_structures"] == 1
    assert set(cleaned["inchikey"]) == {
        ETHANOL_KEY,
        canonicalize_smiles("c1ccccc1")[1],
    }
    ethanol_row = cleaned.loc[cleaned["canonical_smiles"] == "CCO"].iloc[0]
    assert ethanol_row["connectivity"] == "LFQSCWFLJHTTHZ"
