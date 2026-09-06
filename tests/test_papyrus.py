from pathlib import Path

import pandas as pd
from rdkit import Chem

from molecule_gen.data.papyrus import (
    MORDRED_ID_COL,
    canonicalize_molecules,
    mordred_tsv_to_parquet,
    parse_papyrus_sd,
)


ETHANOL_KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
ETHANOL_CONN = "LFQSCWFLJHTTHZ"


def _write_sd(path: Path, smiles: str, connectivity: str, inchikey: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol.SetProp("SMILES", smiles)
    mol.SetProp("connectivity", connectivity)
    mol.SetProp("InChIKey", inchikey)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_parse_papyrus_sd(tmp_path: Path):
    sd_path = tmp_path / "tiny.sd"
    _write_sd(sd_path, "CCO", ETHANOL_CONN, ETHANOL_KEY)
    frame = parse_papyrus_sd(sd_path)
    assert len(frame) == 1
    assert frame.iloc[0]["connectivity"] == ETHANOL_CONN
    assert frame.iloc[0]["canonical_smiles"] == "CCO"
    assert frame.iloc[0]["inchikey"] == ETHANOL_KEY


def test_canonicalize_molecules_fills_inchikey(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "connectivity": [ETHANOL_CONN],
            "canonical_smiles": ["OCC"],
            "inchikey": [""],
        }
    )
    cleaned, stats = canonicalize_molecules(frame)
    assert stats["n_unique"] == 1
    assert cleaned.iloc[0]["canonical_smiles"] == "CCO"
    assert cleaned.iloc[0]["inchikey"] == ETHANOL_KEY
    assert cleaned.iloc[0]["connectivity"] == ETHANOL_CONN


def test_mordred_tsv_to_parquet(tmp_path: Path):
    tsv = tmp_path / "mordred.tsv"
    tsv.write_text(
        f"{MORDRED_ID_COL}\tSLogP\tnAtom\n"
        f"{ETHANOL_CONN}\t-0.0014\t9\n"
        "UHOVQNZJYSORNB\t1.6866\t12\n"
    )
    out = tmp_path / "mordred.parquet"
    n_rows, n_desc = mordred_tsv_to_parquet(tsv, out, chunksize=1)
    assert n_rows == 2
    assert n_desc == 2
    frame = pd.read_parquet(out)
    assert list(frame.columns) == [MORDRED_ID_COL, "SLogP", "nAtom"]
    ethanol = frame.loc[frame[MORDRED_ID_COL] == ETHANOL_CONN].iloc[0]
    assert ethanol["nAtom"] == 9.0
    assert abs(ethanol["SLogP"] + 0.0014) < 1e-9
