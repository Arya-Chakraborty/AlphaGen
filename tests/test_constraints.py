from pathlib import Path

import pandas as pd
import pytest

from molecule_gen.constraints.score import (
    range_penalty,
    score_frame,
    score_smiles,
    score_vector,
)
from molecule_gen.constraints.spec import SpecError, load_spec, parse_spec
from molecule_gen.descriptors.names import local_name_to_papyrus, papyrus_name_to_local


def test_alias_roundtrip():
    assert papyrus_name_to_local("n6aliphRing") == "n6ARing"
    assert local_name_to_papyrus("n6ARing") == "n6aliphRing"


def test_parse_spec_maps_alias_and_warns():
    spec = parse_spec(
        {
            "n": 3,
            "descriptors": {
                "SLogP": {"min": 0, "max": 3},
                "n6aliphRing": {"min": 0, "max": 2},
                "nHBAcc": {"min": 0, "max": 5},
            },
        }
    )
    assert spec.names() == ["SLogP", "n6ARing", "nHBAcc"]
    assert spec.ranges[1].source_name == "n6aliphRing"
    assert any("nHBAcc" in warning for warning in spec.warnings)


def test_parse_spec_blocks_detour():
    with pytest.raises(SpecError, match="detour"):
        parse_spec({"n": 1, "descriptors": {"DetourIndex": {"min": 0, "max": 10}}})
    with pytest.raises(SpecError, match="detour"):
        parse_spec({"n": 1, "descriptors": {"SpAbs_Dt": {"min": 0, "max": 10}}})


def test_parse_spec_rejects_unknown_and_bad_bounds():
    with pytest.raises(SpecError, match="Unknown"):
        parse_spec({"descriptors": {"notARealDesc": {"min": 0, "max": 1}}})
    with pytest.raises(SpecError, match="greater"):
        parse_spec({"descriptors": {"MW": {"min": 10, "max": 1}}})


def test_load_spec(tmp_path: Path):
    path = tmp_path / "spec.json"
    path.write_text('{"n": 2, "descriptors": {"MW": {"min": 10, "max": 20}}}')
    spec = load_spec(path)
    assert spec.n == 2
    assert spec.names() == ["MW"]


def test_score_vector_in_and_out_of_range():
    spec = parse_spec(
        {
            "n": 1,
            "descriptors": {
                "SLogP": {"min": 1.0, "max": 3.0},
                "MW": {"min": 100.0, "max": 200.0},
            },
        }
    )
    inside = score_vector({"SLogP": 2.0, "MW": 150.0}, spec)
    assert inside["in_range"] is True
    assert inside["penalty"] == 0.0
    assert inside["n_passed"] == 2

    outside = score_vector({"SLogP": 5.0, "MW": 150.0}, spec)
    assert outside["in_range"] is False
    assert outside["descriptors"]["SLogP"]["passed"] is False
    assert outside["descriptors"]["MW"]["passed"] is True
    assert outside["penalty"] == range_penalty(5.0, spec.ranges[0])

    missing = score_vector({"SLogP": 2.0, "MW": None}, spec)
    assert missing["in_range"] is False
    assert missing["descriptors"]["MW"]["penalty"] == 1.0


def test_score_frame_vectorized():
    spec = parse_spec(
        {
            "n": 1,
            "descriptors": {
                "SLogP": {"min": 1.0, "max": 3.0},
                "MW": {"min": 100.0, "max": 200.0},
            },
        }
    )
    frame = pd.DataFrame(
        {
            "connectivity": ["A", "B", "C"],
            "SLogP": [2.0, 5.0, None],
            "MW": [150.0, 150.0, 150.0],
        }
    )
    scored = score_frame(frame, spec)
    assert bool(scored.loc[0, "in_range"]) is True
    assert scored.loc[0, "penalty"] == 0.0
    assert bool(scored.loc[1, "in_range"]) is False
    assert scored.loc[2, "penalty"] == 1.0


def test_score_invalid_smiles():
    spec = parse_spec({"n": 1, "descriptors": {"MW": {"min": 1, "max": 100}}})
    with pytest.raises(SpecError, match="Invalid SMILES"):
        score_smiles("not-a-molecule", spec)


@pytest.mark.integration
def test_score_ethanol_local():
    pytest.importorskip("mordred")
    spec = parse_spec(
        {
            "n": 1,
            "descriptors": {
                "MW": {"min": 40.0, "max": 50.0},
                "nHeavyAtom": {"min": 3, "max": 3},
                "nHBDon": {"min": 1, "max": 1},
            },
        }
    )
    result = score_smiles("CCO", spec)
    assert result["canonical_smiles"] == "CCO"
    assert result["source"] in {"cache", "local"}
    assert result["in_range"] is True
    assert result["penalty"] == 0.0
    assert 46.0 <= result["descriptors"]["MW"]["value"] <= 46.2
