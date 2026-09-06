import pytest

from molecule_gen.constraints.spec import parse_spec
from molecule_gen.descriptors.names import FEATURED_DESCRIPTORS, allowed_descriptor_names


def test_allowed_descriptor_names_excludes_detour():
    names = allowed_descriptor_names()
    assert "MW" in names
    assert "SLogP" in names
    assert "DetourIndex" not in names
    assert all(not name.endswith("_Dt") for name in names)
    for name in FEATURED_DESCRIPTORS:
        assert name in names


def test_health_and_descriptors():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from molecule_gen.api.app import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    listed = client.get("/descriptors")
    assert listed.status_code == 200
    payload = listed.json()
    assert "MW" in payload["featured"]
    assert "MW" in payload["names"]

    svg = client.get("/depict", params={"smiles": "CCO"})
    assert svg.status_code == 200
    assert "svg" in svg.headers["content-type"]
    assert b"<svg" in svg.content.lower()

    bad = client.post(
        "/generate",
        json={"n": 1, "descriptors": {"MW": {"min": 10, "max": 1}}},
    )
    assert bad.status_code == 400


def test_parse_spec_does_not_clamp_to_corpus():
    spec = parse_spec({"n": 2, "descriptors": {"MW": {"min": -50, "max": 9000}}})
    assert spec.ranges[0].minimum == -50
    assert spec.ranges[0].maximum == 9000
