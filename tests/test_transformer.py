from pathlib import Path

import pandas as pd
import pytest

from molecule_gen.constraints.spec import parse_spec
from molecule_gen.data.clean import canonicalize_smiles, connectivity_from_inchikey
from molecule_gen.generate.run import generate_molecules
from molecule_gen.generate.selfies_util import smiles_to_selfies, split_selfies
from molecule_gen.generate.transformer import SelfiesTokenizer, default_checkpoint_path


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


def test_tokenizer_roundtrip():
    encoded = smiles_to_selfies("CCO")
    assert encoded is not None
    tokenizer = SelfiesTokenizer.from_selfies_list([encoded])
    ids = tokenizer.encode(encoded)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == encoded
    assert len(split_selfies(encoded)) + 2 == len(ids)


def test_generate_uses_transformer_prior(tmp_path):
    spec = parse_spec({"n": 1, "descriptors": {"MW": {"min": 40, "max": 50}}})
    out = tmp_path / "hits.parquet"
    manifest = generate_molecules(
        spec,
        population_size=4,
        generations=0,
        seed=0,
        prior="transformer",
        out_path=out,
        evaluate_fn=_fake_eval,
        sample_fn=lambda n: ["CCO", "CCC"][:n],
    )
    assert manifest["prior"] == "transformer"
    assert manifest["n_hits"] == 1
    hits = pd.read_parquet(out)
    assert list(hits["canonical_smiles"]) == ["CCO"]
    assert list(hits["origin"]) == ["transformer"]


def test_generate_default_prior_is_transformer(tmp_path):
    spec = parse_spec({"n": 1, "descriptors": {"MW": {"min": 40, "max": 50}}})
    events: list[dict] = []
    manifest = generate_molecules(
        spec,
        population_size=4,
        generations=0,
        seed=0,
        out_path=tmp_path / "hits.parquet",
        evaluate_fn=_fake_eval,
        sample_fn=lambda n: ["CCO", "CCC"][:n],
        on_progress=events.append,
    )
    assert manifest["prior"] == "transformer"
    assert any(event["type"] == "sampling" for event in events)
    assert any(event["type"] == "ga_start" for event in events)
    assert any(event["type"] == "done" for event in events)


torch = pytest.importorskip("torch")


def test_tiny_gpt_loss_drops(tmp_path: Path):
    from molecule_gen.generate.sample import sample_valid_smiles
    from molecule_gen.generate.train import train_transformer

    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CCOC", "CCCO"]
    ckpt = tmp_path / "tiny.pt"
    extra = train_transformer(
        preset="tiny",
        epochs=40,
        batch_size=4,
        lr=3e-3,
        seed=0,
        device_name="cpu",
        out_path=ckpt,
        smiles_list=smiles,
        log_every=1000,
    )
    assert ckpt.exists()
    assert extra["n_parameters"] > 0
    assert extra["loss"] is not None
    assert extra["loss"] < 3.0
    sampled = sample_valid_smiles(
        8, checkpoint=ckpt, seed=1, device_name="cpu", max_draws=64
    )
    assert isinstance(sampled, list)


def test_cached_checkpoint_loads_once(tmp_path, monkeypatch):
    from molecule_gen.generate import sample as sample_mod

    pytest.importorskip("torch")
    ckpt = default_checkpoint_path()
    if not ckpt.exists():
        pytest.skip("no trained checkpoint")

    calls: list[object] = []
    real = sample_mod.load_checkpoint

    def counting(path=None, **kwargs):
        calls.append(path)
        return real(path, **kwargs)

    monkeypatch.setattr(sample_mod, "load_checkpoint", counting)
    sample_mod.cached_checkpoint.cache_clear()
    first = sample_mod.cached_checkpoint(ckpt)
    second = sample_mod.cached_checkpoint(ckpt)
    assert first is second
    assert len(calls) == 1
