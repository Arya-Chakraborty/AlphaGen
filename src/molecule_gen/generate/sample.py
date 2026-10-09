from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from molecule_gen.generate.selfies_util import selfies_to_canonical_smiles
from molecule_gen.generate.transformer import (
    default_checkpoint_path,
    load_checkpoint,
    pick_device,
    require_torch,
)
from molecule_gen.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_processed_dir

EOS_CHECK_EVERY = 8
SAMPLE_PARQUET = "transformer_samples.parquet"
SAMPLE_MANIFEST = "transformer_samples_manifest.json"


def sample_parquet_path() -> Path:
    return PROCESSED_DIR / SAMPLE_PARQUET


def sample_manifest_path() -> Path:
    return PROCESSED_DIR / SAMPLE_MANIFEST


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@lru_cache(maxsize=2)
def _cached_checkpoint(path: Path, mtime: float):
    return load_checkpoint(path, map_location="cpu")


def cached_checkpoint(path: Path | None = None):
    """load_checkpoint, reused across calls. Reloads if the file changes on disk."""
    dest = path or default_checkpoint_path()
    try:
        mtime = dest.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _cached_checkpoint(dest, mtime)


cached_checkpoint.cache_clear = _cached_checkpoint.cache_clear


def generate_selfies_batch(
    model,
    tokenizer,
    *,
    n: int,
    temperature: float = 1.0,
    top_k: int = 50,
    device=None,
) -> list[str]:
    torch = require_torch()
    F = torch.nn.functional
    model.eval()
    block = model.config.block_size
    idx = torch.full((n, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    with torch.no_grad():
        for step in range(block - 1):
            logits, _loss = model(idx[:, -block:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits[:, tokenizer.pad_id] = -float("inf")
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if step % EOS_CHECK_EVERY == 0 and bool(
                (idx[:, 1:] == tokenizer.eos_id).any(dim=1).all().item()
            ):
                break
    return [tokenizer.decode(row.tolist()) for row in idx.cpu()]


def sample_valid_smiles(
    n: int,
    *,
    checkpoint: Path | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
    device_name: str | None = None,
    seed: int = 0,
    max_draws: int | None = None,
    sample_fn: Callable[[int], list[str]] | None = None,
) -> list[str]:
    """Draw unique canonical SMILES from the SELFIES GPT (or a test sampler)."""
    if n < 1:
        return []
    if sample_fn is not None:
        seen: list[str] = []
        have: set[str] = set()
        for smiles in sample_fn(max(n * 4, n)):
            if smiles and smiles not in have:
                have.add(smiles)
                seen.append(smiles)
            if len(seen) >= n:
                break
        return seen[:n]

    torch = require_torch()
    torch.manual_seed(seed)
    device = pick_device(device_name)
    model, tokenizer, _extra = cached_checkpoint(checkpoint)
    model.to(device)
    budget = max_draws if max_draws is not None else max(n * 20, n)
    unique: list[str] = []
    have: set[str] = set()
    remaining = budget
    while len(unique) < n and remaining > 0:
        take = min(32, remaining, (n - len(unique)) * 4)
        remaining -= take
        for selfies in generate_selfies_batch(
            model,
            tokenizer,
            n=take,
            temperature=temperature,
            top_k=top_k,
            device=device,
        ):
            smiles = selfies_to_canonical_smiles(selfies)
            if smiles is None or smiles in have:
                continue
            have.add(smiles)
            unique.append(smiles)
            if len(unique) >= n:
                break
    return unique


def sample_transformer(
    *,
    n: int = 64,
    checkpoint: Path | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 0,
    device_name: str | None = None,
    out_path: Path | None = None,
    sample_fn: Callable[[int], list[str]] | None = None,
) -> dict[str, Any]:
    """Sample valid unique SMILES and write a parquet + manifest."""
    ensure_processed_dir()
    draws_attempted = max(n * 20, n) if sample_fn is None else n
    smiles = sample_valid_smiles(
        n,
        checkpoint=checkpoint,
        temperature=temperature,
        top_k=top_k,
        device_name=device_name,
        seed=seed,
        max_draws=draws_attempted,
        sample_fn=sample_fn,
    )
    dest = out_path or sample_parquet_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if smiles:
        import pandas as pd

        pd.DataFrame({"canonical_smiles": smiles}).to_parquet(dest, index=False)
    elif dest.exists():
        dest.unlink()

    manifest = {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "n_requested": n,
        "n_valid_unique": len(smiles),
        "checkpoint": str(checkpoint or default_checkpoint_path()),
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "output_parquet": _relpath(dest) if dest.exists() else None,
        "smiles": smiles,
    }
    sample_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"sampled {len(smiles)} / {n} unique valid SMILES",
        file=sys.stderr,
    )
    if dest.exists():
        print(f"wrote {dest}", file=sys.stderr)
    return manifest
