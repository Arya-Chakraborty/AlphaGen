from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from molecule_gen.generate.corpus import load_training_selfies
from molecule_gen.generate.selfies_util import MAX_SELFIES_TOKENS
from molecule_gen.generate.transformer import (
    SelfiesTokenizer,
    TransformerError,
    build_model,
    config_from_preset,
    default_checkpoint_path,
    pick_device,
    require_torch,
    save_checkpoint,
)
from molecule_gen.paths import PROJECT_ROOT, ensure_checkpoints_dir


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _encode_batch(
    sequences: list[str],
    tokenizer: SelfiesTokenizer,
    block_size: int,
    torch,
):
    pad = tokenizer.pad_id
    encoded = [tokenizer.encode(s, max_len=block_size) for s in sequences]
    batch_x = []
    batch_y = []
    for ids in encoded:
        x = ids[:-1]
        y = ids[1:]
        if len(x) < block_size - 1:
            x = x + [pad] * (block_size - 1 - len(x))
            y = y + [pad] * (block_size - 1 - len(y))
        else:
            x = x[: block_size - 1]
            y = y[: block_size - 1]
        batch_x.append(x)
        batch_y.append(y)
    return (
        torch.tensor(batch_x, dtype=torch.long),
        torch.tensor(batch_y, dtype=torch.long),
    )


def train_transformer(
    *,
    preset: str = "small",
    limit: int | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
    device_name: str | None = None,
    out_path: Path | None = None,
    smiles_list: list[str] | None = None,
    max_tokens: int = MAX_SELFIES_TOKENS,
    log_every: int = 20,
) -> dict[str, Any]:
    """Train a decoder-only SELFIES GPT. Does not take descriptor ranges."""
    torch = require_torch()
    ensure_checkpoints_dir()
    torch.manual_seed(seed)
    device = pick_device(device_name)
    sequences = load_training_selfies(
        limit=limit, max_tokens=max_tokens, smiles_list=smiles_list
    )
    if not sequences:
        raise TransformerError("Training corpus is empty")
    tokenizer = SelfiesTokenizer.from_selfies_list(sequences)
    config = config_from_preset(preset, tokenizer.vocab_size, block_size=max_tokens + 2)
    model = build_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    print(
        f"training {preset} SELFIES GPT: {model.n_parameters():,} params, "
        f"{len(sequences):,} molecules, device={device}",
        file=sys.stderr,
    )
    model.train()
    n_seen = 0
    last_loss = None
    steps = 0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(sequences)).tolist()
        for start in range(0, len(sequences), batch_size):
            batch_seq = [sequences[i] for i in perm[start : start + batch_size]]
            if not batch_seq:
                continue
            x, y = _encode_batch(batch_seq, tokenizer, config.block_size, torch)
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            _logits, loss = model(x, y, pad_id=tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.item())
            n_seen += len(batch_seq)
            steps += 1
            if steps == 1 or steps % log_every == 0:
                print(
                    f"epoch {epoch}/{epochs} step {steps} loss={last_loss:.4f}",
                    file=sys.stderr,
                )

    dest = out_path or default_checkpoint_path()
    extra = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "n_molecules": len(sequences),
        "epochs": epochs,
        "steps": steps,
        "loss": last_loss,
        "device": str(device),
        "seed": seed,
        "n_parameters": model.n_parameters(),
    }
    save_checkpoint(dest, model=model, tokenizer=tokenizer, extra=extra)
    extra["checkpoint"] = _relpath(dest)
    print(f"wrote {dest}", file=sys.stderr)
    return extra
