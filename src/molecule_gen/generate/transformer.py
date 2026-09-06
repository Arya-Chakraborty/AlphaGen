from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from molecule_gen.generate.selfies_util import (
    MAX_SELFIES_TOKENS,
    join_selfies,
    robust_alphabet,
    split_selfies,
)
from molecule_gen.paths import CHECKPOINTS_DIR, ensure_checkpoints_dir

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
DEFAULT_CHECKPOINT = "selfies_gpt.pt"


class TransformerError(RuntimeError):
    pass


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise TransformerError(
            'PyTorch is required. Install with: pip install -e ".[train]"'
        ) from exc
    return __import__("torch")


def pick_device(name: str | None = None):
    torch = require_torch()
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_checkpoint_path() -> Path:
    return CHECKPOINTS_DIR / DEFAULT_CHECKPOINT


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = MAX_SELFIES_TOKENS + 2
    dropout: float = 0.1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, dict[str, float | int]] = {
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 64, "dropout": 0.0},
    "small": {"n_layer": 4, "n_head": 4, "n_embd": 256},
    "medium": {"n_layer": 8, "n_head": 8, "n_embd": 512},
}


def config_from_preset(preset: str, vocab_size: int, block_size: int | None = None) -> GPTConfig:
    key = preset.strip().lower()
    if key not in PRESETS:
        raise TransformerError(f"Unknown preset {preset!r}. Choose from: {sorted(PRESETS)}")
    kwargs = dict(PRESETS[key])
    kwargs["vocab_size"] = vocab_size
    if block_size is not None:
        kwargs["block_size"] = block_size
    return GPTConfig(**kwargs)  # type: ignore[arg-type]


class SelfiesTokenizer:
    """Maps SELFIES symbols to ids. Special tokens occupy 0..3."""

    def __init__(self, tokens: list[str]):
        special = [PAD, BOS, EOS, UNK]
        seen: set[str] = set(special)
        ordered = list(special)
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        self.token_to_id = {token: idx for idx, token in enumerate(ordered)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def encode(
        self,
        selfies: str,
        *,
        add_special: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in split_selfies(selfies)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None and len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.eos_id:
                break
            token = self.id_to_token.get(int(idx), "")
            if token and token != UNK:
                tokens.append(token)
        return join_selfies(tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": [self.id_to_token[i] for i in range(self.vocab_size)],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SelfiesTokenizer:
        tokens = list(payload["tokens"])
        return cls(tokens)

    @classmethod
    def from_selfies_list(cls, sequences: list[str]) -> SelfiesTokenizer:
        vocab = set(robust_alphabet())
        for selfies in sequences:
            vocab.update(split_selfies(selfies))
        return cls(sorted(vocab))


def _build_model_class():
    torch = require_torch()
    nn = torch.nn
    F = torch.nn.functional

    class CausalSelfAttention(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            if config.n_embd % config.n_head != 0:
                raise TransformerError("n_embd must be divisible by n_head")
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.resid_dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            batch, seq, width = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            head = width // self.n_head
            q = q.view(batch, seq, self.n_head, head).transpose(1, 2)
            k = k.view(batch, seq, self.n_head, head).transpose(1, 2)
            v = v.view(batch, seq, self.n_head, head).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            y = y.transpose(1, 2).contiguous().view(batch, seq, width)
            return self.resid_dropout(self.c_proj(y))

    class MLP(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    class Block(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class SelfiesGPT(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            self.config = config
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
            self.drop = nn.Dropout(config.dropout)
            self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def n_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def forward(self, idx, targets=None, pad_id: int = 0):
            _batch, seq = idx.size()
            if seq > self.config.block_size:
                raise TransformerError(
                    f"Sequence length {seq} exceeds block_size {self.config.block_size}"
                )
            pos = torch.arange(0, seq, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
            for block in self.blocks:
                x = block(x)
            logits = self.lm_head(self.ln_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=pad_id,
                )
            return logits, loss

    return SelfiesGPT


def build_model(config: GPTConfig):
    SelfiesGPT = _build_model_class()
    return SelfiesGPT(config)


def save_checkpoint(
    path: Path,
    *,
    model,
    tokenizer: SelfiesTokenizer,
    extra: dict[str, Any] | None = None,
) -> Path:
    torch = require_torch()
    ensure_checkpoints_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": model.config.as_dict(),
        "tokenizer": tokenizer.as_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path | None = None, *, map_location=None):
    torch = require_torch()
    dest = path or default_checkpoint_path()
    if not dest.exists():
        raise TransformerError(
            f"Checkpoint not found at {dest}. Train first: python -m molecule_gen train-transformer"
        )
    payload = torch.load(dest, map_location=map_location or "cpu", weights_only=False)
    tokenizer = SelfiesTokenizer.from_dict(payload["tokenizer"])
    config = GPTConfig(**payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["model"])
    return model, tokenizer, payload.get("extra") or {}
