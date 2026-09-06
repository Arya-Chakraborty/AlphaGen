from __future__ import annotations

from functools import lru_cache
from random import Random

from molecule_gen.data.clean import canonicalize_smiles

MAX_SELFIES_TOKENS = 80


class SelfiesError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def robust_alphabet() -> tuple[str, ...]:
    import selfies as sf

    return tuple(sorted(sf.get_semantic_robust_alphabet()))


def smiles_to_selfies(smiles: str) -> str | None:
    import selfies as sf

    try:
        encoded = sf.encoder(smiles)
    except Exception:
        return None
    if not encoded:
        return None
    return encoded


def selfies_to_canonical_smiles(selfies: str) -> str | None:
    import selfies as sf

    try:
        raw = sf.decoder(selfies)
    except Exception:
        return None
    parsed = canonicalize_smiles(raw)
    if parsed is None:
        return None
    return parsed[0]


def split_selfies(selfies: str) -> list[str]:
    import selfies as sf

    return list(sf.split_selfies(selfies))


def join_selfies(tokens: list[str]) -> str:
    return "".join(tokens)


def encode_roundtrip(smiles: str) -> tuple[str, str] | None:
    """SMILES → SELFIES → canonical SMILES. None if either step fails."""
    parsed = canonicalize_smiles(smiles)
    if parsed is None:
        return None
    canon, _inchikey = parsed
    encoded = smiles_to_selfies(canon)
    if encoded is None:
        return None
    decoded = selfies_to_canonical_smiles(encoded)
    if decoded is None:
        return None
    return decoded, encoded


def mutate_selfies(
    selfies: str,
    rng: Random,
    *,
    alphabet: tuple[str, ...] | None = None,
    max_tokens: int = MAX_SELFIES_TOKENS,
) -> str:
    tokens = split_selfies(selfies)
    symbols = alphabet or robust_alphabet()
    if not symbols:
        return selfies
    if not tokens:
        return symbols[rng.randrange(len(symbols))]
    op = rng.choice(("replace", "insert", "delete"))
    if op == "delete" and len(tokens) > 1:
        del tokens[rng.randrange(len(tokens))]
    elif op == "insert" and len(tokens) < max_tokens:
        tokens.insert(rng.randrange(len(tokens) + 1), symbols[rng.randrange(len(symbols))])
    else:
        tokens[rng.randrange(len(tokens))] = symbols[rng.randrange(len(symbols))]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return join_selfies(tokens)


def crossover_selfies(left: str, right: str, rng: Random) -> str:
    a = split_selfies(left)
    b = split_selfies(right)
    if len(a) < 2 or len(b) < 2:
        return left
    i = rng.randrange(1, len(a))
    j = rng.randrange(1, len(b))
    child = a[:i] + b[j:]
    if len(child) > MAX_SELFIES_TOKENS:
        child = child[:MAX_SELFIES_TOKENS]
    return join_selfies(child)
