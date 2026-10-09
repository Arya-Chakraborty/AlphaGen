from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from molecule_gen.paths import PROCESSED_DIR, PROJECT_ROOT, ensure_processed_dir

ENV_PARQUET_NAME = "papyrus_env.parquet"
ENV_MANIFEST_NAME = "papyrus_env_manifest.json"
MORGAN_RADIUS = 2
DEFAULT_ENV_SAMPLE = 200_000
DEFAULT_MIN_COUNT = 3

# Chemistry that is wrong, not merely unusual. Each rule matches <0.16% of Papyrus;
# hydroxylamine (1.26%) and acyclic N-N (0.94%) were measured and dropped as too
# common in real drugs to gate on.
UNSTABLE_SMARTS: tuple[tuple[str, str], ...] = (
    ("peroxide", "[OX2]-[OX2]"),
    ("n_o_n", "[#7]-[OX2]-[#7]"),
    ("hypervalent_s", "[#16H1]=*"),
    ("acyclic_enol", "[CX3;!R]=[CX3;!R]-[OX2H]"),
    ("geminal_diol", "[OX2H]-[CX4]-[OX2H]"),
    ("n_n_n_acyclic", "[NX3;!R]-[NX3;!R]-[NX3;!R]"),
    ("phosphine", "[PX3](-[#6,#1])(-[#6,#1])-[#6,#1]"),
)


class PlausibilityError(RuntimeError):
    pass


def env_set_path() -> Path:
    return PROCESSED_DIR / ENV_PARQUET_NAME


def env_manifest_path() -> Path:
    return PROCESSED_DIR / ENV_MANIFEST_NAME


@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[str, Any], ...]:
    from rdkit import Chem

    compiled = []
    for name, smarts in UNSTABLE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise PlausibilityError(f"Bad SMARTS for {name}: {smarts}")
        compiled.append((name, patt))
    return tuple(compiled)


@lru_cache(maxsize=1)
def _generator():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS)


def _mol(smiles: str | None):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    return Chem.MolFromSmiles(str(smiles).strip())


def unstable_motifs_in_mol(mol) -> list[str]:
    """Names of unstable motifs in an already-parsed Mol. Empty for None."""
    if mol is None:
        return []
    return [name for name, patt in _patterns() if mol.HasSubstructMatch(patt)]


def unstable_motifs(smiles: str | None) -> list[str]:
    """Names of unstable motifs in the molecule. Empty for valid or unparseable input."""
    return unstable_motifs_in_mol(_mol(smiles))


@lru_cache(maxsize=4)
def environment_set(path: Path | None = None) -> frozenset[int] | None:
    """Morgan environments seen in Papyrus. None if the artifact is not built yet."""
    target = path or env_set_path()
    if not target.exists():
        return None
    return frozenset(int(value) for value in pd.read_parquet(target)["env"])


def _environments(mol) -> list[int]:
    counts = _generator().GetSparseCountFingerprint(mol).GetNonzeroElements()
    return [int(key) for key in counts]


def unseen_fraction(smiles: str | None, *, path: Path | None = None) -> float | None:
    """Fraction of the molecule's atom environments never seen in Papyrus."""
    known = environment_set(path)
    if known is None:
        return None
    mol = _mol(smiles)
    if mol is None:
        return None
    envs = _environments(mol)
    if not envs:
        return None
    return sum(1 for env in envs if env not in known) / len(envs)


def plausibility(smiles: str | None, *, path: Path | None = None) -> dict[str, Any]:
    """Hard-rule verdict plus corpus novelty. unseen_frac is None without the artifact."""
    motifs = unstable_motifs(smiles)
    return {
        "plausible": not motifs,
        "motifs": motifs,
        "unseen_frac": unseen_fraction(smiles, path=path),
    }


def build_env_set(
    *,
    n: int = DEFAULT_ENV_SAMPLE,
    min_count: int = DEFAULT_MIN_COUNT,
    seed: int = 0,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Collect Morgan environments seen at least min_count times in a Papyrus sample."""
    from molecule_gen.data.papyrus import molecules_parquet_path

    ensure_processed_dir()
    source = molecules_parquet_path()
    if not source.exists():
        raise PlausibilityError(
            f"Molecule table not found at {source}. Run fetch-papyrus first."
        )

    out_path = env_set_path()
    if not force and out_path.exists() and env_manifest_path().exists():
        existing = json.loads(env_manifest_path().read_text())
        if existing.get("n") == n and existing.get("min_count") == min_count:
            print(
                f"already have {existing.get('n_environments', '?')} environments; "
                "pass --force to rebuild",
                file=sys.stderr,
                flush=True,
            )
            return existing

    frame = pd.read_parquet(source, columns=["canonical_smiles"])
    n_take = min(n, len(frame))
    sample = frame.canonical_smiles.sample(n=n_take, random_state=seed).tolist()
    print(f"scanning {n_take:,} Papyrus molecules", file=sys.stderr, flush=True)

    counts: Counter[int] = Counter()
    n_failed = 0
    for smiles in tqdm(
        sample,
        desc="environments",
        unit="mol",
        file=sys.stderr,
        disable=not show_progress,
    ):
        mol = _mol(smiles)
        if mol is None:
            n_failed += 1
            continue
        counts.update(_environments(mol))

    keep = [env for env, count in counts.items() if count >= min_count]
    if not keep:
        raise PlausibilityError("No environments survived min_count")
    pd.DataFrame({"env": pd.Series(keep, dtype="uint64")}).to_parquet(
        out_path, index=False
    )
    environment_set.cache_clear()
    print(
        f"wrote {len(keep):,} environments (of {len(counts):,} distinct) to {out_path}",
        file=sys.stderr,
        flush=True,
    )

    manifest = {
        "source_parquet": str(source.relative_to(PROJECT_ROOT)),
        "output_parquet": str(out_path.relative_to(PROJECT_ROOT)),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "n_scanned": n_take,
        "n_failed": n_failed,
        "seed": seed,
        "min_count": min_count,
        "radius": MORGAN_RADIUS,
        "n_distinct": len(counts),
        "n_environments": len(keep),
    }
    env_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
