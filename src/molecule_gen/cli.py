from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from molecule_gen.data.papyrus import (
    MORDRED_NAME,
    MORDRED_SIZE,
    STRUCTURES_NAME,
    STRUCTURES_SIZE,
    file_help,
    fetch_papyrus,
    import_mordred,
    molecules_parquet_path,
    mordred_parquet_path,
)
from molecule_gen.constraints.lookup import draft_spec_from_percentiles
from molecule_gen.constraints.score import score_connectivity, score_smiles
from molecule_gen.constraints.spec import load_spec
from molecule_gen.generate.corpus import (
    build_selfies_corpus,
    selfies_manifest_path,
    selfies_parquet_path,
)
from molecule_gen.generate.plausibility import (
    DEFAULT_ENV_SAMPLE,
    DEFAULT_MIN_COUNT,
    build_env_set,
    env_manifest_path,
    env_set_path,
)
from molecule_gen.generate.run import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION,
    generate_molecules,
    hits_manifest_path,
    hits_parquet_path,
    parse_smiles_seeds,
)
from molecule_gen.generate.sample import (
    sample_manifest_path,
    sample_parquet_path as transformer_sample_parquet_path,
    sample_transformer,
)
from molecule_gen.generate.train import train_transformer
from molecule_gen.generate.transformer import PRESETS, default_checkpoint_path
from molecule_gen.descriptors.compare import (
    compare_local_to_papyrus,
    compare_manifest_path,
)
from molecule_gen.descriptors.mordred import (
    DEFAULT_BATCH_SIZE,
    compute_mordred_sample,
    sample_parquet_path,
)
from molecule_gen.descriptors.names import is_detour_descriptor, resolve_descriptor_name
from molecule_gen.paths import PROCESSED_DIR, RAW_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="molecule-gen")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch-papyrus",
        help="Download Papyrus unique 2D structures and write a molecule parquet.",
    )
    fetch.add_argument(
        "--force",
        action="store_true",
        help="Re-download and rebuild even if output already exists.",
    )
    fetch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Parse only the first N molecules (debug).",
    )
    fetch.add_argument(
        "--no-download",
        action="store_true",
        help="Do not fetch from Zenodo; require a local file in data/raw/.",
    )

    mordred_import = sub.add_parser(
        "import-mordred",
        help="Import Papyrus precomputed Mordred 2D descriptors to parquet.",
    )
    mordred_import.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the parquet already exists.",
    )
    mordred_import.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import only the first N rows (debug).",
    )
    mordred_import.add_argument(
        "--no-download",
        action="store_true",
        help="Do not fetch from Zenodo; require a local file in data/raw/.",
    )
    mordred_import.add_argument(
        "--chunksize",
        type=int,
        default=25_000,
        help="Rows per conversion chunk (default 25000).",
    )

    sample = sub.add_parser(
        "mordred-sample",
        help="Compute local Mordred 2D for a sample of Papyrus molecules (cached by InChIKey).",
    )
    sample.add_argument("--n", type=int, default=2000, help="Sample size (default 2000).")
    sample.add_argument("--seed", type=int, default=0, help="Random seed.")
    sample.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Molecules per Mordred batch.",
    )
    sample.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the sample parquet even if it already exists (cache is still used).",
    )

    compare = sub.add_parser(
        "compare-mordred",
        help="Compute local Mordred 2D on a sample and compare to the Papyrus table.",
    )
    compare.add_argument("--n", type=int, default=100, help="Sample size (default 100).")
    compare.add_argument("--seed", type=int, default=0, help="Random seed.")
    compare.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Molecules per Mordred batch.",
    )

    score = sub.add_parser(
        "score",
        help="Score a SMILES or Papyrus connectivity against a range spec.",
    )
    score.add_argument("--spec", required=True, help="Path to a range JSON spec.")
    score.add_argument("--smiles", default=None, help="SMILES to score.")
    score.add_argument(
        "--connectivity",
        default=None,
        help="Papyrus connectivity (first InChIKey block) to look up.",
    )

    suggest = sub.add_parser(
        "suggest-ranges",
        help="Draft a spec from Papyrus percentiles of named descriptors.",
    )
    suggest.add_argument(
        "--names",
        required=True,
        help="Comma-separated Mordred names (local or Papyrus aliases).",
    )
    suggest.add_argument("--n", type=int, default=10, help="How many molecules to ask for.")
    suggest.add_argument(
        "--low",
        type=float,
        default=0.10,
        help="Low percentile (default 0.10).",
    )
    suggest.add_argument(
        "--high",
        type=float,
        default=0.90,
        help="High percentile (default 0.90).",
    )
    suggest.add_argument(
        "--out",
        default=None,
        help="Optional path to write the drafted spec JSON.",
    )

    generate = sub.add_parser(
        "generate",
        help="Generate molecules that match a range spec.",
    )
    generate.add_argument("--spec", required=True, help="Path to a range JSON spec.")
    generate.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
        help="GA population size (default 64).",
    )
    generate.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help="Max GA generations (default 20). 0 = seeds only.",
    )
    generate.add_argument("--seed", type=int, default=0, help="Random seed.")
    generate.add_argument(
        "--smiles",
        default=None,
        help="Comma-separated SMILES to use as seeds instead of Papyrus.",
    )
    generate.add_argument(
        "--prior",
        choices=("papyrus", "transformer"),
        default="transformer",
        help="Seed source when --smiles is omitted (default transformer).",
    )
    generate.add_argument(
        "--checkpoint",
        default=None,
        help="SELFIES GPT checkpoint (default checkpoints/selfies_gpt.pt).",
    )
    generate.add_argument(
        "--no-near-miss",
        action="store_true",
        help="Do not mix Papyrus near-misses into transformer seeds.",
    )
    generate.add_argument(
        "--out",
        default=None,
        help="Output parquet path (default data/processed/generate_hits.parquet).",
    )

    build_sf = sub.add_parser(
        "build-selfies",
        help="Encode Papyrus SMILES to a SELFIES training corpus.",
    )
    build_sf.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Encode only the first N molecules (debug).",
    )
    build_sf.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the parquet already exists.",
    )

    train = sub.add_parser(
        "train-transformer",
        help="Train a small local SELFIES GPT (not Llama).",
    )
    train.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="small",
        help="Model size: tiny, small (default, laptop), medium (~25M params).",
    )
    train.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Train on at most N molecules (omit for the full corpus).",
    )
    train.add_argument("--epochs", type=int, default=1, help="Training epochs (default 1).")
    train.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32).")
    train.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    train.add_argument("--seed", type=int, default=0, help="Random seed.")
    train.add_argument(
        "--device",
        default=None,
        help="cpu, mps, or cuda (default: pick the best available).",
    )
    train.add_argument(
        "--out",
        default=None,
        help="Checkpoint path (default checkpoints/selfies_gpt.pt).",
    )

    sample_tf = sub.add_parser(
        "sample-transformer",
        help="Sample valid SMILES from a trained SELFIES GPT.",
    )
    sample_tf.add_argument("--n", type=int, default=64, help="How many unique SMILES (default 64).")
    sample_tf.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default checkpoints/selfies_gpt.pt).",
    )
    sample_tf.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    sample_tf.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default 50).")
    sample_tf.add_argument("--seed", type=int, default=0, help="Random seed.")
    sample_tf.add_argument("--device", default=None, help="cpu, mps, or cuda.")
    sample_tf.add_argument(
        "--out",
        default=None,
        help="Output parquet (default data/processed/transformer_samples.parquet).",
    )

    env_set = sub.add_parser(
        "build-env-set",
        help="Collect the Morgan atom environments seen in Papyrus (plausibility reference).",
    )
    env_set.add_argument(
        "--n",
        type=int,
        default=DEFAULT_ENV_SAMPLE,
        help=f"Molecules to scan (default {DEFAULT_ENV_SAMPLE}).",
    )
    env_set.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help=f"Times an environment must appear to count as seen (default {DEFAULT_MIN_COUNT}).",
    )
    env_set.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    env_set.add_argument("--force", action="store_true", help="Rebuild even if present.")

    serve = sub.add_parser(
        "serve",
        help="Run the FastAPI backend for the web UI.",
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")

    urls = sub.add_parser(
        "papyrus-urls",
        help="Print the Zenodo URLs and expected local paths.",
    )

    args = parser.parse_args(argv)

    if args.command == "fetch-papyrus":
        manifest = fetch_papyrus(
            force=args.force, limit=args.limit, download=not args.no_download
        )
        print(json.dumps(manifest, indent=2))
        print(f"parquet: {molecules_parquet_path()}", file=sys.stderr)
        print(f"raw dir: {RAW_DIR}", file=sys.stderr)
    elif args.command == "import-mordred":
        manifest = import_mordred(
            force=args.force,
            limit=args.limit,
            download=not args.no_download,
            chunksize=args.chunksize,
        )
        print(json.dumps(manifest, indent=2))
        print(f"parquet: {mordred_parquet_path()}", file=sys.stderr)
        print(f"processed dir: {PROCESSED_DIR}", file=sys.stderr)
    elif args.command == "mordred-sample":
        manifest = compute_mordred_sample(
            n=args.n,
            seed=args.seed,
            batch_size=args.batch_size,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2))
        print(f"parquet: {sample_parquet_path()}", file=sys.stderr)
        print(f"processed dir: {PROCESSED_DIR}", file=sys.stderr)
    elif args.command == "compare-mordred":
        report = compare_local_to_papyrus(
            n=args.n, seed=args.seed, batch_size=args.batch_size
        )
        print(json.dumps(report, indent=2))
        print(f"report: {compare_manifest_path()}", file=sys.stderr)
    elif args.command == "score":
        if bool(args.smiles) == bool(args.connectivity):
            parser.error("score requires exactly one of --smiles or --connectivity")
        spec = load_spec(args.spec)
        for warning in spec.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if args.smiles:
            result = score_smiles(args.smiles, spec)
        else:
            result = score_connectivity(args.connectivity, spec)
        print(json.dumps(result, indent=2))
    elif args.command == "suggest-ranges":
        names = []
        for part in args.names.split(","):
            text = part.strip()
            if not text:
                continue
            name = resolve_descriptor_name(text)
            if is_detour_descriptor(name):
                parser.error(
                    f"{name} is a detour-matrix descriptor and is not allowed in specs"
                )
            names.append(name)
        if not names:
            parser.error("provide at least one descriptor name")
        draft = draft_spec_from_percentiles(
            names, n=args.n, low=args.low, high=args.high
        )
        if args.out:
            Path(args.out).write_text(json.dumps(draft, indent=2) + "\n")
            print(f"wrote {args.out}", file=sys.stderr)
        print(json.dumps(draft, indent=2))
    elif args.command == "generate":
        spec = load_spec(args.spec)
        for warning in spec.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        manifest = generate_molecules(
            spec,
            population_size=args.population,
            generations=args.generations,
            seed=args.seed,
            smiles_seeds=parse_smiles_seeds(args.smiles),
            prior=args.prior,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            mix_near_misses=not args.no_near_miss,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(manifest, indent=2))
        print(f"hits: {hits_parquet_path() if args.out is None else args.out}", file=sys.stderr)
        print(f"manifest: {hits_manifest_path()}", file=sys.stderr)
    elif args.command == "build-selfies":
        manifest = build_selfies_corpus(limit=args.limit, force=args.force)
        print(json.dumps(manifest, indent=2))
        print(f"parquet: {selfies_parquet_path()}", file=sys.stderr)
        print(f"manifest: {selfies_manifest_path()}", file=sys.stderr)
    elif args.command == "train-transformer":
        extra = train_transformer(
            preset=args.preset,
            limit=args.limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device_name=args.device,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(extra, indent=2))
        print(f"checkpoint: {args.out or default_checkpoint_path()}", file=sys.stderr)
    elif args.command == "sample-transformer":
        manifest = sample_transformer(
            n=args.n,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            device_name=args.device,
            out_path=Path(args.out) if args.out else None,
        )
        shown = dict(manifest)
        shown.pop("smiles", None)
        print(json.dumps(shown, indent=2))
        print(
            f"samples: {transformer_sample_parquet_path() if args.out is None else args.out}",
            file=sys.stderr,
        )
        print(f"manifest: {sample_manifest_path()}", file=sys.stderr)
    elif args.command == "build-env-set":
        manifest = build_env_set(
            n=args.n,
            min_count=args.min_count,
            seed=args.seed,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2))
        print(f"environments: {env_set_path()}", file=sys.stderr)
        print(f"manifest: {env_manifest_path()}", file=sys.stderr)
    elif args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit(
                'FastAPI/uvicorn missing. Install with: pip install -e ".[api]"'
            ) from exc
        uvicorn.run(
            "molecule_gen.api.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    elif args.command == "papyrus-urls":
        print(file_help(STRUCTURES_NAME, STRUCTURES_SIZE))
        print()
        print(file_help(MORDRED_NAME, MORDRED_SIZE))
