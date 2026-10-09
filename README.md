# molecule_gen_biomed

Generate molecules that match user-specified **Mordred 2D** descriptor ranges.

Papyrus 05.7 is the training / lookup corpus. See `PLAN.md` for the full build order.

Papyrus is CC BY-SA 4.0 ([doi:10.5281/zenodo.13987985](https://doi.org/10.5281/zenodo.13987985)).

## Setup

```bash
cd molecule_gen_biomed
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Phase 1 — fetch Papyrus molecules

Downloads the unique 2D structures (`05.7_combined_2D_set_without_stereochemistry.sd.xz`, ~454 MB) from Zenodo, RDKit-sanitizes them, and writes one row per connectivity.

```bash
python -m molecule_gen fetch-papyrus
```

If you prefer to download the file yourself:

```bash
python -m molecule_gen papyrus-urls
```

Save it as `data/raw/05.7_combined_2D_set_without_stereochemistry.sd.xz`, then run `fetch-papyrus` again (it will use the local copy).

Output:

- `data/raw/05.7_combined_2D_set_without_stereochemistry.sd.xz`
- `data/processed/papyrus_molecules.parquet` — `connectivity`, `canonical_smiles`, `inchikey`
- `data/processed/papyrus_molecules_manifest.json`

~1.27 million unique 2D molecules. Join key for descriptors is **connectivity** (first block of the InChIKey).

## Phase 2 — import Papyrus Mordred 2D

Imports the precomputed 1,613 Mordred 2D descriptors (`05.7_combined_2D_moldescs_mordred2D.tsv.xz`, ~3.1 GB).

```bash
python -m molecule_gen import-mordred
```

Same local-file pattern: drop the TSV.xz in `data/raw/` if you do not want the CLI to fetch it.

Output:

- `data/raw/05.7_combined_2D_moldescs_mordred2D.tsv.xz`
- `data/processed/papyrus_mordred2d.parquet` — `connectivity` + 1613 descriptors
- `data/processed/papyrus_mordred2d_manifest.json`

## Phase 3 — local Mordred for new molecules

New SMILES (generated later) are scored with local Mordred 2D (`mordredcommunity`, `ignore_3D=True`) and cached by InChIKey.

```bash
python -m molecule_gen mordred-sample --n 200
```

Output:

- `data/processed/mordred_sample.parquet`
- `data/processed/mordred_cache.sqlite`

## Phase 4 — generate (SELFIES GPT + GA)

Default seeds come from the trained SELFIES GPT mixed with Papyrus **near-misses** (molecules that miss the box by a little). Mutations stay in ordinary organic chemistry (no boron, no formal charges). `--no-near-miss` uses the GPT only. `--prior papyrus` restores lookup-first seeding.

```bash
python -m molecule_gen generate --spec examples/tight.json
```

`--prior papyrus` restores lookup-first seeding. `--smiles` replaces seeds. `--generations 0` is seeds only.

## Phase 5 — SELFIES GPT prior

Small local decoder-only transformer. It proposes molecules. The GA still hits the Mordred ranges. Not Llama.

```bash
pip install -e ".[train]"
python -m molecule_gen build-selfies
python -m molecule_gen train-transformer --preset small --epochs 2
python -m molecule_gen sample-transformer --n 64
python -m molecule_gen generate --spec examples/tight.json
```

Weights go to `checkpoints/selfies_gpt.pt`.

## Plausibility

Two signals, deliberately split.

**Hard rules** reject chemistry that is simply wrong (peroxides, `N-O-N`, hypervalent sulfur, acyclic enols, geminal diols, `N-N-N`, phosphines). These run inside `is_reasonable_molecule`, so every mutant and seed is gated. Each rule was measured against Papyrus and matches under 0.16% of real molecules; hydroxylamine (1.26%) and acyclic `N-N` (0.94%) were tried and dropped as too common in real drugs to gate on.

**Corpus novelty** is reported, not gated: the fraction of a molecule's Morgan (r=2) atom environments never seen in Papyrus. Build the reference set once:

```bash
python -m molecule_gen build-env-set --n 200000
```

Output: `data/processed/papyrus_env.parquet` (~86k environments). Every hit then carries `unseen_frac` in the parquet and the API.

Calibration, measured on held-out Papyrus: median 0.019, p90 0.097, p99 0.231. Real drugs score 0.000–0.033. Anything above ~0.15 is worth a look; the GA can reach 0.4+ while still satisfying every descriptor range. `--n` and `--min-count` are the knobs.

## Phase 6 — web UI

Backend (FastAPI) and frontend (Next.js) run as two processes.

```bash
pip install -e ".[api]"
python -m molecule_gen serve
```

```bash
cd web
npm install
npm run dev
```

Open http://127.0.0.1:3000. The UI talks to http://127.0.0.1:8000. Papyrus min/max for each descriptor is shown as a reference only; typed ranges are not clamped to the corpus.
