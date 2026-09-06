# Molecule generation — stepwise implementation guide

Goal: user picks Mordred 2D descriptors and ranges → system generates molecules whose computed descriptors fall in those ranges.

Papyrus 05.7 (and later a ZINC slice) are **training / lookup data** only. A generated molecule may already exist in those databases. That is fine.

Do these steps in order. Do not skip ahead to the transformer until the scoring loop works.

---

## Step 0 — Project setup

- Create a Python environment (3.10+).
- Install core libraries: `rdkit`, `pandas`, `pyarrow`, `selfies`, `mordredcommunity`.
- Add later, only when needed: `torch`, `scikit-learn` / `xgboost`.
- Layout to grow into:

```text
molecule_gen_biomed/
  PLAN.md
  data/                 # downloads, parquet caches (gitignored)
  checkpoints/          # trained transformer weights
  src/
    descriptors/        # Mordred 2D + later surrogate
    generate/
      ga.py             # genetic algorithm (hits the ranges)
      transformer.py    # small SELFIES GPT (proposes candidates)
      train.py          # train the transformer on Papyrus SELFIES
      sample.py         # draw seed molecules from the transformer
  notebooks/
```

The transformer lives in `src/generate/`. It is a small local SELFIES GPT, not Llama. It proposes molecules. The GA in the same folder steers them into the Mordred ranges.

---

## Step 1 — Get Papyrus molecules (training corpus)

- Download Papyrus 05.7 unique 2D structures from Zenodo record 13987985:
  `05.7_combined_2D_set_without_stereochemistry.sd.xz` (~454 MB).
- Keep: connectivity, canonical SMILES, InChIKey.
- Do not download activity tables or 3D files unless needed later.

Exit check: a table of ~1.27M unique 2D molecules on disk.

---

## Step 2 — Clean and canonicalize

Papyrus is already standardized. Still:

- Parse with RDKit; drop anything that fails sanitize.
- Store canonical SMILES.
- Recompute InChIKey. Connectivity (first InChIKey block) is the join key to Mordred.

Save as `data/processed/papyrus_molecules.parquet`.

Exit check: invalid rows gone; one row per connectivity.

---

## Step 3 — Mordred 2D table + local calculator

Two sources, same 1,613 2D names (`ignore_3D=True`):

- **Lookup:** import Papyrus `05.7_combined_2D_moldescs_mordred2D.tsv.xz` (~3.1 GB) keyed by connectivity.
- **New molecules:** compute locally with `mordredcommunity`. Cache by InChIKey. Never recompute the same molecule.

Exit check: same molecule twice hits the cache; descriptor names and counts look right (~1613).

---

## Step 4 — Constraint spec

Define the user request as a small JSON object:

- selected descriptor names (e.g. 50)
- min/max for each
- how many molecules to return

Add a checker: given a descriptor vector, return which ranges are satisfied and a numeric penalty for those that are not.

This checker is the source of truth. Not “is it in Papyrus.”

Exit check: you can score a known molecule against a made-up range file.

---

## Step 5 — Generation v0 (no neural net)

Do not train a model yet.

- Start from random valid Papyrus molecules as a seed population.
- Represent them as SELFIES (so mutations stay valid).
- Genetic algorithm:
  - mutate / crossover SELFIES
  - decode to SMILES
  - drop invalid structures
  - score with Mordred (or cache / Papyrus lookup)
  - keep molecules closest to the ranges
- Repeat until N molecules sit inside all selected ranges, or you hit a generation limit.
- Deduplicate the **output batch**. Existing-in-Papyrus is allowed.

Exit check: for a **loose** range (e.g. a few common properties), the GA returns at least one in-range molecule.

---

## Step 6 — Fast surrogate (so GA can scale)

Local Mordred is much faster than PaDEL, but a surrogate still helps if the inner loop is huge.

Deferred for now: local Mordred is ~17 mol/s and is not the bottleneck yet.

- From the Papyrus Mordred table, train a small model: fingerprint → the selected descriptors.
- GA uses the surrogate to rank mutants.
- Re-run **real Mordred** only on the elite / final candidates.
- Keep a molecule only if real Mordred is in range.

Exit check: surrogate ranking agrees reasonably with Mordred on a holdout; finals are Mordred-verified.

---

## Step 7 — Transformer (small SELFIES GPT)

This is **our** transformer. Put it here, after the GA already works.

**Where:** `src/generate/transformer.py` + `train.py` + `sample.py`  
**Weights:** `checkpoints/`  
**What it is:** a small decoder-only GPT (tens of millions of params) trained on Papyrus SELFIES. Not Llama 3, not a frontier API.

**What it does:** next-token-predict molecule strings, then sample a pool of plausible candidates.

**What it does not do:** hit Mordred ranges. That stays the GA.

Steps:

- Convert cleaned Papyrus SMILES → SELFIES (`python -m molecule_gen build-selfies`).
- Train the transformer on those strings (`python -m molecule_gen train-transformer`).
- Use it as a **prior**: sample many candidates (`sample-transformer`) → those become the GA’s starting population (`generate --prior transformer`).

Llama stays optional and only for a chat UI later. It is not this transformer.

Exit check: sampled SELFIES decode to mostly valid, mostly unique molecules.

---

## Step 8 — Wire the full loop

```text
ranges in
  → transformer samples seed molecules     (src/generate/transformer.py)
  → GA evolves them toward the ranges      (src/generate/ga.py)
  → verify with Mordred 2D
  → return SMILES + actual descriptor values vs requested ranges
```

No Papyrus/ZINC rejection step.

Exit check: `python -m molecule_gen generate --spec examples/tight.json` uses the SELFIES GPT as the prior (override with `--prior papyrus`).

---

## Step 9 — UI / API (last)

- FastAPI backend (`python -m molecule_gen serve`) and Next.js frontend (`web/`) run separately.
- User picks descriptors and types min/max. Papyrus corpus min/max is shown as a **reference only**, not a hard limit.
- Live GA generations stream to the UI; the same log prints on the backend.

---

## Order of truth

1. Training data (Papyrus) and Mordred scoring must work before generation.
2. GA + Mordred must work before the transformer.
3. Transformer (SELFIES GPT in `src/generate/`) before Llama or any frontier API.
4. Success = in-range descriptors, not “absent from a database.”
5. UI last.
