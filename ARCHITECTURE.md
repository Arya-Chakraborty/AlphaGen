# Architecture of a Descriptor-Constrained Molecule Generator

**molecule-gen** is a local software system that proposes small organic molecules whose computed physicochemical properties fall inside user-specified ranges. A user selects Mordred two-dimensional descriptors (for example molecular weight, SLogP, topological polar surface area, rotatable-bond count) and a minimum and maximum for each. The system returns unique molecules that satisfy every selected range, together with the actual descriptor values.

The generator is a two-stage pipeline. A small decoder-only transformer, trained on Papyrus molecules written as SELFIES, samples chemically plausible seed structures. A genetic algorithm then mutates those seeds until the Mordred scores lie in the requested box. Papyrus is training and lookup data only. A hit that already exists in Papyrus is accepted. Success is defined as *in-range descriptors*, not as novelty relative to a database.

This document describes the full architecture so that a reader who has not seen the source code can follow the data, the models, and the search procedure.

---

## 1. Problem and design principles

Drug-like molecules are commonly filtered by simple 2D properties: size, lipophilicity, polarity, flexibility, ring count, hydrogen-bond donors and acceptors. Those quantities can be computed from a 2D graph; they do not require a 3D conformer. The practical task is therefore:

> Given a set of named 2D descriptors and an interval for each, produce *N* distinct valid molecules whose computed values all lie in those intervals.

Several constraints shaped the design:

1. **The scorer is the source of truth.** A molecule is a hit if and only if its Mordred 2D vector satisfies every range. Membership in Papyrus or ZINC is irrelevant.
2. **The language model does not optimize properties.** The transformer learns the distribution of drug-like strings. Range matching is left to an explicit search (the genetic algorithm) that can be scored with the same calculator the user cares about.
3. **Mutations must stay chemically valid.** The search representation is SELFIES, not SMILES, so most edits still decode to a molecule.
4. **New molecules are scored locally.** Papyrus supplies a precomputed Mordred table for corpus lookup. Generated structures are computed with the same 2D Mordred calculator (`mordredcommunity`, `ignore_3D=True`) and cached by InChIKey.
5. **The stack stays small and local.** The transformer is a GPT-style model of a few million parameters trained on this machine. It is not a frontier API and not Llama.

---

## 2. System overview

```text
                    user ranges (JSON spec)
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Seed population       │
                 │  SELFIES GPT  (~50%)   │
                 │  Papyrus near-misses   │
                 │  (optional: Papyrus    │
                 │   lookup, or pasted    │
                 │   SMILES)              │
                 └────────────┬───────────┘
                              │ scored candidates
                              ▼
                 ┌────────────────────────┐
                 │  Genetic algorithm     │
                 │  mutate / crossover    │
                 │  SELFIES → SMILES      │
                 │  chemistry filter      │
                 │  Mordred 2D + penalty  │
                 └────────────┬───────────┘
                              │
                              ▼
                    N unique in-range SMILES
                    + values vs requested min/max
```

The same loop is exposed as a command-line tool (`python -m molecule_gen generate`) and as a FastAPI service with a Next.js front end. Offline commands prepare Papyrus, import Mordred tables, encode SELFIES, and train the transformer.

Roles of the main pieces:

| Component | Responsibility | What it does not do |
|---|---|---|
| Constraint spec | Names, min/max, how many molecules | Invent structures |
| Mordred 2D + penalty | Decide whether a molecule is in range | Propose candidates |
| RDKit | Parse, sanitize, canonical SMILES, InChIKey, 2D drawings, organic filter | Compute the 1,613 Mordred properties |
| SELFIES GPT | Sample plausible drug-like strings | Hit descriptor ranges |
| Genetic algorithm | Steer seeds into the ranges | Learn chemistry from scratch |
| Papyrus 05.7 | Training corpus and optional lookup | Filter “already known” molecules out |

---

## 3. Data foundation: Papyrus 05.7

Training and lookup use **Papyrus 05.7** (release 2024.2; Zenodo record 13987985; CC BY-SA 4.0; Béquignon et al., *J. Cheminform.* 2023). Papyrus aggregates standardized small-molecule bioactivity data. This project uses only the unique 2D structures and the precomputed Mordred 2D table. Activity labels, 3D coordinates, and protein descriptors are not required.

### 3.1 Structures

The file `05.7_combined_2D_set_without_stereochemistry.sd.xz` (~454 MB) is downloaded from Zenodo (or supplied locally). Each structure is parsed with RDKit, sanitized, written as canonical SMILES, and given a recomputed InChIKey. Invalid molecules are dropped. The result is one row per **connectivity** — the first 14-character block of the InChIKey, which identifies the 2D graph without stereochemistry.

Output: `data/processed/papyrus_molecules.parquet`  
Columns: `connectivity`, `canonical_smiles`, `inchikey`  
Size: on the order of **1.27 million** unique 2D molecules.

Connectivity is the join key to the descriptor table.

### 3.2 Precomputed Mordred 2D

Papyrus also publishes `05.7_combined_2D_moldescs_mordred2D.tsv.xz` (~3.1 GB): **1,613 two-dimensional Mordred descriptors** for the same molecules. The importer streams the TSV into parquet, keyed by connectivity.

Output: `data/processed/papyrus_mordred2d.parquet`

This table is used for (a) scoring Papyrus seeds without recomputing Mordred, and (b) showing corpus min/max in the user interface as a *reference*, not a clamp on the user’s ranges.

Papyrus ships other descriptor files (Mold2, CDDD, ECFP6 fingerprints, 3D Mordred). They are not used. In particular, Papyrus’s RDKit contribution is ECFP6 bit vectors, not RDKit’s physicochemical descriptor list.

### 3.3 Local Mordred for new molecules

Generated SMILES are not in the Papyrus table (or may differ in tautomer/canonical form). They are scored with **mordredcommunity**, the maintained Python implementation of Mordred, using `Calculator(descriptors, ignore_3D=True)` so that only 2D properties are requested. RDKit supplies the molecule object; Mordred performs the descriptor arithmetic.

Results are stored in SQLite (`data/processed/mordred_cache.sqlite`) keyed by InChIKey. The same molecule is never recomputed. A small sample command (`mordred-sample`) and a comparison command (`compare-mordred`) exist to check that local names and values line up with the Papyrus table. A few ring-count names differ between the Papyrus Mordred export and current mordredcommunity (`aliphRing` versus `ARing`, and similar). Those aliases are mapped in code. Detour-matrix descriptors are banned from specs because the two calculators disagree on them.

Approximate local throughput is on the order of 17 molecules per second for a full 2D vector. That is fast enough that a learned surrogate (fingerprint → selected descriptors) was designed but **not implemented**.

---

## 4. Molecular representation

Three string representations appear in the pipeline.

**SMILES** is the interchange format (input, output, Papyrus storage). RDKit parses SMILES, sanitizes, and emits a canonical SMILES plus InChIKey. Identity and de-duplication use the InChIKey.

**SELFIES** (Self-Referencing Embedded Strings; Krenn et al., 2020) is the *generative* representation. Every SELFIES string decodes to a valid molecule under the SELFIES grammar. The transformer emits SELFIES tokens; the genetic algorithm inserts, deletes, replaces, and crosses over SELFIES tokens. After each edit the string is decoded to SMILES and re-canonicalized. Sequences longer than **80 SELFIES tokens** are truncated or dropped so that training and mutation stay inside a fixed context window.

**RDKit molecule objects** sit under both. They are used for sanitization, InChIKey, the conservative organic filter, Mordred input, and SVG depiction in the UI. RDKit’s own `Descriptors` module (MolWt, MolLogP, TPSA, …) is **not** the scoring function.

The organic filter, applied to GPT seeds and to GA mutants (not to the raw Papyrus corpus), keeps ordinary neutral organics: C, N, O, S, P, and halogens; 6–80 heavy atoms; no formal charges; no radicals; no carbon with two or more double bonds (a cheap allene / cumulative-double-bond check). Mutation tokens are restricted to the corresponding SELFIES alphabet (no boron, no charged atoms).

---

## 5. Constraint specification and scoring

A request is a JSON object, for example:

```json
{
  "n": 5,
  "descriptors": {
    "SLogP":  {"min": 2.8, "max": 3.2},
    "MW":     {"min": 360, "max": 390},
    "TopoPSA":{"min": 70,  "max": 90},
    "nRot":   {"min": 4,   "max": 5},
    "nRing":  {"min": 3,   "max": 3}
  }
}
```

`n` is how many distinct in-range molecules to return. Descriptor names may be local mordredcommunity names or Papyrus aliases. Featured names used in examples include `MW`, `SLogP`, `TopoPSA`, `nRot`, `nRing`, `nHBDon`, `nHBAcc`, and `nAtom`.

Given a descriptor vector, the scorer assigns each range a **penalty**:

- 0 if the value lies in \([min, max]\)
- \(((delta / width))^2\) if it lies outside, where \(width = max - min\) (or 1 if the interval is a point)
- 1.0 if the value is missing

The molecule penalty is the sum over selected descriptors. A molecule is **in range** only if every selected descriptor passes. Fitness for ranking is the pair

\[
(\text{0 if in range else 1},\ \text{total penalty})
\]

so any fully feasible molecule outranks every partial miss, and among misses the closest ones rank first.

Descriptor values for a candidate are resolved in order:

1. SQLite cache by InChIKey  
2. Papyrus parquet by connectivity — **only if lookup is allowed**  
3. Local Mordred on the SMILES, then write the cache  

During genetic-algorithm evolution, Papyrus lookup is turned **off**. Seeds may use the table; mutants are always computed locally so the score matches the structure that was just created.

---

## 6. The transformer prior (SELFIES GPT)

### 6.1 What it is

The prior is a **small decoder-only transformer** trained to predict the next SELFIES token, in the style of GPT. It is implemented in PyTorch in this repository. It is not Llama, not a hosted API, and it never sees descriptor ranges during training.

The network is:

- token embedding + learned positional embedding  
- \(N\) transformer blocks, each **pre-norm**: LayerNorm → causal self-attention → residual, then LayerNorm → MLP (GELU, 4× hidden width) → residual  
- final LayerNorm and a linear language-model head  
- causal attention via PyTorch `scaled_dot_product_attention(..., is_causal=True)`

Context length is 82 (80 SELFIES tokens plus beginning- and end-of-sequence tokens). Dropout is 0.1 on the default models.

Three size presets:

| Preset | Layers | Heads | Embedding | Typical use |
|---|---|---|---|---|
| `tiny` | 2 | 2 | 64 | Tests / debugging |
| `small` (default) | 4 | 4 | 256 | Laptop / local GPU |
| `medium` | 8 | 8 | 512 | Larger local run (~25 million parameters) |

The default `small` model is on the order of a few million parameters.

### 6.2 Tokenizer

SELFIES is already a tokenized language: symbols such as `[C]`, `[=O]`, `[Branch1]`, `[Ring1]`. The tokenizer maps those symbols to integer ids. Special tokens occupy ids 0–3: `<pad>`, `<bos>`, `<eos>`, `<unk>`. The vocabulary is the SELFIES semantic-robust alphabet plus any extra symbols observed in the training corpus.

Encoding of one molecule is `<bos> …tokens… <eos>`, truncated to the block size if needed. Decoding stops at `<eos>` and ignores pad/unknown.

### 6.3 Training corpus

Papyrus canonical SMILES are encoded to SELFIES (`build-selfies`). Strings that fail to encode, or that exceed 80 tokens, are skipped. The result is `data/processed/papyrus_selfies.parquet`.

Training (`train-transformer`) loads that corpus (or encodes on the fly if the parquet is missing). A tokenizer is built from the sequences. Each batch is packed as next-token pairs: input \(x = tokens[:-1]\), target \(y = tokens[1:]\), padded to the block size. The loss is **token-level cross-entropy**, ignoring pad.

Optimization:

- AdamW, learning rate \(3 \times 10^{-4}\), \(\beta = (0.9, 0.95)\), weight decay 0.1  
- gradient clipping at global norm 1.0  
- default batch size 32  
- default one epoch; two epochs are suggested in the project README for a first local run  
- device is chosen automatically: MPS, else CUDA, else CPU  

Weights, tokenizer, and a small training manifest are written to `checkpoints/selfies_gpt.pt`.

The model is **unconditional**. It is not told about SLogP or molecular weight. After training it can sample strings that look like Papyrus molecules.

### 6.4 Sampling

To draw seeds, the model is placed in eval mode. Each sequence starts at `<bos>`. At every step the last-position logits are divided by a temperature (default 1.0), the pad token is masked out, **top-\(k\)** filtering is applied (default \(k = 50\)), and a token is drawn from the softmax. Generation stops when every sequence in the batch has emitted `<eos>`, or when the block is full.

Each decoded SELFIES string is converted to canonical SMILES. Invalid or duplicate SMILES are discarded. Drawing continues until the requested number of unique valid SMILES is reached, or a budget of attempts is exhausted (about \(20 \times N\)). For generation, those SMILES are further passed through the organic filter described in Section 4.

A standalone command `sample-transformer` writes a parquet of SMILES for inspection without running the genetic algorithm.

---

## 7. The genetic algorithm

The genetic algorithm is the component that **hits the ranges**. It operates on a population of already-scored candidates, each carrying canonical SMILES, SELFIES, InChIKey, descriptor values, penalty, and origin.

### 7.1 Initialization

Default prior (`--prior transformer`):

1. Sample about half the seed budget from the SELFIES GPT and keep chemically reasonable unique SMILES.  
2. Mix in the other half as **Papyrus near-misses**: corpus molecules that are *not* already in range, ranked by how many constraints they already pass and then by lowest penalty. These give the search a foothold near the box without simply returning molecules that already satisfy the spec.  
3. Deduplicate by InChIKey and score GPT seeds against the spec.

`--no-near-miss` uses the GPT only. `--prior papyrus` ranks the whole corpus (in-range first, then lowest penalty) and is the original “generation v0” path from before the transformer existed. `--smiles` replaces seeds with user-provided structures. `--generations 0` returns scored seeds and does not evolve.

If the requested number of hits is already present in the seed set, the algorithm stops immediately.

### 7.2 One generation

Defaults: population 64, 20 generations, elite fraction 25% (at least two elites), crossover probability 0.3.

1. Sort the population by fitness (in-range first, then penalty).  
2. Copy the elites into the next generation.  
3. Fill the remaining slots with children:  
   - pick a random elite parent;  
   - with probability 0.3, one-point **crossover** of two elite SELFIES strings; otherwise **mutate** (replace, insert, or delete one SELFIES token, using the restricted organic alphabet);  
   - decode to SMILES; drop if invalid, unreasonable, or a duplicate InChIKey;  
   - score with local Mordred (cache-backed);  
   - if fully in range, record as a hit.  
4. Emit a progress event (generation index, hit count, best penalty, a preview of the population).

The loop stops when `n` unique in-range molecules exist, or the generation limit is reached. Output is de-duplicated by InChIKey and written to `data/processed/generate_hits.parquet` plus a JSON manifest (requested vs found, seed counts, spec, prior used).

Existing-in-Papyrus is allowed. The output batch is de-duplicated; the corpus is not used as a rejection filter.

---

## 8. End-to-end generation procedure

Putting the pieces together, a typical run (`python -m molecule_gen generate --spec examples/tight.json`) does the following.

1. **Load the spec** and resolve descriptor names.  
2. **Sample GPT seeds**, filter to reasonable organics, score against the spec (cache / Papyrus / local Mordred).  
3. **Load Papyrus near-misses** for the same spec and encode them to SELFIES.  
4. **Run the GA** until *N* hits or the generation cap. Mutants are always locally Mordred-scored.  
5. **Write hits**: SMILES, InChIKey, connectivity, per-descriptor values and pass/fail flags, penalty, origin (`transformer`, `near_miss`, `mutant`, …), and generation number.

Example specs in the repository:

- `examples/tight.json` — narrow drug-like box (SLogP 2.8–3.2, MW 360–390, …), five molecules.  
- `examples/loose.json` — Papyrus 10th–90th percentiles of the same properties, ten molecules.  
- `examples/ethanol.json` — a tiny box used as a sanity check.

Offline preparation, in order, is: fetch Papyrus structures → import Mordred parquet → (optional) local Mordred sample / compare → encode SELFIES → train transformer → generate.

---

## 9. User interface and API

The command-line interface covers every stage (`fetch-papyrus`, `import-mordred`, `build-selfies`, `train-transformer`, `sample-transformer`, `generate`, `score`, `suggest-ranges`, `serve`).

The web application is two processes:

- **Backend:** FastAPI on port 8000. `POST /generate` starts a background job (one generation run at a time). Progress is streamed as Server-Sent Events (`sampling`, `near_miss`, `ga_start`, `generation`, `done` / `error`). Other endpoints list allowed descriptor names, return Papyrus corpus min/max/percentiles as reference statistics, and render a SMILES string as an RDKit SVG.  
- **Frontend:** Next.js (React) on port 3000. The user searches Mordred names, types min/max, sets *N*, population, and generations, and watches the GA log and structures update live. Corpus ranges are labelled as reference only; typed limits are not clamped.

The API generation path always uses the transformer prior and requires `checkpoints/selfies_gpt.pt` to exist.

---

## 10. Software organization

Python package `molecule-gen` (Python ≥ 3.10), source under `src/molecule_gen/`:

```text
data/          Papyrus download, RDKit cleaning, parquet writers
descriptors/   local Mordred, SQLite cache, name aliases, Papyrus comparison
constraints/   spec JSON, penalty, Papyrus column lookup
generate/      SELFIES utilities, chemistry filter, GPT, training, sampling,
               seeding, genetic algorithm, orchestrator
api/           FastAPI jobs and SSE
cli.py         all of the above as subcommands
```

Supporting pieces: `web/` (Next.js UI), `examples/` (spec files), `tests/` (unit and a few Mordred integration tests), `data/` and `checkpoints/` (runtime artifacts, gitignored).

Core libraries: RDKit, pandas, PyArrow, SELFIES, mordredcommunity. Optional extras: PyTorch for training/sampling, FastAPI/uvicorn for the API.

---

## 11. What is intentionally out of scope

- **Novelty vs. a database.** Generated molecules may match Papyrus. That is accepted.  
- **3D docking, synthesizability, or target activity.** Only 2D Mordred ranges are optimized.  
- **PaDEL or RDKit native descriptors as the oracle.** The spec, the Papyrus table, and the GA fitness are Mordred 2D. RDKit is the chemistry engine underneath, not the property calculator.  
- **A fingerprint→descriptor surrogate** inside the GA. Planned if Mordred becomes the bottleneck; currently unused.  
- **A large language model in the generation loop.** Llama was considered only as a possible later chat UI, not as the molecule prior.

---

## 12. How to read the pipeline in one sentence

Papyrus teaches a small SELFIES language model what drug-like molecules look like; a genetic algorithm, scored by Mordred 2D on RDKit molecules, edits those proposals until every user-chosen property sits inside the requested range.

---

## References

- Béquignon, O. J. M. et al. Papyrus: a large-scale curated dataset aimed at bioactivity predictions. *Journal of Cheminformatics* 15, 3 (2023). Dataset: Papyrus 05.7, [doi:10.5281/zenodo.13987985](https://doi.org/10.5281/zenodo.13987985).  
- Moriwaki, H. et al. Mordred: a molecular descriptor calculator. *Journal of Cheminformatics* 10, 4 (2018). Local calculator: `mordredcommunity`.  
- Krenn, M. et al. Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. *Machine Learning: Science and Technology* 1, 045024 (2020).  
- Landrum, G. RDKit: Open-source cheminformatics.  
- Vaswani, A. et al. Attention is all you need. *NeurIPS* (2017); Radford, A. et al. Improving language understanding by generative pre-training (GPT).  
- Yap, C. W. PaDEL-Descriptor: An open source software to calculate molecular descriptors and fingerprints. *Journal of Computational Chemistry* 32, 1466–1474 (2011). Not used in this system; cited only as the slower Java alternative that Mordred replaces here.
