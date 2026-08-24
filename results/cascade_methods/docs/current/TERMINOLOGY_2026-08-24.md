# Terminology: what the field calls the things we have been naming ourselves
**2026-08-24** · raised after the last meeting · applies to every deck, doc and paper draft from here

Two of our habitual terms are internal coinages that a reader in this field will either not
recognise or will read as meaning something else. This file fixes the mapping once so it stops
drifting between documents.

---

## 1. "cell" → **benchmark** (or *dataset*)

**Not a domain convention.** It is an internal reporting term of ours: a "reporting cell" meant one
dataset × one answer format, which mattered when we reported multiple-choice and open-ended arms
side by side. Outside this repo it reads as a table cell, or as biology.

The medical-VQA literature says **dataset** or **benchmark**, and reports *per-dataset* results —
VQA-RAD, SLAKE, PathVQA, PMC-VQA are all introduced as datasets and evaluated as benchmarks
(BESTMVQA, arXiv 2312.07867; PMC-VQA, arXiv 2305.10415).

Where the format distinction still matters, the convention is a **split** or **subset**, not a new
noun: "the open-ended split of SLAKE", and the field already uses **open-ended** vs **closed-ended**
exactly as we do.

| ours | use instead |
|---|---|
| "eight cells" | **eight benchmarks** (all open-ended splits) |
| "the omnimed cell" | **OmniMedVQA**, or *the OmniMedVQA open-ended split* |
| "reporting cell" | **benchmark** |
| "per-cell" | **per-benchmark** |
| "in-domain cell" / "new-domain cell" | **benchmark seen in training** / **held-out benchmark** |

Keep `cell_*.json` artifact filenames as they are — those are paths, and renaming them would break
every reference. Cite the file, describe the benchmark.

## 2. "the MLP head" → **probe** (what it is) + **verifier** (what it does)

**"Head" is misleading here.** In the literature a *head* is a layer trained jointly with, and
attached to, a backbone — a classification head, a value head, a reward head. Ours is none of those:
it is a standalone module over hidden states of a **frozen** Lingshu-7B that we never fine-tune.

**"MLP" alone is worse**, because it names the architecture and not the role. Every reader will ask
"an MLP doing what?"

The field has two precise terms and we need both, because they answer different questions:

- **Probe / probing classifier** — what it physically is. Training a lightweight classifier
  (linear or MLP) on frozen hidden states to predict a property is *exactly* the probing-classifier
  setup, and an MLP in that role is a **non-linear probe** (see the probing literature, e.g. LPASS
  arXiv 2505.24451; PEP arXiv 2608.08024, which probes frozen hidden states for hallucination).
- **Verifier** — what it does. In test-time-compute work, the model that scores sampled candidates
  so one can be selected is the **verifier** (equivalently an outcome reward model), and
  verifier-style search *is* best-of-N selection. Pointwise scoring of each candidate is
  **pointwise reranking**.

| ours | use instead |
|---|---|
| "the MLP head" (first mention) | **an MLP probe on frozen hidden states, used as a best-of-N verifier** |
| "the head" (thereafter) | **the probe verifier**, or just **the verifier** |
| "head logits/scores" | **verifier scores** |
| "head − greedy" | **verifier − greedy decoding** |

⚠️ This repo already uses **verifier** for the LoRA model (`lora_verifier_pooled4`). When both appear
in one document, say **probe verifier** vs **LoRA verifier**. Never let "the verifier" be ambiguous.

## 3. The rest of the vocabulary

| ours | field term | note |
|---|---|---|
| "pool" | **candidate set** (candidate pool is also used) | "the 8-candidate set" |
| "string prior" | **answer-prior baseline** | P(correct \| answer string); define at first use |
| "greedy" | **greedy decoding** | spell it out at least once |
| "oracle@8" | **oracle@8** ✓ | already conventional in best-of-N work |
| "self-consistency" | **self-consistency** ✓ | conventional (majority vote over samples) |
| "coverage" | **coverage** ✓ | conventional: is a correct answer present at all |
| "selection skill" / "sampling penalty" | ours, and that is fine | a decomposition we introduce; define it explicitly and do not imply it is standard |
| "escalation / cascade" | **cascade**, **routing** ✓ | conventional |

## 4. One sentence that gets the framing right

> A lightweight MLP probe reads the frozen hidden states of a 7B medical VLM and acts as a
> **best-of-N verifier**, scoring each of 8 sampled candidate answers so one can be selected,
> evaluated across eight open-ended medical VQA benchmarks against greedy decoding, an answer-prior
> baseline, and self-consistency.

Sources consulted 2026-08-24: BESTMVQA (arXiv 2312.07867) · PMC-VQA (arXiv 2305.10415) ·
Scalable Best-of-N Selection via Self-Certainty (arXiv 2502.18581) · LLMs for Reranking: A Survey
(TechRxiv 176300630) · LPASS (arXiv 2505.24451) · Prompt Embedding Probes (arXiv 2608.08024).
