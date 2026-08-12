# Progress — August 12, 2026 (the unified vision-aware 7B-only pipeline — refuted, and the premise with it)

> **Follows `progress_August_10.md`.** The direction came from Leo and it was a good one: *"the whole
> idea is to use only 7B + verifier, using one pipeline for both MCQ and open-text to match 32B while
> using less vram than 32B... this is a VLM research but we seem to be ignoring the Vision part for our
> verifier, even though that info is included into the embeddings of the model, maybe if we inject the
> vision signal into the verifier, it can help."* Six attacks ran. **The pipeline does not match the 32B
> — it falls 0.0404 short, significantly — and the vision hypothesis is refuted with a mechanism rather
> than a flat null: the verifier was never ignoring the image.** What the round *did* produce is a
> measured deployability frontier with a hardware-class discontinuity in it, and that is the paper.
> Round doc: `results/cascade_methods/docs/current/UNIFIED_VISION_VERIFIER_2026-08-12.md`.
> Every number names its artifact. Abstention appears nowhere.

---

## 1. The answer, first line

**A unified, vision-aware, 7B-only pipeline reaches macro 0.616278 against always-32B-direct's 0.656672:
−0.040395 [−0.052275, −0.028427], a significant loss.** It closes **32.2% [20.1%, 44.8%]** of the 0.0596
gap. Vision injection contributes **+0.000681 [−0.003831, +0.005217]** of macro — 1.14% of the gap, CI
spanning zero in both directions.

I re-derived that headline myself before believing it, from the raw per-item vectors, without importing
`sevenb_only_frontier.py`:

- `genframe_selector.py` reload verify: **pass, max abs deviation 4.47971781336598e-07**, and the
  reloaded heads reproduce the 2026-08-04 training-run logits **bit-exact at 0.0**.
- Macro baselines recomputed cell-by-cell from `_selector_rerun_parts/vec_disjoint.npz`: always-7B
  **0.597087**, always-32B-direct **0.656672**, gap **0.059586**. Identical.
- Frontier macro from my own script: **0.616278**, deviation **−3.233e-07** (6-dp rounding).
- My own paired item bootstrap, nboot=10,000: **−0.040395 [−0.052302, −0.028510]** vs direct and
  **+0.019191 [+0.011863, +0.026714]** vs always-7B. The artifact says [−0.052275, −0.028427] and
  [+0.012003, +0.026682]. Bootstrap RNG, nothing else.
- And the check I most wanted: an **alignment audit** between the two data sources. The selector's item
  order and the npz's open cells agree at **max abs deviation 0** on all three cells (645/200/1500), so
  the pairing in the bootstrap is real and not assumed.

Nothing was recomputed by hand. Both GPUs were busy all session (load average 22), so every check above
is CPU-side off frozen artifacts. No GPU job launched, nothing killed, `MedEvalKit/` untouched,
`freeze_selector.py` **not** run.

---

## 2. The vision hypothesis — and why it was wrong in an interesting way

I want to be careful here, because the hypothesis was not silly and the refutation is not "we tried and
it didn't work". The starting fact was real: the clean LoRA has **192 vision-tower tensors = 7,219,200
params = 15.2%** of its capacity, and it got there *by accident of naming* (`q/k/v/o_proj`,
`up/gate/down_proj` match blocks in both the LM and the ViT). No verifier had ever been designed around
the vision signal.

The attack built the obvious things: six vision-token concat arms, a learned cross-attention scorer over
the 6×6 patch grid, a capacity ablation of the deployed LoRA, and a laterality diagnostic. The primary arm
was named by a **pre-registered train-only 5-fold image-grouped CV** (`L_Vmean`, and it is genuinely the
CV maximum at 0.6750161364339039), with the eval-best arm (`L_prod_sim`) reported separately as a declared
anti-conservative second rule. 10 seeds everywhere. Both pre-registered falsification conditions fired.

**Then it asked the question I think is the actual contribution of the day: is the language-side
representation vision-blind at all?** Confound-free — both arms trained *and* tested in-distribution on
their own cache, folds computed once from the real cache and reused verbatim for the noise arm, identical
trainer, 10 seeds [`_visverif_parts/langside_image_dependence.json`]:

| arm | cv_sel_eff | cv_AUROC |
|---|---:|---:|
| real images | **0.804084** | 0.773203 |
| noise images | **0.780050** | 0.745898 |
| contrast | **+0.024035, 10/10 seeds positive** | **+0.027305, 10/10 positive** |

**The premise was false.** A causal LM attending over vision tokens already puts the image into the
language-side vector, which is exactly what the verifier reads. There is no blindness to fix, so explicit
injection is redundant — and that, not bad luck, is why all seven new architectures land inside the
±0.021 seed spread that is by now a field constant across ~27 architectures.

Four measurements make it concrete, and the second is my favourite thing in the round:

1. Zeroing **all 96 vision-tower LoRA modules** of the deployed verifier: **+0.015810 [−0.019763,
   +0.055336]**. *No harm. Point estimate positive.* The 15.2% of capacity sitting on the ViT does
   nothing measurable.
2. **All 192 vision tensors are `visual.blocks.*.mlp.{down,gate,up}_proj`.** Qwen2.5-VL names ViT
   attention `attn.qkv`/`attn.proj` — which is not in `target_modules`. **The ViT's attention was never
   adaptable under this recipe.** The spatial-mixing part that a laterality question actually needs has
   had *zero* capacity this whole time. That is a one-line fix and it is next step #2.
3. The xattn scorer localises **99.6–99.8% of its attention mass on one of 36 patches** (entropy 0.18% of
   uniform) and the attended position does not move sign-consistently between 'left'- and 'right'-bearing
   candidates. It found *something* to stare at; it is not laterality.
4. Similarity in the generator's **own** vision space scores correctness at **0.4346–0.4426 AUROC —
   below chance** — while relevance scores 0.581–0.598. The SigLIP/PubMedCLIP/BiomedCLIP failure
   reproduces *inside the model's own representation space*.

**The honest caveat, which I insisted go in the doc: "already present" is not "well used."** My own
arithmetic — the head sits 0.127824 above the measured random-pick floor (0.804084 vs 0.676260), and the
image is **0.024035 of that, i.e. 18.8%**. A real but minority contributor. And laterality is still the
weakest stratum at **0.613043** against 0.817186 on short non-laterality items, and nothing this round
moved it. So the open question is not *how to get the vision signal in*. It is **why a representation
that demonstrably contains it still cannot separate "Right." from "Left."** Different attack. Don't
conflate them.

---

## 3. The unification itself — the rule works, the mechanism doesn't

The brief's idea was to define the candidate set per format (given options for MCQ, sampled answers for
open text) while keeping one scorer, one decision rule, no format branch, no sampling luck. **The rule
does unify. The mechanism loses.** Scoring the given options is worse than the 7B's own argmax on 2 of 4
option cells and never better: PMC **−0.0758 [−0.0897, −0.0625] SIG**, PATH_VQA_closed **−0.1689
[−0.1886, −0.1499] SIG**, MedXpert +0.0025 n.s., VQA_RAD_closed −0.0518 n.s. Zero wins
[`unified_pipeline_2026-08-12.json`].

And it is **not** a floor artifact — candidate AUROC is 0.583–0.800, so the verifier *has* ranking signal
over the options. It is simply worse than the generator's argmax. When the generator's own answer is
folded in, the cross-fit global λ goes to **1.0**: the verifier is given zero effective weight.

That attack also found a **grader defect that killed its own only positive**, and I want it recorded
because it is exactly the failure mode CLAUDE.md warns about. The option branch is graded `pick == gold`;
the baselines go through MedEvalKit's extractor, which reduces a bare `"C:"` to the empty string
(`utils/utils.py:111-112`) and falls through to difflib. On 6,000 PMC items the graders disagree on **69**
items for the 7B and **94** for the 32B. The fusion arm's PMC win: **+0.0132 [+0.0072, +0.0192] SIG**
against the harness grader, **+0.0030 [−0.0023, +0.0083] NOT SIGNIFICANT** against a repaired one.
**The entire win was the grader.** Good catch, honestly reported, self-inflicted wound taken.

*My knock-on arithmetic, flagged as NOT propagated:* under the repaired grader the PMC gap widens by
+0.0045, so the macro gap would move **0.059586 → 0.060148**. The frontier is internally consistent on the
harness-graded basis and I left it there rather than half-propagating a correction. `MedEvalKit/` was not
modified.

---

## 4. Five of eight cells have no 7B-side lever, and that is a measurement

The part of this round I trust most.

| cell | 7B-only best | 32B-direct | shortfall |
|---|---:|---:|---:|
| PMC_VQA | 0.542656 | 0.551780 | −0.009124 |
| SLAKE_closed | 0.825359 | 0.858852 | −0.033493 |
| VQA_RAD_closed | 0.780876 | 0.852590 | −0.071713 |
| PATH_VQA_closed | 0.840869 | 0.889054 | −0.048186 |
| MedXpertQA-MM | 0.261500 | 0.306500 | −0.045000 |
| SLAKE_open | 0.778295 | 0.818605 | −0.040310 |
| VQA_RAD_open | 0.510000 | 0.600000 | −0.090000 |
| PATH_VQA_open | 0.390667 | 0.376000 | **+0.014667 ← we win** |

On every MCQ cell the candidate set is **complete by construction** — gold is always among the options,
or in {yes,no} — so the coverage wall is **0** and the oracle-over-candidates is a vacuous **1.0**. The
honest MCQ ceiling is therefore the best *measured* 7B-only mechanism, and every one ever tried is
≤ greedy: verifier-over-options loses (above), MCQ TTA is **−0.0078** summed, sampling best-of-N is
structurally dead (PMC verifier pick 0.4325 below greedy 0.5060; MedXpert oracle@8 0.5365 below its own
luck floor 0.6808). Positive-part decomposition of the −0.0404: **selection 0.0163 / coverage 0.0000 /
pure capability 0.0259.** **~64% of what is left is capability the 7B does not have.**

And the framing number that ends the argument: with the MCQ cells at always-7B, **oracle@8** on the three
open cells closes only **104.95%** of the gap, and parity needs a **uniform open sel_eff of 0.988342** —
against a field constant of 0.78–0.81. **Open-text selection cannot carry this, even perfectly.**

---

## 5. The frontier, and the cliff in the middle of it

| configuration | macro | Δ vs 32B-direct | VRAM (d) | compute | 32B resident? |
|---|---:|---|---:|---:|:--:|
| always-7B | 0.597087 | −0.059586 | **23.4206** ᵐ | 0.2188× | **NO** |
| **best 7B-only** (frozen selector, bo8 on open) | **0.616278** | **−0.040395 [−0.052275, −0.028427]** | **23.4206** ᵐ | **1.4199×** | **NO** |
| ↳ same at 4-bit weights | not measured on macro-8 | — | **10.9792** ʳ | 1.4199× | **NO** |
| min escalation that ties (6/8 cells, 17.3% items) | 0.657365 | +0.000693 [−0.002214, +0.003646] TIE | ≥84.7452 ʳ | — | **YES** |
| pre-gen router, honest nested CV | 0.655945 | −0.000728 [−0.006614, +0.005268] | 77.999 wts | **0.7427×** | **YES** |
| shipped accuracy-max | 0.657500 | +0.0008 [−0.0022, +0.0037] TIE | 77.999 wts | 1.7400× | **YES** |
| **always-32B-direct** | **0.656672** | 0 | **72.6023** ᵐ | 1.0000× | **YES** |

ᵐ measured on a clean exclusive card; ʳ reconstructed ((c) + the 1.3835 GiB context offset, shared card).
Compute for the 7B-only row is derived: 8 generations + **7.636674** full-7B-pass-eq/question for the
selector stack [`genframe_head_ens8/recipe.json`] ⇒ macro 6.488753 ⇒ **1.4199×** direct.

**The structural fact of the round: accuracy and compute are continuous in the escalation rate, VRAM is
not.** Escalate **0.47% of items** and you are in a different hardware class, because the 32B has to be
somewhere. 7B weights 15.4937 + 32B weights 62.3125 = **77.8062 GiB** (both measured), +5.5555 derived
activations = 83.3617, +context = **84.7452**, against a **79.1384 GiB** usable A100. The
measured-only version needs no derived term at all: 18.7644 + 62.3125 = **81.0769**, over capacity
**before the 32B runs a single forward pass**. Resolution cannot fix it — the co-residency wall is a
*weights* wall, still short by 0.2579 GiB at cap20. Load-on-demand cannot fix it — most favourable
measured load **139.8 s = 74.4× a forward pass**, giving 24.15 s per query averaged over all traffic
against 0.325 s with the 32B on a second card. And the 4-bit strong leg, the one lever that could have
built a middle rung, **breaks the tie** (macro 0.657365 → 0.655034, CI lower bound −0.002214 → −0.006266)
— though read that correctly: the paired swap delta is **−0.002331 [−0.006028, +0.001417]**, spanning
zero. The tie was marginal (0.000686 of slack); NF4 is *not certified to preserve it*, which is not the
same as damaging accuracy.

**There is no measured operating point between 0.616278 at 23.42 GiB and a tie at ≥84.75 GiB.**

---

## 6. Corrections I am making (rule 7)

**C1 — I corrected the VRAM headline, and it is my own side of the round that was wrong.**
`vram_levers_2026-08-12.json:headline_round2` reports the nf4 unified arm at **8.5905 GiB / 8.45×
smaller**. That (d) is the peak over **12 open-text items**, worst driver **1,200 vision tokens**. But the
8-cell macro also needs the **MCQ leg**, whose worst item (MedXpert MM-1561) carries **46,816 vision
tokens** — and the same file's own `arms.nf4.mcq_by_cap.cap16384` records **(d) 10.9792**. The honest
whole-suite figures are **bf16 23.4206 = 3.10× smaller** and **nf4 10.9792 = 6.61×**. The fit verdicts
survive (10.9792 < 11.63 usable on a 12 GB board) but with **0.65 GiB** of headroom, so the safe claim is
a **16 GB** board. The brief's "3.9× VRAM advantage" is likewise the open-arm-only ratio. **Use 3.10×.**
Same failure mode as §9.6: a number correct in its own scope, quoted outside it.

**C2 — the "short-answer failure mode" that motivated the whole attack does not hold on the clean pool.**
It came from the **contaminated pooled4** verifier on a different n=1,064 pool. On the clean disjoint
verifier over n=2,345 the length pattern is **monotone decreasing**: 1-word golds are the verifier's
*strongest* stratum (**0.826396**), 2–3 words 0.713178, 4–8 words 0.500000. The weak stratum is
**laterality (0.613043)**; laterality items merely happen to be short. **Length was a confounder.**

**C3 — the grader defect, §3 above.** Recorded, not propagated, direction stated.

---

## 7. What I recommend, and why

Leo asked to choose between *"beat 32B outright"* and *"match 32B cheaply."* **Neither, as stated — but a
third framing is measured, robust, and unclaimed.**

*Beat it outright* is not the direction. The target is **0.0596**; eight pre-registered attacks over two
rounds failed to close **0.0029**, a target 20× smaller. This round adds eight more failures at the same
wall, and ~64% of the residual is capability on cells where the coverage wall is provably 0. Selection
efficiency 0.78–0.81 is not a number one more architecture moves.

*Match it cheaply* is nearly won and is not news: shipped accuracy-max already ties at +0.0008, and the
pre-gen router reaches 0.655945 at **0.7427×** compute. But both keep the 32B resident — *cheap* means
FLOPs, and the box is still a 78 GiB box.

**The claim the data supports:**

> A single 7B model plus a 47.6M-parameter LoRA verifier, with no large model anywhere in the system,
> recovers **32.2% [20.1%, 44.8%]** of the 7B→32B gap and runs in **23.42 GiB — 3.10× less VRAM** than
> always-32B-direct's measured 72.60, and **6.61× less at 4-bit (10.98 GiB)**, moving the deployment from
> an 80 GB datacentre card to a 24 GB (or 16 GB) board. The price is **−0.0404 [−0.0523, −0.0284]** macro,
> of which ~64% is generator capability and **0% is coverage**. And VRAM is **discontinuous** in the
> escalation rate: 0.47% of items requires ≥84.75 GiB, and both mitigations — load-on-demand and a 4-bit
> strong leg — are measured and both fail.

Two of its three legs are negative results with mechanisms, which is this project's established
contribution shape. The discontinuity claim is new. It needs **no** win over 32B-direct to be true.

**What the paper must not say:** that the 7B-only pipeline matches always-32B-direct; that it is cheaper
in FLOPs (**1.4199×**); that vision injection helps; or that it is 8.45× smaller (**3.10× / 6.61×**).

---

## 8. Ranked next steps

1. **Generator capability on the 4 MCQ cells** — 0.0207 of macro with no 7B-side mechanism at all.
   Distillation or domain FT; no verifier can touch it. Fourth confirmation that generator work outranks
   verifier work.
2. **Laterality, framed correctly** — add `attn.qkv` / `attn.proj` to the verifier's `target_modules` so
   the ViT's attention has non-zero capacity for the first time. One line, well-motivated by the
   structural finding, and *not* the arm this round refuted. ~30.5% of vqa_rad_open turns on a laterality
   token.
3. **Measure the unified nf4 arm's (d) directly**, clean exclusive card, MedEvalKit default cap, item pool
   including MedXpert. 10.9792 is a reconstruction with 0.65 GiB of headroom and it is now load-bearing.
4. **4-bit accuracy on the 5 unmeasured cells**, all 3 open cells included. If nf4 is free suite-wide the
   headline becomes 10.98 GiB / 6.61× rather than 23.42 / 3.10×. (The open NF4 rows scored with
   `use_llm_judge=False` read 0.000 for **both** arms — not accuracy, never quote them.)
5. **Verifier-resolution sweep with a matched control on the full n=2,345 pool.** n=600 found monotone
   degradation and a significant hit on the ≤3-word stratum (−0.0339 [−0.0627, −0.0052]) but could not
   resolve the cap320 rung.
6. **Do NOT** spend more compute on vision-injection architectures, listwise/pairwise/set-aware scorers,
   or (choice)(why). All measured; all inside the field constant or worse.

---

## 9. Housekeeping

- Doc: `results/cascade_methods/docs/current/UNIFIED_VISION_VERIFIER_2026-08-12.md`.
- Artifacts this round: `vision_verifier_2026-08-12.json` (+ `_visverif_parts/`),
  `sevenb_only_frontier_2026-08-12.json` (+ `_frontier_verify_parts/`), `vram_levers_2026-08-12.json`
  (+ `_vram_levers_parts/`), `unified_pipeline_2026-08-12.json` (+ `_unified_pipeline_parts/`),
  `min_escalation_2026-08-12.json`, `pregen_router_2026-08-12.json`, `shrink_strong_leg_2026-08-12.json`,
  `cost_decomposition_2026-08-12.json`, `pmcvqa_grader_defect_2026-08-12.json`.
- ⚠️ **`main` is still 63+ commits ahead of `origin/main`, and `ckpts/`, `feats_hidden/`, `logs/` have
  zero tracked files.** `feats_hidden/` (4.4 GB), `ckpts/train/genframe_head_ens8/` (29.4 MB) and
  `ckpts/train/lora_verifier_disjoint/` (190 MB) are the inputs to every number above and **a push does
  not protect them.** Copy to `/data` or external storage. Top-priority chore, unchanged.
- ⚠️ **Never run `freeze_selector.py`** (rewrites the frozen selector; a refit is a fresh seed draw).
  ⚠️ **Never score a visual LoRA under vLLM** (drops all 192 `visual.*` modules; 0.775204 HF vs 0.702997).
- `MedEvalKit/` untouched. No process was killed on either shared card.
