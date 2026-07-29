# Knowledge-augmentation (RAG) feasibility — dead, for an informative reason

> New-method-loop, 2026-06-25 (Direction 4: the one lever that ADDS information rather than re-selecting
> model outputs, so it is not a-priori bound by the luck floor). Zero-GPU feasibility check on existing
> checkpoints (consistent judge, this session). Conclusion: NOT viable, because the apparent open-ended
> headroom is capacity-bound or benchmark artifact, not a retrievable knowledge gap.

## Why this lever was worth checking
Selection/sampling/synthesis are all luck-floored ([[openended-selection-luckfloor]]) — they only re-rank
the model's OWN outputs. Retrieval ADDS external facts, so IF errors are knowledge-limited it could fix
what selection cannot. Feasibility = is there a pool of errors that is (a) beyond the sampling luck floor
(oracle@8 wrong = "genuinely unknown") AND (b) genuinely knowledge-limited (not perception / not artifact)?

## (a) There IS a genuinely-unknown pool, concentrated in knowledge questions (SLAKE content_type)
| group | n | greedy | oracle@8 | genuinely-unknown |
|---|---|---|---|---|
| KNOWLEDGE (KG + Abnormality) | 150 | 0.593 | 0.773 | **22.7%** |
| perception (Modality/Position/Organ/…) | 495 | 0.780 | 0.911 | 8.9% |

Knowledge questions have 2.5× the genuinely-unknown rate. Encouraging — until you ask what those errors ARE.

## (b) But they are NOT knowledge-limited — two independent kills
**Kill 1 — SLAKE: the gap is capacity, not knowledge.** Of the 7B's genuinely-unknown errors (12.1% of
SLAKE), the higher-capacity 32B fixes only **37%**, and **equally across knowledge (38%) and perception
(36%)**. If the deficit were knowledge, the 32B (more medical facts) would fix knowledge questions
preferentially — it does not. The 32B's help is general capacity (matches the §5.7 recoverability bound),
and the residual hard-for-both is only ~7.6% of SLAKE. SLAKE KG questions are also perception-confounded
(the `triple` is templated `[vhead, relation, ktail]` — you must READ the organ/disease from the image,
then look up the relation), so "knowledge-unknown" is partly mis-perception.

**Kill 2 — PathVQA: the difficulty is GENUINE, not artifact (hypothesis TESTED and REFUTED).** PathVQA-open
has a massive genuinely-unknown pool — **48.3%**, **42.9% hard-for-both** (32B fixes only 11% of it). A
14-case eyeball SUGGESTED these were decontextualized caption-extraction artifacts (e.g. "what does process
begin as?" → gold `'a focus of microabscess in a vascular loop…'`). **So I ran a systematic LLM audit**
(`run_artifact_audit.py`, neutral 32B classifies each Q+gold as ANSWERABLE vs ARTIFACT) **and it REFUTED the
artifact hypothesis:**
- PathVQA 80% labeled "artifact" — but ARTIFACT-labeled questions have HIGHER accuracy (greedy 0.369, 32B
  0.426) than ANSWERABLE-labeled (greedy 0.144, 32B 0.174), and artifacts are NOT enriched in hard-for-both
  (0.87×, slightly depleted). So the "artifact" label tracks *terse fragment-style* (which the model does
  BETTER on), NOT unanswerability. The difficulty lives in the WELL-FORMED answerable questions (acc 0.144).
- SLAKE 24.5% artifact, and there it DOES correlate sensibly (artifact 0.620 < answerable 0.774) — a modest
  real signal, de-artifacted SLAKE acc 0.774.

So PathVQA open-ended difficulty is **genuine model limitation** (these models are simply weak at open-ended
pathology), not a caption artifact. My small-sample impression was wrong; the systematic test corrected it.

## Conclusion (corrected)
The add-information lever is **not viable**: SLAKE genuinely-unknown errors are capacity-bound (32B fixes
them equally across knowledge/perception → not a knowledge gap), and PathVQA difficulty is genuine
(answerable questions are the hardest; 32B barely helps; oracle 0.144→0.308 unharvestable). RAG would target
a pool that is either capacity-bound or genuinely beyond these models — low EV, killed before GPU spend.

**Honest correction:** an earlier draft of this doc claimed PathVQA headroom was "dominated by caption
artifact." The systematic audit REFUTES that (answerable questions are harder; artifacts not enriched in
hard-for-both). The corrected, stronger finding: the open-ended headroom is **genuine but unharvestable** —
real model limitation + luck-floored selection ([[openended-selection-luckfloor]]) + capacity-bound
recoverability, NOT a cleanable artifact. De-artifacting does not reopen a method win (clean SLAKE still
luck-floored 0.758<0.879; clean PathVQA still 0.144→0.308 the 32B can't capture). This is the deepest, most
robust finding of the loop, now confirmed across GATE, ACTION, SELECTION, SYNTHESIS, and RETRIEVAL: **no
training-free method harvests the open-ended headroom because the bottleneck is genuine "which answer is
right?" knowledge the models lack, not a signal/selection/information problem.**
