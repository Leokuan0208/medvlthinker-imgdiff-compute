# ACC — the math (scores, expected cost, latency, energy)

All constants below are real: FLOPs from token counts × params; latency/energy from measured batch-1
runs (`src/cascade/measure_config.py`) + `ckpts/rt_cascade_cap320.jsonl`. CPU-reproducible from the
checkpoints. Scope of the parity claim: the 4 competent perception benchmarks (MMMU/MedXpert excluded).

## 1. Scores (the gating signals)
Each model, on one greedy decode of the answer letter, returns logprobs over candidate letters
`opt_logprobs = {A: ℓ_A, B: ℓ_B, …}` (natural log). Sort descending ℓ₍₁₎ ≥ ℓ₍₂₎ ≥ ….

The gate signal ACC uses is the **margin** (`harness.signals_from_logprobs`):
```
m = ℓ₍₁₎ − ℓ₍₂₎            (top-1 minus top-2 logprob)
```
(Other benchmarked signals use the softmax pᵢ = e^{ℓᵢ}/Σⱼ e^{ℓⱼ}: top1prob = p₍₁₎; entropy = −Σ pᵢ ln pᵢ;
gini = 1 − Σ pᵢ². The bake-off found margin as good as any — see gate_compare.)

Escalate at a tier iff `m < τ` (low margin ⇒ unconfident).

ACC-v2's think-tier gate uses **cross-model agreement** instead of single-model margin:
```
disagree = 1[ pred_7B-nt  ≠  pred_32B-nt ]
s₁       = disagree + ε·(−m_32B-nt),     ε = 1e-6
escalate to think iff s₁ > τ₁
```
With τ₁≈1 the integer part selects disagreements; the tiny ε·(−margin) tiebreak lets τ₁ pick only the
**lowest-margin disagreements** (so think fires ~14%, not the full ~32% disagreement rate).

Only fitted parameters: the scalars τ₀, τ₁, chosen on held-out calibration to reach parity
(always-32B-think accuracy) at minimum latency.

## 2. The cascade and its expected cost (32B-no-think is GATED, not run every time)
```
run T0 (7B-nt@cap320), always ;  if m₀ ≥ τ₀ → output, STOP
else run T1 (32B-nt@cap320)   ;  if not think-gated → output, STOP
else run T2 (32B-think@fullres)
```
Let e₀ = P(escalate past T0), e₁ = P(reach think). For ANY metric with per-tier cost c_T0,c_T1,c_T2:
```
        C = c_T0 + e₀·c_T1 + e₁·c_T2          (expected per-query cost)
```
T1 (32B-no-think) runs only on the e₀ escalated subset, and it is the SUBSTITUTE for the expensive
think pass, not an addition. Measured per-tier costs (ALL-6):

| tier | meanP | meanG | FLOPs | latency | energy |
|---|---|---|---|---|---|
| T0 7B-nt@cap320 | 388 | 2 | 0.01e15 | 0.21s | 25J |
| T1 32B-nt@cap320 | 388 | 2 | 0.03e15 | 0.34s | 65J |
| T2 32B-think@fullres | 685 | 391 | 0.07e15 | 26.6s | 6994J |

FLOPs of one run = `2·N·(P + G)` with N₇=7.6e9, N₃₂=33.0e9; P = prompt tokens (incl. vision), G = generated.
backbone% = Σ(cascade FLOPs) / Σ(always-32B-think FLOPs).

T2 latency ≈ 80× T1 (serial decode of ~391 think tokens), so latency ≈ e₁·26.6s. Plugging real
operating points (parity, ALL-6):
- **Deployed 2-tier (7B-nt → 32B-think):** e₁=69% → latency = 0.21 + 0.69·26.6 = **19.4s**
- **ACC-v2 (7B-nt → 32B-nt → 32B-think):** e₀=84%, e₁=14% →
  latency = 0.21 + 0.84·0.34 + 0.14·26.6 = 0.21+0.29+4.55 = **5.05s**

The win is the third term collapsing (e₁: 69%→14%) because 32B-no-think ≈ 32B-think in accuracy on
perception VQA, so it absorbs the escalations that don't need reasoning. The added T1 term (0.29s) is
negligible — that's why "adding a 32B-no-think tier" *lowers* total cost.

**FLOPs vs latency:** FLOPs only drops 79%→55%, but latency drops 19.4s→5.05s, because FLOPs counts the
parallel vision PREFILL heavily (every escalation pays it) while latency is dominated by the SERIAL
think DECODE that ACC mostly avoids.

## 3. Energy (measured, not assumed)
`measure_config.py` runs a `PowerSampler` thread that polls NVML GPU power every 25 ms during each
batch-1 query and integrates by the trapezoid rule over the query window:
```
E_query = Σ_k (P_k + P_{k+1})/2 · (t_{k+1} − t_k)   [Joules]
```
Per-config fit E(G) = α_e·G + β_e:
- 7B-nt@cap320: ~25 J (const, G≈2)
- 32B-nt@cap320: ~65 J (const)
- 32B-think@fullres: 18.17·G − 107.5 J ≈ 6994 J at G≈391 (≈254 W × 27.5 s — and 254 W matches the
  rt_cascade NVML power readings, a cross-check)

Cascade energy uses the same expected-cost formula E = c_T0 + e₀·c_T1 + e₁·c_T2:
- Deployed: 25 + 0.69·6994 = **5048 J**
- ACC-v2: 25 + 0.84·65 + 0.14·6994 = 25+55+1189 = **1268 J**

## Honesty caveat
FLOPs is exact. Latency and energy are CALIBRATED: fit from real batch-1 measurements (the think term
from rt_cascade's 5440 escalated queries, R²=0.99; the no-think tiers from measure_config.py), then
applied per-query via the expected-cost formula — not a single end-to-end wall-clock of the full ACC
pipeline. Hence "calibrated wall-clock latency/energy." Reproduce: `python3 src/cascade_methods/acc_v2.py`.

---

# Trained verifier — the math (selection, training, gap captured)

The second contribution. Notation: image v, question q, a candidate output a (a free-text answer, or a
bounding box). A generator M produces N i.i.d. samples a₁…a_N ~ M(·|v,q) at temperature T.

## 4. Verifier score
A LoRA-fine-tuned VLM verifier with parameters φ scores one candidate by the probability it assigns to the
token "Yes" vs "No" at the final position of the prompt "…Is the proposed answer correct? Answer Yes or No.":
```
s_φ(v,q,a) = P_φ(Yes | v,q,a) = e^{z_Yes} / (e^{z_Yes} + e^{z_No})
```
where z_Yes, z_No are the final-token logits. (Box-verifier: a is a box drawn on the image, same prompt
"does this red box localize the {target}?".)

## 5. Training objective
The verifier is trained on per-sample correctness labels y ∈ {0,1} from a neutral LLM judge (free-text) or
from IoU (boxes): y = 1[ judge(a)=correct ]  (free-text);  y = 1[ IoU(a, gold) ≥ θ ], θ = 0.3 (boxes).
Minimize binary cross-entropy on the Yes/No token (LoRA params φ only; base frozen):
```
L(φ) = − Σ_(v,q,a,y)  [ y · log s_φ(v,q,a) + (1−y) · log(1 − s_φ(v,q,a)) ]
```

## 6. Best-of-N selection
Return the candidate the verifier scores highest:
```
â = argmax_{i∈1..N}  s_φ(v,q,aᵢ)
```
Baselines on the same N samples: greedy = a₁ (the T=0 / first sample); self-consistency = the majority /
medoid answer; random = a uniformly random sample.

## 7. Oracle gap and fraction captured
With judged accuracy acc(·):
```
greedy      = acc(a₁)
oracle@N    = E[ max_{i} 1(aᵢ correct) ]          (an oracle picks a correct sample iff one exists)
selector    = acc(â)
gap-captured = (selector − greedy) / (oracle@N − greedy)
```
The "luck floor" finding: for every *training-free* selector, selector ≈ random ≤ greedy, so gap-captured ≈ 0.
The trained verifier achieves gap-captured = 0.49 (free-text, pooled-4), 0.39–0.53 (SLAKE boxes),
0.77–0.78 (MS-CXR boxes).

## 8. Verifier discrimination (AUROC)
Over all (sample, label) pairs, the verifier's ranking quality is
```
AUROC = P( s_φ(a⁺) > s_φ(a⁻) ),   a⁺ correct, a⁻ incorrect
```
Measured AUROC = 0.924 (n = 8512 samples), vs ≈ 0.5 for training-free self-verification — direct evidence
the signal is learnable but not zero-shot-surfaceable.

## 9. Test-time scaling and significance
Best-of-K accuracy is non-decreasing in K for a fixed verifier on nested sample sets; empirically it rises
0.385→0.501 (K:1→8) while random stays flat (~0.39). Significance by question-level bootstrap (B = 2000
resamples): CI = [P₂.₅, P₉₇.₅] of (selector − baseline). Free-text gain +0.116 [+0.092, +0.139] (n=1064);
MS-CXR box gain +0.191 [+0.152, +0.232] (n=435) — both exclude 0.
