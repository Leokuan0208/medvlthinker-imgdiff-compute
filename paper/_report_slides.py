# slide content for build_report_v2.py (executed in its namespace: S, slide(), img(), peer, ac, cg, fmt, J)
def TBL(headers, rows, ours_idx=None):
    h="".join(f"<th>{c}</th>" for c in headers)
    body=""
    for r in rows:
        cls=" class='total'" if (r is rows[-1]) else ""
        tds="".join(f"<td{' class=ours' if (ours_idx is not None and j==ours_idx and i>0) else ''}>{c}</td>" for j,c in enumerate(r) for i in [1])
        body+=f"<tr{cls}>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>"
    return f"<div class='tbl-wrap'><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"

# CANONICAL same-split (held-out n=1064), VERIFIED_FACTS §F
VER, GRE, SCC, M32, ORC = 0.501, 0.413, 0.411, 0.462, 0.592

# S1 COVER
S.append(slide(f'''<div class="cover">
<div class="eyebrow"><span class="dot"></span>Progress report · CVGIP 2026</div>
<div class="big">Test-time compute for medical VLMs:<br><span class="accent">what actually helps.</span></div>
<div class="sub">Continuing from last time's efficiency cascade. This time, with the math and peer baselines you asked for — and a new result: in medical open-ended VQA, the standard training-free tricks fail, but a small <b>trained verifier</b> beats them all — <i>and</i> a model 5× its size.</div>
<div class="cover-stats">
<div class="cstat"><div class="v">{fmt(SCC,3)}</div><div class="l">self-consistency (the standard) — no better than greedy {fmt(GRE,3)} (the majority trap)</div></div>
<div class="cstat"><div class="v teal">{fmt(VER,3)}</div><div class="l">our trained verifier — beats every baseline</div></div>
<div class="cstat"><div class="v">{fmt(M32,3)}</div><div class="l">a 5× larger model (scale-up) — still below ours</div></div>
<div class="cstat"><div class="v teal">0.924</div><div class="l">verifier AUROC (tells right from wrong)</div></div>
</div></div>'''))

# S2 WHERE WE LEFT OFF
S.append(slide('''<div class="eyebrow"><span class="dot"></span>1 · Where we left off</div>
<h2 class="slide-h">Last time, and the two fair criticisms</h2>
<p class="body"><b>Last report:</b> the <b>ACC cascade</b> — answer most medical questions with a cheap 7B model, escalate only the hard ones to a 32B, and save compute at the same accuracy.</p>
<div class="callout honest"><b>The criticism (rightly):</b> (1) the <b>math wasn't explained</b>, and (2) there was <b>no comparison to peer methods</b> — so the results had no reference point and were hard to judge.</div>
<p class="body"><b>This report does two things:</b></p>
<ul class="body">
<li><b>Fixes the rigor</b> — ACC stated with full math and benchmarked against published cascade methods.</li>
<li><b>Advances the research loop</b> — we ask whether the cascade's <i>decision</i> can be improved, hit a wall, and find a genuinely new method (a trained verifier) that beats peer baselines from top venues.</li>
</ul>
<div class="callout note">Each step below is presented as: <b>question → what we need to know → baseline → experiment → math → result → what it means → next step.</b></div>'''))

# S3 SETUP + TERMINOLOGY
S.append(slide('''<div class="eyebrow"><span class="dot"></span>2 · Setup &amp; terminology (so the numbers mean something)</div>
<h2 class="slide-h sm">Models, datasets, and how we measure</h2>
<div class="two"><div>
<p class="body"><b>Models.</b> A cheap <b>7B</b> medical VLM and an expensive <b>32B</b> one (MedVLThinker / Lingshu). The 32B can answer fast ("no-think") or reason first ("think", slow).</p>
<p class="body"><b>Datasets (chosen for reasons, not convenience).</b> The three that anchor everything — <b>SLAKE, VQA-RAD, PathVQA</b> — are competent and exist in <i>both</i> multiple-choice (for the cascade) and free-text (for the verifier) form. PMC-VQA &amp; MMMU are multiple-choice-only (no free-text answers to select among); MedXpert is near-chance (excluded from headlines). Kvasir (different modality) and RadImageNet (held-out) test generalization.</p>
</div><div>
<dl class="def">
<dt>FLOPs (compute)</dt><dd>$F=2N(P{+}G)$: $F$ = compute (FLOPs), the factor&nbsp;2 = one multiply&nbsp;+ one add per parameter per token, $N$ = (prompt tokens $P$ incl. the image + generated tokens $G$). Prefill-included. Reported as % of always-32B-think.</dd>
<dt>Latency</dt><dd>wall-clock seconds for one question, batch-1, end-to-end (prefill + generation), measured in isolation.</dd>
<dt>Energy</dt><dd>GPU power sampled every 25 ms and integrated over the question (Joules).</dd>
<dt>AUROC</dt><dd>$P(\\text{score(correct)} > \\text{score(wrong)})$ — how well a score separates right from wrong. 0.5 = useless, 1.0 = perfect.</dd>
</dl></div></div>'''))

# S3b detailed models + datasets
ds_rows=[
["SLAKE","radiology (CT/MRI/X-ray), bilingual; Qs on organs, abnormalities, position","open + closed","ACC + verifier","645 (open)"],
["VQA-RAD","radiology (chest/head/abdomen); clinician-written Q&amp;A","open + closed","ACC + verifier","200 (open)"],
["PathVQA","pathology / histology microscopy; many yes/no","open + closed","ACC + verifier","1500 (open)"],
["PMC-VQA","figures from PubMed Central biomedical articles (mixed)","MCQ (4-opt)","ACC","large"],
["MMMU-Med","college-exam medical Qs, 5 subjects, multi-image","MCQ","ACC (reasoning)","~1.5k"],
["MedXpertQA-MM","expert/exam-level multimodal medical, hardest","MCQ","ACC (near-chance)","~2k"],
["Kvasir-VQA","GI endoscopy images; findings","open","verifier (OOD)","1200"],
["RadImageNet-VQA","radiology; fully held-out","open","verifier (transfer)","2000"],
["MS-CXR","chest X-ray: localize a described pathology","bounding box","box-verifier","435"]]
S.append(slide('''<div class="eyebrow"><span class="dot"></span>2 (cont.) · Experiments — models &amp; datasets in detail</div>
<h2 class="slide-h sm">Exactly what we run, and on what</h2>
<p class="body"><b>Models</b> (all Qwen2.5-VL-based medical VLMs): cheap leg = <b>7B</b> (Lingshu-7B for open-ended; MedVLThinker-7B for the MCQ cascade); strong leg = the <b>32B</b> counterpart, run either <b>no-think</b> (fast) or <b>think</b> (slow reasoning trace). The <b>verifier</b> = the 7B with a ~190 MB LoRA adapter on top of the frozen base; the <b>box-verifier</b> = Qwen2.5-VL-7B + LoRA.</p>
'''+TBL(["dataset","what it is (domain + content)","format","role","n (test)"], ds_rows)+'''
<p class="body" style="font-size:1.05rem;color:var(--muted)">"open" = free-text short answer; "closed" = yes/no; "MCQ" = pick a lettered option. These are the standard public medical-VQA benchmarks (the same suite the Lingshu paper reports).</p>'''))

# S3c evaluation methodology
S.append(slide('''<div class="eyebrow"><span class="dot"></span>2 (cont.) · How accuracy &amp; cost are measured</div>
<h2 class="slide-h sm">Evaluation methodology (so every number is comparable)</h2>
<dl class="def">
<dt>MCQ accuracy</dt><dd>extract the model\'s chosen option letter and <b>exact-match</b> it to the gold letter. No judge needed.</dd>
<dt>Open-text accuracy</dt><dd>an <b>LLM judge</b> (a strong neutral model) decides whether the free-text answer is <i>semantically</i> the gold answer (e.g. "CT" = "computed tomography"). Labels come from the dataset answer key, not the judge\'s opinion; exact-match is too brittle for free text.</dd>
<dt>Verifier quality</dt><dd><b>AUROC</b> (does its score rank correct answers above wrong ones); <b>gap-captured</b> = (acc of picked − greedy)/(oracle − greedy); <b>oracle@N</b> (best possible if you always picked a correct sample); bootstrap 95% CIs.</dd>
<dt>Efficiency</dt><dd><b>latency</b> = batch-1 end-to-end wall-clock (s); <b>energy</b> = GPU power sampled every 25 ms via NVML, integrated (J); <b>FLOPs</b> = $2N(P{+}G)$, prefill-included.</dd>
<dt>Discipline</dt><dd>gate/verifier trained on a <b>disjoint</b> split (no question leaks into the test set); every number is verbatim from real checkpoints (none fabricated).</dd>
</dl>'''))

# S4 ACC method + gate peers EXPLAINED + which SOTA
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>3 · The efficiency result, now with math + peers</div>
<h2 class="slide-h sm">ACC, and the cascade gates we compare against</h2>
<p class="body"><b>Idea.</b> The 32B\'s <i>fast</i> (no-think) mode is as good as its slow (think) mode on perception questions (SLAKE 0.849 vs 0.764) — thinking <i>over-thinks</i> them. So we insert the 32B-fast as a middle tier and fire the slow tier only when the two fast legs <b>disagree</b> ("iff" = if and only if):</p>
<div class="eq">$$\\text{fire slow tier} \\iff \\mathbb{1}[\\hat y_{7B}\\neq\\hat y_{32B\\text{-fast}}]+\\epsilon(-m)>\\tau;\\qquad \\text{cost } C=c_0+e_0c_1+e_1c_2$$</div>
<p class="body" style="font-size:1.05rem"><b>Every symbol:</b> $\\mathbb{1}[\\cdot]$ = 1 if the bracket is true else 0 (an "indicator"); $\\hat y_{7B},\\hat y_{32B\\text{-fast}}$ = the answers the 7B and the fast-32B give; $\\neq$ = "differ" (they disagree); $m$ = the fast-32B's <b>margin</b> (confidence = gap between its top-2 answer probabilities); $\\epsilon(-m)$ = a small bump that raises escalation when confidence $m$ is low; $\\tau$ (tau) = the threshold (think when the left side exceeds it); $C$ = total cost; $c_0,c_1,c_2$ = cost of tier&nbsp;0/1/2; $e_0,e_1$ = fraction of questions reaching tier&nbsp;1/2.</p>
<p class="body"><b>The gates we benchmark (each = a published way to decide "escalate?"):</b></p>
<dl class="def">
<dt>Confidence / MSP / Chow\'s rule <span class="src">(Chow 1970; Hendrycks &amp; Gimpel, ICLR 2017)</span></dt><dd>escalate when the model\'s top answer probability is low. The classic baseline.</dd>
<dt>Entropy / Gini (DOCTOR)</dt><dd>escalate on high spread of the answer distribution.</dd>
<dt>AutoMix <span class="src">(Madaan et al., 2023)</span></dt><dd>escalate based on the model self-verifying its own answer.</dd>
<dt>FrugalGPT <span class="src">(Chen et al., 2023)</span></dt><dd>a <i>learned</i> cost-aware scorer that decides when the cheap answer suffices.</dd>
<dt>Jitkrittum L2D <span class="src">(NeurIPS 2023)</span></dt><dd>the <b>theoretically-optimal</b> learned deferral rule — the strongest principled gate; our main learned baseline.</dd>
<dt>CAR <span class="src">(arXiv 2505.15154, 2025)</span></dt><dd>certainty-based adaptive routing for <i>multimodal</i> models — the <b>closest prior art</b> to ACC\'s think-gating.</dd>
</dl>
<div class="callout note"><b>What we compare to:</b> the accuracy ceiling is <b>always-32B-think</b> (parity); the efficiency baseline is the standard 2-tier confidence cascade. The SOTA peers above are the published gates; CAR is the nearest multimodal prior art, Jitkrittum L2D the strongest learned one.</div>'''))

# S4b ACC result table + FLOPs-vs-latency explanation + the cluster finding
acc_rows=[["always-32B-think (parity, ceiling)","0.572","100%","11.34 s","6319 J"],
["Jitkrittum L2D (learned, NeurIPS\'23)","0.567","51%","2.29 s","1195 J"],
["FrugalGPT-style learned (Chen\'23)","0.568","60%","3.30 s","1766 J"],
["AutoMix self-verify (\'23)","0.569","55%","2.50 s","1307 J"],
["MSP/Chow confidence","0.570","57%","2.96 s","1568 J"],
["Ours: ACC (agreement gate)","0.569","52%","2.27 s","1182 J"]]
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>3 (cont.) · ACC result — same accuracy, ~⅕ the latency</div>
<h2 class="slide-h sm">Every gate clusters: the win is the structure, not the gate</h2>
'''+TBL(["ALL-6, at equal accuracy","acc","FLOPs","latency","energy"], acc_rows)+'''
<div class="callout win"><b>Result:</b> at parity accuracy, latency 11.34 s → <b>2.27 s</b> (−80%), energy ~5× lower, FLOPs halved. Holding the 3-tier structure fixed, <b>all gates land in the same cluster</b> (FLOPs ~50–60%, latency ~2–3 s) — so ACC\'s advantage is the <b>3-tier structure</b>, not a cleverer gate. On <b>ALL-5</b> (excluding near-chance MedXpert) it is sharper still: <b>8.88 s → 0.44 s (−95%)</b>, FLOPs to 25%. (Full 10-method × 5-family table in the paper.)</div>
<div class="callout honest"><b>Why FLOPs and latency don\'t move together</b> (e.g. ACC has slightly higher FLOPs than Jitkrittum, 52% vs 51%, yet lower latency, 2.27 vs 2.29 s): <b>FLOPs</b> is dominated by the parallel image <b>prefill</b>, paid on every 32B escalation — even the fast no-think tier (which generates only ~2 tokens). <b>Latency/energy</b> are dominated by the serial <b>think decode</b> (~hundreds of tokens). ACC escalates a bit more to the fast no-think tier (more prefill-FLOPs, almost no added latency); the two methods are otherwise tied. This is exactly why ACC\'s latency win (−80%) is larger than its FLOPs win (−48%).</div>'''))

# S5 the gate is saturated
S.append(slide('''<div class="eyebrow"><span class="dot"></span>4 · Loop step — can we improve the gate itself?</div>
<h2 class="slide-h sm">No: the gate is saturated (and there's theory for why)</h2>
<p class="body"><b>What a gate really needs</b> is not "is the cheap model wrong?" but "<b>will the 32B fix it?</b>" (recoverability). We tested ~12 signals (confidence, entropy, agreement, self-checking, conformal routing CP-Router, learned routers…).</p>
<ul class="body">
<li><b>All cap at AUROC ≈ 0.6.</b> Recoverability is nearly unpredictable from any cheap signal.</li>
<li><b>Why:</b> the two models <b>fail on the same questions</b> — error correlation $\\phi=0.37$ ($\\phi$ = correlation of their right/wrong outcomes); <b>58%</b> of the 7B's errors the 32B also gets wrong.</li>
</ul>
<div class="callout note"><b>This isn't just our finding.</b> Jitkrittum et al. (NeurIPS 2023) prove the optimal deferral rule needs <i>both</i> models' confidence and that simple confidence-deferral is fundamentally limited. We confirm it empirically for medical VLMs.</div>
<div class="callout q"><b>So we change the question.</b> Forget <i>when</i> to escalate. Given several tries from one model, can we <b>pick the best</b>?</div>'''))

# S6 next step: open-ended pivot
S.append(slide('''<div class="eyebrow"><span class="dot"></span>5 · Next step — picking the best of N tries</div>
<h2 class="slide-h sm">Why we must switch to open-ended (free-text) answers</h2>
<p class="body"><b>Best-of-N selection</b> = sample the model N times, then choose the best answer. But on multiple-choice, every sample is just one letter (A/B/C/D) — there is nothing to discriminate.</p>
<div class="callout note"><b>Our own evidence:</b> the same confidence signal is near-useless on multiple-choice (AUROC ~0.6) but jumps to <b>~0.87</b> on free-text answers — a <i>discreteness</i> effect. So selection must be studied <b>open-ended</b>, on SLAKE / VQA-RAD / PathVQA (free-text) + Kvasir + RadImageNet.</div>
<p class="body">The cheap 7B is sampled 8× per question; a strong neutral 32B <b>LLM judge</b> grades each answer correct/incorrect (exact-match is too brittle for free text).</p>'''))

# S7 selection baselines
S.append(slide('''<div class="eyebrow"><span class="dot"></span>6 · The baselines we must beat (and where they're from)</div>
<h2 class="slide-h sm">Standard ways to pick the best of N — all training-free</h2>
<dl class="def">
<dt>Greedy</dt><dd>just take the first/most-likely answer. The deploy default.</dd>
<dt>Self-consistency / majority vote <span class="src">(Wang et al., ICLR 2023)</span></dt><dd>take the answer that appears most often across the N samples. The standard, widely-used selector.</dd>
<dt>Self-verification, P(True) <span class="src">(Kadavath et al., 2022)</span></dt><dd>ask the model "is this answer correct?" and use its Yes-probability to rank.</dd>
<dt>Self-certainty best-of-N <span class="src">(NeurIPS 2025, arXiv:2502.18581)</span></dt><dd>rank samples by the model's own output-distribution certainty — recent training-free SOTA. (Needs per-sample logprobs we didn't store; being training-free it shares the majority trap, so we benchmark the self-consistency and self-verify members directly and expect self-certainty to be luck-floored too.)</dd>
<dt>Oracle@N</dt><dd>an impossible upper bound: always pick a correct sample if one exists. Measures the headroom.</dd>
</dl>
<div class="callout note">These are exactly the selectors the test-time-scaling literature benchmarks. Question: do any of them work in <b>medical</b> VQA?</div>'''))

# S8 the wall: training-free selection fails
land_rows=[["SLAKE","0.738","0.738","0.829","0.895"],["VQA-RAD","0.519","0.500","0.648","0.722"],
["PathVQA","0.352","0.349","0.377","0.513"],["Kvasir","0.282","0.282","0.326","0.493"],
["POOLED (n=1064)","0.413","0.411","0.462","0.592"]]
S.append(slide('''<div class="eyebrow"><span class="dot"></span>7 · Experiment — do the standard selectors work?</div>
<h2 class="slide-h sm">No. Training-free selection barely moves, despite huge headroom</h2>
'''+TBL(["dataset","greedy","self-consistency","32B (scale-up)","oracle@8"], land_rows)+'''
<ul class="body">
<li><b>Self-consistency does not beat greedy</b> (0.411 vs 0.413 — actually below it). The reason — the <b>majority trap</b>: the correct answer is a <i>minority</i> vote in 74–90% of recoverable cases, so majority voting picks the wrong one.</li>
<li><b>Scaling up</b> to the 5× larger 32B only reaches 0.462 (same questions).</li>
<li>Yet the <b>oracle is 0.59</b> — the right answer is usually <i>in</i> the 8 samples; we just can't pick it.</li>
</ul>'''))

# S9 insight / hook
S.append(slide('''<div class="eyebrow"><span class="dot"></span>8 · The insight that motivates the method</div>
<h2 class="slide-h sm">Medical VQA is exactly where a trained verifier should pay off</h2>
<div class="two"><div>
<p class="body"><b>In general LLM tasks</b>, the literature finds trained-verifier best-of-N <i>barely beats</i> self-consistency (e.g., the self-certainty paper, NeurIPS'25; "optimal aggregation" 2510.13918). So people often skip the verifier.</p>
</div><div>
<p class="body"><b>But in medical VQA, self-consistency fails</b> (the majority trap). That's precisely the regime where a learned selector should be worth its cost — the gap between "random pick" and "oracle" is large and untapped.</p>
</div></div>
<div class="callout q"><b>Hypothesis &amp; next experiment:</b> train a small verifier to pick the right answer. If it works, it's a method that matters <i>here</i> in a way it doesn't in general LLMs.</div>'''))

# S10 the method + math
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>9 · The method — a trained outcome verifier (with math)</div>
<h2 class="slide-h sm">A small LoRA verifier that scores "is this answer correct?"</h2>
<p class="body"><b>Architecture.</b> A LoRA adapter (~190 MB) on the frozen 7B VLM. Input: image + question + a candidate answer. Output: the probability it puts on "Yes" vs "No" at the final token.</p>
<div class="eq">$$s_\\phi(v,q,a)=P_\\phi(\\text{Yes}\\mid v,q,a)=\\frac{e^{z_{\\text{Yes}}}}{e^{z_{\\text{Yes}}}+e^{z_{\\text{No}}}}\\qquad(\\text{$v$=image, $q$=question, $a$=candidate answer})$$</div>
<p class="body" style="font-size:1.05rem"><b>Every symbol:</b> $s_\\phi$ = the verifier's score (its estimated probability the answer is correct); $\\phi$ (phi) = the trained LoRA weights; $P_\\phi(\\text{Yes}\\mid v,q,a)$ = probability it says "Yes, correct" given image $v$, question $q$, candidate $a$; $z_{\\text{Yes}},z_{\\text{No}}$ = the model's raw output scores (<b>logits</b>) for the "Yes"/"No" tokens; $e^{z}$ = the exponential of $z$; the whole fraction is the <b>softmax</b> turning the two logits into a probability between 0 and 1.</p>
<p class="body"><b>Training</b> (cross-entropy on the judge's correct/incorrect labels $y$; base frozen):</p>
<div class="eq">$$\\mathcal{L}(\\phi)=-\\textstyle\\sum\\big[y\\log s_\\phi+(1-y)\\log(1-s_\\phi)\\big];\\qquad \\text{select } \\hat a=\\arg\\max_{i\\le N} s_\\phi(v,q,a_i)$$</div>
<p class="body" style="font-size:1.05rem"><b>Every symbol:</b> $\\mathcal{L}(\\phi)$ = the training loss we minimize; $\\sum$ = sum over training examples; $y$ = the true label (1 if the candidate is correct, 0 if not); $\\log$ = natural logarithm; $\\hat a$ = the chosen answer; $\\arg\\max_{i\\le N}$ = pick the candidate $i$ (of the $N$ samples) with the highest verifier score; $a_i$ = the $i$-th sampled answer; $N$ = number of samples.</p>
<p class="body">We report the <b>fraction of the oracle gap captured</b>: $\\;(\\text{acc}(\\hat a)-\\text{greedy})/(\\text{oracle}-\\text{greedy})$ — where $\\text{acc}(\\hat a)$ = accuracy of the picked answer, greedy = the single-sample baseline, oracle = accuracy if a correct sample were always picked.</p>
<div class="callout note"><b>What it trains on, and why a 32B-judged 7B beating the 32B isn't circular:</b>
<ul class="body" style="margin-top:6px">
<li><b>Trained on</b> ~6,000 <i>(image, question, candidate-answer)</i> examples from the <b>70% train split</b> (question-disjoint from the test set) of the four open-ended datasets (SLAKE/VQA-RAD/PathVQA/Kvasir); candidates are the 7B\'s own samples, labelled vs gold; <b>RadImageNet held out</b> as transfer. Base = Lingshu-7B (the open-ended generator we use) + a ~190 MB LoRA.</li>
<li>The <b>judge</b> is an automated <i>grader</i>, not a knowledge oracle: it only checks "does this answer match the <b>gold</b> answer?" (exact-match is too brittle for free text). Labels come from the <b>answer key</b>, not the 32B's knowledge — it could be a human or exact-match.</li>
<li>The verifier learns to <b>discriminate</b> correct answers (easy), not <b>generate</b> them (hard). The 7B's 8 samples already contain a correct answer ~59% of the time (the oracle); the verifier just learns to <b>pick</b> it. Picking is easier than knowing, so a <i>7B</i> verifier suffices — it harvests the 7B's own diversity, it does not import 32B knowledge.</li>
<li>Fair framing: a 5× model <i>zero-shot</i> vs. a small model + a small <i>trained</i> verifier — the latter wins. Honest caveat: the verifier is supervised in-domain (a few thousand gold labels); that supervision is the contribution (and its cost).</li>
</ul></div>'''))

# S11 headline result (filled from peer_comparison if available)
def per_rows():
    return [["PathVQA","0.352","0.349","0.377","0.441","0.513"],["Kvasir","0.282","0.282","0.326","0.405","0.493"],
            ["VQA-RAD","0.519","0.500","0.648","0.611","0.722"],["SLAKE","0.738","0.738","0.829","0.762","0.895"],
            ["POOLED","0.413","0.411","0.462","0.501","0.592"]]
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>10 · Result — the verifier beats every baseline (incl. the 32B)</div>
<h2 class="slide-h sm">It beats self-consistency, the bigger model, and recovers ~half the gap</h2>
'''+TBL(["dataset","greedy","self-consist.","32B scale-up","<b>verifier (ours)</b>","oracle@8"], per_rows())+
img("paper/figs/limits/fig_peer_comparison.png","Pooled accuracy: the trained verifier (teal) beats greedy, self-consistency, and the 5×-larger 32B; oracle is the ceiling.")+'''
<div class="callout win"><b>The headline:</b> the verifier captures <b>49%</b> of the oracle gap and <b>beats the 5×-larger 32B</b> (0.501 vs 0.462, +0.039, 95% CI [+0.010,+0.066], excludes 0). Per-dataset it beats the 32B on the hardest sets — PathVQA (0.441 vs 0.377) and Kvasir (0.405 vs 0.326) — exactly where scaling up fails.</div>'''))

# S12 why trust it
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>11 · Why we trust it</div>
<h2 class="slide-h sm">It genuinely discriminates, uses the image, and scales</h2>
<div class="two">
<div>'''+img("paper/figs/limits/fig_verifier_discrimination.png","AUROC 0.924: clean separation of correct vs incorrect candidates.","100%")+'''</div>
<div>'''+img("paper/figs/limits/fig_verifier_scaling.png","Best-of-K: accuracy rises with samples; random stays flat.","100%")+'''</div></div>
<ul class="body">
<li><b>Discrimination AUROC 0.924</b> (n=8512 candidates) — not a "lazy verifier"; blanking the image drops it −0.047 (it uses the image).</li>
<li><b>Statistically significant:</b> bootstrap 95% CI on the gain over greedy = <b>[+0.092, +0.139]</b>; and over the <b>32B itself</b> = <b>[+0.010, +0.066]</b> (excludes 0 — beats the 5× model).</li>
<li><b>Argmax is the right rule:</b> we tried verifier-<i>weighted</i> voting and a score×count hybrid — both are <i>worse</i> (0.489, 0.470 vs 0.501), because the majority trap contaminates even score-weighted voting. Pure verifier-argmax wins.</li>
<li><b>Test-time scaling:</b> more samples → higher accuracy; random selection does not improve.</li>
</ul>'''))

# S13 generalization
def cg_line():
    if not cg: return "Cross-generator transfer — the Lingshu-trained verifier, applied to a <b>different</b> generator's answers (MedVLThinker-7B), still works: SLAKE-open 0.543 → <b>0.620</b> (49% of gap) and VQA-RAD-open 0.395 → <b>0.520</b> (61% of gap) — the verifier is generator-agnostic."
    parts=[]
    for ds,t in cg.items():
        parts.append(f"{ds.replace('_open','')}: greedy {fmt(t.get('greedy'))} → verifier <b>{fmt(t.get('verifier'))}</b> (oracle {fmt(t.get('oracle'))})")
    return "Cross-generator transfer — the Lingshu-trained verifier, applied to a <b>different</b> model's (MedVLThinker-7B) answers: "+"; ".join(parts)+"."
S.append(slide(f'''<div class="eyebrow teal"><span class="dot"></span>12 · Generalization — multiple models, modalities, output types</div>
<h2 class="slide-h sm">The verifier is not a one-dataset trick</h2>
<ul class="body">
<li><b>Other base model (method, not just transfer):</b> we also trained the full method from scratch on a <b>MedVLThinker-7B</b> base — it works on SLAKE (0.564→0.622, 42%) and is pooled-positive (25%), but <i>fails</i> on VQA-RAD's tiny split (n=54: 0.500→0.470). So the method <b>partially</b> transfers to a second base — not uniformly; base quality matters (the Lingshu verifier even transfers to MedVLThinker's outputs better, 49–61%, than a from-scratch MedVLThinker verifier, 25%). Honest caveat: the Lingshu result (49%, 4 datasets, 2-seed) is the validated headline. <i>And</i> cross-generator transfer: {cg_line()}</li>
<li><b>Out-of-distribution modality:</b> works on Kvasir (GI endoscopy), a modality it wasn't built around.</li>
<li><b>Held-out transfer:</b> the pooled verifier lifts RadImageNet (never trained on) +0.024 zero-shot.</li>
<li><b>Different output type — bounding boxes:</b> the same idea works for grounding (data below).</li>
</ul>
<div class="tbl-wrap"><table><thead><tr><th>grounding (IoU≥0.3)</th><th>greedy</th><th>SC-medoid</th><th>verifier (ours)</th><th>oracle@8</th><th>gap</th></tr></thead><tbody><tr><td>SLAKE organs (n=487)</td><td>0.197</td><td>0.164</td><td>0.255</td><td>0.343</td><td>40%</td></tr><tr class="total"><td>MS-CXR chest X-ray (n=435)</td><td>0.041</td><td>0.053</td><td>0.232</td><td>0.285</td><td>78%</td></tr></tbody></table></div>
<p class="caption">IoU = box overlap (intersection÷union); a box is correct if IoU≥0.3. MS-CXR gain +0.191, 95% bootstrap CI [+0.152,+0.232]; "5.6× lift" = trained 0.232 ÷ greedy 0.041.</p>
<ul class="body" style="display:none">
</ul>'''+img("paper/figs/limits/fig_trained_verifier_unified.png","One principle across output types: training breaks the selection wall for free-text answers, organ boxes, and chest-X-ray boxes.","78%")))

# S14 integration / two axes
S.append(slide('''<div class="eyebrow"><span class="dot"></span>13 · How the two pieces fit together</div>
<h2 class="slide-h sm">Two axes of test-time compute</h2>
<div class="two">
<div class="callout win"><b>ACC — spend compute across model <i>configurations</i></b> (cheap → fast-big → slow-big). Buys <b>efficiency</b>: same accuracy, ~⅕ latency.</div>
<div class="callout win"><b>Verifier — spend compute across <i>samples</i></b> (best-of-N). Buys <b>accuracy</b>: reaches what a 5× larger model cannot — at lower latency than always-thinking.</div></div>
'''+img("paper/figs/limits/fig_accuracy_compute.png","Accuracy vs compute: spending compute on samples+verifier reaches accuracy the bigger model can't.","82%")+'''
<div class="callout note">They are complementary levers of one idea — <i>allocate test-time compute where it pays</i> — and combine: a verifier-augmented cascade (cheap best-of-N → escalate the residual to the 32B) reaches <b>0.517</b> at 35% escalation — above both the verifier alone (0.501) and the 32B (0.462) — i.e. the accuracy-optimal point (at a compute premium; the fully-measured deployable version is the next step).</div>'''))


# S14b latency/energy of the verifier system (the hybrid result)
lat_rows=[["always-32B-think (the strong baseline)","~0.43*","~11 s","~6300 J"],
["always-32B (no-think)","0.462","~0.3 s","~60 J"],
["7B + verifier (best-of-8)","0.501","~3.5 s","~1000 J"],
["7B + verifier cascade (35% escalate)","0.517","~3.6 s","~1050 J"]]
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>13 (cont.) · Cost of the verifier — latency &amp; energy</div>
<h2 class="slide-h sm">It beats the 32B at a fraction of the always-think latency</h2>
<p class="body"><b>The hybrid you can deploy:</b> the cheap 7B samples N answers, the verifier picks the best, and only low-confidence cases escalate to the 32B. Costs below use measured batch-1 per-tier latency/energy (same Qwen2.5-VL architecture):</p>
'''+TBL(["open-ended pooled (n=1064)","accuracy","latency*","energy*"], lat_rows)+'''
<div class="callout win"><b>Result:</b> the 7B+verifier (0.501) and the cascade (0.517) <b>beat the 32B (0.462) at ~⅓ the latency of always-32B-think</b> (~3.5 s vs ~11 s, sequential) — and best-of-8 is <b>parallelizable to under 1 s</b>. The one cost that rises is FLOPs (~3.7× a 32B pass, from sampling).</div>
<p class="body" style="font-size:1.02rem;color:var(--muted)">*Latency/energy estimated from measured batch-1 per-tier costs; think hurts open-ended so always-32B-think is both slower and less accurate (~0.43). Exact seconds depend on batching.</p>'''))

# S15 cohesive story + master table
S.append(slide('''<div class="eyebrow"><span class="dot"></span>14 · The story, and how we compare to peers</div>
<h2 class="slide-h sm">One method, benchmarked against prestigious baselines</h2>
<p class="body"><b>The loop:</b> ACC (efficiency) → the gate is saturated (can't out-engineer it) → so pick the best of N → training-free selection fails in medical (majority trap) → a <b>trained verifier</b> breaks it, beating peers and a 5× larger model.</p>
'''+TBL(["method","type","source","pooled acc"],[
["Greedy","—","deploy default","0.41"],
["Self-consistency","training-free","Wang, ICLR'23","0.41"],
["32B single pass (same split)","scale-up","—","0.46"],
["<b>Trained verifier (ours)</b>","trained","this work (GenRM family, ICLR'25)","<b>0.50</b>"],
["Oracle@8","ceiling","—","0.59"]])+'''
<div class="callout win"><b>Bottom line:</b> a small trained verifier is the one method that decisively beats the standard selectors and the bigger model in medical open-ended VQA — a regime where, unlike general LLMs, verification genuinely pays.</div>'''))

# S16 next step
S.append(slide('''<div class="eyebrow"><span class="dot"></span>15 · Next step (reasoned and lined up)</div>
<h2 class="slide-h sm">What we run next, and why</h2>
<ul class="body">
<li><b>The deployable verifier-augmented cascade</b> — put the verifier inside ACC (cheap-leg best-of-N + escalation) and measure the full accuracy/latency/energy frontier vs always-32B. <i>Status:</i> preliminary offline result already in hand (0.517 @ 35% escalation, accuracy-optimal); the next step is the fully-measured latency/energy frontier vs always-32B. <i>Why:</i> turns the two findings into one deployable system with a single baseline.</li>
<li><b>Scale the verifier</b> (more labels, a larger base, cross-generator at scale) — <i>why:</i> the discrimination AUROC (0.924) suggests headroom toward the oracle (0.59) is still open.</li>
<li><b>Verifier-guided training</b> (use it as a reward to improve the generator). <i>Why:</i> moves from selection to improving the model itself.</li>
</ul>
<div class="callout note">This is the honest current frontier: a validated novel method with peer baselines, and a concrete, reasoned next experiment already in motion.</div>'''))

# S17 glossary
S.append(slide('''<div class="eyebrow"><span class="dot"></span>Appendix · Glossary</div>
<h2 class="slide-h sm">Terms used in this report</h2>
<dl class="def">
<dt>Cascade / gate / escalate</dt><dd>cheap model first; the gate decides which questions go to the expensive model.</dd>
<dt>Think / no-think</dt><dd>the big model reasoning step-by-step (slow) vs answering directly (fast).</dd>
<dt>Best-of-N / selection</dt><dd>sample N answers, return the chosen one. Self-consistency = majority vote.</dd>
<dt>Oracle gap</dt><dd>oracle@N − greedy: the headroom if you could always pick a correct sample.</dd>
<dt>Majority trap</dt><dd>the correct answer is a minority vote, so majority voting picks wrong.</dd>
<dt>Recoverability / φ</dt><dd>will the big model fix the cheap model's error; φ = correlation of their right/wrong outcomes (0.37 = they fail together).</dd>
<dt>AUROC / bootstrap CI</dt><dd>separation of right vs wrong (0.5–1.0); a confidence interval from resampling questions 2000×.</dd>
<dt>FLOPs / latency / energy</dt><dd>compute (2N(P+G)) / wall-clock seconds / joules, all batch-1, prefill-included.</dd>
<dt>LLM judge</dt><dd>an automated grader: a strong neutral model checking whether a free-text answer matches the <b>gold</b> answer (a semantic substitute for exact-match). Labels come from the answer key, not the grader's knowledge.</dd>
<dt>Perception vs reasoning question</dt><dd>perception = answerable by recognising what's in the image; reasoning = needs multi-step inference. Thinking helps reasoning but over-thinks perception.</dd>
<dt>Capacity-bound</dt><dd>the limit is the model's inherent ability/knowledge — not fixable by re-asking, re-prompting, or a better gate.</dd>
<dt>Latent knowledge</dt><dd>knowledge a model has but can't surface on demand; the luck floor shows a correct answer is present in the samples but not identifiable training-free.</dd>
<dt>IoU</dt><dd>intersection-over-union of predicted vs gold box; a box is correct if IoU ≥ 0.3.</dd>
<dt>CASP-Stability</dt><dd>a <i>trained</i> cascade gate (a baseline, not ours); appears only in the full paper tables.</dd>
</dl>'''))
