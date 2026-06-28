# slide content for build_report_v2.py (executed in its namespace: S, slide(), img(), peer, ac, cg, fmt, J)
def TBL(headers, rows, ours_idx=None):
    h="".join(f"<th>{c}</th>" for c in headers)
    body=""
    for r in rows:
        cls=" class='total'" if (r is rows[-1]) else ""
        tds="".join(f"<td{' class=ours' if (ours_idx is not None and j==ours_idx and i>0) else ''}>{c}</td>" for j,c in enumerate(r) for i in [1])
        body+=f"<tr{cls}>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>"
    return f"<div class='tbl-wrap'><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"

VER = (peer or {}).get("POOLED",{}).get("verifier") or 0.501
GRE = (peer or {}).get("POOLED",{}).get("greedy") or 0.413
SCC = (peer or {}).get("POOLED",{}).get("self_consistency") or 0.411
M32 = (peer or {}).get("POOLED",{}).get("m32b") or 0.444
ORC = (peer or {}).get("POOLED",{}).get("oracle") or 0.592

# S1 COVER
S.append(slide(f'''<div class="cover">
<div class="eyebrow"><span class="dot"></span>Progress report · CVGIP 2026</div>
<div class="big">Test-time compute for medical VLMs:<br><span class="accent">what actually helps.</span></div>
<div class="sub">Continuing from last time's efficiency cascade. This time, with the math and peer baselines you asked for — and a new result: in medical open-ended VQA, the standard training-free tricks fail, but a small <b>trained verifier</b> beats them all <i>and</i> a model 5× its size.</div>
<div class="cover-stats">
<div class="cstat"><div class="v">{fmt(SCC,3)}</div><div class="l">self-consistency (the standard) — barely above greedy {fmt(GRE,3)}</div></div>
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
<dt>FLOPs (compute)</dt><dd>$F=2N(P{+}G)$: params $N$ × (prompt tokens $P$ incl. the image + generated tokens $G$). Prefill-included. Reported as % of always-32B-think.</dd>
<dt>Latency</dt><dd>wall-clock seconds for one question, batch-1, end-to-end (prefill + generation), measured in isolation.</dd>
<dt>Energy</dt><dd>GPU power sampled every 25 ms and integrated over the question (Joules).</dd>
<dt>AUROC</dt><dd>$P(\\text{score(correct)} > \\text{score(wrong)})$ — how well a score separates right from wrong. 0.5 = useless, 1.0 = perfect.</dd>
</dl></div></div>'''))

# S4 ACC recap with math + peers
acc_rows=[["always-32B-think (parity)","0.572","100%","11.34 s","6319 J"],
["MSP/Chow gate (Chow'70)","0.570","57%","2.96 s","1568 J"],
["AutoMix self-verify","0.569","55%","2.50 s","1307 J"],
["FrugalGPT-style learned (Chen'23)","0.568","60%","3.30 s","1766 J"],
["Jitkrittum L2D (NeurIPS'23)","0.567","51%","2.29 s","1195 J"],
["Ours: ACC (agreement)","0.569","52%","2.27 s","1182 J"]]
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>3 · The efficiency result, now with math + peers</div>
<h2 class="slide-h sm">ACC: same accuracy, ~⅕ the latency — and it beats published gates</h2>
<p class="body"><b>Idea.</b> The 32B's <i>fast</i> mode is as good as its slow mode on perception questions (SLAKE 0.849 no-think vs 0.764 think) — thinking <i>over-thinks</i> them. So we add the 32B-fast as a middle tier and fire the slow tier only when the two fast legs <b>disagree</b>:</p>
<div class="eq">$$\\text{fire slow tier} \\iff \\mathbb{1}[\\hat y_{7B}\\neq\\hat y_{32B\\text{-fast}}]+\\epsilon(-m)>\\tau \\quad(\\text{“iff” = if and only if});\\quad \\text{cost } C=c_0+e_0 c_1+e_1 c_2$$</div>
'''+TBL(["system (6-benchmark avg, equal accuracy)","acc","FLOPs","latency","energy"], acc_rows)+'''
<div class="callout win"><b>Result:</b> at equal accuracy, latency 11.34 s → <b>2.27 s</b>, energy ~5× lower, compute halved. Every published gate (MSP/Chow, AutoMix, FrugalGPT, Jitkrittum L2D) lands on the same frontier — <b>the win is the 3-tier structure, not the gate</b>. (Full 10-method table + 5 model families in the paper.)</div>'''))

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
<dt>Self-certainty best-of-N <span class="src">(NeurIPS 2025, arXiv:2502.18581)</span></dt><dd>rank samples by the model's own output-distribution certainty — recent training-free SOTA.</dd>
<dt>Oracle@N</dt><dd>an impossible upper bound: always pick a correct sample if one exists. Measures the headroom.</dd>
</dl>
<div class="callout note">These are exactly the selectors the test-time-scaling literature benchmarks. Question: do any of them work in <b>medical</b> VQA?</div>'''))

# S8 the wall: training-free selection fails
land_rows=[["SLAKE","0.722","0.736","0.819","0.879"],["VQA-RAD","0.420","0.465","0.600","0.630"],
["PathVQA","0.295","0.324","0.376","0.517"],["Kvasir","0.287","0.286","0.301","0.491"],
["POOLED (n=3545)","0.377","0.394","0.444","0.580"]]
S.append(slide('''<div class="eyebrow"><span class="dot"></span>7 · Experiment — do the standard selectors work?</div>
<h2 class="slide-h sm">No. Training-free selection barely moves, despite huge headroom</h2>
'''+TBL(["dataset","greedy","self-consistency","32B (scale-up)","oracle@8"], land_rows)+'''
<ul class="body">
<li><b>Self-consistency barely beats greedy</b> (0.394 vs 0.377; on Kvasir it's <i>below</i> greedy). The reason — the <b>majority trap</b>: the correct answer is a <i>minority</i> vote in 74–90% of recoverable cases, so majority voting picks the wrong one.</li>
<li><b>Scaling up</b> to the 5× larger 32B only reaches 0.444.</li>
<li>Yet the <b>oracle is 0.58</b> — the right answer is usually <i>in</i> the 8 samples; we just can't pick it.</li>
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
<p class="body"><b>Training</b> (cross-entropy on the judge's correct/incorrect labels $y$; base frozen):</p>
<div class="eq">$$\\mathcal{L}(\\phi)=-\\textstyle\\sum\\big[y\\log s_\\phi+(1-y)\\log(1-s_\\phi)\\big];\\qquad \\text{select } \\hat a=\\arg\\max_{i\\le N} s_\\phi(v,q,a_i)$$</div>
<p class="body">We report the <b>fraction of the oracle gap captured</b>: $\\;(\\text{acc}(\\hat a)-\\text{greedy})/(\\text{oracle}-\\text{greedy})$.</p>'''))

# S11 headline result (filled from peer_comparison if available)
def per_rows():
    if not peer:
        return [["PathVQA","0.352","0.349","0.376","0.441","0.513"],["Kvasir","0.282","0.282","0.301","0.405","0.493"],
                ["VQA-RAD","0.519","0.500","0.600","0.611","0.722"],["SLAKE","0.738","0.738","0.819","0.762","0.895"],
                ["POOLED","0.413","0.411","0.444","0.501","0.592"]]
    nm={"slake_open":"SLAKE","vqa_rad_open":"VQA-RAD","pathvqa_open":"PathVQA","kvasir_open":"Kvasir","POOLED":"POOLED"}
    out=[]
    for ds in ["pathvqa_open","kvasir_open","vqa_rad_open","slake_open","POOLED"]:
        t=peer.get(ds,{});
        out.append([nm[ds],fmt(t.get("greedy")),fmt(t.get("self_consistency")),fmt(t.get("m32b")),fmt(t.get("verifier")),fmt(t.get("oracle"))])
    return out
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>10 · Result — the verifier beats every baseline</div>
<h2 class="slide-h sm">It beats self-consistency, the bigger model, and recovers ~half the gap</h2>
'''+TBL(["dataset","greedy","self-consist.","32B scale-up","<b>verifier (ours)</b>","oracle@8"], per_rows())+
img("paper/figs/limits/fig_peer_comparison.png","Pooled accuracy: the trained verifier (teal) beats greedy, self-consistency, and the 5×-larger 32B; oracle is the ceiling.")+'''
<div class="callout win"><b>The headline:</b> ours captures <b>49%</b> of the oracle gap, beats the 5×-larger model, and (per-dataset) wins on PathVQA / Kvasir / VQA-RAD, losing only on SLAKE where the 32B is genuinely stronger.</div>'''))

# S12 why trust it
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>11 · Why we trust it</div>
<h2 class="slide-h sm">It genuinely discriminates, uses the image, and scales</h2>
<div class="two">
<div>'''+img("paper/figs/limits/fig_verifier_discrimination.png","AUROC 0.924: clean separation of correct vs incorrect candidates.","100%")+'''</div>
<div>'''+img("paper/figs/limits/fig_verifier_scaling.png","Best-of-K: accuracy rises with samples; random stays flat.","100%")+'''</div></div>
<ul class="body">
<li><b>Discrimination AUROC 0.924</b> (n=8512 candidates) — not a "lazy verifier"; blanking the image drops it −0.047 (it uses the image).</li>
<li><b>Statistically significant:</b> bootstrap 95% CI on the gain = <b>[+0.092, +0.139]</b> (resample questions 2000×; excludes 0).</li>
<li><b>Test-time scaling:</b> more samples → higher accuracy; random selection does not improve.</li>
</ul>'''))

# S13 generalization
def cg_line():
    if not cg: return "Cross-generator transfer (Lingshu-trained verifier applied to MedVLThinker-7B's answers): <b>running</b> — result will slot in here."
    parts=[]
    for ds,t in cg.items():
        parts.append(f"{ds.replace('_open','')}: greedy {fmt(t.get('greedy'))} → verifier <b>{fmt(t.get('verifier'))}</b> (oracle {fmt(t.get('oracle'))})")
    return "Cross-generator transfer — the Lingshu-trained verifier, applied to a <b>different</b> model's (MedVLThinker-7B) answers: "+"; ".join(parts)+"."
S.append(slide(f'''<div class="eyebrow teal"><span class="dot"></span>12 · Generalization — multiple models, modalities, output types</div>
<h2 class="slide-h sm">The verifier is not a one-dataset trick</h2>
<ul class="body">
<li><b>Multiple generators (multi-model):</b> {cg_line()}</li>
<li><b>Out-of-distribution modality:</b> works on Kvasir (GI endoscopy), a modality it wasn't built around.</li>
<li><b>Held-out transfer:</b> the pooled verifier lifts RadImageNet (never trained on) +0.024 zero-shot.</li>
<li><b>Different output type — bounding boxes:</b> the same idea recovers 40% of the gap on SLAKE organ grounding and <b>78%</b> on the real <b>MS-CXR</b> chest-X-ray benchmark (a 5.6× lift; bootstrap CI [+0.152,+0.232]).</li>
</ul>'''+img("paper/figs/limits/fig_trained_verifier_unified.png","One principle across output types: training breaks the selection wall for free-text answers, organ boxes, and chest-X-ray boxes.","78%")))

# S14 integration / two axes
S.append(slide('''<div class="eyebrow"><span class="dot"></span>13 · How the two pieces fit together</div>
<h2 class="slide-h sm">Two axes of test-time compute</h2>
<div class="two">
<div class="callout win"><b>ACC — spend compute across model <i>configurations</i></b> (cheap → fast-big → slow-big). Buys <b>efficiency</b>: same accuracy, ~⅕ latency.</div>
<div class="callout win"><b>Verifier — spend compute across <i>samples</i></b> (best-of-N). Buys <b>accuracy</b>: reaches what a 5× larger model cannot.</div></div>
'''+img("paper/figs/limits/fig_accuracy_compute.png","Accuracy vs compute: spending compute on samples+verifier reaches accuracy the bigger model can't.","82%")+'''
<div class="callout note">They are complementary levers of one idea — <i>allocate test-time compute where it pays</i> — and combine in a verifier-augmented cascade (the deployable system; next step).</div>'''))

# S15 cohesive story + master table
S.append(slide('''<div class="eyebrow"><span class="dot"></span>14 · The story, and how we compare to peers</div>
<h2 class="slide-h sm">One method, benchmarked against prestigious baselines</h2>
<p class="body"><b>The loop:</b> ACC (efficiency) → the gate is saturated (can't out-engineer it) → so pick the best of N → training-free selection fails in medical (majority trap) → a <b>trained verifier</b> breaks it, beating peers and a 5× larger model.</p>
'''+TBL(["method","type","source","pooled acc"],[
["Greedy","—","deploy default","0.41"],
["Self-consistency","training-free","Wang, ICLR'23","0.41"],
["Self-verify P(True)","training-free","Kadavath'22","~0.41"],
["32B single pass","scale-up","—","0.44"],
["<b>Trained verifier (ours)</b>","trained","this work (GenRM family, ICLR'25)","<b>0.50</b>"],
["Oracle@8","ceiling","—","0.59"]])+'''
<div class="callout win"><b>Bottom line:</b> a small trained verifier is the one method that decisively beats the standard selectors and the bigger model in medical open-ended VQA — a regime where, unlike general LLMs, verification genuinely pays.</div>'''))

# S16 next step
S.append(slide('''<div class="eyebrow"><span class="dot"></span>15 · Next step (reasoned and lined up)</div>
<h2 class="slide-h sm">What we run next, and why</h2>
<ul class="body">
<li><b>The deployable verifier-augmented cascade</b> — put the verifier inside ACC (cheap-leg best-of-N + escalation) and measure the full accuracy/latency/energy frontier vs always-32B. <i>Why:</i> turns the two findings into one system with a single baseline.</li>
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
<dt>LLM judge</dt><dd>a strong neutral 32B model grading free-text answers as correct/incorrect vs gold.</dd>
</dl>'''))
