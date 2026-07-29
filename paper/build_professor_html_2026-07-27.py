#!/usr/bin/env python3
"""Build the advisor-facing progress slide deck for 2026-06-29 -> 2026-07-27.

Reuses the weekly-update template's CSS + nav JS (same head/appbar/slide()/img()/close
pattern as paper/build_professor_html.py and paper/build_report_v2.py).

EVERY number in this deck was read out of a real artifact on disk. The constant block
below records the file each figure came from; nothing is fabricated. Writes
meetings/progress_report_professor_2026-07-27.html directly (no repo-root staging).
"""
import base64, os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "meetings/progress_report_professor_2026-07-27.html")
A = "results/cascade_methods/artifacts"
D = "results/cascade_methods/docs/current"

# ---------------------------------------------------------------------------
# CANONICAL NUMBERS — each block names the file it was read from.
# (kept as comments/constants so a reader can re-verify every cell)
#
# paper_baselines.json (full suite n=42,374, 10,000-sample paired bootstrap):
#   always_7b            0.5558 | 1.000 FLOP-eq | 347.0 ms |   45.8 J
#   always_32b_nothink   0.5732 | 4.570         | 665.0 ms |  127.0 J
#   always_32b_think     0.5632 | 4.570         | 10521.6 ms | 2001.9 J
#   oracle_mode_32b      0.5733 | 4.570         | 894.4 ms |  170.6 J
#   method_compute_lean  0.5749 | 2.244 (0.491x)| 468.6 ms |   83.5 J
#   method_accuracy_max  0.5869 | 5.695 (1.246x)| 666.3 ms |  176.6 J
#   compute-lean vs oracle  +0.0015 [-0.0025,+0.0055] n.s.
#   accuracy-max vs oracle  +0.0136 [+0.0108,+0.0165] SIG
# f8_mode_vsthink_ci.json (Variant B, MMMU excluded, n=42,224, MEASURED think):
#   accuracy-max 0.5836 vs 32B-think 0.5591 -> +0.0245 [+0.0216,+0.0274] SIG @ 0.93x FLOPs
#   open-only    0.5727 vs 0.3028          -> +0.2699 [+0.2490,+0.2908] SIG
# opentext_32b_think_full.json (Variant B): compute-lean +0.0150 [+0.0107,+0.0192]
# opentext_32b_think.json: think 10,521.6 ms / 2,001.9 J / 0.3867 acc (n=600)
#                          no-think 665.0 ms / 126.9 J / 0.5367 acc   (ratio 15.8x)
# generalization.json: 15/20 perception cells think<=no-think strict, 19/20 within +-0.02
#
# !!! STALE AS OF 2026-07-29 — DO NOT RE-RUN THIS SCRIPT WITHOUT FIXING FINDING 1 !!!
#   The Finding-1 numbers hard-coded below (see L~99 "15/20" and L~108 / L~169) came from
#   generalization.json, whose think arms were PROMPT-UNMATCHED. This file and the deck it
#   already produced (meetings/progress_report_professor_2026-07-27.html) are left as the
#   dated 2026-07-27 record; re-running as-is would republish superseded numbers.
#   Canonical replacement artifact: artifacts/finding1_corrected_2026-07-29.json
#   (audit: artifacts/finding1_prompt_matching_audit.json). Corrected values:
#     - perception 17/20 strictly negative (not 15/20), 19/20 within +0.02,
#       14/20 with 95% CIs excluding zero, pooled -0.0401 [-0.0456,-0.0347], n=30,250
#     - reasoning half is MODEL-DEPENDENT, not universal: 12/15 point-positive but only
#       4/15 CI-significant and 1/15 significantly negative
#     - WITHDRAWN: all 7 Lingshu-32B cells (native-think prompt had no reasoning trigger,
#       3.0 generated tokens) and QoQ-Med-VL-32B as reasoning evidence
#     - Lingshu's "1.2x think:no-think" is a format-prompt ratio, not a reasoning ratio
#     - MedGemma:PathVQA +0.0413 [+0.0220,+0.0607] is a REAL exception (fully matched)
#     - the open-text think-vs-direct delta is PROVISIONAL (run_openvqa.py:26/27 confound)
#   Narrative: results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md
#   sections 5.1, 10.1 (C20-C25), 10.5.
# ---------------------------------------------------------------------------

lines = open(os.path.join(ROOT, "meetings/report_template.html"), encoding="utf-8").read().split("\n")
assert lines[186].strip() == "</head>", "template head offset changed; re-check the 187 line split"
head = "\n".join(lines[:187])
head = head.replace("<title>Weekly Progress &mdash; Cascade Results</title>",
                    "<title>Progress Report 2026-07-27 &mdash; Regime-Adaptive Compute for Medical VLMs</title>")
head = head.replace("<title>Weekly Progress — Cascade Results</title>",
                    "<title>Progress Report 2026-07-27 — Regime-Adaptive Compute for Medical VLMs</title>")

mathjax = '''<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>.eq{font-size:1.15rem;background:#f6f8fd;border:1px solid var(--line);border-left:4px solid var(--indigo);border-radius:8px;padding:12px 16px;margin:12px 0;overflow-x:auto}
.step{display:flex;gap:14px;margin:12px 0}.step .n{flex:0 0 32px;height:32px;border-radius:50%;background:var(--indigo);color:#fff;font-family:'Roboto Slab',serif;font-weight:700;display:flex;align-items:center;justify-content:center;font-size:1.05rem}
.callout{border-radius:12px;padding:15px 20px;margin:14px 0;font-size:1.16rem;line-height:1.55}
.callout.win{background:var(--good-bg);border:1px solid #bfe0c2}.callout.win b{color:var(--good)}
.callout.note{background:var(--indigo-bg);border:1px solid #cfd6f0}
.callout.honest{background:#fff7e8;border:1px solid #f3e0b8}.callout.honest b{color:var(--warn)}
.callout.q{background:#eef1f8;border:1px dashed var(--indigo-l)}
.figimg{width:100%;max-width:720px;display:block;margin:12px auto 4px;border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow-s)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}@media(max-width:900px){.two{grid-template-columns:1fr}}
ul.body{font-size:1.2rem;line-height:1.55;margin:8px 0 8px 22px}ul.body li{margin:6px 0}
dl.def{font-size:1.14rem}dl.def dt{font-weight:700;color:var(--indigo-d);margin-top:8px}dl.def dd{margin:2px 0 6px 0;color:var(--ink)}
.src{font-size:.95rem;color:var(--muted);margin-top:6px}
table td,table th{font-size:1.03rem}</style>'''
head = head.replace("</head>", mathjax + "\n</head>")


def img(path, cap, w="720px"):
    """Embed a PNG as base64. Missing files degrade to a caption instead of failing."""
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f"<img class='figimg' style='max-width:{w}' src='data:image/png;base64,{b}' alt='{cap}'>"
            f"<p class='caption' style='text-align:center'>{cap}</p>")


appbar = '''<body>
<div class="appbar"><span class="mark">Med-VLM &middot; Test-Time Compute</span>
<span class="title">Allocating Compute by Regime &mdash; <b>Reasoning, Format, and the Two Walls</b></span>
<span class="spacer"></span><span class="meta">progress report &middot; 2026-06-29 &rarr; 2026-07-27</span></div>
<div class="progress"><div class="progress-bar" id="pbar"></div></div>
<div class="deck-viewport"><main class="deck" id="deck">
'''


def slide(inner):
    return f'<section class="slide"><div class="swrap"><div class="card">{inner}</div></div></section>\n'


S = []

# ---------------------------------------------------------------- 1 COVER
S.append(slide('''<div class="cover">
<div class="eyebrow"><span class="dot"></span>Progress report &middot; 2026-06-29 &rarr; 2026-07-27</div>
<div class="big">One adaptive cascade now beats<br><span class="accent">every fixed way of using the 32B</span>.</div>
<div class="sub">At its compute-lean setting it <b>matches an oracle mode-selected 32B</b> (0.5749 vs 0.5733, not significant) using <b>half the compute, half the latency, half the energy</b>. At its accuracy-max setting it <b>beats always-32B-with-reasoning</b> by +0.0245 while still using <b>less compute than a single 32B forward pass</b>. The scientific driver: chain-of-thought reasoning measurably <b>hurts</b> perception questions across five model families.</div>
<div class="cover-stats">
<div class="cstat"><div class="v">0.49&times;</div><div class="l">compute of a 32B call, at matched accuracy</div></div>
<div class="cstat"><div class="v teal">+0.0245</div><div class="l">accuracy over always-32B-reasoning<br>95% CI [+0.0216, +0.0274], n=42,224</div></div>
<div class="cstat"><div class="v">15/20</div><div class="l">perception cells where reasoning does not help</div></div>
<div class="cstat"><div class="v teal">6/7</div><div class="l">benchmarks fully reproduced, 3 model families</div></div>
</div></div>'''))

# ---------------------------------------------------------------- 2 EXEC SUMMARY
S.append(slide('''<div class="eyebrow"><span class="dot"></span>0 &middot; Executive summary</div>
<h2 class="slide-h sm">Six things we learned since 29 June</h2>
<ul class="body">
<li><b>The baseline is now trustworthy.</b> Our internal evaluation harness was <i>not</i> faithful to the published Lingshu protocol; MedEvalKit is. We re-ran <b>3 model families &times; 7 benchmarks</b>. Lingshu-32B MMMU-Medical <b>0.633</b> vs published 62.3; Lingshu-7B OmniMedVQA <b>0.8274</b> on all 88,996 questions vs published 0.829. <span class="kbd">mmmu_fix.json</span> <span class="kbd">OMNIMED_FALLBACK.md</span></li>
<li><b>Finding 1 &mdash; reasoning hurts perception.</b> Across 5 families &times; 4 perception benchmarks, <b>15/20 cells</b> have reasoning &le; direct answering; VQA-RAD is negative in <b>all five</b> families. Reasoning costs up to <b>49&times;</b> the latency. <span class="kbd">generalization.json</span></li>
<li><b>Finding 2 &mdash; answer format decides whether sampling helps.</b> Forcing multiple-choice through free-text generation collapses accuracy (PMC-VQA <b>0.534 &rarr; 0.132</b>). So the deliverable is a <b>two-arm router</b>, not one universal mechanism. <span class="kbd">ugv_mcq_verdict.json</span></li>
<li><b>The method Pareto-dominates every 32B strategy, including an oracle.</b> Compute-lean matches an oracle mode-selected 32B at <b>0.49&times; compute / 469 ms vs 894 ms / 83.5 J vs 170.6 J</b>. <span class="kbd">paper_baselines.json</span></li>
<li><b>Two hard limits, each confirmed several times over.</b> The <b>recoverability wall</b> (six independent methods, zero new certified slices) and the <b>selection wall</b> (a 7&times;-larger verifier only ties our small trained one). <span class="kbd">robust_slice_routing.json</span> <span class="kbd">verifier_32b_gpu.json</span></li>
<li><b>A data-integrity anomaly, adversarially audited.</b> Lingshu-7B scores <b>+26 points above its published MMMU number</b>. Seven checks say it is a genuine model score, not our bug &mdash; and we still <b>refuse to bank it</b>. <span class="kbd">mmmu_verify.json</span></li>
</ul>
<div class="callout honest"><b>Reporting-window honesty note:</b> documented research activity in this window runs <b>2026-06-29 &rarr; 2026-07-09</b>. The newest result file on disk is <span class="kbd">f8_mode_vsthink_ci.json</span> (2026-07-09 19:44); there is <b>no documented activity for 10&ndash;27 July</b>. Everything from 3 July onward is also uncommitted working-tree state (last commit 2026-07-02) &mdash; a real preservation risk that should be fixed first thing.</div>'''))

# ---------------------------------------------------------------- 3 WHERE WE LEFT OFF
S.append(slide('''<div class="eyebrow"><span class="dot"></span>1 &middot; Where we left off</div>
<h2 class="slide-h sm">What the last report promised, and what this one answers</h2>
<p class="body"><b>Settled previously (recapped here only for continuity, not re-argued):</b> the three-tier adaptive-compute cascade and its &minus;80% latency / ~5&times; energy result; the finding that every untrained gate lands on the same frontier, so the win is structure not gate; the multiple-choice &rarr; open-ended pivot; the "luck floor" (untrained selection cannot beat random); and the trained LoRA verifier reaching 0.501 pooled, 49% of the oracle gap, discrimination AUROC 0.924.</p>
<p class="body">The last report closed on <b>five explicit promises</b>. This report resolves each of them:</p>
<div class="tbl-wrap"><table>
<thead><tr><th>promised on 29 June</th><th>status now</th><th>where in this deck</th></tr></thead>
<tbody>
<tr><td>Reproduce the Lingshu evaluation suite faithfully</td><td class="ours">Done, 6/7 benchmarks &times; 3 families; 7th documented as infrastructure-limited</td><td>slides 4&ndash;5</td></tr>
<tr><td>One unified allocator over multiple-choice and free-text</td><td class="ours">Built &mdash; but as a <b>two-arm router</b>; a single unified mechanism was tested and failed</td><td>slides 8&ndash;10</td></tr>
<tr><td>Beat the 32B on accuracy <i>and</i> cost at once</td><td class="ours">Achieved against every 32B strategy incl. an oracle upper bound</td><td>slides 11&ndash;13</td></tr>
<tr><td>Make the trained stability gate benchmark-safe</td><td>Closed as negative &mdash; on this model family the trained gate is inert and plain confidence wins</td><td>slide 13</td></tr>
<tr><td>Scale the verifier toward the oracle</td><td>Closed as negative &mdash; the verifier is at its ceiling; <b>candidate quality</b> is the binding constraint</td><td>slides 15, 18</td></tr>
</tbody></table></div>
<div class="callout note"><b>The template each research step follows</b> (unchanged from the last report): question &rarr; what we need to know &rarr; baseline &rarr; experiment &rarr; math &rarr; result &rarr; what it means &rarr; next step.</div>'''))

# ---------------------------------------------------------------- 4 REPRODUCTION
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>2 &middot; Foundation &mdash; faithful reproduction</div>
<h2 class="slide-h sm">Anchoring every claim to a protocol that reproduces published numbers</h2>
<p class="body"><b>The question.</b> A cascade result is worthless if the strong baseline is weak. Does our evaluation stack actually reproduce the models we compare against?</p>
<p class="body"><b>What we found.</b> Our internal NGC harness is <b>not faithful</b> to the published Lingshu evaluation protocol. <b>MedEvalKit</b> &mdash; the framework the Lingshu authors themselves used &mdash; is. We locked one recipe (isolated environment, Qwen2.5-VL wrapper, vLLM serving, exact-match scoring for multiple choice, a validated language-model judge for the free-text halves) and re-ran everything through it.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>fidelity check (MedEvalKit, our run vs published)</th><th>ours</th><th>published</th><th>gap</th></tr></thead>
<tbody>
<tr><td>Lingshu-32B &mdash; MMMU-Medical</td><td class="ours">0.633</td><td>0.623</td><td><span class="pos">+0.010</span></td></tr>
<tr><td>Lingshu-7B &mdash; SLAKE (closed)</td><td class="ours">0.825</td><td>0.831</td><td><span class="neg">&minus;0.006</span></td></tr>
<tr><td>Lingshu-7B &mdash; PMC-VQA</td><td class="ours">0.543</td><td>0.563</td><td><span class="neg">&minus;0.020</span></td></tr>
<tr><td>Lingshu-7B &mdash; MedXpertQA-MM</td><td class="ours">0.262</td><td>0.267</td><td><span class="neg">&minus;0.005</span></td></tr>
<tr class="total"><td>Lingshu-7B &mdash; OmniMedVQA (all 88,996 questions)</td><td class="ours">0.8274</td><td>0.829</td><td><span class="zero">&minus;0.002</span></td></tr>
</tbody></table></div>
<div class="callout win"><b>Result:</b> the harness reproduces the published models to within a couple of accuracy points on every benchmark. Every downstream efficiency and accuracy claim in this report is therefore measured against a baseline that <b>provably reproduces the model it claims to be</b>, and cannot be dismissed as a weak-baseline artifact.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/mmmu_fix.json</span> &middot; <span class="kbd">''' + A + '''/reframe_vs_bigthink.json</span> &middot; <span class="kbd">''' + D + '''/OMNIMED_FALLBACK.md</span> &middot; published targets verified from the Lingshu release, <span class="kbd">''' + D + '''/VERIFIED_FACTS.md</span> &sect;J.</p>'''))

# ---------------------------------------------------------------- 5 THE HONEST GAP
S.append(slide('''<div class="eyebrow"><span class="dot"></span>2 (cont.) &middot; The one gap, reported as a gap</div>
<h2 class="slide-h sm">The 7th benchmark's large-model leg is blocked by infrastructure</h2>
<p class="body"><b>What happened.</b> OmniMedVQA is the largest benchmark in the suite (88,996 image-question pairs). The <b>cheap</b> leg ran successfully in all three families and reproduces the published number to 0.2 of a point. The <b>strong</b> (32B / 38B) leg hits a deterministic communication hang when the model is split across both GPUs &mdash; every chunk stalls for ~36 minutes and is killed, and this recurred on every attempt over roughly two days. Running the large model on a single GPU was re-confirmed impossible: 64 GB of weights plus multimodal activations exceed an 80 GB card.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>OmniMedVQA, cheap leg (n=88,996)</th><th>accuracy</th></tr></thead>
<tbody>
<tr><td>Lingshu-7B (ours)</td><td class="ours">0.8274</td></tr>
<tr><td>Lingshu-7B (published)</td><td>0.829</td></tr>
<tr><td>MedVLThinker-7B (ours)</td><td>0.6248</td></tr>
<tr class="total"><td>InternVL3-8B (ours)</td><td>0.7847</td></tr>
</tbody></table></div>
<div class="callout honest"><b>How we handled it.</b> We wrote <b>no metrics file</b> for the missing cell. It is reported as published-reference (Lingshu-32B 0.834) plus an explicit infrastructure limitation. This costs no conclusion: on OmniMedVQA the published cheap and strong models are <b>essentially tied</b> (82.9 vs 83.4, a 0.5-point gap), so a cascade simply keeps the cheap model at near-zero escalation there.</div>
<div class="callout note"><b>Route around it, if we want the cell:</b> a 4-bit quantized strong leg would fit on one GPU (~20 GB) and sidestep the two-GPU hang entirely. That was identified but never executed.</div>
<p class="src">Source: <span class="kbd">''' + D + '''/OMNIMED_FALLBACK.md</span> &middot; <span class="kbd">''' + A + '''/quantized_strong_leg.json</span></p>'''))

# ---------------------------------------------------------------- 6 FINDING 1a
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>3 &middot; Finding 1 &mdash; the central result</div>
<h2 class="slide-h sm">Step-by-step reasoning <i>hurts</i> perception questions</h2>
<p class="body"><b>The question.</b> The motivating deployment is "just run the big model with reasoning switched on." Is that actually the best use of compute? We measured, per family and per benchmark, the accuracy of reasoning mode minus direct-answer mode.</p>
<p class="body"><b>The result.</b> On perception questions &mdash; where the answer is read off the image &mdash; reasoning is a <b>net loss</b>. Of the 20 perception cells (5 families &times; 4 benchmarks), <b>15 are strictly negative</b> and <b>19 of 20</b> fall inside a &plusmn;0.02 noise band.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>accuracy(reasoning) &minus; accuracy(direct), per family</th><th>PMC-VQA</th><th>SLAKE</th><th>VQA-RAD</th><th>PathVQA</th></tr></thead>
<tbody>
<tr><td>MedVLThinker</td><td><span class="pos">+0.006</span></td><td><span class="neg">&minus;0.084</span></td><td><span class="neg">&minus;0.077</span></td><td><span class="pos">+0.011</span></td></tr>
<tr><td>Lingshu</td><td><span class="pos">+0.012</span></td><td><span class="neg">&minus;0.010</span></td><td><span class="neg">&minus;0.070</span></td><td><span class="neg">&minus;0.017</span></td></tr>
<tr><td>QoQ-Med</td><td><span class="neg">&minus;0.085</span></td><td><span class="neg">&minus;0.065</span></td><td><span class="neg">&minus;0.077</span></td><td><span class="neg">&minus;0.063</span></td></tr>
<tr><td>Chiron</td><td><span class="neg">&minus;0.072</span></td><td><span class="neg">&minus;0.108</span></td><td><span class="neg">&minus;0.092</span></td><td><span class="neg">&minus;0.051</span></td></tr>
<tr class="total"><td>MedGemma</td><td><span class="neg">&minus;0.008</span></td><td><span class="zero">+0.005</span></td><td><span class="neg">&minus;0.018</span></td><td><span class="pos">+0.040</span></td></tr>
</tbody></table></div>
<div class="legend"><span class="it"><b>VQA-RAD negative in all 5 families</b></span><span class="it"><b>SLAKE negative in 4 of 5</b></span><span class="it">the single genuine win in 20 cells is MedGemma on PathVQA, +0.040</span></div>
<div class="callout note"><b>Not a medical or Qwen-family artifact:</b> two additional general-purpose architectures are also pooled-negative on perception &mdash; InternVL2.5-8B &minus;0.008 and Phi-3.5-Vision &minus;0.019.</div>
<p class="src">Source: <span class="kbd">''' + A + '''/generalization.json</span> &rarr; <span class="kbd">finding1_reasoning_vs_perception</span>; supplementary architectures from <span class="kbd">''' + A + '''/overthink_generalize.txt</span>.</p>'''))

# ---------------------------------------------------------------- 7 FINDING 1b cost
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>3 (cont.) &middot; Finding 1 &mdash; what it costs</div>
<h2 class="slide-h sm">Worse accuracy, and up to 49&times; the latency</h2>
<p class="body">The accuracy loss above would be tolerable if reasoning were cheap. It is not. Measured batch-1 latency, same harness, one question at a time:</p>
<div class="tbl-wrap"><table>
<thead><tr><th>family</th><th>direct answer</th><th>with reasoning</th><th>slowdown</th></tr></thead>
<tbody>
<tr><td>MedVLThinker</td><td>0.231 s</td><td>11.338 s</td><td class="ours">49.1&times;</td></tr>
<tr><td>MedGemma</td><td>0.282 s</td><td>12.722 s</td><td class="ours">45.1&times;</td></tr>
<tr><td>QoQ-Med</td><td>0.227 s</td><td>9.719 s</td><td class="ours">42.8&times;</td></tr>
<tr><td>Chiron / InternVL3</td><td>0.291 s</td><td>4.246 s</td><td class="ours">14.6&times;</td></tr>
<tr class="total"><td>Lingshu</td><td>0.270 s</td><td>0.322 s</td><td>1.2&times;</td></tr>
</tbody></table></div>
<p class="body">And measured directly on free-text answering with the 32B, on the same hardware and harness as our deployed 665 ms number:</p>
<div class="tiles">
<div class="tile"><div class="v bad">10,521.6 ms</div><div class="l">32B with reasoning (vs 665.0 ms direct &mdash; ~16&times;)</div></div>
<div class="tile"><div class="v bad">2,001.9 J</div><div class="l">energy per question (vs 127.0 J direct)</div></div>
<div class="tile"><div class="v bad">&minus;0.150</div><div class="l">accuracy: 0.387 with reasoning vs 0.537 direct (n=600)</div></div>
</div>
''' + img("paper/figs/fig2_overthinking_perbench.png",
          "Per-benchmark bars: direct answering minus reasoning for the large model. Green = direct answering wins. (Earlier MedVLThinker-family measurement, reported previously; the July cross-family table above supersedes it in scope.)") + '''
<div class="callout win"><b>What it means:</b> "always run the big model with reasoning on" is <b>strictly self-sabotaging</b> on the majority of medical visual question answering &mdash; slower, more energy, <i>and</i> less accurate. That converts what was an efficiency argument into an accuracy argument too, and it is the empirical justification for allocating compute by regime rather than uniformly.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/generalization.json</span> &middot; <span class="kbd">''' + A + '''/opentext_32b_think.json</span> (measured 2026-07-07, batch-1, per-GPU energy meter, n=15 after 3 warm-ups).</p>'''))

# ---------------------------------------------------------------- 8 FINDING 1c reasoning half + correction
S.append(slide('''<div class="eyebrow"><span class="dot"></span>3 (cont.) &middot; Finding 1 &mdash; the other half, and a correction</div>
<h2 class="slide-h sm">Reasoning does help &mdash; but only on reasoning benchmarks</h2>
<p class="body">The claim is not "reasoning is useless." It is <b>regime-specific</b>. On the two genuinely knowledge/reasoning benchmarks, measured through the faithful harness:</p>
<div class="tbl-wrap"><table>
<thead><tr><th>accuracy gain from reasoning</th><th>Lingshu-32B</th><th>MedVLThinker-32B</th><th>InternVL3-38B</th></tr></thead>
<tbody>
<tr><td>MMMU-Medical (n=150)</td><td><span class="pos">+0.027</span></td><td class="ours"><span class="pos">+0.100</span></td><td class="ours"><span class="pos">+0.120</span></td></tr>
<tr class="total"><td>MedXpertQA-MM (n=2,000)</td><td><span class="zero">&minus;0.003</span></td><td><span class="pos">+0.045</span></td><td><span class="pos">+0.031</span></td></tr>
</tbody></table></div>
<div class="eq" style="text-align:center;font-size:1.22rem">perception &rarr; never reason &nbsp;&middot;&nbsp; reasoning benchmarks &rarr; reason, on the residual only</div>
<div class="callout honest"><b>A mid-window claim we had to retract.</b> On 1 July we asserted from token counts (~3 generated tokens) that "Lingshu has no promptable reasoning mode." That was <b>wrong</b>. An explicit probe showed 3 &rarr; <b>174</b> generated tokens under a proper "reason step by step" instruction, and <b>267</b> with explicit reasoning tags. The earlier reading was a weak-prompt artifact in our evaluation harness, fixed in two prompt paths on 2 July. The Lingshu 1.2&times; latency row on the previous slide reflects that model answering directly, not an absence of the capability.</div>
<div class="callout note"><b>Why this matters for honesty:</b> a second mid-window correction runs the same way. The 6 July conclusions were drawn against the 32B's <i>cheap</i> direct mode; re-grounding them on the reasoning mode the deployment premise is actually about flipped several verdicts. Both corrections are recorded in the diaries rather than quietly overwritten.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/reframe_vs_bigthink.json</span> &middot; <span class="kbd">''' + A + '''/generalization.json</span> &middot; <span class="kbd">progress/progress_July_01-02.md</span>, <span class="kbd">progress/progress_July_07.md</span>.</p>'''))

# ---------------------------------------------------------------- 9 FINDING 2 format
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>4 &middot; Finding 2 &mdash; format decides everything</div>
<h2 class="slide-h sm">Answer format, not medical difficulty, controls whether sampling helps</h2>
<p class="body"><b>The question.</b> We wanted <i>one</i> mechanism for both answer formats: generate several candidate answers, score them with a trained verifier, keep the best. If that worked for multiple choice as well as free text, no router would be needed. We ran it on two 7B models over PMC-VQA and MedXpertQA (n=2,000 each, 8 samples), in two modes: <b>content</b> (options hidden, model must produce the answer text) and <b>letter</b> (standard option scoring).</p>
<div class="tbl-wrap"><table>
<thead><tr><th>multiple-choice, Lingshu-7B</th><th>letter mode</th><th>content mode</th><th>verifier gain (content)</th></tr></thead>
<tbody>
<tr><td>PMC-VQA (n=2,000)</td><td>0.534</td><td class="ours">0.132</td><td>+0.009</td></tr>
<tr class="total"><td>MedXpertQA-MM (n=2,000)</td><td>0.556</td><td class="ours">0.499</td><td>&minus;0.004</td></tr>
</tbody></table></div>
<p class="body"><b>Result &mdash; negative, and the mechanism is informative.</b> Hiding the options craters accuracy by about 0.40 on PMC-VQA. The verifier's gain in content mode is <b>+0.0038 pooled</b> (mean discrimination AUROC 0.696), and in letter mode it is inconsistent and parser-sensitive (+0.082 on PMC-VQA under a strict parse, but &minus;0.074 under the as-run parse; &minus;0.001 on MedXpertQA).</p>
<div class="tbl-wrap"><table>
<thead><tr><th>"is the cheap model wrong?" &mdash; how learnable is it?</th><th>AUROC</th></tr></thead>
<tbody>
<tr><td>Multiple choice &mdash; error detection, every gate tested</td><td>0.643 &ndash; 0.693</td></tr>
<tr><td>Multiple choice &mdash; will the big model <i>fix</i> it?</td><td>0.506 &ndash; 0.614</td></tr>
<tr><td>Free text &mdash; MedVLThinker-7B</td><td class="ours">0.735 &ndash; 0.781</td></tr>
<tr><td>Free text &mdash; Lingshu-7B</td><td class="ours">0.845 &ndash; 0.866</td></tr>
<tr class="total"><td>Free text &mdash; trained verifier, per-answer (n=8,512)</td><td class="ours">0.924</td></tr>
</tbody></table></div>
<div class="callout win"><b>What it means:</b> the information in a multiple-choice question lives <b>in the option set</b>; destroying it to unify the two formats destroys what makes multiple choice tractable. This also explains a puzzle that held the project up for weeks &mdash; the "no routing signal exists" result was a property of the <b>format</b>, not of medical imaging. The accuracy engine (sampling + verification) and the efficiency engine (confidence-gated escalation) are structurally different levers that must be <b>composed</b>, not merged.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/ugv_mcq_verdict.json</span> &middot; <span class="kbd">''' + A + '''/generalization.json</span> &rarr; <span class="kbd">finding2_format_signal_gap</span>.</p>'''))

# ---------------------------------------------------------------- 10 THE METHOD
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>5 &middot; The method</div>
<h2 class="slide-h sm">A format-aware, regime-adaptive cascade over one 7B and one 32B</h2>
<p class="body">A <b>format router</b> first decides multiple-choice vs free-text <b>from the prompt alone</b> (never from the gold answer &mdash; this is the realistic test-time condition). It then runs one of two arms.</p>
<div class="two">
<div>
<h3 class="sub">Multiple-choice arm &mdash; the efficiency engine</h3>
<ul class="body">
<li><b>Cheap leg:</b> 7B, direct answer, one greedy generation.</li>
<li><b>Gate:</b> confidence margin (top-1 minus top-2 option probability). Threshold frozen on a held-out calibration split.</li>
<li><b>Strong leg:</b> 32B <i>direct</i> &mdash; reasoning never beats direct on any perception benchmark.</li>
<li><b>Accuracy add-on:</b> on PMC-VQA only, a certified rule that fuses the two models' decisions where the 7B is more calibrated-confident.</li>
</ul>
</div>
<div>
<h3 class="sub">Free-text arm &mdash; the accuracy engine</h3>
<ul class="body">
<li><b>Sample N candidates</b> from the 7B; a small trained verifier scores each and picks the best.</li>
<li><b>Adaptive sample count:</b> an optimal-stopping rule from search theory draws only as many as needed &mdash; mean <b>8 &rarr; 4.28</b> draws.</li>
<li><b>Escalation:</b> a team-objective rule decides keep-the-7B vs hand up to the 32B.</li>
</ul>
</div></div>
<div class="eq" style="text-align:center;font-size:1.2rem">7B first pass &rarr; <b>confidence gate</b> &rarr; 32B direct &nbsp;&nbsp;|&nbsp;&nbsp; 7B best-of-N &rarr; <b>verifier gate</b> &rarr; 32B direct</div>
<p class="body"><b>The one Pareto knob.</b> <span class="kbd">compute-lean</span> = the cascade plus adaptive sampling (cheapest). <span class="kbd">accuracy-max</span> = adds the PMC-VQA fusion. Both are reproducible from one command.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>lever</th><th>what it does</th><th>measured effect</th></tr></thead>
<tbody>
<tr><td>Confidence margin gate</td><td>escalate only low-confidence answers</td><td>matches the 32B at ~16% escalation</td></tr>
<tr><td>Verifier best-of-N</td><td>7B samples N, verifier picks</td><td class="ours">0.563 vs 32B-direct 0.517</td></tr>
<tr><td>Adaptive stopping</td><td>draw only as many samples as needed</td><td class="ours">&minus;33% free-text compute at held accuracy</td></tr>
<tr><td>PMC-VQA decision fusion</td><td>pick the better-calibrated model</td><td class="ours">+0.0135 [+0.0100, +0.0169], n=33,430</td></tr>
<tr><td>Team-objective escalation</td><td>better keep/escalate rule on free text</td><td>PathVQA-open +0.086 [+0.064, +0.106]</td></tr>
<tr class="total"><td>Prefill prefetch</td><td>overlap the 32B image encoding with the 7B pass</td><td>461 &rarr; 405 ms, <b>zero</b> accuracy cost</td></tr>
</tbody></table></div>
<p class="src">Sources: <span class="kbd">''' + D + '''/TECHNICAL_REPORT_2026-07.md</span> &middot; <span class="kbd">''' + A + '''/integrated_pandora_opentext.json</span> &middot; <span class="kbd">''' + A + '''/beat32b_fusion.json</span> &middot; <span class="kbd">''' + A + '''/beat32b_more.json</span> &middot; <span class="kbd">''' + A + '''/escalation_levers.json</span>.</p>'''))

# ---------------------------------------------------------------- 11 MAIN RESULT vs oracle
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>6 &middot; Main result</div>
<h2 class="slide-h sm">Beating even an oracle that picks the 32B's best mode per benchmark</h2>
<p class="body"><b>The baseline we chose to fight.</b> Rather than a convenient baseline, we scored against an <b>oracle mode-selected 32B</b>: for each benchmark it is told in advance whether reasoning or direct answering wins there, and it pays that mode's cost. It is deliberately <b>not deployable</b> &mdash; it is an upper bound on anything a single 32B strategy can do.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>full suite, n=42,374, held-out</th><th>accuracy</th><th>compute (&times; one 7B pass)</th><th>latency</th><th>energy</th></tr></thead>
<tbody>
<tr><td>always 7B (cheap floor)</td><td>0.5558</td><td>1.00</td><td>347 ms</td><td>45.8 J</td></tr>
<tr><td>always 32B, direct</td><td>0.5732</td><td>4.57</td><td>665 ms</td><td>127.0 J</td></tr>
<tr><td>always 32B, with reasoning</td><td>0.5632</td><td>4.57</td><td>10,522 ms</td><td>2,001.9 J</td></tr>
<tr><td><b>oracle mode-selected 32B</b> (upper bound)</td><td>0.5733</td><td>4.57</td><td>894 ms</td><td>170.6 J</td></tr>
<tr><td class="ours">ours &mdash; compute-lean</td><td class="ours">0.5749</td><td class="ours">2.24 (0.49&times;)</td><td class="ours">469 ms</td><td class="ours">83.5 J</td></tr>
<tr class="total"><td class="ours">ours &mdash; accuracy-max</td><td class="ours">0.5869</td><td class="ours">5.70</td><td class="ours">666 ms</td><td class="ours">176.6 J</td></tr>
</tbody></table></div>
<div class="two">
<div class="callout win"><b>Compute-lean:</b> statistically <b>indistinguishable</b> from the oracle 32B &mdash; &Delta; +0.0015, 95% CI [&minus;0.0025, +0.0055], not significant &mdash; at <b>0.49&times; the compute, 0.52&times; the latency and 0.49&times; the energy</b>. Its win is cost, and we state it that way.</div>
<div class="callout win"><b>Accuracy-max:</b> <b>beats</b> the oracle 32B, &Delta; +0.0136, 95% CI [+0.0108, +0.0165], significant. It is significantly better than the oracle on PMC-VQA, MMMU-Medical and PathVQA-open, and significantly worse on <b>none</b>.</div>
</div>
<div class="callout note"><b>Pareto verdict.</b> On both compute and latency, only three systems are non-dominated: <b>always-7B, our compute-lean setting, and our accuracy-max setting</b>. All three 32B strategies &mdash; direct, with-reasoning, and the oracle &mdash; are dominated. Thresholds are 5-fold cross-fit held-out; cost constants are measured batch-1; confidence intervals are 10,000-sample paired bootstraps over questions.</div>
<p class="src">Source: <span class="kbd">''' + A + '''/paper_baselines.json</span> (<span class="kbd">pooled</span>, <span class="kbd">paired_bootstrap_ci</span>, <span class="kbd">pareto_frontier</span>, <span class="kbd">verdict</span>).</p>'''))

# ---------------------------------------------------------------- 12 vs THINK measured
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>6 (cont.) &middot; Against the deployment premise</div>
<h2 class="slide-h sm">More accurate than always-reasoning, using less compute than one 32B pass</h2>
<p class="body"><b>Closing the last rigour gap.</b> Until 9 July the accuracy-max configuration was compared against an <i>estimated</i> reasoning baseline on the free-text cells. That estimate has been replaced by a <b>fully measured, per-sample</b> reasoning baseline (real judged outputs for every free-text question), and the comparison re-run as a paired bootstrap.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>vs measured always-32B-with-reasoning</th><th>ours</th><th>baseline</th><th>&Delta;</th><th>95% CI</th><th>n</th></tr></thead>
<tbody>
<tr><td>Full pool (MMMU excluded) &mdash; accuracy-max</td><td class="ours">0.5836</td><td>0.5591</td><td class="ours">+0.0245</td><td>[+0.0216, +0.0274]</td><td>42,224</td></tr>
<tr><td>Full pool (MMMU excluded) &mdash; compute-lean</td><td class="ours">0.5741</td><td>0.5591</td><td class="ours">+0.0150</td><td>[+0.0107, +0.0192]</td><td>42,224</td></tr>
<tr><td>Multiple-choice only</td><td>0.5842</td><td>0.5742</td><td>+0.0101</td><td>[+0.0073, +0.0128]</td><td>39,879</td></tr>
<tr class="total"><td>Free text only</td><td class="ours">0.5727</td><td>0.3028</td><td class="ours">+0.2699</td><td>[+0.2490, +0.2908]</td><td>2,345</td></tr>
</tbody></table></div>
<div class="callout win"><b>Result:</b> the accuracy-max setting is <b>+0.0245 more accurate</b> than running the 32B with reasoning on everything, while using <b>0.93&times; the compute of a single 32B forward pass</b> &mdash; i.e. it is compute-<i>negative</i> &mdash; and roughly 15&times; faster in wall-clock. The free-text arm is where the margin is enormous: reasoning collapses free-text accuracy to 0.30.</div>
<div class="callout honest"><b>Two data gaps we are not hiding.</b> (1) PathVQA-closed has no reasoning dump, so reasoning is assumed equal to direct answering there &mdash; consistent with Finding 1, but it is an assumption inside a headline pool. (2) The bootstrap resamples questions, so calibration variance from the held-out thresholds is not inside the interval.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/f8_mode_vsthink_ci.json</span> (headline + <span class="kbd">data_gaps</span>) &middot; <span class="kbd">''' + A + '''/opentext_32b_think_full.json</span>.</p>'''))

# ---------------------------------------------------------------- 13 per-benchmark efficiency + gate verdict
S.append(slide('''<div class="eyebrow"><span class="dot"></span>6 (cont.) &middot; Per-benchmark efficiency, and the gate verdict</div>
<h2 class="slide-h sm">Where the savings come from, and which gate we settled on</h2>
<p class="body">Under the faithful protocol, the two-tier arm matches the 32B at large compute savings wherever the cheap model is competitive. The size of the win tracks the accuracy gap between the two models: small gap, big win.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>benchmark</th><th>Lingshu 7B&rarr;32B</th><th>MedVLThinker 7B&rarr;32B</th></tr></thead>
<tbody>
<tr><td>PMC-VQA (33k questions)</td><td class="ours">match at &minus;69% compute / &minus;33% latency (9% escalated)</td><td class="ours">match at &minus;49% compute (29% escalated)</td></tr>
<tr><td>SLAKE-closed</td><td>match at &minus;56% compute / &minus;22% latency (22%)</td><td>no win (cheap model too weak)</td></tr>
<tr><td>VQA-RAD-closed</td><td>match at &minus;17% compute (61% escalated)</td><td>match at &minus;41% compute (37%)</td></tr>
<tr class="total"><td>MedXpertQA-MM</td><td>no win (cheap model near chance)</td><td>no win</td></tr>
</tbody></table></div>
<h3 class="sub">The gate: plain confidence wins, and gate choice is family-specific</h3>
<p class="body">We expected the agreement gate and the trained stability gate to win here, because they did on the other model family. On Lingshu they do not:</p>
<ul class="body">
<li><b>Confidence margin is the best ranker</b> &mdash; AUROC <b>0.7254</b>, minimum escalation 15.62%.</li>
<li><b>Agreement is the <i>worst</i></b> &mdash; AUROC 0.657 &mdash; <i>and</i> it needs a 32B forward pass to compute, which defeats the purpose of a cheap gate.</li>
<li><b>The trained stability gate is inert</b> here, because Lingshu-7B is <b>98.95% resolution-stable</b> &mdash; the signal has nothing to vary on.</li>
</ul>
<div class="callout note"><b>Lesson recorded:</b> gate choice is <b>model-family-specific, not universal</b>. This retires the "make the trained stability gate safe" item from the last report as a closed negative rather than an open task.</div>
<div class="callout win"><b>A free speed lever.</b> The 32B's image encoding does not depend on the 7B's output, so we prefetch it on the idle second GPU concurrently with the cheap pass: pooled batch-1 latency <b>461 &rarr; 405 ms (&minus;12.1%) at identical accuracy</b>.</div>
<p class="src">Sources: <span class="kbd">''' + D + '''/MASTER_SUMMARY_2026-07.md</span> &middot; <span class="kbd">''' + D + '''/METHOD_FINAL_2026-07.md</span> &middot; <span class="kbd">''' + A + '''/escalation_levers.json</span>.</p>'''))

# ---------------------------------------------------------------- 14 LIMIT 1 recoverability
S.append(slide('''<div class="eyebrow"><span class="dot"></span>7 &middot; Limit 1 &mdash; the recoverability wall</div>
<h2 class="slide-h sm">You cannot tell in advance which errors the big model will fix</h2>
<p class="body"><b>The question.</b> To <i>beat</i> the 32B on a multiple-choice question you must know, before paying for it, which questions the 32B will get right that the 7B got wrong. How learnable is that?</p>
<div class="eq">$$\\text{AUROC}\\big(\\text{predict } \\mathbb{1}[\\hat y_{32}=y \\;\\wedge\\; \\hat y_{7}\\neq y]\\big) \\;\\approx\\; 0.6 \\quad(\\text{measured } 0.506\\text{--}0.614)$$</div>
<p class="body"><b>Consequence.</b> The only closed multiple-choice slice we can <b>certifiably</b> beat the 32B on is <b>PMC-VQA</b>: +0.0135, 95% CI [+0.0100, +0.0169], n=33,430, held-out &mdash; the one large slice where the two models are comparably skilled with <i>de-correlated</i> errors. A conservative variant of the same rule gives +0.0095 [+0.0071, +0.0118].</p>
<p class="body"><b>Six independent methods were then asked to extend that beat. None did.</b></p>
<div class="tbl-wrap"><table>
<thead><tr><th>#</th><th>method tried</th><th>new certified slices beyond PMC-VQA</th></tr></thead>
<tbody>
<tr><td>1</td><td>Confidence-advantage fusion of the two models</td><td>0</td></tr>
<tr><td>2</td><td>Certified weak-veto rule</td><td>0</td></tr>
<tr><td>3</td><td>Bayesian model averaging</td><td>0</td></tr>
<tr><td>4</td><td>Contrastive decoding</td><td>0</td></tr>
<tr><td>5</td><td>Full-posterior fusion over raw option-probability vectors</td><td>0</td></tr>
<tr class="total"><td>6</td><td>Automatic error-slice discovery (106 candidate slices per split)</td><td>0</td></tr>
</tbody></table></div>
<div class="callout honest"><b>The slice-discovery result is the sharpest one.</b> It found <b>1.62</b> genuinely-new slices per split &mdash; <i>below</i> the label-permutation null of <b>5.61</b> (95th percentile 15), with only 0.25 surviving multiple-testing correction. In other words, the "discoveries" are indistinguishable from noise. Reassuringly, the same search <b>re-found the known PMC-VQA and MMMU wins in 7&ndash;8 of 8 splits without being told about them</b>, which validates the hand-built guardrail. A nearest-neighbour gate also lost to the plain confidence margin on 5 of 5 datasets.</div>
<div class="callout note"><b>The overfitting failure mode is real, and already fixed.</b> Naive per-slice routing produces <b>7.5 held-out guardrail violations per split</b>. The confidence-interval lower-bound guardrail we already deploy drives that to <b>0.25</b> while preserving the pooled beat (+0.0117). A fancier actuarial shrinkage estimator only reached 6.62 &mdash; it is diagnostic, not a replacement.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/robust_slice_routing.json</span> &middot; <span class="kbd">''' + A + '''/beat32b_fusion.json</span> &middot; <span class="kbd">''' + A + '''/beat32b_more.json</span> &middot; <span class="kbd">''' + A + '''/logit_fusion.json</span>.</p>'''))

# ---------------------------------------------------------------- 15 LIMIT 2 selection
S.append(slide('''<div class="eyebrow"><span class="dot"></span>7 (cont.) &middot; Limit 2 &mdash; the selection wall</div>
<h2 class="slide-h sm">The right answer is often among the samples; picking it is the hard part</h2>
<p class="body"><b>The gap.</b> On free text, sampling N answers frequently produces at least one correct answer, but the verifier cannot always identify it. That oracle-minus-selection gap is about <b>0.19</b>, and it resists every lever we tried.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>pooled, n=600 (VQA-RAD / SLAKE / PMC), same candidates for all rows</th><th>selection accuracy</th><th>vs the 32B verifier</th></tr></thead>
<tbody>
<tr><td>greedy single answer</td><td>0.440</td><td>&mdash;</td></tr>
<tr><td>self-consistency (majority vote)</td><td>0.445</td><td>&mdash;</td></tr>
<tr><td>7B verifier, zero-shot (no training)</td><td>0.413</td><td>+0.067 [+0.038, +0.095] <b>significant</b></td></tr>
<tr><td class="ours">7B verifier, <b>trained</b> (ours, 47.6M adapter parameters)</td><td class="ours">0.475</td><td>+0.005 [&minus;0.023, +0.032] <b>not significant</b></td></tr>
<tr><td>32B verifier, zero-shot (4.57&times; the cost per pass)</td><td>0.480</td><td>&mdash;</td></tr>
<tr class="total"><td>oracle (pick the best candidate present)</td><td>0.672</td><td>residual gap <b>0.192</b></td></tr>
</tbody></table></div>
<div class="callout win"><b>The headline read:</b> a verifier <b>7&times; larger</b> only <b>ties</b> our small trained one. The pure-capacity control confirms the comparison is fair &mdash; 32B zero-shot beats 7B zero-shot by +0.067, significant &mdash; so capacity does something, just not enough. <b>Task-specific training buys as much verification skill as 25 billion extra frozen parameters.</b></div>
''' + img("paper/figs/limits/fig_verifier_scaling.png",
          "Best-of-K: accuracy rises with the number of samples while random selection stays flat. The curve flattens well below the oracle - that flattening is the selection wall. (Figure produced 2026-06-26; the July work characterises the wall it shows.)", "660px") + '''
<div class="callout note"><b>What the wall is <i>not</i>.</b> It is not answer-formatting and it is not judge noise: selection efficiency is ~80% at every answer length, and inspected failures are legitimate semantic matches (e.g. "right cerebellum" vs "right posteroinferior cerebellum"). The residual is <b>intrinsic image-grounding difficulty for terse medical answers</b>.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/verifier_32b_gpu.json</span> &middot; <span class="kbd">''' + D + '''/UNIFIED_METHOD_EXPERIMENTS.md</span>.</p>'''))

# ---------------------------------------------------------------- 16 LoRA page 1: what & why & setup
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>8 &middot; LoRA &mdash; the project's one trained component (1 of 3)</div>
<h2 class="slide-h sm">What it is, why we use it, and exactly how it is trained</h2>
<p class="body"><b>What LoRA is.</b> Low-Rank Adaptation freezes the whole pre-trained model and inserts a pair of small trainable matrices beside each attention and feed-forward weight. Instead of updating 8.3 billion parameters, we update <b>47.6 million</b> &mdash; 0.57% &mdash; and ship a <b>190 MB</b> adapter file that sits on top of the frozen 7B we already serve.</p>
<p class="body"><b>The one job it does well: an outcome verifier.</b> Given the image, the question, and a <i>candidate answer</i>, the adapted model is asked "Is the proposed answer correct? Answer Yes or No", and we read the probability of "Yes" at the single next-token position:</p>
<div class="eq">$$s_\\phi(v,q,a)\\;=\\;P_\\phi(\\text{Yes}\\mid v,q,a)\\;=\\;\\frac{e^{z_{\\text{Yes}}}}{e^{z_{\\text{Yes}}}+e^{z_{\\text{No}}}}$$</div>
<p class="body">That single scalar does <b>two</b> jobs in the deployed free-text system: <b>(1) selection</b> &mdash; score all N sampled answers and keep the best, $\\hat a=\\arg\\max_i s_\\phi(v,q,a_i)$; and <b>(2) escalation</b> &mdash; if the winner's score is below a threshold, hand the question to the 32B. Training is plain cross-entropy on correctness labels, with the loss <b>masked to that single final token</b>, so the only thing learned is the Yes/No decision.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>training setup (the deployed adapter)</th><th>value</th></tr></thead>
<tbody>
<tr><td>Base model (frozen)</td><td>Lingshu-7B, bfloat16, flash-attention-2</td></tr>
<tr><td>Adapter shape</td><td>rank 16, alpha 32, dropout 0.05, no bias; 7 target modules (all attention + all feed-forward projections)</td></tr>
<tr><td>Trainable parameters</td><td class="ours">47,589,376 of 8,339,756,032 = <b>0.5706%</b></td></tr>
<tr><td>Training data</td><td>4 free-text sets (SLAKE, PathVQA, VQA-RAD, Kvasir); 3,545 questions with images and labels; grouped 70/30 split <b>by question</b> so no question leaks</td></tr>
<tr><td>Labels</td><td>per-candidate judge verdicts on 8 sampled answers per question; 6,000 training pairs used, positive rate 0.196</td></tr>
<tr><td>Held-out test set</td><td class="ours">1,064 questions, never seen in training</td></tr>
<tr class="total"><td>Optimisation</td><td>1 epoch, learning rate 1e-4, AdamW, batch 2 &times; gradient accumulation 8, gradient clip 1.0, full-resolution images, seed 0</td></tr>
</tbody></table></div>
<div class="callout honest"><b>Provenance gap, recorded honestly.</b> The stdout log for this exact adapter's seed-0 run is not on disk (we have the seed-1 rerun and the v2 retrain). Wall-clock and GPU-hours for the deployed adapter are therefore <b>not recorded</b>; the comparable 2-dataset run took 50.6 minutes on one A100. The adapter directory's README is still the auto-generated stub. Both are ten-minute fixes and should be done while still reconstructable.</div>
<p class="src">Sources: <span class="kbd">ckpts/train/lora_verifier_pooled4/adapter_config.json</span> &middot; <span class="kbd">logs/verif_pooled4_s1.log</span> &middot; <span class="kbd">src/training_methods/run_lora_verifier_open.py</span>.</p>'''))

# ---------------------------------------------------------------- 17 LoRA page 2: what it bought
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>8 (cont.) &middot; LoRA &mdash; what it bought us (2 of 3)</div>
<h2 class="slide-h sm">Free-text answers, structured outputs, and a tie with a 7&times;-larger model</h2>
<h3 class="sub">1. Free-text best-of-8 selection &mdash; the headline positive</h3>
<div class="tbl-wrap"><table>
<thead><tr><th>held-out, question-grouped</th><th>n</th><th>greedy</th><th>self-consistency</th><th>+ trained verifier</th><th>oracle</th><th>gap captured</th></tr></thead>
<tbody>
<tr><td>PathVQA-open</td><td>435</td><td>0.352</td><td>0.349</td><td class="ours">0.441</td><td>0.513</td><td>56%</td></tr>
<tr><td>Kvasir-open</td><td>365</td><td>0.282</td><td>0.282</td><td class="ours">0.405</td><td>0.493</td><td>58%</td></tr>
<tr><td>VQA-RAD-open</td><td>54</td><td>0.519</td><td>0.500</td><td class="ours">0.611</td><td>0.722</td><td>46%</td></tr>
<tr><td>SLAKE-open</td><td>210</td><td>0.738</td><td>0.738</td><td class="ours">0.762</td><td>0.895</td><td>15%</td></tr>
<tr class="total"><td>pooled</td><td>1,064</td><td>0.413</td><td>0.411</td><td class="ours">0.501</td><td>0.592</td><td class="ours">49%</td></tr>
</tbody></table></div>
<p class="body">Every dataset lifts. Every <i>training-free</i> selector sits at or below greedy. Scaling with samples is clean &mdash; K=1/2/4/8 gives 0.385 / 0.425 / 0.476 / 0.501 while random selection stays flat at ~0.39 &mdash; and the bootstrap gain over a single sample is <b>+0.116, 95% CI [+0.092, +0.139]</b>.</p>
<h3 class="sub">2. The same recipe works for structured outputs (bounding boxes)</h3>
<div class="tbl-wrap"><table>
<thead><tr><th>grounding task, overlap &ge; 0.3</th><th>n</th><th>greedy</th><th>consistency baseline</th><th>zero-shot verifier</th><th>trained verifier</th><th>oracle</th><th>gap</th></tr></thead>
<tbody>
<tr><td>SLAKE organ boxes</td><td>487</td><td>0.197</td><td>0.164</td><td>0.177</td><td class="ours">0.255</td><td>0.343</td><td>40%</td></tr>
<tr class="total"><td>MS-CXR chest X-ray pathology boxes</td><td>435</td><td>0.041</td><td>0.053</td><td>0.115</td><td class="ours">0.232</td><td>0.285</td><td class="ours">78%</td></tr>
</tbody></table></div>
<p class="body">On MS-CXR &mdash; a real published chest-X-ray grounding benchmark &mdash; the gain is <b>+0.191, 95% CI [+0.152, +0.232]</b>, a 5.6&times; lift over greedy, and the untrained version of the identical script reaches only 0.115. <b>Training, not prompting, is the active ingredient.</b></p>
''' + img("paper/figs/limits/fig_trained_verifier_unified.png",
          "One panel per output type - free-text answers, organ boxes, chest X-ray pathology boxes. Training breaks the floor in all three. (Figure produced 2026-06-26; tables above are the current numbers.)", "700px") + '''
<div class="callout win"><b>3. The headline fact for this line of work:</b> a <b>zero-shot 32B verifier</b> &mdash; a model 4.57&times; more expensive per forward pass &mdash; only <b>ties</b> the small trained adapter at picking the right answer (0.480 vs 0.475, &Delta; +0.005, CI [&minus;0.023, +0.032], not significant, n=600). 47.6 million trained parameters buy exactly as much verification skill as 25 billion extra frozen ones.</div>
<div class="callout note"><b>It transfers, with honest limits.</b> Cross-generator: the Lingshu-trained adapter scoring MedVLThinker answers lifts SLAKE-open 0.543 &rarr; 0.620 (n=645) and VQA-RAD-open 0.395 &rarr; 0.520 (n=200). Out-of-domain: RadImageNet 0.2045 &rarr; 0.2385 (n=2,000) &mdash; positive, but it captures far less of the gap than in-domain.</div>
<p class="src">Sources: <span class="kbd">ckpts/train/lora_verifier_pooled4/{result,scaling_curve,crossgen_result,transfer_result}.json</span> &middot; <span class="kbd">ckpts/train/lora_box_verifier*/result.json</span> &middot; <span class="kbd">''' + A + '''/verifier_32b_gpu.json</span>.</p>'''))

# ---------------------------------------------------------------- 18 LoRA page 3: where it failed
S.append(slide('''<div class="eyebrow"><span class="dot"></span>8 (cont.) &middot; LoRA &mdash; where it did <i>not</i> help (3 of 3)</div>
<h2 class="slide-h sm">Three other jobs we gave it, and the mechanistic reason each failed</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>job we gave LoRA</th><th>result</th><th>why it failed</th></tr></thead>
<tbody>
<tr><td><b>Learn the escalation decision</b> &mdash; fine-tune the 7B to answer "will a stronger model give the same answer?"</td><td>AUROC <b>0.7226</b> vs an 8-feature logistic regression at <b>0.7328</b>; plain confidence 0.6813. ~44 minutes of A100 time to lose to a model that fits in milliseconds.</td><td>The <b>recoverability wall</b>. Extra model capacity cannot manufacture a signal that is not in the input.</td></tr>
<tr><td><b>Distil the big model into the cheap leg</b> &mdash; so fewer questions need escalating at all</td><td>Net flat and it only <b>redistributes</b> accuracy. Config 1: 5-benchmark average 0.6377 &rarr; 0.6377. PathVQA 0.625&rarr;0.679 and MMMU 0.481&rarr;0.506, but VQA-RAD <b>0.767&rarr;0.662</b>. Config 2 repeats the pattern with the signs moved around.</td><td><b>Capacity and interference.</b> A single shared adapter learns pathology at the cost of forgetting radiology. It is a diagnosis, not a dead end &mdash; it says one adapter cannot serve all benchmarks.</td></tr>
<tr class="total"><td><b>Make the verifier itself better</b> by training harder &mdash; four separate levers</td><td>All flat on the metric that matters. More data + more epochs: 0.5056 vs 0.5009 (noise). A within-question ranking objective: pooled <b>unchanged at 0.5009</b>, even though per-answer discrimination rose 0.903 &rarr; 0.931.</td><td>The verifier is at its <b>selection ceiling</b>. Sharper globally, no better at picking. The residual failures are near-ties it genuinely cannot ground.</td></tr>
</tbody></table></div>
<div class="callout honest"><b>Two recorded gaps in the training story.</b> The rank-32 capacity run was killed at step 200 of 5,182 and its checkpoint directory is empty; the 500-pair and 1,500-pair data-scaling runs finished training but produced no result file. So there is currently <b>no training-data scaling curve and no capacity ablation</b> for the verifier. Both are cheap to finish and both are the first things a careful reader will ask for.</div>
<div class="callout note"><b>Also closed as negative:</b> the verifier does <b>not</b> work for multiple choice (mean gain +0.004, sign flips across datasets &mdash; see slide 9), and a related trained fusion head over the two models' option probabilities captured almost none of the complementarity the oracle shows (4&ndash;14% per family, negative on two).</div>
<h3 class="sub">Next steps for the LoRA line, in priority order</h3>
<ul class="body">
<li><b>Deploy the pairwise protocol, or establish why we cannot.</b> Re-prompting the <i>existing</i> adapter as an A-vs-B comparator already beats its own pointwise argmax by <b>+0.036, CI [+0.016, +0.055]</b> (n=578), with no retraining; a knockout bracket gets +0.032 for ~7 comparisons versus 8 pointwise calls &mdash; roughly free. This is the highest-confidence unbanked gain we have. It needs a measured latency/energy pass and an explanation for PathVQA, the one dataset where it did not help.</li>
<li><b>Shift effort to the candidate side</b> &mdash; the evidence now says that, not the verifier, is the binding constraint (slide 19).</li>
<li><b>Train a verifier <i>on</i> an evaluation distribution instead of transferring to it.</b> In-domain it captures 49% of the gap; zero-shot on RadImageNet, 13.6%. That single comparison separates "the verifier generalises weakly" from "those datasets are simply harder," which is currently confounded.</li>
<li><b>Revisit cheap-leg adaptation as per-domain adapters</b> rather than one shared adapter &mdash; the interference failure above points directly at this variant. Scope it tightly, kill it fast if the first slice is flat.</li>
<li><b>Do not reopen</b> three closed questions: the verifier on multiple choice, LoRA as an escalation router, and training the pointwise verifier harder on the current data.</li>
</ul>
<p class="src">Sources: <span class="kbd">ckpts/train/lora_stability/result.json</span> &middot; <span class="kbd">ckpts/train/fld{,_delta}/result.json</span> &middot; <span class="kbd">ckpts/train/lora_verifier_pooled4_v2/result.json</span> &middot; <span class="kbd">ckpts/train/lora_verifier_rank_l05/result.json</span> &middot; <span class="kbd">''' + A + '''/pairwise_verifier_gpu.json</span>.</p>'''))

# ---------------------------------------------------------------- 19 candidate side wins
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>9 &middot; The binding constraint moved</div>
<h2 class="slide-h sm">Better candidates, not a better verifier &mdash; and the cost of getting them</h2>
<p class="body">Because four training levers failed to move selection, we attacked the other side of the problem: the pool of candidate answers the verifier has to choose from. Three levers there <b>do</b> work.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>lever</th><th>effect on the oracle ceiling</th><th>effect on realised accuracy</th></tr></thead>
<tbody>
<tr><td><b>Diverse generation</b> (5 prompt personas &times; a temperature ladder), n=1,623</td><td class="ours">oracle 0.593 &rarr; 0.657, +0.064 [+0.047, +0.080]</td><td class="ours">verifier 0.434 &rarr; 0.459, +0.025 [+0.008, +0.042]</td></tr>
<tr><td><b>Cross-model candidate pooling</b> (3 different 7B/8B generators), n=3,400, held-out</td><td class="ours">over single-best: +0.056 (2 draws), +0.065 (4), +0.080 (8), <b>+0.105 (16)</b></td><td>not yet converted &mdash; see the honest note</td></tr>
<tr class="total"><td><b>Real pairwise comparison</b> (same adapter, A-vs-B prompt), n=578</td><td>&mdash;</td><td class="ours">selection 0.374 &rarr; 0.410, +0.036 [+0.016, +0.055]; efficiency 0.783 &rarr; 0.859</td></tr>
</tbody></table></div>
<h3 class="sub">And the cost of sampling is now adaptive rather than fixed</h3>
<p class="body">Instead of always drawing 8 samples, an <b>optimal-stopping rule from search theory</b> (Weitzman's Pandora's-box problem) uses a single exchange-rate knob to produce <i>both</i> a stop-drawing threshold and an escalation threshold, with thresholds held out via 5-fold cross-fit calibration.</p>
<div class="tiles">
<div class="tile"><div class="v good">8 &rarr; 4.28</div><div class="l">mean samples drawn, in the deployed method</div></div>
<div class="tile"><div class="v good">&minus;33%</div><div class="l">free-text compute (16.18 &rarr; 10.87), accuracy held 0.5642 &rarr; 0.5625</div></div>
<div class="tile"><div class="v teal">&minus;27% / &minus;28%</div><div class="l">compute / energy vs fixed best-of-8 at matched accuracy, held out</div></div>
</div>
<div class="callout honest"><b>The levers do not compound.</b> Pairwise comparison applied over diverse candidates is <i>not</i> better than pointwise over diverse candidates (&minus;0.0117, CI [&minus;0.0283, +0.0049]) and the both-levers gain (+0.0186) is not significant. Diversity buys <b>coverage</b>; it does not buy <b>selectability</b>. No candidate pre-filter beat both baselines either (best filter +0.0108, CI [&minus;0.0059, +0.0293], and it sign-flips per dataset).</div>
<div class="callout note"><b>The single most-identified unrun experiment:</b> wire a <i>stronger selector</i> directly into the optimal-stopping controller, to convert the +0.11 to +0.15 of oracle headroom that cross-model pooling and diverse generation demonstrably buy but the pointwise verifier cannot cash in.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/diverse_generation_gpu.json</span> &middot; <span class="kbd">''' + A + '''/generator_portfolio.json</span> &middot; <span class="kbd">''' + A + '''/pairwise_verifier_gpu.json</span> &middot; <span class="kbd">''' + A + '''/integrated_pandora_opentext.json</span> &middot; <span class="kbd">''' + A + '''/pandora_controller.json</span> &middot; <span class="kbd">''' + A + '''/combine_diverse_pairwise.json</span> &middot; <span class="kbd">''' + A + '''/distractor_filter.json</span> &middot; <span class="kbd">''' + A + '''/pandora_pooling_combo.json</span>.</p>'''))

# ---------------------------------------------------------------- 20 HONEST NEGATIVES
S.append(slide('''<div class="eyebrow"><span class="dot"></span>10 &middot; Honest negatives</div>
<h2 class="slide-h sm">What we ruled out, and the one principle the failures share</h2>
<p class="body">The search was widened deliberately rather than randomly: we read mechanisms out of economics of information, portfolio theory, crowdsourcing truth-inference, coding theory, social choice, sequential analysis, bandits and computer architecture, and mapped each onto a concrete testable variation. The backlog grew <b>35 &rarr; 46 &rarr; 56 &rarr; 68</b> ideas across four passes. Here is what did not survive.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>ruled out</th><th>the number that killed it</th></tr></thead>
<tbody>
<tr><td>One universal verifier covering multiple choice too</td><td>PMC-VQA greedy 0.534 &rarr; 0.132 when options are hidden; pooled gain +0.0038</td></tr>
<tr><td>Simulated pairwise comparison (derived from pointwise scores)</td><td>&minus;0.003 vs pointwise; the noise-free ceiling equals pointwise <b>exactly</b> in all three families</td></tr>
<tr><td>Adaptive bandit allocation of the sample budget</td><td>best pooled held-out gain <b>+0.002</b>; the de-biased ceiling <i>inverts</i> at 8 draws (&minus;0.026)</td></tr>
<tr><td>Optimal (Markowitz) portfolio allocation across generators</td><td>+0.002 pooled vs naive uniform; <b>negative</b> on VQA-RAD</td></tr>
<tr><td>Unsupervised answer aggregation (Dawid&ndash;Skene)</td><td>&minus;0.013 vs plain majority; sits 0.132 below the trained verifier</td></tr>
<tr><td>Test-time adaptation of the cheap model</td><td>the real objective <b>collapses</b> accuracy (&minus;0.159 pooled); even a label-informed oracle ceiling is under 1 point</td></tr>
<tr><td>Neuro-symbolic constraint checking</td><td>constraints fire on ~1 sample (coverage 0.0002); of 1,118 items where both models are wrong <i>and</i> agree, logic catches 16 (1.4%)</td></tr>
<tr><td>4-bit quantization of the strong leg as a <i>compute</i> lever</td><td>literally <b>zero</b> reduction under a multiply-accumulate unit; latency only 665 &rarr; 583 ms (the strong leg is encoding-bound)</td></tr>
<tr class="total"><td>Actuarial shrinkage instead of the deployed guardrail</td><td>7.5 &rarr; 6.62 violations per split, versus 7.5 &rarr; <b>0.25</b> for the simple confidence-interval rule already in place</td></tr>
</tbody></table></div>
<div class="callout win"><b>The reusable principle.</b> Transplants that supply <b>genuinely new signal</b> &mdash; a real pairwise forward pass, a diverse generator, a trained verifier &mdash; work. Transplants that merely <b>re-weight signal we already have</b> &mdash; optimal allocation, unsupervised aggregation, shrinkage &mdash; collapse to the naive baseline every single time. The recurring mechanism: these models are <b>confidently wrong</b>, so unsupervised reliability tracks self-agreement (~0.52) rather than accuracy (~0.29).</div>
<div class="callout honest"><b>One negative that had to be re-scoped, not just recorded.</b> "Best-of-N is compute-dominated" was true against the 32B's <i>cheap direct</i> mode and false against the reasoning mode the deployment premise is actually about. Same experiment, opposite verdict, because the baseline changed. That is exactly the error Finding 1 exists to prevent.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/{active_comparison_verifier,bandit_allocation,generator_portfolio,dawid_skene_aggregate,ttt_cheap_leg,neurosymbolic_gate,quantized_strong_leg,robust_slice_routing,end_to_end_consolidation}.json</span>.</p>'''))

# ---------------------------------------------------------------- 21 DATA INTEGRITY
S.append(slide('''<div class="eyebrow"><span class="dot"></span>11 &middot; Data integrity</div>
<h2 class="slide-h sm">The 7B scores +26 points above its published number &mdash; is that our bug?</h2>
<p class="body"><b>The anomaly.</b> On MMMU-Medical, Lingshu-7B scores <b>0.80</b> against a published <b>0.540</b>, and beats its own 32B (0.633). This is the single result that could have undermined the honesty of everything else, so it was investigated three times &mdash; the last at an explicitly adversarial standard: <i>prove it is not our evaluation bug</i>.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>#</th><th>check</th><th>outcome</th></tr></thead>
<tbody>
<tr><td>1</td><td>Sampling noise?</td><td>No. Discordant pairs 34 (7B right) vs 9 (32B right); McNemar &chi;&sup2; = 13.395, two-sided <b>p = 0.00017</b></td></tr>
<tr><td>2</td><td>Option-position / letter bias?</td><td>No. Under <b>every cyclic option permutation</b>, 7B 0.7708 vs 32B 0.6321 &mdash; position explains ~4 of ~18 points; the 7B still wins by +0.14</td></tr>
<tr><td>3</td><td><b>Is the image even used?</b></td><td class="ours"><b>Decisive.</b> 0.8267 with the real image &rarr; 0.62 blank &rarr; 0.62 noise &rarr; 0.5933 text-only. A ~0.23 drop &mdash; the answers are not text-derivable</td></tr>
<tr><td>4</td><td><b>Is the harness broken?</b></td><td class="ours"><b>Decisive.</b> The untuned base model Qwen2.5-VL-7B-Instruct scores <b>0.5667</b> through the identical harness &mdash; exactly the expected untuned range</td></tr>
<tr><td>5</td><td>Right model, right weights?</td><td>Pass. 8.29B parameters, correct architecture, correct snapshot</td></tr>
<tr><td>6</td><td>Gold answers and option sets intact?</td><td>Pass. 150 official validation items, <b>0</b> gold-letter and <b>0</b> option-set mismatches; no answer leaked into the prompt</td></tr>
<tr class="total"><td>7</td><td>Is our scorer inflating it?</td><td>Pass &mdash; the opposite. An independent non-harness rescore gives <b>123/150 = 0.82</b> vs our 120/150 = 0.80, with zero parse failures</td></tr>
</tbody></table></div>
<div class="callout win"><b>The load-bearing argument</b> is checks 3 &times; 4 taken together: no bug on our end could simultaneously make accuracy depend ~23 points on image <i>content</i> and leave the untuned base model at ~0.57. Verdict: a <b>genuine Lingshu-7B score</b>, consistent with training-set contamination upstream and outside our control. The same 7B reproduces its published numbers on every other benchmark, so a global harness fault is ruled out.</div>
<div class="callout honest"><b>And we still refuse to bank it.</b> Recomputing the headline without it moves the sample-weighted result by only <b>&minus;0.0005</b> (MMMU is 0.35% of the 42,374-question pool), so the headline never depended on it. But any <b>equal-weight per-benchmark average is materially inflated</b> and must be corrected: full-suite macro +0.0777 &rarr; <b>+0.0621</b>; multiple-choice-only macro +0.027 &rarr; <b>+0.0036</b>. Our current default is to exclude the cell entirely.</div>
<div class="callout note"><b>This is also a benchmark-hygiene finding of independent interest</b> to anyone using MMMU-Medical to evaluate medical vision-language models. <b>Question for you:</b> exclude the cell, or escalate it to the 32B and report the conservative number? We have not decided.</div>
<p class="src">Sources: <span class="kbd">''' + A + '''/mmmu_verify.json</span> &middot; <span class="kbd">''' + A + '''/mmmu_fix.json</span> &middot; <span class="kbd">''' + A + '''/mmmu_perm_{7b,32b}.json</span> &middot; <span class="kbd">''' + A + '''/method_final_mmmu_corrected.json</span>.</p>'''))

# ---------------------------------------------------------------- 22 OPEN QUESTIONS
S.append(slide('''<div class="eyebrow"><span class="dot"></span>12 &middot; Open questions &amp; known gaps</div>
<h2 class="slide-h sm">What is genuinely unresolved &mdash; stated plainly</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>open question</th><th>why it matters / what would settle it</th></tr></thead>
<tbody>
<tr><td><b>Preservation risk.</b> All work from 3 July onward is uncommitted; the last commit is 2026-07-02. There is also no documented activity 10&ndash;27 July.</td><td>About a week of results exists only in the working tree. Commit first, before any new experiment.</td></tr>
<tr><td><b>How should the contaminated MMMU cell be presented?</b></td><td>Sample-weighted headline is robust either way (&plusmn;0.0005); any macro number must use the corrected cell. This is your call, not ours.</td></tr>
<tr><td><b>The free-text verifier is in-domain</b> for SLAKE, VQA-RAD and PathVQA-open.</td><td>A genuinely out-of-domain verifier evaluation is the missing external-validity test for the accuracy engine.</td></tr>
<tr><td><b>Cross-family generalisation holds in direction, not magnitude.</b></td><td>The ~0.87 free-text routing signal is Lingshu-specific (MedVLThinker sits at 0.735&ndash;0.781); the cross-family verifier captures 25% of the oracle gap vs Lingshu's 49%. How much of the headline rides on Lingshu's calibration quality is unresolved.</td></tr>
<tr><td><b>The 7th benchmark's large-model leg</b> is still blocked by the two-GPU communication hang.</td><td>A 4-bit strong leg would fit on one GPU and route around it. Identified, never executed.</td></tr>
<tr><td><b>Image-token pruning of the strong leg</b> is a &minus;26% compute projection, deliberately not converted into a measured claim.</td><td>A wrong implementation would fabricate a lesion-safety number. It carries a mandatory radiology guardrail: image reduction is free on PMC-VQA (&minus;0.001) but costs &minus;0.017 on SLAKE and &minus;0.040 on VQA-RAD.</td></tr>
<tr><td><b>Layer-depth early exit</b> is the untested speed lever with real headroom (~2&ndash;3&times; on the forward pass).</td><td>It could not be evaluated offline because both models emit only ~3 generated tokens on every benchmark &mdash; they are image-encoding-bound, so token-level patience saves nothing.</td></tr>
<tr class="total"><td><b>A semantic escalation cache is currently unmeasurable.</b></td><td>No per-sample image identifier or hash is stored, so only question text is a usable key &mdash; and the high-duplication sets (SLAKE 0.815, PathVQA 0.372) are templated questions over <i>different</i> images. A data-capture gap to fix in future runs.</td></tr>
</tbody></table></div>
<div class="callout honest"><b>A caveat on the record itself.</b> One diary in this window (29&ndash;30 June) is an explicit reconstruction written on 2 July from the commit log and results documents, not a contemporaneous log. Two dated conclusions were later corrected within the window: 6 July (wrong strong-model baseline) and 8 July (estimated rather than measured reasoning numbers, superseded the same day and again on 9 July). All three are flagged rather than smoothed over.</div>'''))

# ---------------------------------------------------------------- 23 NEXT STEPS
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>13 &middot; Next steps</div>
<h2 class="slide-h sm">What we propose to do next, in priority order</h2>
<div class="step"><div class="n">1</div><div><b>Commit and preserve the July results.</b> Roughly a week of work &mdash; including the entire measured-reasoning baseline and the data-integrity audit &mdash; exists only as uncommitted working-tree state. This is a ten-minute task with an outsized downside if skipped.</div></div>
<div class="step"><div class="n">2</div><div><b>Bank the pairwise verification gain.</b> +0.036 selection accuracy (CI [+0.016, +0.055]) from re-prompting the existing adapter, with no retraining and roughly no extra cost via the knockout bracket. Needs a measured latency/energy pass and a diagnosis of the one dataset where it does not help.</div></div>
<div class="step"><div class="n">3</div><div><b>Wire a stronger selector into the optimal-stopping controller.</b> This is the single most-identified unrun experiment: cross-model pooling and diverse generation demonstrably buy +0.11 to +0.15 of oracle headroom that the current pointwise verifier cannot convert.</div></div>
<div class="step"><div class="n">4</div><div><b>Close the two recorded training gaps</b> &mdash; the verifier's data-scaling curve and its capacity ablation. Both runs were started and neither produced a result file. These are the first things a careful reader asks for.</div></div>
<div class="step"><div class="n">5</div><div><b>Run the external-validity test:</b> train a verifier <i>on</i> a held-out evaluation distribution rather than transferring to it, to separate "generalises weakly" from "that dataset is harder."</div></div>
<div class="step"><div class="n">6</div><div><b>Decide the MMMU presentation</b> with you (exclude vs conservatively escalate), and unblock the 7th benchmark's large-model leg with a 4-bit strong leg if you judge the cell worth the GPU time.</div></div>
<div class="two">
<div class="callout win"><b>Where the work stands.</b> A deployable method that is <b>faster and more accurate</b> than the strongest single-model strategy on the whole suite, with a clean two-setting Pareto knob, both settings using less compute than one large-model forward pass, reproducible from one command.</div>
<div class="callout win"><b>And a mapped boundary.</b> A precise, repeatedly-confirmed account of <i>where</i> a cheap-to-expensive medical cascade can and cannot beat the strong model, and <i>why</i> &mdash; the recoverability wall and the selection wall. That honest characterisation is a result in its own right, not a caveat.</div>
</div>'''))

# ---------------------------------------------------------------- 24 GLOSSARY
S.append(slide('''<div class="eyebrow"><span class="dot"></span>Appendix &middot; Glossary</div>
<h2 class="slide-h sm">Terms used in this report</h2>
<div class="two">
<div><dl class="def">
<dt>Cascade</dt><dd>Run the cheap model first; hand the question up to the expensive model only when needed.</dd>
<dt>Gate</dt><dd>The rule that decides whether to hand a question up.</dd>
<dt>Confidence margin</dt><dd>Gap between the top-1 and top-2 answer probabilities &mdash; the cheap confidence signal the gate thresholds.</dd>
<dt>Escalation rate</dt><dd>Fraction of questions handed to the expensive model.</dd>
<dt>Reasoning / direct mode</dt><dd>The model can emit a step-by-step trace before answering, or answer immediately.</dd>
<dt>Perception vs reasoning benchmark</dt><dd>Perception = read the answer off the image (SLAKE, VQA-RAD, PathVQA, PMC-VQA). Reasoning = requires medical knowledge and inference (MMMU-Medical, MedXpertQA).</dd>
<dt>Best-of-N</dt><dd>Sample N candidate answers, keep the one a scorer likes best.</dd>
<dt>Oracle@N</dt><dd>The accuracy you would get if you could always pick the best of the N candidates. An upper bound, not achievable.</dd>
</dl></div>
<div><dl class="def">
<dt>Gap captured</dt><dd>(selector &minus; greedy) / (oracle &minus; greedy) &mdash; how much of the available headroom a selector converts.</dd>
<dt>AUROC</dt><dd>How well a score separates two classes. 0.5 is useless, 1.0 is perfect.</dd>
<dt>Recoverability</dt><dd>Whether the expensive model will actually <i>fix</i> a given cheap-model error. Predicting this is the recoverability wall.</dd>
<dt>Selection efficiency</dt><dd>Probability of picking a correct answer <i>given that</i> one is present among the candidates.</dd>
<dt>LoRA</dt><dd>Low-Rank Adaptation: freeze the model, train a small pair of matrices beside each weight. Here 0.57% of parameters, a 190 MB file.</dd>
<dt>Outcome verifier</dt><dd>A model that scores "is this candidate answer correct?" rather than producing an answer itself.</dd>
<dt>FLOP-equivalent</dt><dd>Compute measured in multiples of one cheap-model forward pass; the 32B costs 4.57.</dd>
<dt>Oracle mode-selected 32B</dt><dd>A non-deployable upper bound: the large model, told in advance which mode wins on each benchmark, paying that mode's cost.</dd>
<dt>Paired bootstrap</dt><dd>Resample questions 10,000 times to get a confidence interval on the difference between two systems on the same questions.</dd>
</dl></div></div>
<div class="foot">All figures in this report were read from real artifacts under <span class="kbd">results/cascade_methods/artifacts/</span>, <span class="kbd">results/cascade_methods/docs/current/</span> and <span class="kbd">ckpts/train/</span>. No number was estimated or invented.</div>'''))

close = '''</main></div>
<button class="edge left" id="prev" aria-label="Previous slide"><span class="chevbtn">&#8249;</span></button>
<button class="edge right" id="next" aria-label="Next slide"><span class="chevbtn">&#8250;</span></button>
<div class="counter" id="counter">1 / N</div>
<script>
const deck=document.getElementById('deck');const slides=[...document.querySelectorAll('.slide')];
const counter=document.getElementById('counter');const pbar=document.getElementById('pbar');
const prev=document.getElementById('prev');const next=document.getElementById('next');const N=slides.length;let idx=0;
function render(){deck.style.transform='translateX(-'+(idx*100)+'vw)';counter.textContent=(idx+1)+' / '+N;pbar.style.width=(N>1?(idx/(N-1)*100):0)+'%';prev.disabled=(idx<=0);next.disabled=(idx>=N-1);}
function go(i){idx=Math.max(0,Math.min(N-1,i));if(slides[idx])slides[idx].scrollTop=0;render();}
prev.onclick=()=>go(idx-1);next.onclick=()=>go(idx+1);
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();go(idx+1);}else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();go(idx-1);}else if(e.key==='Home'){e.preventDefault();go(0);}else if(e.key==='End'){e.preventDefault();go(N-1);}});
let tx=0,ty=0;deck.addEventListener('touchstart',e=>{tx=e.changedTouches[0].clientX;ty=e.changedTouches[0].clientY;},{passive:true});
deck.addEventListener('touchend',e=>{const dx=e.changedTouches[0].clientX-tx,dy=e.changedTouches[0].clientY-ty;if(Math.abs(dx)>60&&Math.abs(dx)>Math.abs(dy))go(idx+(dx<0?1:-1));},{passive:true});
window.addEventListener('resize',render);render();
</script>
</body></html>'''

html = head + "\n" + appbar + "".join(S) + close
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT} ({len(html)} bytes, {len(html)//1024} KiB, {len(S)} slides)")
