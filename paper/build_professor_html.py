#!/usr/bin/env python3
"""Build the professor-facing slide deck, reusing the weekly-update template's CSS + nav JS."""
import base64, os
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
TPL=os.path.join(ROOT,"weekly-update-results (4).html")
lines=open(TPL,encoding="utf-8").read().split("\n")
head="\n".join(lines[:187])  # through </head>
# inject MathJax for real equations
mathjax='''<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>.eq{font-size:1.18rem;background:#f6f8fd;border:1px solid var(--line);border-left:4px solid var(--indigo);border-radius:8px;padding:12px 16px;margin:12px 0;overflow-x:auto}
.step{display:flex;gap:14px;margin:14px 0}.step .n{flex:0 0 34px;height:34px;border-radius:50%;background:var(--indigo);color:#fff;font-family:'Roboto Slab',serif;font-weight:700;display:flex;align-items:center;justify-content:center;font-size:1.1rem}
.callout{border-radius:12px;padding:16px 20px;margin:16px 0;font-size:1.18rem;line-height:1.55}
.callout.win{background:var(--good-bg);border:1px solid #bfe0c2}.callout.win b{color:var(--good)}
.callout.note{background:var(--indigo-bg);border:1px solid #cfd6f0}.callout.honest{background:#fff7e8;border:1px solid #f3e0b8}.callout.honest b{color:var(--warn)}
.figimg{width:100%;max-width:760px;display:block;margin:14px auto 4px;border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow-s)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:820px){.two{grid-template-columns:1fr}}
ul.body{font-size:1.22rem;line-height:1.6;margin:10px 0 10px 22px}ul.body li{margin:7px 0}</style>'''
head=head.replace("</head>", mathjax+"\n</head>")

def img(path, cap):
    p=os.path.join(ROOT,path)
    if not os.path.exists(p): return f"<p class='caption'>[missing {path}]</p>"
    b=base64.b64encode(open(p,"rb").read()).decode()
    return f"<img class='figimg' src='data:image/png;base64,{b}' alt='{cap}'><p class='caption' style='text-align:center'>{cap}</p>"

appbar='''<body>
<div class="appbar">
  <span class="mark">Med-VLM · CVGIP 2026</span>
  <span class="title">Efficient &amp; Accurate Medical VLMs &mdash; <b>ACC + Trained Verifier</b></span>
  <span class="spacer"></span>
  <span class="meta">structure &amp; a little training beat the gate</span>
</div>
<div class="progress"><div class="progress-bar" id="pbar"></div></div>
<div class="deck-viewport"><main class="deck" id="deck">
'''

def slide(inner): return f'<section class="slide"><div class="swrap"><div class="card">{inner}</div></div></section>\n'
S=[]

# 1 COVER
S.append(slide('''<div class="cover">
<div class="eyebrow"><span class="dot"></span>Progress Report</div>
<div class="big">Structure and a little training<br><span class="accent">beat the gate</span>.</div>
<div class="sub">For medical vision-language models, you cannot out-engineer the cascade's decision with untrained tricks &mdash; they all hit one wall. But two levers give large, real gains: a smarter <b>compute structure</b> (efficiency) and a small <b>trained verifier</b> (accuracy).</div>
<div class="cover-stats">
<div class="cstat"><div class="v">&minus;80%</div><div class="l">latency at equal accuracy (ACC cascade)</div></div>
<div class="cstat"><div class="v teal">~5&times;</div><div class="l">less energy (ACC, at parity)</div></div>
<div class="cstat"><div class="v">40&ndash;78%</div><div class="l">of the oracle gap recovered (trained verifier)</div></div>
<div class="cstat"><div class="v teal">0.924</div><div class="l">verifier AUROC (right vs wrong answers)</div></div>
</div></div>'''))

# 2 PREMISE
S.append(slide('''<div class="eyebrow"><span class="dot"></span>The premise</div>
<h2 class="slide-h">Why this project exists</h2>
<p class="body">A <b>medical VLM</b> takes a medical <b>image</b> (chest X-ray, pathology slide) plus a <b>question</b> and returns an <b>answer</b>. The most accurate models are large "reasoning" models that think step-by-step before answering.</p>
<div class="tiles">
<div class="tile"><div class="v bad">~11 s</div><div class="l">a 32B reasoning model, per question (batch&nbsp;1)</div></div>
<div class="tile"><div class="v bad">~6 kJ</div><div class="l">energy for that one question</div></div>
<div class="tile"><div class="v good">~0.2 s</div><div class="l">a small 7B model, per question</div></div>
</div>
<p class="body">The small model is fast but less accurate; the big model is accurate but slow and power-hungry. <span class="hl">Our goal: match the big model's accuracy for far less compute &mdash; and, separately, raise accuracy cheaply.</span> Deliverable: a CVGIP&nbsp;2026 paper.</p>
<div class="callout note"><b>The standard tool is a cascade:</b> run the cheap model first, and only "escalate" to the expensive model on hard questions. The whole game is (1) <i>which</i> questions to escalate (the <b>gate</b>) and (2) <i>what</i> to run when you do (the <b>action</b>).</div>'''))

# 3 KEY TERMS
S.append(slide('''<div class="eyebrow"><span class="dot"></span>Key terms &amp; metrics</div>
<h2 class="slide-h sm">The vocabulary for this report</h2>
<div class="two">
<div><ul class="body">
<li><b>Cascade</b> &mdash; cheap model first; escalate hard cases.</li>
<li><b>Gate</b> &mdash; the rule deciding "escalate or not."</li>
<li><b>Confidence / margin</b> &mdash; how sure a model is: $m=\\ell_1-\\ell_2$ (gap of top-2 answer log-probs).</li>
<li><b>Think / no-think</b> &mdash; the big model can reason slowly, or answer directly (fast).</li>
<li><b>Oracle gap</b> &mdash; how much better you'd do if you could always pick a model's best of N tries.</li>
</ul></div>
<div><ul class="body">
<li><b>Luck floor</b> &mdash; our key negative: untrained, you can't pick the best try better than random.</li>
<li><b>Verifier</b> &mdash; a small trained helper scoring "is this answer correct?"</li>
<li><b>Best-of-N</b> &mdash; sample N, keep the verifier's favourite.</li>
<li><b>AUROC</b> &mdash; how well a score separates right from wrong (0.5 useless, 1.0 perfect).</li>
<li><b>FLOPs / latency / energy</b> &mdash; three cost measures (math ops / seconds / joules).</li>
</ul></div></div>
<div class="callout note"><b>The whole story in one line:</b> untrained decision rules all hit one wall; <b>structure</b> (ACC) and <b>a little training</b> (the verifier) are the two things that get past it.</div>'''))

# 4 ACC idea + math
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>Step 1 &middot; Method &mdash; ACC (efficiency)</div>
<h2 class="slide-h sm">What we did, and why it works</h2>
<p class="body"><b>What:</b> a three-tier cascade built from <i>compute configurations of the same two models</i>:</p>
<div class="eq" style="text-align:center;font-size:1.3rem">cheap 7B (fast) &rarr; big 32B (<b>fast / no-think</b>) &rarr; big 32B (slow / think)</div>
<p class="body"><b>Why it works:</b> the big model's <i>fast</i> mode is as good as or <i>better than</i> its slow mode on perception questions &mdash; <span class="hl">thinking over-thinks them</span> (SLAKE 0.849 vs 0.764; VQA-RAD 0.853 vs 0.776). So most escalations need <i>capacity</i>, not <i>reasoning</i>; the fast big-model tier absorbs them, and the slow think pass fires <b>only when the two fast legs disagree</b> (~15% of questions).</p>
<p class="body"><b>The math (the agreement gate):</b> with the two fast predictions $\\hat y_{T0},\\hat y_{T1}$,</p>
<div class="eq">$$s_1=\\mathbb{1}[\\hat y_{T0}\\neq\\hat y_{T1}]+\\epsilon\\,(-m_{T1}),\\qquad \\text{fire the slow tier iff } s_1>\\tau_1.$$</div>
<p class="body">Expected cost of the cascade (per-tier cost $c$, escalation rates $e_0,e_1$): $\\;C=c_{T0}+e_0\\,c_{T1}+e_1\\,c_{T2}.$ The savings come from $e_1$ (slow-tier rate) dropping from ~69% to ~15%, and the slow tier costs ~80&times; the fast one.</p>
<div class="callout honest"><b>Honest note:</b> the agreement <i>rule</i> itself is not new (it is shared with prior "agreement-based cascading"). Our contribution is the <b>structure</b> &mdash; inserting the big model's fast mode as a middle tier.</div>'''))

# 5 ACC results
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>Step 1 &middot; Result &mdash; ACC</div>
<h2 class="slide-h sm">Same accuracy, a fraction of the cost</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>system (6-benchmark avg)</th><th>accuracy</th><th>latency</th><th>energy</th><th>FLOPs</th></tr></thead>
<tbody>
<tr><td>always use big-think</td><td>0.572</td><td>11.34 s</td><td>6,319 J</td><td>100%</td></tr>
<tr class="total"><td class="ours">ACC (ours)</td><td class="ours">0.569</td><td class="ours">2.27 s</td><td class="ours">1,182 J</td><td class="ours">52%</td></tr>
</tbody></table></div>
<div class="legend"><span class="it"><b>&minus;80% latency</b></span><span class="it"><b>~5&times; less energy</b></span><span class="it"><b>compute halved</b></span><span class="it">at parity accuracy, never worse than the 7B on any benchmark</span></div>
'''+img("paper/figs/fig1_latency_accuracy_frontier.png","ACC reaches parity accuracy at a fraction of the latency.")+'''
<div class="callout win"><b>What it means:</b> on the 5-benchmark set latency drops 8.9 s &rarr; 0.44 s (&minus;95%). And swapping in <i>any</i> other untrained gate gives the same frontier &mdash; confirming the win is the <b>structure, not the gate</b>.</div>'''))

# 6 the wall
S.append(slide('''<div class="eyebrow"><span class="dot"></span>Step 2 &middot; The wall (why no "smarter gate" exists)</div>
<h2 class="slide-h sm">A dozen untrained signals, one ceiling</h2>
<p class="body"><b>What we did:</b> tried to build a better escalation rule with no extra training &mdash; confidence, entropy, agreement, self-checking, conformal prediction, learned routers, multi-resolution stability, and more.</p>
<p class="body"><b>Result:</b> all <b>12 signals hit the same ceiling</b> (AUROC ~0.5&ndash;0.69). <b>Why:</b> the cheap and big models <i>fail on the same questions</i> &mdash; correlation $\\phi=0.37$; <b>58%</b> of the cheap model's errors the big model also gets wrong. You can tell a question is hard, but not whether the big model would <i>fix</i> it.</p>
<div class="callout note"><b>A useful twist:</b> this looks worse than it is on multiple-choice (a single A/B/C/D letter is too coarse). On <b>open-ended</b> answers the same confidence signal jumps from AUROC ~0.6 to <b>~0.87</b> &mdash; so medical-VLM cascades should be evaluated open-ended.</div>'''))

# 7 luck floor selection
S.append(slide('''<div class="eyebrow"><span class="dot"></span>Step 2 &middot; The luck floor</div>
<h2 class="slide-h sm">You can't pick your best try (without training)</h2>
<p class="body"><b>What we did:</b> sample the model 8 times per question. A correct answer is often present (big oracle gap, e.g. one task 0.730 &rarr; <b>0.879</b> if you could always pick right). Can any untrained rule pick it?</p>
<p class="body"><b>Result &mdash; no.</b> Every untrained selector ties or trails picking <i>at random</i> (0.720): self-check 0.715, big-model re-rank 0.758; none beats the big model's single pass (0.819). A <b>"majority trap"</b> explains it: the correct answer is a <i>minority</i> vote 74&ndash;90% of the time.</p>
<div class="callout honest"><b>The luck floor:</b> the gap is <i>sampling luck, not latent knowledge</i> &mdash; the model doesn't <i>know</i> which try is right. The same holds even for bounding boxes. This is the negative that the next step finally breaks.</div>'''))

# 8 verifier idea + math
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>Step 3 &middot; Method &mdash; the trained verifier (accuracy)</div>
<h2 class="slide-h sm">Train a small helper to pick the right answer</h2>
<p class="body"><b>Idea:</b> the luck floor is a property of <i>frozen</i> models. So <b>train</b> a small helper (a LoRA fine-tune) that scores "is this candidate correct?" and use it to pick the best of N samples.</p>
<p class="body"><b>The math.</b> The verifier scores a candidate by its probability of "Yes":</p>
<div class="eq">$$s_\\phi(v,q,a)=P_\\phi(\\text{Yes}\\mid v,q,a)=\\frac{e^{z_{\\text{Yes}}}}{e^{z_{\\text{Yes}}}+e^{z_{\\text{No}}}}.$$</div>
<p class="body">It is trained on correctness labels $y$ (from an LLM judge for answers; from box-overlap IoU for boxes) by cross-entropy: $\\;\\mathcal{L}(\\phi)=-\\sum[y\\log s_\\phi+(1{-}y)\\log(1{-}s_\\phi)].$ Then best-of-N selection just takes</p>
<div class="eq">$$\\hat a=\\arg\\max_{i\\le N}\\;s_\\phi(v,q,a_i).$$</div>
<p class="body">We measure the fraction of the oracle gap captured: $\\;(\\mathrm{acc}(\\hat a)-\\mathrm{greedy})/(\\mathrm{oracle}-\\mathrm{greedy}).$</p>'''))

# 9 verifier free-text
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>Step 3 &middot; Result &mdash; free-text answers</div>
<h2 class="slide-h sm">49% of the oracle gap, every dataset lifted</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>dataset</th><th>cheap model</th><th>+ trained verifier</th><th>best possible</th></tr></thead>
<tbody>
<tr><td>PathVQA</td><td>0.352</td><td class="ours">0.441</td><td>0.513</td></tr>
<tr><td>Kvasir</td><td>0.282</td><td class="ours">0.405</td><td>0.493</td></tr>
<tr><td>VQA-RAD</td><td>0.519</td><td class="ours">0.611</td><td>0.722</td></tr>
<tr><td>SLAKE</td><td>0.738</td><td class="ours">0.762</td><td>0.895</td></tr>
<tr class="total"><td>average (n=1064)</td><td>0.413</td><td class="ours">0.501</td><td>0.592</td></tr>
</tbody></table></div>
<div class="callout win"><b>What it means:</b> the trained verifier captures <b>49%</b> of the oracle gap, lifting every dataset, and even transfers zero-shot to a 5th dataset it never saw. The bootstrap gain is +0.116, 95% CI [+0.092, +0.139] &mdash; statistically solid.</div>'''))

# 10 verifier boxes
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>Step 3 &middot; Result &mdash; bounding boxes</div>
<h2 class="slide-h sm">The same idea works for grounding &mdash; on a real benchmark</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>task</th><th>cheap</th><th>+ trained verifier</th><th>best possible</th><th>gap captured</th></tr></thead>
<tbody>
<tr><td>SLAKE organs</td><td>0.197</td><td class="ours">0.255</td><td>0.343</td><td>40%</td></tr>
<tr class="total"><td>MS-CXR chest X-ray (real)</td><td>0.041</td><td class="ours">0.232</td><td>0.285</td><td>78%</td></tr>
</tbody></table></div>
'''+img("paper/figs/limits/fig_trained_verifier_unified.png","Training breaks the luck floor across output types: free-text answers, organ boxes, and real chest-X-ray pathology boxes.")+'''
<div class="callout win"><b>What it means:</b> on the real <b>MS-CXR</b> chest-X-ray benchmark the gain is +0.191 (95% CI [+0.152, +0.232]) &mdash; a <b>5.6&times; lift</b>. Training <i>doubles</i> what an untrained verifier achieves. The principle &mdash; "training breaks the luck floor" &mdash; holds for structured outputs too.</div>'''))

# 11 verifier trust
S.append(slide('''<div class="eyebrow teal"><span class="dot"></span>Step 3 &middot; Why we trust it</div>
<h2 class="slide-h sm">It really discriminates &mdash; and beats a 5&times; bigger model</h2>
<div class="two">
<div>'''+img("paper/figs/limits/fig_verifier_discrimination.png","AUROC 0.924: clean separation of correct vs incorrect candidates.")+'''</div>
<div>'''+img("paper/figs/limits/fig_verifier_pareto.png","7B + verifier (0.501) beats the 32B single pass (0.444).")+'''</div>
</div>
<ul class="body">
<li><b>Genuine discriminator:</b> AUROC <b>0.924</b> at scoring individual answers; it <i>uses the image</i> (blanking it hurts).</li>
<li><b>Beats the bigger model:</b> a 7B with the verifier (0.501) beats the <b>5&times; larger</b> 32B's single answer (0.444) &mdash; test-time compute beats parameters where reasoning doesn't help.</li>
<li><b>Scales with compute:</b> more samples &rarr; higher accuracy (random selection stays flat).</li>
</ul>'''))

# 12 conclusion
S.append(slide('''<div class="eyebrow"><span class="dot"></span>Where it stands</div>
<h2 class="slide-h sm">Two contributions, one clean story</h2>
<div class="two">
<div class="callout win"><b>Structure &mdash; ACC (efficiency):</b> &minus;80% latency, ~5&times; energy, compute halved, at parity accuracy. The win is the compute-configuration, not the gate.</div>
<div class="callout win"><b>Training &mdash; the verifier (accuracy):</b> recovers 40&ndash;78% of the oracle gap for answers <i>and</i> boxes (incl. a real chest-X-ray benchmark), a test-time-scaling method that lets a 7B beat a 32B.</div>
</div>
<p class="body">Tied together by the <b>luck floor</b>: untrained routing/selection over a frozen model is bounded; the two levers that move the needle are <b>structure</b> and <b>a little training</b>.</p>
<div class="callout note"><b>Next steps:</b> push the verifier further (a larger verifier, more datasets, reuse across generator models) to strengthen the novelty. Full detail: <span class="kbd">paper/cvgip2026_draft.md</span> &middot; deep-dives in <span class="kbd">results/cascade_methods/</span> &middot; reading order in <span class="kbd">READING_GUIDE.md</span>.</div>'''))

close='''</main></div>
<button class="edge left" id="prev" aria-label="Previous slide"><span class="chevbtn">&#8249;</span></button>
<button class="edge right" id="next" aria-label="Next slide"><span class="chevbtn">&#8250;</span></button>
<div class="counter" id="counter">1 / 12</div>
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

html=head+"\n"+appbar+"".join(S)+close
out=os.path.join(ROOT,"progress_report_professor.html")
open(out,"w",encoding="utf-8").write(html)
print(f"wrote {out} ({len(html)//1024} KiB, {len(S)} slides)")
