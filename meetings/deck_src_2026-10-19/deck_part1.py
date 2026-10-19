"""deck_part1.py — opening, background (the field's vocabulary), key papers. Each slide carries its verbatim
script (say=), what to point at (look=) and the questions it should be able to answer (qa=)."""
from deck_lib import C, fmt, forest, bars, lines
import deck_data as D
import deck_figs as F
import figs_v2 as F2

pm, ci = D.pm, D.ci
import json as _json
# numbers derived during the 2026-10-19 review from the shipped artifact (script + inputs recorded in the file)
R1 = _json.load(open("/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part1.json"))
R1F = "numbers_review_part1.json (derived from final_eval_FINAL_cap320_T07.json by review/scr/w1_micro.py)"


def fit_svg(svg, width):
    """Rescale an SVG's outer width/height attributes to `width` px (viewBox kept, so it scales cleanly)."""
    import re
    m = re.search(r'<svg[^>]*?width="([\d.]+)" height="([\d.]+)"', svg)
    w, h = float(m.group(1)), float(m.group(2))
    return svg.replace(f'width="{m.group(1)}" height="{m.group(2)}"', f'width="{width}" height="{h * width / w:.0f}"', 1)


def kpi(label, val, sub, tone=""):
    return f'<div class="kpi"><span class="l">{label}</span><span class="big {tone}">{val}</span><span class="s">{sub}</span></div>'


def build(d):
    J = D.macro("judge"); X = D.macro("xjudge"); E = D.macro("em"); F1 = D.macro("f1"); L = D.macro("llava_recall")
    SCJ = D.sc_macro("judge")

    # ------------------------------------------------------------------ 1 title
    d.part = "Opening"
    d.add("title", "Title — choosing the right answer from a medical VLM's own samples", f"""
<div class="divider" style="position:absolute;inset:0">
 <div class="num">PROFESSOR REPORT · OCTOBER 2026</div>
 <h1>Choosing the right answer from a medical<br>VLM's own samples</h1>
 <p>A trained probe on Lingshu-7B's hidden states, used as a best-of-8 verifier for open-ended medical VQA —
 the field, the method in full, why each design choice, and how it compares.</p>
 <p style="top:560px;font-size:16px;opacity:.75">Li-Wen (Leo) Kuan · medvlthinker-imgdiff-compute · shipped method: single sampling temperature T = 0.7</p>
</div>""", raw=True, minutes=1.0,
          say="""# Choosing the right answer from a medical VLM's own samples
- Good afternoon. Today I want to do three things.
  - First, the vocabulary of our research area, so every term I use later is defined.
  - Second, our method piece by piece — what it reads, how it decides, and why each piece is the way it is.
  - Third, an honest comparison with what other groups have published.
- The short version: a medical vision-language model writes eight possible answers to a question.
- A small trained classifier that reads the model's internal states picks one of them.""")

    # ------------------------------------------------------------------ 2 the result in one slide
    d.add("oneslide", "The whole talk on one slide",
          f"""
<div class="row">
 <div class="col w2">
  <div class="term t"><b class="k">Method</b><span>8 samples (T = 0.7) → 24 probes on layers 18/20/22 → rank average → top answer</span></div>
  <div class="term"><b class="k">Question</b><span>does a cheap trained selector beat the model's own greedy answer on open-ended medical VQA?</span></div>
  <div class="term v"><b class="k">What is new — and what is not</b><span>Mechanism: not ours (text LLMs; MedProb App. H, 100 items).
  Ours: 8 benchmarks · image cross-fitting · cross-family judge · lenient EM · WED head-to-head.</span></div>
 </div>
 <div class="col">
  {kpi("cross-family judge", pm(X[0]), "MedGemma-27B · never used in training · " + ci(X[1], X[2]), "i")}
  {kpi("training judge", pm(J[0]), f"MedVLThinker-32B · also the label source · {ci(J[1],J[2])} · 6 of 8 significant, 2 tie")}
  {kpi("lenient exact match", pm(E[0]), f"{ci(E[1],E[2])} · loses on 3 benchmarks", "g")}
 </div>
</div>""", "Opening", sub="gain in accuracy over the same model's greedy answer, macro over 8 open-ended medical VQA benchmarks, 36,869 questions",
          src="final_eval_FINAL_cap320_T07.json — macro_over_open8.all.pick_minus_greedy_{judge,xjudge,em}; every question scored by probes that never saw its image",
          minutes=2.5,
          look="Left: the three boxes. Right: the three numbers — the top one is the grader the probe never trained on.",
          say=f"""# Method
- Here is the whole talk on one slide.
- Our model, Lingshu-7B, is a medical vision-language model.
- For each question it writes eight answers, using a bit of randomness.
- A small trained classifier — in the field, a probe — reads the model's internal numbers for each answer and scores it.
- We train 24 of these probes, average their rankings, and return the answer they rank first.
# Question
- Does that beat the model's normal answer — the single most likely one, called the greedy answer?
- It does.
# CROSS-FAMILY JUDGE
- Averaged over eight benchmarks, a grader from a completely different model family, MedGemma-27B, measures a gain of {X[0]:.3f} — about seven points.
- The probe never saw this grader during training.
# TRAINING JUDGE
- The grader whose labels we trained on, a 32-billion-parameter medical model, measures {J[0]:.3f}.
- Because the probe learned from this grader, I treat the MedGemma number as the cleaner one.
# LENIENT EXACT MATCH
- A lenient string-matching score: right if the answer equals or contains the gold answer.
- It also improves, by {E[0]:.3f}: small, but significant on these benchmarks.
# What is new — and what is not
- Using a probe on hidden states to pick among sampled answers has been done for text-only language models, mostly on math.
- One medical paper tried it on a hundred questions.
- What we add is the measurement:
  - eight benchmarks and held-out images;
  - two graders from different families plus string matching;
  - a direct comparison with a published method.
- I will show you exactly where it works and where it does not.""",
          qa=[("Is the +0.08 partly the probe learning the training judge's taste?",
               f"Partly, yes. In this project a newly trained verifier gains about 0.006–0.009 under its training judge from paraphrase drift alone "
               f"(coadapt_verifier_T04_2026-08-14.json). That is why the cross-family judge, never used in training, is quoted first: {pm(X[0])} {ci(X[1],X[2])}. "
               f"The gap to {pm(J[0])} is {J[0]-X[0]:.4f}, the size of that drift. In a September 2×2 each probe scored best under its own training judge "
               f"and the cross-judge gain was +0.0565 to +0.0658 (judge_2x2_2026-09-20.json)."),
              ("Does the exact-match gain generalise beyond these eight benchmarks?",
               f"Not shown. It is significant with images resampled within these eight benchmarks. Treating the benchmarks themselves as the sampling unit "
               f"(t-interval over the 8 deltas, df 7) the judge gains stay positive — training judge {ci(*R1['benchmark_as_unit_t_ci_judge']['value'])}, "
               f"cross-family {ci(*R1['benchmark_as_unit_t_ci_xjudge']['value'])} — but lenient exact match becomes a tie, "
               f"{ci(*R1['benchmark_as_unit_t_ci_em']['value'])} ({R1F}). Dropping OmniMedVQA or RadImageNet alone also makes it a tie (review/rr_lobo.py)."),
              ("Macro vs micro — what is the question-weighted gain?",
               f"Weighting each question equally instead of each benchmark: training judge {pm(R1['micro_pick_minus_greedy_judge']['value'])}, "
               f"cross-family {pm(R1['micro_pick_minus_greedy_xjudge']['value'])}, lenient EM {pm(R1['micro_pick_minus_greedy_em']['value'])} "
               f"(no CI computed; {R1F}). Macro is reported so the 10,121-question Kvasir set cannot dominate."),
              ("Would it work on a new benchmark you have no labels for?",
               "Not shown, and probably not as well. The claim is about new questions and images from these 8 benchmarks, each of which contributed "
               "labelled training questions. Leaving a benchmark out of training gave no benefit over a probe trained on the four-domain set (the PathVQA, SLAKE and VQA-RAD training splits plus Kvasir-VQA) "
               "(pooling slide, Part IV); a new benchmark needs some labelled questions."),
              ("Is 'exact match' strict?", "No — it is lenient: after removing case, punctuation and a few stop-words, an answer counts as correct if it equals the gold "
               "or either string contains the other (em_score in src/audit_2026_09_18/em_rescore_pooled_probe.py, verbatim from run_openvqa.py). "
               "Strict equality has not been computed for the final method — not measured."),
              ("Why compare against greedy and not against a bigger model?",
               "Greedy is what you get from the same model with no extra machinery, so the gain is attributable to the selector alone. "
               "Against Lingshu-32B greedy we only have an older, four-benchmark comparison: under the judge one loss (VQA-RAD), two ties, one win; "
               "under exact match two losses — so we do not claim to beat the 32B (audit_stats_baselines_2026-09-20.json)."),
              ("What is 'macro'?", "Each benchmark counts one eighth regardless of size, so the 10,121-question Kvasir set cannot dominate the average.")])

    # ------------------------------------------------------------------ 3 roadmap
    d.add("roadmap", "Roadmap",
          """
<div class="row">
 <div class="col">
  <div class="box"><h3>I · The field's vocabulary</h3><ul class="b sm"><li>medical VQA, VLMs, decoding</li><li>best-of-N, coverage, selection</li>
  <li>verifiers, probes, judges</li><li>uncertainty ≠ selection</li></ul></div>
  <div class="box"><h3>II · Key papers and their methods</h3><ul class="b sm"><li>trained verifiers, test-time compute</li>
  <li>training-free selection, the voting ceiling</li><li>hidden-state verifiers in LLMs</li><li>medical VQA: self-verification, probes, WED</li></ul></div>
 </div>
 <div class="col">
  <div class="box t"><h3>III · Our method, piece by piece</h3><ul class="b t sm"><li>sampling → re-read → layers → probe → readout</li>
  <li>the math at each step</li><li>evaluation protocol and cost</li></ul></div>
  <div class="box t"><h3>IV · Why each choice</h3><ul class="b t sm"><li>what we tested instead, with numbers</li></ul></div>
 </div>
 <div class="col">
  <div class="box"><h3>V · Results</h3><ul class="b sm"><li>all metrics, per benchmark, fixes vs harms</li></ul></div>
  <div class="box"><h3>VI · Against published work</h3><ul class="b sm"><li>same candidates · same metric code · different protocols</li></ul></div>
  <div class="box"><h3>VII · Honest limits and next steps</h3></div>
 </div>
</div>""", "Opening", minutes=0.7,
          say="""# Roadmap
- The talk has seven parts.
# I · The field's vocabulary
- Parts one and two are background: the words the field uses …
# II · Key papers and their methods
- … and the papers that matter, including how their methods work.
# III · Our method, piece by piece
- Part three is our method step by step, with the math.
# IV · Why each choice
- Part four goes through every design choice and shows what we tried instead.
# V · Results
- Then results.
# VI · Against published work
- Then a careful comparison with published work.
# VII · Honest limits and next steps
- And finally the limits of what we can claim.""")

    # ================================================================== PART I
    d.part = "I · Vocabulary"
    d.add("p1", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART I</div>
<h1>The field's vocabulary</h1><p>Every term used later, defined once, with the name the literature uses.</p></div>""",
          raw=True, minutes=0.2, say="# The field's vocabulary\n- Part one: the vocabulary.")

    # ---- medical VQA
    ex = D.EX
    d.add("medvqa", "Medical visual question answering: two answer formats",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 300px"><img src="data:image/jpeg;base64,{F.IMG64}" style="width:300px;border-radius:6px">
  <div class="note">SLAKE, abdominal CT (question {ex['idx']}).</div></div>
 <div class="col">
  <div class="term"><b class="k">Closed-ended</b><span>answer from a fixed set: yes/no, or a letter among options
  (multiple choice). Grading is trivial: string equality.</span>
  <span class="alt">“Is the liver visible?” → yes</span></div>
  <div class="term t"><b class="k">Open-ended (free text) — this work</b><span>the model writes the answer. Many correct
  strings exist (“large bowel”, “colon”, “large intestine”), so grading needs either exact match after normalisation or a
  grader model (LLM-as-a-judge).</span>
  <span class="alt">“{ex['question']}” → gold: {ex['gold']}</span></div>
  <div class="term v"><b class="k">Why open-ended</b><span>it is what a clinician-facing system faces, and it is where
  selection signals work: in this project, error detection reached AUROC ≈ 0.87 on free text vs ≈ 0.6 on 4-option
  multiple choice.</span></div>
 </div>
</div>""", "I · Vocabulary", src="AUROC figures: project finding 2 (PROJECT_RETROSPECTIVE_2026-07-29.md §5.2), from the earlier multiple-choice phase",
          minutes=1.5,
          say="""# Medical visual question answering: two answer formats
- Visual question answering, VQA: the model gets an image and a question and has to answer.
- In medicine there are two answer formats.
# Closed-ended
- The answer comes from a fixed set — yes or no, or a letter from multiple choices.
- That is easy to grade.
# Open-ended (free text) — this work
- The model writes the answer itself, like the example on the right.
- The gold answer is "Large Bowel", but "colon" or "large intestine" would also be right.
- So grading open answers is itself a problem; I will come back to how we do it.
# Why open-ended
- Two reasons. First, it is the realistic setting.
- Second, in an earlier phase of this project, signals about whether an answer is right worked much better on free text than on four-option multiple choice.""",
          qa=[("Where does the 0.87 vs 0.6 come from?", "From the earlier multiple-choice phase of this project (finding 2 in the project retrospective): "
               "routing/error-detection signals reached AUROC ≈ 0.6 on 4-option MCQ and ≈ 0.87 on free text. It motivated the switch, it is not a result of this method.")])

    # ---- VLM anatomy
    spec = {"img_note": "sides × 28 px", "patch": "14 × 14 px patches",
            "vit": "32 layers · width 1280", "merger": "2 × 2 patches → 1 token",
            "merger2": "2-layer MLP → 3584-d",
            "tokens_note": "one image token per 28 × 28 px  ⇒  ≤ max_pixels / 784 tokens  (≤ 320 at our cap)",
            "llm": "28 blocks · hidden size 3584 · M-RoPE positions"}
    d.add("vlm", "How a VLM answers: Lingshu-7B (Qwen2.5-VL-7B architecture)",
          f"""{F.fig_vlm(spec)}
<div class="row" style="height:auto;margin-top:8px">
 <div class="col"><div class="term"><b class="k">vision encoder (ViT)</b><span>turns image patches into vectors (window attention)</span></div></div>
 <div class="col"><div class="term"><b class="k">merger / projector</b><span>maps them to the language model's width — each becomes an <i>image token</i></span></div></div>
 <div class="col"><div class="term"><b class="k">decoder (LLM)</b><span>reads system + image + question tokens, writes the answer one token at a time</span></div></div>
 <div class="col"><div class="term v"><b class="k">hidden state</b><span>the 3,584-number vector at one position after one block (the <i>residual stream</i>)</span></div></div>
</div>""", "I · Vocabulary", tall=True,
          src="Qwen2.5-VL report Table 1 p.4, §2.1 p.3–5; Lingshu §3 p.9 (built on Qwen2.5-VL-7B-Instruct); token bound derived (14-px patches, 2×2 merge)",
          minutes=2.0,
          look="Left to right along the figure: image, patches, ViT, merger, the token strip, then down to the decoder. End on the violet 'hidden state' card.",
          say="""# How a VLM answers: Lingshu-7B (Qwen2.5-VL-7B architecture)
- This is how our model works.
- Lingshu-7B is a medical fine-tune of Qwen2.5-VL-7B and has exactly its architecture.
# vision encoder (ViT)
- The image is cut into 14-by-14-pixel patches.
- A vision transformer with 32 layers turns each patch into a vector.
# merger / projector
- A small two-layer network, the merger or projector, groups each 2-by-2 block of patches into one vector of size 3,584 — the width of the language model.
- So one image token covers a 28-by-28-pixel region.
- An image capped at our budget of about 250 thousand pixels becomes at most 320 image tokens.
# decoder (LLM)
- The language model reads the system prompt, the image tokens and the question.
- It writes the answer one token at a time.
- It has 28 blocks; after every block, each position has a vector of 3,584 numbers.
# hidden state
- That vector is called the hidden state, or the residual stream.
- Those vectors are what our probe reads — so keep this picture in mind.""",
          qa=[("Is the 320-token bound stated in the Qwen paper?", "No — it is derived: 14-px patches merged 2×2 give one token per 28×28 px, so tokens ≤ max_pixels/784 = 250,880/784 = 320. "
               "Measured: GEMeX averages 303 image tokens at cap320 (FINAL_METHOD_2026-10-16.md)."),
              ("Was Lingshu trained with reinforcement learning?", "No. The released Lingshu-7B is the three supervised stages; RL produced a separate Lingshu-RL "
               "that we do not use (Lingshu report, Fig. 6 caption and §3.1.4).")])

    # ---- decoding and temperature
    xs = [0.2, 0.4, 0.7, 1.0]
    chart = lines(xs, [("oracle@8", C["indigo_l"], D.T_ORACLE)], 0.40, 0.60, 0.05, w=700, h=232, xlab="sampling temperature T",
                  ylab="oracle@8 (judge)", value_fmt=lambda v: f"{v:.3f}")
    chart2 = lines(xs, [("distinct answers", C["teal"], D.T_DISTINCT)], 1.5, 5.5, 1.0, w=700, h=232, xlab="sampling temperature T",
                   ylab="distinct of 8")
    d.add("decoding", "Decoding: greedy vs sampling, and what temperature does",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 420px">
  <div class="term"><b class="k">greedy decoding</b><span>always take the most likely next token. One answer, deterministic. Our baseline.</span></div>
  <div class="term t"><b class="k">temperature sampling</b><span>draw the next token at random from
  $$p_i=\\frac{{e^{{u_i/T}}}}{{\\sum_j e^{{u_j/T}}}}$$ \\(u_i\\) = the model's logit for token i. T → 0 is greedy; larger T spreads probability, so repeated samples differ.
  Ours: plain temperature sampling, T = 0.7, max 64 tokens; no top-p / top-k set (run_openvqa.py).</span></div>
  <div class="term w"><b class="k">not to be confused with</b><span><i>temperature scaling</i>, a calibration method that rescales a classifier's logits after training. Unrelated.</span></div>
 </div>
 <div class="col" style="gap:4px">
  <div class="note"><b>distinct answers among 8 samples</b> (macro mean)</div>{chart2}
  <div class="note"><b>oracle@8</b> — at least one of the 8 is correct (judge, macro)</div>{chart}
  <div class="note">Both y-axes are truncated (they do not start at 0).</div></div>
</div>""", "I · Vocabulary",
          src="head_temp_ensemble_2026-08-30.json — cells.*.{T}.mean_distinct and .oracle, macro over 8 benchmarks, held-out image halves, N = 8",
          minutes=2.0,
          look="Top chart first (distinct answers rise with T), then the bottom chart (coverage rises with T). Point at 0.7 on both.",
          say="""# greedy decoding
- To get several different answers we cannot use greedy decoding.
- Greedy always takes the most likely next token, so it gives the same answer every time.
# temperature sampling
- Instead we sample: the model's scores for the next token become probabilities through this softmax.
- The temperature T controls how spread out they are.
- At low temperature the samples are almost identical; at high temperature they vary more.
# distinct answers among 8 samples (top chart)
- The two charts are our own measurements over the eight benchmarks.
- Top: distinct answers among eight samples rise from about 2 at temperature 0.2 to about 5 at 1.0.
# oracle@8 — at least one of the 8 is correct (bottom chart)
- Bottom: the chance that at least one of the eight is correct rises from 0.46 to 0.57.
- Note that neither axis starts at zero.
- The price, typically, is that each individual sample is less accurate — that is not shown here.
- That trade-off is why the temperature matters.
- We ship one temperature, 0.7, for every question.
- An earlier version picked a different temperature per benchmark; we dropped it.
  - It needs to know which dataset a question comes from, which a deployed system does not.
  - I come back to this in part four.
# not to be confused with
- "Temperature scaling" in the calibration literature is a different thing — it rescales a trained classifier's outputs.
- When I say temperature, I always mean the sampling temperature.""",
          qa=[("Any top-p, top-k or repetition penalty?", "None set: SamplingParams gets temperature, max_tokens = 64, n and logprobs only "
               "(src/labeling/run_openvqa.py:186). Whether vLLM 0.9.0.1 applies the model's own generation_config defaults when SamplingParams is "
               "built by hand was not separately verified."),
              ("Is the temperature chosen per benchmark?", "Not in the shipped method: one T = 0.7 for all questions. A per-benchmark map was adopted in "
               "FINAL_METHOD_2026-10-16.md and reversed on 2026-10-18, because it needs each question's dataset identity at test time; it is kept as a "
               "labelled backup variant. The probe itself needs no dataset label.")])

    # ---- best-of-N, coverage, selection
    d.add("bestofn", "Best-of-N: coverage and selection are two different problems",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 560px">{fit_svg(F.fig_bestofn(D.GREEDY_MACRO, D.PICK_MACRO, D.ORACLE_MACRO), 560)}</div>
 <div class="col">
  <div class="term"><b class="k">best-of-N</b><span>sample N answers, return one chosen by a scoring rule (a.k.a. repeated sampling, n-best reranking).</span></div>
  <div class="term"><b class="k">oracle@N  = pass@N = coverage</b><span>share of questions with ≥ 1 correct answer among the N — the ceiling for any selector.</span></div>
  <div class="term t"><b class="k">selection</b><span>finding it without the gold. Brown: <i>precision</i>; Bay: the <i>identifiability gap</i>.</span></div>
  <div class="term v"><b class="k">selection efficiency (ours)</b><span>selected ÷ oracle@N <b>per benchmark, then averaged</b>: <b>{D.SELEFF:.2f}</b>
  (ratio of the two macro bars: {R1['ratio_of_macro_pick_to_macro_oracle_judge']['value']:.2f}). Not a standard name.</span></div>
 </div>
</div>""", "I · Vocabulary",
          minutes=2.0,
          look="The three bars left to right (greedy, probe pick, oracle), then the two brackets on the right: red = coverage wall, orange = selection gap.",
          say=f"""# best-of-N
- This is the central idea of our field.
- Best-of-N: sample N answers and return one of them.
- It splits into two separate problems.
# oracle@N  = pass@N = coverage
- First, coverage: is a correct answer anywhere among the N?
- Measured by oracle-at-N — the accuracy you would get if someone told you which answer was right.
# selection
- Second, selection: can you find the right one without knowing the answer?
# the bars (greedy · probe pick · oracle)
- These are our real numbers, averaged over the eight benchmarks.
- Greedy decoding is right {D.GREEDY_MACRO:.3f} of the time.
- A correct answer exists among our eight samples {D.ORACLE_MACRO:.3f} of the time — that is the ceiling.
- Our probe picks a correct answer {D.PICK_MACRO:.3f} of the time.
- Red bracket: {1-D.ORACLE_MACRO:.0%} of questions are lost because no sample is right — the coverage wall.
- Orange bracket: {D.ORACLE_MACRO-D.PICK_MACRO:.3f} is lost to imperfect selection.
# selection efficiency (ours)
- The ratio of picked to oracle we call selection efficiency.
- Dividing the two bars gives {R1['ratio_of_macro_pick_to_macro_oracle_judge']['value']:.2f}.
- We compute it benchmark by benchmark and then average, which gives {D.SELEFF:.2f}.
- The difference: one weak benchmark, VQA-Med, pulls the average down.""",
          src=f"final_eval_FINAL_cap320_T07.json — cells.*.all.{{greedy,pick,oracle}}_judge, macro; selection efficiency per benchmark: {R1F}; Brown et al. 2024 p.2; Bay & Yearick 2026 abstract",
          qa=[("Is selection efficiency a standard metric?", "No. Brown et al. call the problem 'precision', Hu (2026) uses 'effective capture' = fixed/recoverable, "
               "which is a different normalisation. Ours is selected/oracle@N per benchmark, then averaged; we always define it."),
              ("Per benchmark?", "pick ÷ oracle@8 under the judge: " + ", ".join(f"{D.SHORT[c]} {v:.2f}" for c, v in R1['mean_of_per_benchmark_pick_over_oracle_judge']['per_benchmark'].items())
               + f" ({R1F})."),
              ("What is the coverage wall per benchmark?", "1 − oracle@8 ranges from 0.12 (SLAKE) to 0.79 (VQA-Med) — see the per-benchmark results slide.")])

    # ---- selector taxonomy
    d.add("selectors", "Who chooses? The selectors the field uses",
          """
<table class="d">
<tr><th></th><th>signal</th><th>trained?</th><th>examples</th><th>status here</th></tr>
<tr><td><b>majority vote / self-consistency</b></td><td>most frequent answer string</td><td>no</td><td>Wang et al. 2023</td><td><span class="tag r">loses to greedy</span></td></tr>
<tr><td><b>LLM consistency / semantic agreement</b></td><td>model or embedding picks the most "central" answer</td><td>no</td><td>USC 2023, EBA 2026, MBR</td><td><span class="tag w">not run</span></td></tr>
<tr><td><b>likelihood / logit scores</b></td><td>model's own log-probability or self-certainty of the answer</td><td>no</td><td>sample-and-rank, self-certainty (Kang 2025)</td><td><span class="tag w">not run</span></td></tr>
<tr><td><b>self-verification</b></td><td>ask the same model "is this correct?" (P(True))</td><td>no</td><td>Kadavath 2022; Verification Mirage 2026</td><td><span class="tag r">failed (July)</span></td></tr>
<tr><td><b>game-theoretic decoding</b></td><td>generator & verifier roles of one model reach agreement</td><td>no</td><td>Consensus Game 2023 → BDG 2024 → WED 2026</td><td><span class="tag g">head-to-head</span></td></tr>
<tr class="hl"><td><b>trained verifier / reward model</b></td><td>a model trained on correct/incorrect labels</td><td>yes</td><td>Cobbe 2021, GenRM, HSRM, SWIFT</td><td><span class="tag t">ours</span></td></tr>
</table>
<div class="row" style="height:auto;margin-top:14px">
 <div class="col"><div class="term"><b class="k">ORM vs PRM</b><span><i>outcome</i> reward model scores the final answer; <i>process</i> reward model scores each reasoning step. Ours: ORM (answers are ~5 tokens, no steps).</span></div></div>
 <div class="col"><div class="term"><b class="k">discriminative vs generative</b><span>a classifier outputs a score (ours) vs an LLM writes "Yes/No" (GenRM).</span></div></div>
 <div class="col"><div class="term t"><b class="k">our place</b><span>a <b>discriminative outcome verifier</b> — implemented as a <b>probe</b> on frozen hidden states.</span></div></div>
</div>""", "I · Vocabulary",
          src="ORM/PRM: Uesato et al. arXiv:2211.14275; Lightman et al. arXiv:2305.20050; Snell et al. 2024; Zhang et al. 2025 GenRM; Kang et al. arXiv:2502.18581; "
              "Jacob et al. arXiv:2310.09139; BDG arXiv:2410.01064; status column from final_eval_FINAL_cap320_T07.json and PROJECT_RETROSPECTIVE N32",
          minutes=2.0,
          say="""# Who chooses? The selectors the field uses
- Who does the choosing? The field has roughly six families — one per row.
# majority vote / self-consistency
- The simplest: return the answer string that appears most often.
# LLM consistency / semantic agreement
- Training-free extensions for free text: ask the model which answer is most consistent, or cluster answers by meaning.
# likelihood / logit scores
- Rank by the model's own likelihood.
# self-verification
- Ask the model to check itself.
# game-theoretic decoding
- One recent medical paper, WED, makes two roles of the same model negotiate.
# trained verifier / reward model
- A model trained on labelled correct and incorrect answers. That is our family.
# ORM vs PRM
- Two distinctions the literature uses. First, outcome versus process.
- An outcome reward model scores the final answer; a process reward model scores every reasoning step.
- Lightman and colleagues: process supervision wins on long math solutions.
- Uesato and colleagues: outcome supervision reaches similar final-answer error with less labelling.
- Our answers are about five tokens long with no steps, so ours is outcome.
# discriminative vs generative
- A discriminative verifier is a classifier that outputs a number.
- A generative one is a language model that writes "yes" or "no".
# our place
- Ours is a discriminative outcome verifier, built as a probe on the model's hidden states.
# status here
- This column already tells you something about results.
- Majority vote actually loses to greedy on our data.
- Asking the model to check itself failed when we tried it earlier.""",
          qa=[("Which training-free selectors were not run, and why does it matter?",
               "USC, embedding agreement (EBA), MBR, and likelihood-based ranking were not run on our pools. Per-candidate log-probabilities were not stored "
               "for the shipped cap320 pools; they exist for the October 1-megapixel-cap pools on 7 of 8 benchmarks (cumlogprob_all, src/labeling/run_openvqa.py:235–241; "
               "RadImageNet has none), so a likelihood-ranking baseline can be run there on CPU — it is first on the owed list. "
               "Self-certainty (Kang et al., NeurIPS 2025), the strongest logit-only best-of-N score for open-ended output, was not run either. "
               "These are honest gaps — a reviewer may ask whether labels are needed at all; the answer-prior and vote comparisons partly address it."),
              ("Why an outcome scorer, when Lightman 2023 and Snell found process reward models better?",
               "Process supervision wins on multi-step math ('process supervision significantly outperforms outcome supervision', Lightman, arXiv:2305.20050). "
               "Our answers are ~5 tokens with no steps, so a process reward has nothing to score; Uesato 2022 found 'pure outcome-based supervision produces "
               "similar final-answer error rates with less label supervision' (arXiv:2211.14275). We claim nothing about long-chain settings."),
              ("When was self-verification tested?", "In July with the earlier LoRA-era pools: a zero-shot P(True)-type selector scored below random pick (0.715 vs 0.720 selection efficiency) "
               "and below greedy (0.319 vs 0.343). Not re-run on the final pools (PROJECT_RETROSPECTIVE N32).")])

    # ---- probing
    d.add("probing", "What a probe is — and what it does not prove",
          r"""
<div class="row">
 <div class="col">
  <div class="term"><b class="k">probe</b><span>small classifier on a <b>frozen</b> network's internal vectors</span><span class="alt">Alain & Bengio 2016; Q-Probe §2</span></div>
  <div class="term"><b class="k">linear probe</b><span>\( \sigma(w^\top h + b) \) — the diagnostic standard</span></div>
  <div class="term t"><b class="k">MLP probe (ours)</b><span>\( w_2^\top\,\mathrm{GELU}(W_1 h + b_1) + b_2 \) — one hidden layer, width 256</span></div>
  <div class="term v"><b class="k">correctness is decodable — an old finding</b><span>Azaria & Mitchell · CCS · Kadavath: one statement each, not N answers</span></div>
 </div>
 <div class="col">
  <div class="term w"><b class="k">decodable ≠ causal</b><span>readable from the vector ≠ used by the model (Belinkov 2022)</span></div>
  <div class="term w"><b class="k">control task & selectivity</b><span>real accuracy − accuracy on random labels (Hewitt & Liang 2019)</span>
  <span class="alt">not yet run for our probe — on the limits slide</span></div>
  <div class="term r"><b class="k">transfer</b><span>“error detectors fail to generalize across datasets” (Orgad 2024)</span></div>
  <div class="term v"><b class="k">our use is operational</b><span>we judge the probe only by the accuracy of the answers it chooses</span></div>
 </div>
</div>""", "I · Vocabulary", src="Q-Probe arXiv:2402.14688 §2 p.2; Alain & Bengio arXiv:1610.01644; Hewitt & Liang EMNLP 2019; Belinkov, Computational Linguistics 48(1) 2022; "
              "Azaria & Mitchell arXiv:2304.13734; Burns et al. arXiv:2212.03827 (ICLR 2023); Kadavath et al. arXiv:2207.05221; Orgad et al. arXiv:2410.02707 (abstracts, via guide cards)",
          minutes=2.4,
          qa=[("Hidden states encoding correctness is old (Azaria & Mitchell, CCS, Orgad). What is different?",
               "Decodability is old and we cite it. Those works classify one statement in isolation; we rank the N answers to the same question, which is "
               "a harder within-question task — CASE shows AUC falling from 0.610 to 0.502 on LogiQA once question identity is removed. We have not yet "
               "reported within-question AUROC for our probe; it is on the gap list."),
              ("Your probe fails to transfer to a left-out benchmark. Is that expected?",
               "It matches Orgad et al. ('such error detectors fail to generalize across datasets'). Kossen et al.'s semantic-entropy probes 'generalize better "
               "to out-of-distribution data than previous probing methods that directly predict model accuracy' — an auxiliary target worth trying; untested here.")],
          say="""# probe
- The core object of our method is a probe.
- In the interpretability literature: a small classifier that takes a network's internal vectors as input and is trained to predict some property.
- The network itself is frozen — never changed.
# linear probe
- The classic version: one weighted sum and a sigmoid.
# MLP probe (ours)
- Ours has one hidden layer of width 256, which makes it an MLP probe.
- A note on words: say "one hidden layer of width 256", not "a two-layer MLP" — people count layers differently.
# correctness is decodable — an old finding
- That hidden states carry information about whether a statement is true is an old finding.
- Azaria and Mitchell read true or false from a model's activations with 71 to 83 percent accuracy.
- The contrast-consistent search of Burns and colleagues found it without labels, about four percent over zero-shot prompting.
- Kadavath and colleagues at Anthropic trained models to predict whether they know an answer.
- But each of them scores one statement on its own.
- We need something narrower: among eight answers to the same question, which is right.
# transfer
- Orgad and colleagues warn that such detectors often fail to carry over to a new dataset.
- We see the same thing, as I will show.
# decodable ≠ causal
- Two cautions from that literature. First: being able to decode something from a vector does not mean the model uses it.
- It is correlation, not mechanism.
# control task & selectivity
- Second: a probe with enough capacity can memorise labels.
- The careful check is a control task: train the same probe on random labels and measure the difference, called selectivity.
- We have not run that control yet; I list it among the gaps.
# our use is operational
- We do not claim anything about how the model thinks.
- We use the probe as a tool to choose an answer.
- We judge it only by how often the chosen answer is right on images it never saw.""")

    # ---- grading
    d.add("judges", "Grading open answers: LLM judges and exact match",
          """
<div class="row">
 <div class="col">
  <div class="term"><b class="k">LLM-as-a-judge</b><span>reference-guided single-answer grading: question + gold + one candidate → correct / incorrect</span><span class="alt">MT-Bench §3.1</span></div>
  <div class="term"><b class="k">lenient exact match (“EM”)</b><span>equal <b>or either string contains the other</b>, after normalisation</span></div>
  <div class="term"><b class="k">token-F1, token recall</b><span>word-set overlap; token recall = LLaVA-Med's open metric</span></div>
 </div>
 <div class="col">
  <table class="d"><tr><th>judge</th><th>role</th><th>family</th></tr>
  <tr class="hl"><td>MedVLThinker-32B</td><td><b>training judge</b>: gave the probe's labels</td><td>Qwen2.5-VL (Lingshu's lineage)</td></tr>
  <tr><td>MedGemma-27B-it</td><td><b>cross-family judge</b>: evaluation only</td><td>Gemma 3</td></tr></table>
  <div class="note" style="margin-top:6px">Both: text-only, see the gold, temperature 0, correct iff P(“Yes”) ≥ P(“No”).</div>
  <div class="term w" style="margin-top:10px"><b class="k">known judge biases</b><span>position · verbosity (checked) · self-preference → cross-family judge</span></div>
 </div>
</div>""", "I · Vocabulary",
          src="MT-Bench arXiv:2306.05685 §3; src/labeling/run_judge.py (MedGemma via the same script); EM/F1: src/audit_2026_09_18/em_rescore_pooled_probe.py; "
              "length bias: AUDIT_2026-09-18.md §2.2 — full provenance in the script",
          minutes=2.0,
          say="""# LLM-as-a-judge
- Open answers can be right in many wordings, so we need a grader.
- The field's standard is an LLM judge: it sees the question, the gold answer and one candidate, and says correct or incorrect.
- In the MT-Bench paper's vocabulary: reference-guided single-answer grading.
# judge · role · family (table)
- We use two judges, and they play different roles.
- MedVLThinker-32B produced the probe's training labels, so I call it the training judge.
  - It comes from the same Qwen family as our generator; a probe trained on its labels can learn its taste.
- So every headline number is also graded by MedGemma-27B, from Google's Gemma family, never used in training — the cleaner number.
- Both judges read text only; they do not see the image.
# lenient exact match (“EM”)
- We also report a lenient string-matching score: right if the answer equals the gold or either contains the other.
- Lenient means a partial answer contained in the gold also passes, and a correct paraphrase fails.
# token-F1, token recall
- And LLaVA-Med's token recall, so we can compare with their paper.
- In our files we call these different metrics "currencies"; from now on I just say metric.
- I never show a judge number without the others.
# known judge biases
- Position bias needs two answers side by side, so it cannot happen here.
- Verbosity bias — preferring longer answers — we checked: always picking the longest answer loses 0.029.
- Self-preference and learning the training judge's taste are why we have the second judge.
- This is also the field's norm: each benchmark paper uses its own LLM judge.
  - GPT-4.1 for Lingshu, Mistral-Large for RadImageNet-VQA, Qwen3-30B for Kvasir-VQA-x1, GPT-4o for GEMeX.""",
          qa=[("Has the judge been validated against humans?", "No human has rated this method's picks. Earlier spot checks, on July pools: an LLM rater "
               "(results/cascade_methods/claude_judge/) on SLAKE and VQA-RAD, and an audit of 50 + 90 PathVQA items (pathvqa_judge_audit.json). On the 50 "
               "direct (short) answers the judge was wrong 2 times: 0.04, Wilson 95% [0.011, 0.135] (computed for 2/50); on 90 long reasoning-mode answers "
               "it erred more (up to 0.28 on rejections). Who labelled them is not recorded, so we do not call it human. Nothing at all on the other 5 benchmarks. "
               "The cross-family judge agreeing in sign is our main defence; a human rating is listed as a gap."),
              ("How often do the two judges agree?", "On September training candidates (112,770 rows) the two judges agree on 92.9% of labels, Cohen's κ = 0.79 (judge_2x2_2026-09-20.json train_label_agreement_AB and train_label_pos_rate; κ from the marginals, numbers_review_part2.json judge_agreement). MedGemma says 'correct' more often (23.9% vs 19.4%). Not yet measured on the October final pools. On the final method, the two judges' per-benchmark deltas agree in sign on 7 of 8 benchmarks; on VQA-RAD the training judge gives "
               "+0.0000 and MedGemma +0.0100 (final_eval_FINAL_cap320_T07.json cells.*.all)."),
              ("How does the judge decide?", "Deterministically: temperature 0, text-only prompt in run_judge.py; correct iff the first-token probability "
               "of 'Yes' is at least that of 'No'. MedGemma-27B-it runs through the same script and prompt."),
              ("Why was MedVLThinker-32B chosen as judge?", "It was the labelling script's default, not a deliberate choice (AUDIT_2026-09-18.md §1). It is text-only and sees the gold answer.")])

    # ---- uncertainty vs selection
    d.add("uncert", "Related but different: uncertainty estimation is not selection",
          """
<table class="d">
<tr><th></th><th>confidence / uncertainty estimation</th><th>best-of-N selection (ours)</th></tr>
<tr><td>question asked</td><td>“is <i>this</i> answer (or question) likely wrong?”</td><td>“which of the N answers to this question is right?”</td></tr>
<tr><td>unit scored</td><td>one answer, or the question itself</td><td>every candidate of one question</td></tr>
<tr><td>what matters</td><td>ranking across questions (AUROC) and calibration (ECE)</td><td>ranking within one question only</td></tr>
<tr><td>examples</td><td>semantic entropy (Farquhar, Nature 2024), semantic-entropy probes (SEPs, Kossen 2024), verbalised confidence; P(True) is per answer, so it <i>can</i> rank — that is the self-verification selector</td><td>verifiers, reward models, our probe</td></tr>
</table>
<div class="row" style="height:auto;margin-top:14px">
 <div class="col"><div class="term"><b class="k">semantic entropy cannot pick</b><span>it is one number per question, identical for all 8 candidates. Its only per-candidate quantity is cluster size — a majority vote by meaning.</span></div></div>
 <div class="col"><div class="term"><b class="k">calibration cannot reorder</b><span>Platt / temperature scaling are monotone: they change the numbers, never which candidate is first (Byun et al. 2026, Table 2).</span></div></div>
 <div class="col"><div class="term t"><b class="k">shared mechanism</b><span>hidden-state probes appear on both sides; ours is the per-candidate correctness variant used as a within-question ranker.</span></div></div>
</div>""", "I · Vocabulary", src="Farquhar et al. Nature 630 (2024) Eq. 3 p.8; Kossen et al. arXiv:2406.15927; Byun et al. arXiv:2604.02543 Table 2 p.7; Geng et al. NAACL 2024",
          minutes=1.5,
          qa=[("If calibration cannot reorder, why test Platt averaging later?", "Calibrating one probe cannot reorder its own scores. Calibrating each of "
               "the 24 probes before averaging them can change the order of the average. That is what the readout comparison tested, and it tied under both judges."),
              ("Isn't P(True) an uncertainty method?", "It scores one answer at a time, so it can rank candidates — used that way it is the "
               "self-verification selector, which failed on our earlier pools (0.715 vs 0.720 random-pick selection efficiency; PROJECT_RETROSPECTIVE N32).")],
          say="""# confidence / uncertainty estimation (left column)
- One neighbouring field is easy to confuse with ours: uncertainty estimation.
- Those methods ask "is this answer probably wrong?" — one score per answer or per question.
- They are judged by how well they separate right from wrong across many questions.
# best-of-N selection (ours) (right column)
- We ask a different question: among the eight answers to one question, which is right?
- Only the order within one question matters.
# semantic entropy cannot pick
- That difference has consequences.
- Semantic entropy, the best-known uncertainty method, gives one number per question, the same for all eight candidates.
- So it cannot choose between them.
# calibration cannot reorder
- Calibration methods only stretch the scores, so they never change which answer is first.
# shared mechanism
- The shared ingredient: hidden-state probes appear in both areas.
- Ours is the variant that scores each candidate's correctness.""")

    # ---- glossary summary
    d.add("glossary", "Glossary — the terms as used in this talk",
          """
<table class="d xs" style="columns:2">
<tr><th>term</th><th>meaning here</th><th>term</th><th>meaning here</th></tr>
<tr><td>VLM / LVLM</td><td>vision-language model (image + text in, text out)</td><td>probe</td><td>small classifier on a frozen model's internal vectors</td></tr>
<tr><td>greedy decoding</td><td>always the most likely token; our baseline answer</td><td>verifier / reward model</td><td>a scorer of candidate answers</td></tr>
<tr><td>sampling temperature T</td><td>spread of the next-token distribution (we use 0.7)</td><td>ORM / PRM</td><td>scores the final answer / each reasoning step</td></tr>
<tr><td>best-of-N</td><td>sample N, return one chosen by a scorer</td><td>class-weighted BCE</td><td>binary cross-entropy with correct answers up-weighted by (1−π)/π ≈ 3.8 — the probe's loss</td></tr>
<tr><td>oracle@N (pass@N, coverage)</td><td>share with ≥1 correct sample</td><td>rank averaging</td><td>average each probe's within-question rank (a Borda count)</td></tr>
<tr><td>selection efficiency</td><td>selected ÷ oracle@N, per benchmark, then averaged (ours)</td><td>teacher-forced pass</td><td>one forward pass over prompt + a given answer to read its states; no sampling</td></tr>
<tr><td>hidden state / residual stream</td><td>3,584-number vector at a position after a block</td><td>deep (seed) ensemble</td><td>the same probe trained from 8 random seeds; here also across 3 layers</td></tr>
<tr><td>layer ℓ</td><td>output of decoder block ℓ (of 28)</td><td>z-score</td><td>subtract the training mean, divide by the training std, per dimension</td></tr>
<tr><td>mean pooling</td><td>average over the answer's token positions</td><td>LLM-as-a-judge</td><td>grader model that compares answer and gold</td></tr>
<tr><td>cap320</td><td>image budget max_pixels = 250,880 (≤ 320 image tokens)</td><td>training / cross-family judge</td><td>MedVLThinker-32B (gave the labels) / MedGemma-27B (evaluation only)</td></tr>
<tr><td>1-megapixel cap</td><td>max_pixels = 1,003,520 (≤ 1,280 image tokens); “full resolution” in our files</td><td>lenient EM</td><td>equal or containment match after normalisation</td></tr>
<tr><td>self-consistency</td><td>majority vote over samples</td><td>metric (“currency”)</td><td>which grader a number is measured in</td></tr>
<tr><td>cross-fitting</td><td>score each image with probes trained without it (= 2-fold grouped CV, grouped by image)</td><td>clustered bootstrap</td><td>CI resampling images, not questions</td></tr>
<tr><td>half A / half B</td><td>the two image halves; fold 0 scores half A, fold 1 scores half B</td><td>macro</td><td>equal weight per benchmark</td></tr>
<tr><td>FLOP-eq</td><td>compute in units of one greedy Lingshu-7B answer at cap320</td><td>harness</td><td>how a cost was measured (vLLM shared prefill vs HF separate passes)</td></tr>
</table>""", "I · Vocabulary", minutes=0.5,
          src="π = share of correct training rows 0.2071 / 0.2085 → (1−π)/π = 3.83 / 3.80 (numbers_review_part1.json, review/scr/w1_pi.py); head_sweep.py:172–174; "
              "FLOP-eq unit: free_head_2026-08-16.json DELIVERABLE_3_cost.unit",
          say="""# Glossary — the terms as used in this talk
- This slide collects the terms in one place.
- I will not read it; it is here for reference while we go through the method.""")

    # ================================================================== PART II
    d.part = "II · Key papers"
    d.add("p2", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART II</div>
<h1>Key papers and their methods</h1><p>186 papers catalogued (194 cards in the domain guide). These are the ones the method and the comparison rest on, read in full text.</p></div>""",
          raw=True, minutes=0.2, src="domain guide cards_*.json: 194 cards, 186 distinct papers (review/scr/w1_count.py)", say="# Key papers and their methods\n- Part two: the papers this work stands on, and how their methods work.")

    # ---- lineage
    d.add("lineage", "The idea is old: rerank a generator's candidates with a trained scorer",
          """
<svg viewBox="0 0 1190 500" width="1190" height="500">
 <line x1="30" y1="245" x2="1170" y2="245" stroke="#cdd4e3" stroke-width="2"/>
 """ + "".join(
              f'<circle cx="{x}" cy="245" r="8" fill="{col}"/><text x="{x}" y="{y}" text-anchor="middle" font-size="14.5" font-weight="700" fill="#1d2030">{a}</text>'
              f'<text x="{x}" y="{y+20}" text-anchor="middle" font-size="13" fill="#5b6472">{b}</text><text x="{x}" y="{y+38}" text-anchor="middle" font-size="13" fill="#5b6472">{c}</text>'
              f'<line x1="{x}" y1="{245 + (-9 if y < 245 else 9)}" x2="{x}" y2="{y + (48 if y < 245 else -18)}" stroke="#cdd4e3"/>'
              for x, y, a, b, c, col in [
                  (85, 60, "Collins & Koo 2005", "parsing: rerank n-best", "parses, boosting", "#9aa1ad"),
                  (210, 330, "CEM 2020 · R-EBM 2021", "speech: trained scorers", "on frozen decoder features", "#9aa1ad"),
                  (335, 60, "Cobbe · WebGPT 2021", "trained verifier best-of-100;", "reward model best-of-64", "#5c6bc0"),
                  (460, 330, "AlphaFold 2021", "best of 5 structures", "by predicted confidence", "#9aa1ad"),
                  (585, 60, "Q-Probe 2024", "linear probe reranks", "k samples (code)", "#5c6bc0"),
                  (710, 330, "SWIFT · LiLaVe 2025", "hidden-state reward", "models (math)", "#5c6bc0"),
                  (830, 60, "HSRM · CASE · VLA 2026", "readers on hidden states;", "VLA: probe selects actions", "#5c6bc0"),
                  (955, 330, "MedProb · DualRead 2026", "medical VLM probes: MCQ,", "confidence; App. H: 100 items", "#7a5cc7"),
                  (1100, 60, "this work", "probe verifier, medical", "VLM, open, 8 benchmarks", "#00897b")]) + "".join(
              f'<circle cx="{lx}" cy="470" r="7" fill="{col}"/><text x="{lx+14}" y="475" font-size="13.5" fill="#5b6472">{name}</text>'
              for lx, name, col in [(300, "reranking before LLMs", "#9aa1ad"), (520, "trained scorers on LLMs", "#5c6bc0"),
                                    (740, "medical VLMs", "#7a5cc7"), (900, "this work", "#00897b")]) + """
</svg>""", "II · Key papers",
          src="Collins & Koo, Comp. Ling. 31(1) 2005; Li et al. arXiv:2010.11428 (CEM), 2103.14152 (R-EBM); Cobbe arXiv:2110.14168; Nakano arXiv:2112.09332; Jumper, Nature 2021; "
              "arXiv:2402.14688, 2505.12225 (SWIFT, KDD 2026), 2504.16760, 2608.30841, 2608.17124, 2605.28527 (VLA probe), 2609.04336, 2609.06419",
          minutes=1.5,
          look="Walk the timeline left to right, alternating top and bottom labels; the colour legend at the bottom groups them.",
          say="""# The idea is old: rerank a generator's candidates with a trained scorer
- The core idea is much older than large language models.
# Collins & Koo 2005
- In 2005 Collins and Koo took a parser's list of candidate parses and trained a second model to rerank them.
# CEM 2020 · R-EBM 2021
- Speech recognition did related things: a trained scorer reads the frozen recogniser's internal features.
  - To estimate confidence, in 2020.
  - To rescore the list of candidate transcripts, in 2021.
# Cobbe · WebGPT 2021
- Cobbe and colleagues brought trained verifiers to language models in 2021.
- The same year, WebGPT used a learned reward model to pick the best of up to 64 long free-text answers.
# AlphaFold 2021
- AlphaFold picks the best of five protein structures by its own predicted confidence.
# Q-Probe 2024 · SWIFT · LiLaVe 2025
- From 2024 on, several groups put probes on language models' hidden states — Q-Probe, SWIFT, LiLaVe, HSRM, CASE.
# HSRM · CASE · VLA 2026
- In 2026 the idea also reached vision: a linear probe on a frozen vision-language-action robot policy chose among sampled action sequences.
# MedProb · DualRead 2026
- Two papers probed medical vision-language models — mostly for multiple choice and confidence.
- Plus a hundred-question demonstration of reranking open answers in MedProb's appendix.
# this work
- So I want to be clear from the start: our contribution is not the mechanism.""",
          qa=[("Hasn't a probe on a frozen vision-language model already been used as a best-of-N selector?",
               "Yes, outside VQA: 'What Frozen VLAs Already Know About Success' (arXiv:2605.28527) uses a linear probe on a frozen Pi0.5 policy as a "
               "test-time selector over sampled action prefixes — 'on push-plate, success rises from 26.7% under greedy decoding to 44.3%' (abstract, via "
               "AUDIT_2026-09-18_appendix/lit-vlm-med.md [V1-a]). In medical VQA, MedProb App. H reranked open answers on 100 items. We claim the measurement, not the idea."),
              ("Why not N = 64 or more?", "More samples raise coverage but also the chance of fooling the scorer — reward over-optimisation (Gao, Schulman "
               "& Hilton 2022); Cobbe saw accuracy fall beyond 400 samples. Our largest measured N is 32 on one benchmark; see the N slide.")])

    # ---- one slide per key paper: the paper's own overview figure + a labelled fact block
    def fig_b64(name):
        import base64, os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figs", name)
        return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

    def facts(rows, cols=1):
        lab = ("font-family:'Roboto Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;"
               "color:var(--indigo-d);font-weight:600;padding-top:3px")
        cells = "".join(f'<div style="{lab}">{k}</div><div style="min-width:0">{v}</div>' for k, v in rows)
        if cols == 1:
            return (f'<div style="display:grid;grid-template-columns:112px 1fr;gap:9px 14px;font-size:15.5px;line-height:1.4;'
                    f'color:#2b3045">{cells}</div>')
        half = (len(rows) + 1) // 2
        return ('<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 30px">'
                + facts(rows[:half]) + facts(rows[half:]) + '</div>')

    def paper_slide(sid, title, fig, cap, rows, src, say, look, minutes=2.0, qa=(), wide=False, fig_h=430):
        img = (f'<img src="{fig_b64(fig)}" alt="{cap}" style="max-width:100%;max-height:{fig_h}px;display:block;margin:0 auto;'
               f'border:1px solid var(--line);border-radius:6px;background:#fff">')
        capd = f'<div class="note" style="text-align:center;margin-top:6px;font-size:13px">{cap}, shown for illustration</div>'
        if wide:
            body = f'<div>{img}{capd}</div><div style="margin-top:16px">{facts(rows, cols=2)}</div>'
        else:
            body = (f'<div class="row"><div class="col w3" style="justify-content:center">{img}{capd}</div>'
                    f'<div class="col w2">{facts(rows)}</div></div>')
        d.add(sid, title, body, "II · Key papers", src=src, say=say, look=look, minutes=minutes, qa=list(qa))

    # ---- Cobbe 2021
    paper_slide("cobbe", "Trained verifiers: Cobbe et al. 2021 (OpenAI)", "cobbe_fig4.png",
                "Figure 4 from Cobbe et al. 2021 (p.7): the verifier training pipeline",
                [("domain", "text · grade-school math word problems (numeric answers)"),
                 ("models", "GPT-3 6B and 175B generators, fine-tuned; verifier = a same-size LM, full fine-tune"),
                 ("method", "sample 100 solutions per training problem → label each by its final answer → train a verifier → "
                            "at test time return the highest-scored of 100 (T = 0.7)"),
                 ("benchmarks", "GSM8K (introduced here: 7.5K train / 1K test)"),
                 ("result", "6B + verifier slightly beats a fine-tuned 175B — “approximately equivalent to a 30x model size increase” (§6)"),
                 ("against us", "token-level verifier beat solution-level; accuracy fell past 400 samples (<i>reward over-optimisation</i>)"),
                 ("for us", "our recipe: outcome labels → trained scorer → argmax, with a 918k-parameter probe as the verifier")],
                "Cobbe et al. arXiv:2110.14168: Fig. 4 p.7; §4.2 p.8, §4.3 p.9, §5.1 p.10, §6 p.12, Table 1 p.16 (PAPER_NOTES_A §1.1); Gao et al. arXiv:2210.10760",
                """# Trained verifiers: Cobbe et al. 2021 (OpenAI)
- Two founding papers. First, Cobbe and colleagues at OpenAI, 2021, established the recipe we follow.
# Figure 4 (left)
- Point at the three columns: train the generator; generate and label a hundred solutions per problem; train the verifier.
- Each solution is labelled correct or incorrect by its final answer only.
# domain
- Text only: grade-school math word problems, where the answer is a number.
# models
- GPT-3 generators; the verifier is a second language model of the same size, fully fine-tuned.
# method
- Sample a hundred solutions to each training problem.
- Label each one correct or incorrect by its final answer, and train a verifier to predict that label.
- At test time, return the highest-scored of a hundred samples.
# benchmarks
- GSM8K, a benchmark they released with this paper.
# result
- On grade-school math, a 6-billion model with a verifier slightly beat a fine-tuned 175-billion model.
  - They describe it as worth a 30-times increase in model size.
# against us
- Two findings.
  - Their token-level verifier was better than a whole-answer verifier.
  - Past 400 samples accuracy fell: the generator produced answers that fooled the verifier — reward over-optimisation.
# for us
- Cobbe's recipe — outcome labels, a trained scorer, pick the highest score.
- But the verifier is replaced by a small probe on our generator's own frozen internal states.""",
                "Figure on the left: walk the three numbered columns; then read the fact rows top to bottom.",
                minutes=2.0)

    # ---- Brown 2024
    paper_slide("brown", "Coverage vs precision: Brown et al. 2024 (Large Language Monkeys)", "brown_fig1.png",
                "Figure 1 from Brown et al. 2024 (p.2): repeated sampling, then a verifier",
                [("domain", "text · math, formal proofs (Lean4), code"),
                 ("models", "Llama-3 8B / 8B-Instruct / 70B-Instruct, Gemma 2B / 7B, Pythia, DeepSeek-Coder-V2; nothing trained"),
                 ("method", "repeated sampling up to 10,000 samples; split best-of-N into <i>coverage</i> (any sample right) and "
                            "<i>precision</i> (finding it)"),
                 ("benchmarks", "GSM8K, MATH, MiniF2F-MATH, CodeContests, SWE-bench Lite"),
                 ("result", "MATH, Llama-3-8B-Instruct: coverage 82.9% → 98.44%; majority vote / reward models only 40.50% → 41.41% (p.3, Fig. 7)"),
                 ("for us", "coverage = our oracle@8; selection is what the probe does")],
                "Brown et al. arXiv:2407.21787: Fig. 1 p.2; §2–4, p.3, Fig. 7 p.10 (PAPER_NOTES_A §1.2)",
                """# Coverage vs precision: Brown et al. 2024 (Large Language Monkeys)
- Brown and colleagues, 2024, showed why selection is the hard part.
# Figure 1 (top)
- Point at the two stages: the model generates many candidates; a verifier has to find a right one.
- Problem 1 under the first stage is coverage; problem 2 under the verifier is precision.
# domain
- Text only: math, formal proofs and code.
# models
- Several open models, Llama-3, Gemma and others; nothing is trained.
# method
- They sample up to ten thousand times per problem and measure the two problems separately.
# benchmarks
- GSM8K and MATH for math, MiniF2F for proofs, CodeContests and SWE-bench for code.
# result
- With ten thousand samples from Llama-3-8B-Instruct, a correct answer exists for 98 percent of the MATH problems.
- But voting or off-the-shelf reward models only got from 40.5 to 41.4 percent.
# for us
- Coverage grows; selection does not keep up. That split frames all our results.""",
                "Figure on top: the two numbered problems; then the fact rows, ending on result.", minutes=1.5, wide=True)

    # ---- Snell 2024
    paper_slide("snell", "Test-time compute vs model size: Snell et al. 2024", "snell_fig2.png",
                "Figure 2 from Snell et al. 2024 (p.8): three ways to search against a verifier",
                [("domain", "text · competition math"),
                 ("models", "PaLM 2-S* base; verifier = a process reward model (PRM), PaLM 2-S* fine-tuned; a revision model"),
                 ("method", "every method = proposer + verifier; pick the search strategy per question difficulty (“compute-optimal”)"),
                 ("benchmarks", "MATH (Lightman et al. split, 500 test problems)"),
                 ("result", "≈ best-of-N at up to 4× less compute; beats a ~14× larger model only where the small model sometimes succeeds"),
                 ("against us", "PRM beat ORM, more so as N grew; we use one fixed N and T"),
                 ("our data", "OmniMedVQA: oracle@8 0.70 → +0.160 · VQA-Med: oracle@8 0.21 → −0.004 (a selection failure too, Part V)")],
                "Snell et al. arXiv:2408.03314 (ICLR 2025): Fig. 2 p.8, abstract, Fig. 4 p.10, Fig. 9 p.15, App. D p.22, App. F p.24–25 (PAPER_NOTES_A §1.3); "
                "ours: final_eval_FINAL_cap320_T07.json cells.{omnimed_open,vqamed_open}",
                """# Test-time compute vs model size: Snell et al. 2024
- Snell and colleagues framed the modern debate: spend compute at test time, or on a bigger model?
# Figure 2 (left)
- Point at the three panels: best-of-N, beam search, lookahead search.
- Green nodes are kept by the verifier, red ones rejected; best-of-N on the left is our setting.
# domain
- Text only: competition math.
# models
- Every method is a proposer plus a verifier; their verifiers are process reward models, which score each reasoning step.
# method
- They choose the strategy per question according to its difficulty.
# benchmarks
- The MATH benchmark.
# result
- Smart test-time search can match plain best-of-N with four times less compute.
- And it can beat a 14-times-larger model — but only, in their words, "on problems where a smaller base model attains somewhat non-trivial success rates".
- On the hardest questions, they say, pretraining is the more effective way.
# against us
- Step-level verifiers beat outcome-level ones in their data, more so as N grows.
- Their method chooses the budget per question; ours uses one fixed setting.
- Their default way of combining scores: add up the verifier's score over identical answers.
- We tested it against our own readout; I show it in part four.
# our data
- On OmniMedVQA, a correct answer is among the eight samples 70 percent of the time — and we gain 16 points.
- On VQA-Med, that is only 21 percent — and we gain nothing.
  - Though I will show that there the probe also fails to find the correct answers that do exist.""",
                "Figure on the left: point at best-of-N (left panel); then result and our data.", minutes=2.0)

    # ---- self-consistency (Wang 2023)
    paper_slide("wang", "Majority vote: self-consistency (Wang et al. 2023)", "wang_fig1.png",
                "Figure 1 from Wang et al. 2023 (p.2): greedy chain of thought vs sampled paths and a vote",
                [("domain", "text · arithmetic, commonsense and symbolic reasoning (chain of thought)"),
                 ("models", "UL2-20B, GPT-3 (code-davinci-001/002), LaMDA-137B, PaLM-540B; few-shot, nothing trained"),
                 ("method", "sample m = 40 reasoning paths; majority vote over the final answers; needs answers that match exactly"),
                 ("benchmarks", "GSM8K, SVAMP, AQuA, AddSub, MultiArith, ASDiv, CSQA, StrategyQA, ARC"),
                 ("result", "PaLM-540B, GSM8K: 56.5 → 74.4 (Table 2, p.5)"),
                 ("for us", "the standard training-free baseline; on our open answers it loses to greedy (next slides)")],
                "Wang et al., ICLR 2023, arXiv:2203.11171: Fig. 1 p.2, p.3–5, Table 2 p.5 (PAPER_NOTES_DE E1)",
                """# Majority vote: self-consistency (Wang et al. 2023)
- The simplest selector is majority vote, which the field calls self-consistency.
# Figure 1 (top)
- Top row: greedy chain of thought gives one wrong answer.
- Bottom: several sampled reasoning paths, then the most common final answer wins.
# domain
- Text only: arithmetic, commonsense and symbolic reasoning.
# models
- Large language models used few-shot; nothing is trained.
# method
- Sample forty reasoning paths and take the majority answer. It needs answers that can be matched exactly.
# benchmarks
- Nine reasoning benchmarks, GSM8K among them.
# result
- On grade-school math with a big model it raised accuracy from 56 to 74 percent.
- Because a correct number tends to come up again and again.
# for us
- It is the baseline every selector must beat; on our open medical answers it does worse than greedy.""",
                "Figure on top: greedy row, then the sampled paths converging on one answer.", minutes=1.3, wide=True, fig_h=300)

    # ---- USC (Chen 2023)
    paper_slide("usc", "Voting for free text: Universal Self-Consistency (Chen et al. 2023)", "usc_fig1.png",
                "Figure 1 from Chen et al. 2023 (p.2): the USC workflow",
                [("domain", "text · math, code (text-to-SQL, Python), summarisation, open QA"),
                 ("models", "PaLM 2-L, gpt-3.5-turbo; nothing trained"),
                 ("method", "concatenate 8 sampled answers; ask the same LLM to “select the most consistent response”; one extra call"),
                 ("benchmarks", "GSM8K, MATH, BIRD-SQL, ARCADE, GovReport, SummScreen, TruthfulQA"),
                 ("result", "GSM8K, PaLM 2-L: greedy 85.7, majority vote 90.4, USC 90.2 — oracle over the same 8: 96.2 (Tables 1, 7)"),
                 ("also free text", "EBA (embedding clusters), self-certainty (own logits)"),
                 ("for us", "not run on our pools — an owed baseline")],
                "Chen et al. arXiv:2311.17311: Fig. 1 p.2, p.4, Table 1 p.5, Table 7 p.13 (PAPER_NOTES_DE E2); Ontalvilla et al. arXiv:2606.12003; Kang et al. arXiv:2502.18581",
                """# Voting for free text: Universal Self-Consistency (Chen et al. 2023)
- Majority vote needs exact repeats; USC is the field's answer for free text.
# Figure 1 (top)
- Point at the prompt box: all sampled responses are pasted in, and the model is asked which one is most consistent.
# domain
- Text only: math, code, summaries and open questions.
# models
- PaLM 2 and GPT-3.5; nothing is trained.
# method
- One extra call to the same model per question.
# benchmarks
- Seven text benchmarks, from GSM8K to TruthfulQA.
# result
- On math it matches majority vote, and both stay well below the best of the same eight samples.
# also free text
- EBA picks the centre of the largest cluster of answer embeddings.
- Self-certainty scores each sample from the model's own output probabilities.
# for us
- We have not run these on our pools, and that is a gap.""",
                "Figure on top: the USC prompt box in the middle.", minutes=1.2, wide=True)

    # ---- voting and its ceiling
    sc_rows = []
    for c in D.OPEN8:
        a = D.cell(c)
        sc_rows.append((D.SHORT[c], [(a["sc_minus_greedy_judge"], *a["ci_sc_minus_greedy_judge"])]))
    sc_rows.append(("<tspan font-weight='700'>macro</tspan>", [(SCJ[0], SCJ[1], SCJ[2])]))
    d.add("voting", "Training-free selection and its ceiling: majority vote, USC, embeddings",
          f"""
<div class="row">
 <div class="col">
  <div class="paper"><div class="cite">Wang et al. ICLR 2023</div><h3>Self-consistency = majority vote</h3>
  <span class="note">\\( \\hat a = \\arg\\max_a \\sum_i \\mathbb 1[a_i=a] \\) · GSM8K 56.5 → 74.4 · needs exact repeats</span></div>
  <div class="paper"><div class="cite">Bay & Yearick 2026 · arXiv:2606.28661</div><h3>The modal ceiling</h3>
  <span class="note">voting is capped at \\( \\pi_{{mode}}=P[\\text{{mode is correct}}] \\), whatever N</span></div>
  <div class="paper"><div class="cite">Chen 2023 (USC) · Ontalvilla 2026 (EBA) · Kang 2025 (self-certainty)</div><h3>For free text</h3>
  <span class="note">consistency by LLM, embedding cluster, or own logits — none run on our pools</span></div>
 </div>
 <div class="col">{forest(sc_rows, -0.08, 0.04, 0.02, w=560, rowh=30, title="majority vote − greedy, training judge (shipped pools)", value_fmt=lambda v: fmt(v,4))}
 <div class="note">Open medical answers rarely repeat, and the correct answer is often a minority: majority vote <b>loses</b> to greedy ({pm(SCJ[0])} {ci(SCJ[1],SCJ[2])}, LOSS).</div></div>
</div>""", "II · Key papers", src="final_eval_FINAL_cap320_T07.json cells.*.all.sc_minus_greedy_judge (+CI), macro_over_open8.all.sc_minus_greedy_judge; Wang 2023 Table 2 p.5; Bay Prop. 5 Eq. 13 p.10",
          minutes=1.5,
          look="Right chart: most dots sit left of the zero line; end on the bold macro row at the bottom (hollow dot = not significant).",
          say=f"""# Self-consistency = majority vote
- The simplest selector is majority vote, which the field calls self-consistency.
- On math it is very strong: on grade-school math with a big model it raised accuracy from 56 to 74 percent.
- Because a correct number tends to come up again and again.
# The modal ceiling
- Bay and Yearick showed its hard ceiling.
- As you add samples, voting converges to the most frequent answer.
- So it can never beat the rate at which the most frequent answer is correct. A learned scorer is not bound by that.
# majority vote − greedy, training judge (shipped pools)
- The chart is majority vote minus greedy on our pools.
- On most benchmarks it is negative, and averaged it loses {abs(SCJ[0]):.4f}.
- Free-text medical answers rarely repeat exactly, and the right answer is often a minority.
  - Like "Large Bowel", which appeared once in our example.
# For free text
- There are training-free methods designed for free text:
  - USC asks the language model which answer is the most consistent with the others;
  - EBA picks the centre of the largest cluster of answer embeddings;
  - self-certainty scores each sample from the model's own output probabilities.
- We have not run them, and that is a gap.""",
          qa=[("LiLaVe and CASE found pure best-of-n can fall below voting. Why not here?",
               f"Their answers repeat (math, multiple choice), so voting is strong. On our open answers voting itself loses to greedy ({pm(SCJ[0])} {ci(SCJ[1],SCJ[2])}), "
               "and the probe beats voting by +0.0893 [+0.0811, +0.0972] under the training judge and +0.0136 [+0.0053, +0.0216] under lenient EM (extra_numbers.json).")])

    # ---- Q-Probe
    paper_slide("qprobe", "The first probe reranker: Q-Probe (Li et al., ICML 2024)", "qprobe_fig1.png",
                "Figure 1 from Li et al. 2024 (p.2): one feature vector per sampled program, scored by a probe",
                [("domain", "text · code generation (also grade-school math, human preferences)"),
                 ("models", "Code-LLaMA-7B, frozen; probe = linear, 4,096 weights, on layer 26's embedding of prompt + completion"),
                 ("method", "sample k = 48 programs; the probe scores each; sample from a softmax over scores (β → 0 = argmax)"),
                 ("benchmarks", "MBPP, HumanEval, GSM8K; HH / OASST / SHP preferences (GPT-4 win rate)"),
                 ("result", "MBPP pass@1 0.29 → 0.46 with the policy-gradient loss (greedy 0.38) (Table 1, p.6)"),
                 ("against us", "pointwise cross-entropy 0.40 &lt; within-question loss 0.46; 2–3-layer MLPs overfit on preference data")],
                "Li et al. arXiv:2402.14688 (ICML 2024): Fig. 1 p.2, §4–5 p.4–5, Table 1 p.6, §7.2 p.9 (PAPER_NOTES_BC C3)",
                """# The first probe reranker: Q-Probe (Li et al., ICML 2024)
- Q-Probe, in 2024, is the earliest paper to rerank a frozen model's samples with a probe.
# Figure 1 (top)
- Point left to right: the model writes several programs; each gets its own feature vector; the probe turns the vectors into probabilities; one program is picked.
- That is our pipeline, with programs instead of medical answers.
# domain
- Text only: code generation, plus some math and preference data.
# models
- A frozen code model with a linear probe on one middle layer.
# method
- It reranks 48 sampled programs; as the softmax temperature goes to zero this becomes our argmax.
# benchmarks
- Two code benchmarks, MBPP and HumanEval, plus GSM8K and preference data.
# result
- On MBPP, success with one program rises from 0.29 to 0.46.
# against us
- Their pointwise loss — the kind we use — did worse than a within-question loss.
- And deeper probes overfit on their preference data.""",
                "Figure on top: follow the colours from each program to its feature strip to the probe.", minutes=1.4, wide=True, fig_h=250)

    # ---- SWIFT
    paper_slide("swift", "Rewards from hidden states: SWIFT (Guo et al., KDD 2026)", "swift_fig1.png",
                "Figure 1 from Guo et al. 2026 (p.2): a large reward model vs SWIFT's linear head on all layers",
                [("domain", "text · math, code, commonsense, symbolic reasoning, safety"),
                 ("models", "Llama-3.2-3B, Llama-3.1-8B, Ministral-8B, frozen; head = linear gate + linear token reward, 1.8–3.0×10⁵ parameters"),
                 ("method", "concatenate every layer's state at each answer token → gate-weighted mean of token rewards → BCE on "
                            "correctness → argmax of N"),
                 ("benchmarks", "MATH, GSM8K, AQuA_RAT, Imbue Code, HellaSwag, CoinFlip, PKU-SafeRLHF"),
                 ("result", "MATH average 57.5 vs 52.9 for the strongest baseline reward model (Skywork-Llama3.1-8B), 51.0 EurusRM-7B (Table 1)"),
                 ("supports us", "BCE was the best of 5 losses (Table 7)"),
                 ("against us", "the learned gate beat a plain mean by 2.7 points (Table 8)")],
                "Guo et al. arXiv:2505.12225 (KDD 2026): Fig. 1 p.2, §4.2 p.3–4, Tables 1, 7, 8 (PAPER_NOTES_BC C2; arxiv.org/html/2505.12225)",
                """# Rewards from hidden states: SWIFT (Guo et al., KDD 2026)
- SWIFT is the closest single text-only neighbour.
# Figure 1 (left)
- Top half: the usual way, a separate billion-parameter reward model reads the output text.
- Bottom half: SWIFT taps every transformer block, concatenates, and a linear head with a gate gives the reward.
# domain
- Text only: math, code, commonsense and safety tasks.
# models
- Three open 3-to-8-billion models, frozen; the head has about a hundred thousand to three hundred thousand weights.
# method
- It reads every token of every layer and pools them with a learned gate.
- It trains with binary cross-entropy — the same kind of loss as ours; ours additionally up-weights the correct answers.
# benchmarks
- Seven text benchmarks, with MATH as the headline.
# result
- On MATH it beats the strongest trained reward model in their table.
# supports us
- Our loss was their best of five.
# against us
- Learning which tokens to weight beat a plain average.""",
                "Figure on the left: top half (big reward model) vs bottom half (taps on every block).", minutes=1.5)

    # ---- HSRM
    paper_slide("hsrm", "A small reader on hidden states: HSRM (Li & Zhu 2026)", "hsrm_fig1.png",
                "Figure 1 from Li &amp; Zhu 2026 (p.3): hidden states at reasoning-step boundaries feed a 2M-parameter reader",
                [("domain", "text · math reasoning"),
                 ("models", "Qwen3 1.7B–14B and Llama-3 1B–8B generators, frozen; reader = 2-layer Transformer → mean → linear, 2.1–3.4 M parameters"),
                 ("method", "read final-layer states at the end of each reasoning step; pairwise (Bradley–Terry) loss within a problem; best-of-8"),
                 ("benchmarks", "GSM8K, MATH-500, AIME, OlympiadBench"),
                 ("result", "≥ a 55M text verifier in 15 of 16 settings; Llama-8B GSM8K 88.3 vs 85.2 (oracle 96.0) (Table 3, p.9)"),
                 ("supports us", "a mean-pooled MLP reader ≈ their Transformer reader (Table 10)"),
                 ("against us", "Bradley–Terry 85.9 &gt; BCE 83.9 on GSM8K (Table 9)")],
                "Li &amp; Zhu arXiv:2608.30841: Fig. 1 p.3, §3 p.3–4, Table 3 p.9, Tables 9–10 p.14–15 (PAPER_NOTES_BC C1)",
                """# A small reader on hidden states: HSRM (Li & Zhu 2026)
- HSRM is the closest published analogue to our pipeline.
# Figure 1 (top)
- Point at the yellow boxes: the hidden states at the end of each reasoning step go into the 2-million-parameter reader.
- The bottom row is the expensive alternative: re-reading the text with a separate verifier.
# domain
- Text only: math word problems and competition math.
# models
- Frozen Qwen3 and Llama generators.
# method
- It reads the states at the end of each reasoning step with a small Transformer and picks the best of eight — the same N as ours.
# benchmarks
- Four math benchmarks, from GSM8K to olympiad problems.
# result
- It matches or beats a much larger text verifier in fifteen of sixteen settings.
# supports us
- The reader's architecture matters much less than what you feed it.
  - A mean-pooled MLP reader did about as well as their Transformer reader.
# against us
- HSRM found a pairwise ranking loss, Bradley–Terry, better than our loss — 85.9 versus 83.9 on grade-school math.""",
                "Figure on top: the yellow step-boundary states → HSRM 2M box.", minutes=1.5, wide=True, fig_h=250)

    # ---- hidden-state verifiers in LLMs
    d.add("hsv", "Hidden-state verifiers side by side: Q-Probe, SWIFT, HSRM — and ours",
          """
<table class="d sm">
<tr><th>paper</th><th>what is read</th><th>layer</th><th>pooling</th><th>head · loss</th><th>params</th><th>N</th><th>result (own units)</th></tr>
<tr><td><b>Q-Probe</b><br><span class="note">ICML 2024</span></td><td>prompt+completion embedding</td><td>26 of Code-LLaMA-7B</td><td>not stated</td><td>linear · policy-gradient (best), CE</td><td>4,096</td><td>k = 48</td><td>MBPP pass@1 0.29 → 0.46 (greedy 0.38)</td></tr>
<tr><td><b>SWIFT</b><br><span class="note">KDD 2026 · arXiv 2505.12225</span></td><td>every response token</td><td>all layers, concatenated</td><td>gated mean</td><td>linear gate + reward · <b>BCE</b> (best of 5 losses)</td><td>1.8–3.0×10⁵</td><td>4–64</td><td>MATH avg 57.5 vs 52.9 for the strongest baseline RM (Skywork-Llama3.1-8B), 51.0 EurusRM-7B</td></tr>
<tr><td><b>HSRM</b><br><span class="note">arXiv 2608.30841</span></td><td>states at reasoning-step boundaries</td><td>final (top-4 best)</td><td>2-layer Transformer, then mean</td><td>linear · Bradley–Terry (BCE worse)</td><td>2.1–3.4 M</td><td>8</td><td>≥ a 55M text verifier in 15/16 settings; Llama-8B GSM8K 88.3 vs 85.2</td></tr>
<tr class="hl"><td><b>ours</b></td><td>the answer's own tokens (teacher-forced)</td><td>18, 20, 22 of 28</td><td>mean</td><td>MLP 3584→256→1 · class-weighted BCE · ×24</td><td>918,017 each</td><td>8</td><td>macro over greedy, 8 medical VQA benchmarks: +0.080 training judge, +0.071 cross-family judge</td></tr>
</table>
<div class="row" style="height:auto;margin-top:12px">
 <div class="col"><div class="term g"><b class="k">supports our design</b><span>HSRM: mean-pooled MLP ≈ Transformer reader · SWIFT: BCE best of 5 losses</span></div></div>
 <div class="col"><div class="term r"><b class="k">cuts against it</b><span>HSRM: Bradley–Terry > BCE · Q-Probe: within-question loss > CE · SWIFT: gated > mean pooling</span></div></div>
</div>""", "II · Key papers", src="Q-Probe Table 1 p.6, §7.2; SWIFT Tables 1 (MATH Avg. column, arxiv.org/html/2505.12225), 7, 8; HSRM Tables 3, 9, 10 (PAPER_NOTES_BC); ours final_eval_FINAL_cap320_T07.json",
          minutes=2.5,
          say="""# Hidden-state verifiers side by side: Q-Probe, SWIFT, HSRM — and ours
- These are our three closest relatives, all on text-only language models.
# Q-Probe
- Q-Probe, in 2024, put a linear probe on a frozen code model's embeddings to rerank 48 sampled programs.
# SWIFT
- SWIFT reads every token of every layer and pools them with a learned gate.
- It trains with binary cross-entropy — the same kind of loss as ours; ours additionally up-weights the correct answers.
# HSRM
- HSRM reads the states at the end of each reasoning step with a small Transformer and picks the best of eight — the same N as ours.
# ours
- The last row is us, so you can compare column by column.
- We read only the answer's own tokens, at three middle layers, with a simple mean, and a one-hidden-layer network.
# cuts against it
- I want to be honest about what these papers say against our choices.
- HSRM found a pairwise ranking loss, Bradley–Terry, better than our loss — 85.9 versus 83.9 on grade-school math.
- Q-Probe found a within-question loss better than a pointwise one, and saw MLPs overfit on preference data.
- SWIFT found that learning which tokens to weight beats a plain average, by 2.7 points.
# supports our design
- On the other side, SWIFT found our loss the best of five.
- HSRM found the reader's architecture matters much less than what you feed it.
  - A mean-pooled MLP reader did about as well as their Transformer reader.
- I will come back to each of these when I justify our choices.""",
          qa=[("Is the BCE-vs-ranking-loss question settled?", "No. SWIFT finds BCE best; HSRM and Q-Probe find within-question losses better. Our own BCE-vs-Bradley–Terry comparison "
               "is confounded (Bradley–Terry sees 38.9% of rows and 3.26× fewer steps), so we do not claim BCE is better; a ranking loss on the final recipe is untested.")])

    # ---- CASE (+ LiLaVe, ReProbe)
    paper_slide("case", "When hidden-state selection beats voting: CASE (Wang, Hong, Bagci 2026)", "case_fig1.png",
                "Figure 1 from Wang, Hong &amp; Bagci 2026 (p.8): voting vs CASE on the same 5 candidates",
                [("domain", "text · multiple-choice and short-answer QA: logic, medicine, math"),
                 ("models", "12 LLMs incl. Qwen2.5 1.5–14B, Llama-3-8B, OpenBioLLM-8B/70B, Med42-8B; gate = logistic regression on the answer-token state"),
                 ("method", "score each of N candidates with the gate, pick the maximum; <b>within-question AUC</b> predicts the gain over voting"),
                 ("benchmarks", "LogiQA, MedQA, MedMCQA, PubMedQA, MATH-500, GSM8K, GPQA"),
                 ("result", "OpenBioLLM-8B, MedMCQA medium-difficulty bin: +19.1 over voting; ordinary CV AUC 0.610 → grouped 0.542 → within-question 0.502 (LogiQA)"),
                 ("against us", "logistic 0.596 ≥ MLP-256 0.537 (small data); token-mean “averages the signal away”; voting overtakes it by N = 16 on the medium bin"),
                 ("related", "LiLaVe: trees on hidden states (AUC 0.86, GSM8K) · ReProbe: our head shape, BCE, LLM-judge labels"),
                 ("our gap", "within-question AUROC never reported")],
                "Wang, Hong, Bagci arXiv:2608.17124: Fig. 1 p.8; HTML §3.1–3.3, §6.1, §6.4, §6.6, Tables 1, 4; LiLaVe arXiv:2504.16760 Table 1, §4.4; "
                "ReProbe arXiv:2511.06209 §3 (PAPER_NOTES_I G1–G3)",
                """# When hidden-state selection beats voting: CASE (Wang, Hong, Bagci 2026)
- CASE, from August 2026, asks exactly when choosing by hidden states beats majority voting.
# Figure 1 (left)
- Left box: voting counts surface answers and picks the wrong majority A.
- Right box: CASE reads each candidate's answer-token state, a logistic gate scores it, and the minority correct answer B wins.
# domain
- Text only: multiple-choice and short-answer questions, including medical exams.
# models
- Twelve language models, general and medical, with a logistic gate.
# method
- Key contribution: within-question AUC — how well the probe orders the candidates of the same question.
- That metric predicts how much you gain over voting.
# benchmarks
- Seven text benchmarks, four of them medical.
# result
- Ordinary cross-validation badly overstates probes: a probe can learn which questions are easy instead of which answers are right.
  - 0.61 drops to 0.50 when measured properly.
- On a medical multiple-choice set, the medium-difficulty questions, it gained 19 points over voting.
# against us
- A small logistic regression, 0.596, beat a 256-unit MLP, 0.537, in their small-data setting.
- Averaging over tokens hurt in their multiple-choice setting.
- Also, on those medium questions, voting caught up and overtook them by N of 16.
# related
- LiLaVe uses gradient-boosted trees on the hidden states of the last 16 tokens of five layers.
- In their data plain best-of-n eventually fell below voting, and weighted voting was best.
- ReProbe has the same classifier shape as ours.
  - One Transformer layer, a mean over the span, then linear, GELU, linear.
  - Trained with binary cross-entropy on labels from an LLM judge.
- It matches process reward models up to 810 times larger.
# our gap
- Our evaluation avoids the leakage they warn about by splitting by image.
- But we have not reported their within-question AUC, and we should.""",
                "Figure on the left: the red voting box vs the CASE box; then result and against us.", minutes=2.2)

    # ---- Verification Mirage
    paper_slide("medside", "Medical self-verification fails: Verification Mirage (Jin et al. 2026)", "mirage_fig1.png",
                "Figure 1 from Jin et al. 2026 (p.2): the generator–verifier study pipeline",
                [("domain", "vision-language · medical VQA"),
                 ("models", "6 VLMs: Qwen2.5-VL-7B, Gemma-3, Phi-4-Multimodal, MedGemma, HuatuoGPT-Vision, Lingshu (size not stated); "
                            "verifiers prompted, nothing trained"),
                 ("method", "the model answers; the same model, in a fresh context, judges “correct / incorrect / uncertain”; oracle labels by "
                            "Gemini-2.5-Flash; a 4-turn feedback loop"),
                 ("benchmarks", "VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA"),
                 ("result", "Lingshu accepts <b>62.2–72.5%</b> of its own wrong answers; larger same-family verifier: no significant reduction "
                            "(p = 0.782); 69.5–87.1% of wrong answers locked in over 4 turns"),
                 ("for us", "why our selector is <b>trained</b>, not prompted")],
                "Jin et al. arXiv:2605.10850: Fig. 1 p.2, §3–4, Tables 1, 11–15, App. B.4 (PAPER_NOTES_BC B2)",
                """# Medical self-verification fails: Verification Mirage (Jin et al. 2026)
- On the medical side, three papers set the context.
# Figure 1 (left)
- Panel a: the generator answers, then a verifier with the same weights says correct or incorrect.
- Panel b: five medical VQA datasets and seven question types; panel c: where verifiers fail.
# domain
- Images plus text: medical visual question answering, the same field as ours.
# models
- Six medical and general VLMs, including Lingshu; nothing is trained.
# method
- Verification Mirage asked medical VLMs, including Lingshu, to check their own answers.
# benchmarks
- Five medical VQA datasets, three of which we also use.
# result
- Lingshu accepted 62 to 73 percent of its own wrong answers.
- A bigger model of the same family did not significantly help.
- Over four turns of conversation, the wrong answers got locked in.
# for us
- That is why our selector is trained on labels rather than prompted.
- One point to keep apart: in part four, a separate 32-billion model judging our 7B's candidates without training does better than a 7B doing the same.
  - That does not conflict with Mirage.
  - Mirage measures whether a model accepts its own wrong answers when it re-checks them.
  - We measure best-of-8 selection accuracy, with a different model as the judge.
  - Both point the same way: training, not prompting, is what makes a selector work.""",
                "Figure on the left: panel a (generator → verifier); then the result row.", minutes=1.8,
                qa=[("Did Verification Mirage use Lingshu-7B?", "The paper does not state Lingshu's size; do not call it 7B."),
                    ("Why not just ask the model, or use its likelihood — no labels needed?",
                     "Prompted self-verification: Mirage shows Lingshu accepts 62–73% of its own wrong answers. Plain log-probability ranking and self-certainty "
                     "have not been run on our pools — an owed baseline on the gap list.")])

    # ---- MedProb
    paper_slide("medprob", "Probing medical VLMs: MedProb (Nourbakhsh et al. 2026)", "medprob_fig2.png",
                "Figure 2 from Nourbakhsh et al. 2026 (p.5): VLMs evaluated (left) and the probe vs prompting (right)",
                [("domain", "vision-language · medical VQA, closed questions as multiple choice"),
                 ("models", "28 VLMs incl. Qwen2.5-VL 3/7/32B, MedVLThinker 3/7/32B; probe = logistic regression on the last <i>input</i> token, per layer"),
                 ("method", "read the state before any answer is written and classify the option — a classifier, not a selector"),
                 ("benchmarks", "PathVQA, SLAKE, VQA-RAD (closed, as MCQ); OmniMedVQA"),
                 ("result", "probe beats prompting by &gt;20 points in several cases; linear ≈ MLP (App. L)"),
                 ("App. H", "open reranking on 100 OmniMedVQA items, Qwen2.5-VL-7B 0.09 → 0.13 — “an initial demonstration”; N, grader, probe type not stated"),
                 ("for us", "the only earlier medical instance of probe reranking of open answers")],
                "Nourbakhsh et al. arXiv:2609.04336: Fig. 2 p.5, §3 p.3–5, p.6, Table 10 p.26, App. H p.27 Table 14, App. L Table 21 (PAPER_NOTES_BC C4)",
                """# Probing medical VLMs: MedProb (Nourbakhsh et al. 2026)
- MedProb probes medical vision-language models directly.
# Figure 2 (top)
- Left panel: the many general and medical VLMs they test.
- Right panel, middle row: a linear classifier on the model's internal state, instead of letting it write an answer.
# domain
- Images plus text: medical VQA, but closed questions turned into multiple choice.
# models
- Twenty-eight VLMs; the probe is a logistic regression.
# method
- It reads the state before any answer is written, so it is a classifier, not a selector.
# benchmarks
- The closed halves of PathVQA, SLAKE and VQA-RAD, plus OmniMedVQA.
# result
- A simple linear probe on a medical VLM's internal state answers closed multiple-choice questions better than the model's own generated answer.
- By more than 20 points in several cases.
# App. H
- In an appendix they tried reranking open answers on a hundred OmniMedVQA questions, from 0.09 to 0.13.
  - They call it an initial demonstration; the number of samples, the grader and the probe type are not stated.
# for us
- It is the closest medical precedent; we make no priority claim.""",
                "Figure on top: the MedProb row in the right panel; then App. H.", minutes=1.6, wide=True, fig_h=270,
                qa=[("Is MedProb App. H comparable to our OmniMedVQA result?", "No: 100 test items, unstated grader and recipe, baseline is a single sample rather than greedy. "
                     "Ours on OmniMedVQA: 8,883 questions, greedy 0.516 → 0.677 judge. We make no priority claim — its authors call it 'an initial demonstration'.")])

    # ---- DualRead
    paper_slide("dualread", "Teacher-forced replay for confidence: DualRead (Xie et al. 2026)", "dualread_fig2.png",
                "Figure 2 from Xie et al. 2026 (p.3): replaying the answer, reading states before and after it",
                [("domain", "vision-language · medical VQA, confidence of one answer"),
                 ("models", "GRPO-trained Qwen3-VL-2B and MedGemma1.5-4B, frozen; small MLP / linear readers on final-layer states"),
                 ("method", "replay the greedy answer with teacher forcing; read the state before and after the answer; fuse into one "
                            "confidence. It does not choose among answers."),
                 ("benchmarks", "SLAKE-test (in-distribution); VQA-RAD, PathVQA, PMC-VQA, OmniMedVQA, ReXVQA, VQA-Med 2019 (out of distribution)"),
                 ("result", "AUROC 0.952 on SLAKE vs 0.666 averaged out of distribution (Qwen3-VL-2B, Table 1)"),
                 ("for us", "the same teacher-forced re-read as ours; its drop on new datasets is a warning about transfer")],
                "Xie et al. arXiv:2609.06419: Fig. 2 p.3, method §, Table 1 (PAPER_NOTES_I G4)",
                """# Teacher-forced replay for confidence: DualRead (Xie et al. 2026)
- DualRead re-reads a frozen medical VLM's answer with teacher forcing — the same re-reading we do.
# Figure 2 (left)
- Panel a, middle band: the image, question and answer tokens replayed through the frozen model.
- An orange read before the answer and a purple read after it are fused into one confidence.
# domain
- Images plus text: medical VQA, but scoring the confidence of a single answer.
# models
- Two small medical VLMs, frozen, with small readers on the last layer.
# method
- But only to calibrate confidence; it never chooses between answers.
# benchmarks
- Trained on SLAKE, tested on SLAKE and six other medical sets.
# result
- Its quality drops sharply on new datasets: AUROC 0.95 in distribution, 0.67 outside.
# for us
- Same re-read, different job; and a warning that probes may not transfer to new datasets.""",
                "Figure on the left: panel a, the replay band and the two reads.", minutes=1.3)

    # ---- WED
    paper_slide("wed", "A published medical selector: WED (Hagen et al. 2026)", "wed_fig1.png",
                "Figure 1 from Hagen et al. 2026 (p.3): generator and verifier rankings converge; the Wasserstein rule stops earlier",
                [("domain", "vision-language · open-ended medical VQA"),
                 ("models", "Qwen3-VL-2B / 4B / 8B, Gemma-3-4B, MedGemma-4B; nothing trained — the same VLM plays generator and verifier"),
                 ("method", "sample ≤ 8 distinct candidates (≤ 16 draws), merge by meaning → score each as generator and as verifier → "
                            "the two update toward each other (a game) → stop when Wasserstein-1 &lt; 0.2 → return the top candidate"),
                 ("benchmarks", "VQA-RAD and PathVQA open-ended test subsets, yes/no excluded; graded by Grok 4.1 Fast, EM, F1"),
                 ("result", "−1.2 to +5.0 points over greedy (judge, Table 1 p.8) — “gains diminish with size”"),
                 ("for us", "evaluated on two of our open benchmarks → we ran its code on <b>our</b> candidates (Part VI)")],
                "Hagen et al. arXiv:2605.18313: Fig. 1 p.3, §2 p.3–6, p.5 (≤ 8 candidates, ≤ 16 calls), p.7, Table 1 p.8, authors' code BDG.py (PAPER_NOTES_DE E4); "
                "Jacob et al. arXiv:2310.09139; BDG arXiv:2410.01064",
                """# A published medical selector: WED (Hagen et al. 2026)
- Two recent papers do test-time selection specifically for medical VQA.
# Figure 1 (top)
- Left: a CT image and candidate answers; synonyms like liver and hepatic region are merged.
- Middle: the generator's and the verifier's rankings of the candidates.
- Right: the rankings converge over iterations; the Wasserstein rule stops earlier than exact agreement.
# domain
- Images plus text: open-ended medical VQA, like ours.
# models
- Small VLMs; WED is training-free.
# method
- It samples up to eight distinct candidates, using up to sixteen draws at temperatures 0.5 and 1.0, and merges answers with the same meaning.
- Then the same model plays two roles:
  - a generator that scores how likely each candidate is;
  - a verifier that says whether it is correct.
- The two roles adjust toward each other in a game until their rankings agree.
  - Measured with a Wasserstein distance, so that synonyms count as agreement.
- The game itself comes from earlier work, the Consensus Game and the Bayesian Decoding Game; WED's addition is the stopping rule.
# benchmarks
- The open questions of VQA-RAD and PathVQA, without yes/no.
# result
- Graded by the Grok model, it reports one to five points over greedy on most small models.
- And a small loss for the largest one on VQA-RAD — so gains shrink as models get bigger.
# for us
- WED matters most for us: the only published selector evaluated on two of our open benchmarks, VQA-RAD and PathVQA.
- So rather than compare numbers across papers, we ran WED's own code on our candidates.
- I show that result in part six.""",
                "Figure on top: candidates → two rankings → convergence; then the method row.", minutes=2.0, wide=True, fig_h=250)

    # ---- Best-of-Evidence
    paper_slide("boe", "Claim-level checking: Best-of-Evidence (Zhang et al. 2026)", "boe_fig1.png",
                "Figure 1 from Zhang et al. 2026 (p.1): a fixed candidate pool, a claim graph, budgeted evidence checks",
                [("domain", "vision-language · medical VQA (open and multiple choice)"),
                 ("models", "generator Qwen3-VL-30B-A3B (16 samples, T = 1.1); evidence judge Qwen3-VL-235B-A22B"),
                 ("method", "keep the 16-sample pool; check individual claims (modality, anatomy …) under a budget; update each "
                            "candidate's score; pick the maximum"),
                 ("benchmarks", "VQA-Med (training partition), PathVQA open, PMC-VQA (options hidden), MedXpertQA-MM"),
                 ("result", "vs majority vote: +0.26 to +0.58 points; 2 of 4 significant (Table 2, p.9)"),
                 ("for us", "its baseline is voting, not greedy — the matching number for us is probe − vote (Part VI)")],
                "Zhang et al. arXiv:2607.20950: Fig. 1 p.1, §5.1 p.7, Table 1 p.7, Table 2 p.9, App. B.2 (PAPER_NOTES_BC B3)",
                """# Claim-level checking: Best-of-Evidence (Zhang et al. 2026)
- The second medical selector.
# Figure 1 (top)
- Left: a VLM generates a fixed pool of candidates.
- Middle: their claims form a graph; a budget controller decides which claims to check.
- Right: the evidence updates each candidate's score, and the best one is selected.
# domain
- Images plus text: medical VQA.
# models
- Samples sixteen answers and uses a 235-billion-parameter judge to check individual claims — modality, anatomy and so on.
# method
- The pool stays fixed; only the checking is budgeted.
# benchmarks
- Four medical sets; its VQA-Med items come from the training split.
# result
- It gains under one point over majority vote, significant on two of four datasets.
# for us
- No greedy baseline, so the fair comparison for us is probe minus vote.""",
                "Figure on top: left to right, pool → claim graph → budgeted checks → selection.", minutes=1.3, wide=True, fig_h=250)

    # ---- generator and judges provenance + gap table
    d.add("gap", "Where this work sits",
          """
<table class="d">
<tr><th></th><th>(A) probe on frozen<br>hidden states</th><th>(B) selects among<br>sampled answers</th><th>(C) pooled over the<br>answer's own tokens</th><th>(D) no extra<br>forward pass</th><th>image + medical +<br>open-ended</th></tr>
<tr><td>Q-Probe 2024</td><td>✓</td><td>✓</td><td>?</td><td>✗ (separate pass)</td><td>✗</td></tr>
<tr><td>SWIFT (KDD 2026)</td><td>✓</td><td>✓</td><td>✓ (gated)</td><td>not stated</td><td>✗</td></tr>
<tr><td>HSRM 2026</td><td>✓</td><td>✓</td><td>step boundaries</td><td>✓ claimed</td><td>✗</td></tr>
<tr><td>CASE 2026</td><td>✓ (linear)</td><td>✓</td><td>answer token</td><td>at deployment</td><td>medical text only</td></tr>
<tr><td>MedProb 2026</td><td>✓</td><td>App. H only</td><td>✗ (input token)</td><td>—</td><td>✓ (MCQ)</td></tr>
<tr><td>DualRead 2026</td><td>✓</td><td>✗</td><td>✗</td><td>✗ (replay)</td><td>✓ (confidence)</td></tr>
<tr><td>VLA probe 2026</td><td>✓ (linear)</td><td>✓ action prefixes vs greedy</td><td>—</td><td>—</td><td>robot actions, not VQA</td></tr>
<tr><td>VHS 2026</td><td>✓</td><td>✓ (image generation)</td><td>—</td><td>✓ (no decode / re-encode)</td><td>image generation, not VQA</td></tr>
<tr class="hl"><td><b>ours</b></td><td>✓ (MLP)</td><td>✓</td><td>✓ (mean)</td><td><b>by design; not yet as run</b></td><td>✓</td></tr>
</table>
<div class="term t" style="margin-top:14px"><b class="k">contribution, stated narrowly</b><span>an evaluation, not a mechanism: 8 benchmarks · 36,869 questions · cross-fitting · 2 judges + EM · WED head-to-head</span></div>""",
          "II · Key papers", src="literature/DOMAIN_GUIDE_2026-09-30.md §4.1 (branch worktree-lit-domain-package); PAPER_NOTES_BC/I; VLA probe arXiv:2605.28527; VHS arXiv:2603.22492; "
              "novelty withdrawn: AUDIT_2026-09-18_appendix/lit-vlm-med.md C2 — full provenance in the script",
          minutes=1.5,
          say="""# Where this work sits (table columns A–D)
- Putting it together. The literature's hidden-state verifiers share four properties:
  - (A) a probe on frozen states;
  - (B) used to select among samples;
  - (C) pooled over the answer's own tokens;
  - (D) in principle no extra pass through the model, because the states were already computed while generating.
- The text-only papers have most of these.
- The medical papers mostly probe the model without selecting free-text answers.
- In vision the idea also exists, for robot actions and for image generation.
# ours
- Our row has all of them except one honest asterisk, column D.
- In our experiments we re-read each answer in a separate pass, so "no extra pass" is true by design, not yet as run.
- For an earlier probe, capturing the states inside generation changed 24 of 2,345 picks and left accuracy unchanged.
- For the shipped probe, re-extracting the features at the generation resolution is running now.
# contribution, stated narrowly
- So the contribution is the evaluation, not the mechanism.
- MedProb's appendix tried this on a hundred questions.
- We measure it on eight benchmarks, with held-out images.
- With a second grader from another model family, plus string matching.
- And against a published selector on identical candidates.""",
          qa=[("Is this just HSRM / SWIFT / Q-Probe with a different backbone?",
               "In mechanism, yes, and we say so. What differs is the setting and the measurement: an image-conditioned model, short free-text answers "
               "with no exact checker, labels from an LLM judge, 8 benchmarks cross-fitted by image, a cross-family judge and lenient EM, and a controlled "
               "comparison with a published medical selector. Some of their findings cut against our choices: HSRM's Bradley–Terry > BCE, SWIFT's gate > mean, "
               "Q-Probe's within-question loss > CE."),
              ("Why not fine-tune the generator on the same labels instead?",
               "A frozen generator keeps the model unchanged and the verifier swappable. A generator LoRA-SFT baseline exists only from an earlier era "
               "(+0.0142 at 1× cost, ckpts/cheapleg/scores_*) and has not been re-run against the final method; it is owed. Do not set it beside +0.0798 as a ranking.")])
