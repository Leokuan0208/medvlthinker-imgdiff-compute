"""deck_part3.py — Part V results, Part VI comparison with published work, Part VII limits/next, appendix."""
import json
from deck_lib import C, fmt, forest, bars, lines
import deck_data as D

pm, ci = D.pm, D.ci
EXN = json.load(open(f"{D.DK}/extra_numbers.json"))
# statistics computed for this deck from the shipped per-question records (part3_numbers.py, part3_rows.py)
RV = {k: v["value"] for k, v in json.load(open(f"{D.DK}/numbers_review_part3.json")).items()}
RVA = RV  # round-2 keys (r2_part3_extra.py) live in the same file
RVSRC = "numbers_review_part3.json (part3_numbers.py on final_eval_FINAL_cap320_T07_records.json; image-clustered bootstrap, 10,000 resamples)"


def build(d):
    fin = D.fin
    J, X, E, F1, L = (D.macro(k) for k in ["judge", "xjudge", "em", "f1", "llava_recall"])
    h1 = D.fold1["macro"] if "macro" in D.fold1 else D.fold1
    cap = RV["judge.headroom_captured"]
    hdr = RV["judge.headroom_oracle_minus_greedy"]
    rows_tr = RV["training_rows_by_source"]
    wa = RV["wed.game_active_subset"]

    # ================================================================== PART V
    d.part = "V · Results"
    d.add("p5", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART V</div>
<h1>Results</h1><p>Shipped method: N = 8, T = 0.7, cap320, 24 probes, rank average + vote tie-break.<br>36,869 questions, every one scored by probes that never saw its image (exact pixel match).</p></div>""",
          raw=True, minutes=0.2, say="# Results\n- Part five: results.")

    rows = ""
    for lab, k in [("cross-family judge (MedGemma-27B) — never used in training", "xjudge"),
                   ("judge (MedVLThinker-32B) — also the training-label source", "judge"),
                   ("lenient exact match (equality or containment)", "em"),
                   ("token-F1", "f1"), ("LLaVA-Med open recall (its code)", "llava_recall")]:
        m = D.macro(k)
        f = h1[k]
        rows += (f"<tr><td>{lab}</td><td class='n'><b>{pm(m[0])}</b></td><td class='n'>{ci(m[1], m[2])}</td><td class='n'>{m[4]}/8</td>"
                 f"<td class='n'>{pm(f['macro'])} {ci(f['ci'][0], f['ci'][1])}</td></tr>")
    d.add("headline", "Headline: significant macro gains under every metric",
          f"""
<table class="d"><tr><th>metric</th><th class="n">gain over greedy</th><th class="n">95% CI</th><th class="n">benchmarks Δ &gt; 0 (point est.)</th><th class="n">image half B only</th></tr>{rows}</table>
<div class="row" style="margin-top:16px">
 <div class="col"><div class="term"><b class="k">per benchmark, judge</b><span>6 significant wins · 2 ties (VQA-RAD, VQA-Med)</span></div></div>
 <div class="col"><div class="term w"><b class="k">lenient exact match</b><span>small gain; loses on Kvasir-x1, GEMeX and VQA-Med</span></div></div>
</div>""", "V · Results", src="final_eval_FINAL_cap320_T07.json macro_over_open8.all.pick_minus_greedy_*; final_fold1_headline_2026-10-16.json policies.T07.macro",
          minutes=2.0,
          look="the first row (cross-family judge), then the last column (image half B).",
          say=f"""# Headline: significant macro gains under every metric
- The headline: the gain over the model's own greedy answer, averaged over the eight benchmarks.
# cross-family judge (MedGemma-27B) — never used in training
- Under a judge from another model family, MedGemma-27B, the method gains {X[0]:.4f}.
- The probe never saw this judge during training. That is the number I lead with.
# judge (MedVLThinker-32B) — also the training-label source
- Under our main judge, MedVLThinker-32B, the gain is {J[0]:.4f}.
- But that judge also produced the probe's training labels, so it is not independent.
# lenient exact match (equality or containment)
- Lenient exact match counts an answer right if it equals or contains the gold string; it gains {E[0]:.4f}.
# token-F1
- Token overlap gains {F1[0]:.4f}; LLaVA-Med's recall metric {L[0]:.4f}.
- All five intervals exclude zero.
# benchmarks Δ > 0 (point est.)
- This column counts positive point estimates, not significant wins.
# per benchmark, judge
- Per benchmark, under the judge, six of eight are significant wins and two are ties.
# image half B only
- The last column uses only image half B.
- Half B played no part in the September choices — layers, probe size, loss — nor in the readout tie-break or the temperature decision.
- The October check that kept the cap320 image budget used both halves.
- The gains on half B are, if anything, slightly larger.
# lenient exact match
- Exact match gains much less than the judges.
- Per benchmark it loses on Kvasir, GEMeX and VQA-Med; I will explain why on the next slide.""",
          qa=[("Isn't the +0.08 partly the probe learning the judge's taste?",
               "See the answer on the opening slide: the cross-family judge, which was never used in training, gives "
               f"{X[0]:+.4f} [{X[1]:+.4f}, {X[2]:+.4f}]; the 0.009 gap to the training judge matches the measured paraphrase drift (0.006–0.009)."),
              ("Were hyperparameters tuned on the test data?",
               f"The September choices (layers, width, loss, T) were made on image half A. Half B alone: judge {pm(h1['judge']['macro'])} {ci(*h1['judge']['ci'])}, "
               f"larger than half A ({RV['judge.fold0_macro_pt']:+.4f}). The October keep-cap320 check used both halves but kept the incumbent."),
              ("Is the exact-match gain real?",
               f"Significant on these eight benchmarks. It depends on OmniMedVQA and RadImageNet: drop either and it becomes a tie "
               f"({RV['em.leave_one_benchmark_out']['omnimed_open']['pt']:+.4f} {ci(*RV['em.leave_one_benchmark_out']['omnimed_open']['ci'])}; "
               f"{RV['em.leave_one_benchmark_out']['radimagenet_open']['pt']:+.4f} {ci(*RV['em.leave_one_benchmark_out']['radimagenet_open']['ci'])}). "
               f"With benchmarks as the sampling unit it is {ci(*RV['em.benchmark_as_unit_t_interval']['ci'])}, a tie. Source: {RVSRC}.")])

    # per benchmark forest
    rws = []
    for c in D.OPEN8:
        a = D.cell(c)
        rws.append((D.SHORT[c], [(a["pick_minus_greedy_judge"], *a["ci_pick_minus_greedy_judge"]),
                                 (a["pick_minus_greedy_xjudge"], *a["ci_pick_minus_greedy_xjudge"]),
                                 (a["pick_minus_greedy_em"], *a["ci_pick_minus_greedy_em"])]))
    eb = RV["em.per_benchmark_bonferroni"]
    d.add("perbench", "Per benchmark: where it helps, where it ties, where exact match disagrees",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 700px">{forest(rws, -0.10, 0.20, 0.05, w=700, rowh=46, series=[("judge", C["indigo"]), ("MedGemma", C["teal"]), ("lenient EM", C["tap"])], xlabel="probe − greedy (accuracy)", value_fmt=lambda v: fmt(v,3))}
 <div class="note">filled = 95% CI excludes 0 · hollow = tie · value shown = judge</div></div>
 <div class="col">
  <div class="term g"><b class="k">large gains</b><span>OmniMedVQA · RadImageNet · GEMeX · Kvasir-x1: +0.11 to +0.16</span></div>
  <div class="term"><b class="k">ties</b><span>VQA-RAD +0.000 · VQA-Med −0.004 — correct samples exist</span></div>
  <div class="term w"><b class="k">exact-match losses</b><span>Kvasir-x1, GEMeX: sentence golds · VQA-Med: real</span></div>
  <div class="note">Bonferroni ×8: 6 judge wins hold (even ×40); GEMeX EM loss → tie {ci(*eb['gemex_open']['ci_bonf8'])}</div>
 </div>
</div>""", "V · Results", src="final_eval_FINAL_cap320_T07.json cells.*.all.{pick_minus_greedy,ci_pick_minus_greedy}_{judge,xjudge,em}; Bonferroni: " + RVSRC,
          minutes=2.0,
          look="filled vs hollow markers; the indigo (judge) and teal (MedGemma) dots move together, the violet (lenient exact match) ones do not.",
          say="""# Per benchmark: where it helps, where it ties, where exact match disagrees
- Per benchmark; filled markers are significant, hollow ones are ties.
# large gains
- The largest gains, 11 to 16 points under the judge and 10 to 15 under MedGemma, are on OmniMedVQA, RadImageNet, GEMeX and Kvasir.
- The two judges agree closely.
# ties
- Two benchmarks tie: VQA-RAD, which has only 200 questions so its interval is wide, and VQA-Med.
- Neither tie is for lack of correct answers among the samples.
- I will show in two slides that the probe simply fails to pick them there.
# exact-match losses
- Exact match disagrees on three benchmarks.
- On Kvasir and GEMeX the gold answers are whole sentences generated by a language model.
  - A correct answer in different words fails string matching, while both judges accept it.
- On VQA-Med the exact-match loss is small but real.
# Bonferroni ×8
- If you correct for testing eight benchmarks, all six judge wins remain.
- The GEMeX exact-match loss becomes a tie.""",
          qa=[("You test 8 benchmarks × 5 metrics. Multiple comparisons?",
               f"Bonferroni: the 6 judge wins survive even a 40-test correction (smallest lower bound SLAKE "
               f"{RV['judge.per_benchmark_bonferroni']['slake_open']['ci_bonf40'][0]:+.4f}). MedGemma-SLAKE does not at ×40 "
               f"{ci(*RV['xjudge.per_benchmark_bonferroni']['slake_open']['ci_bonf40'])}. EM: GEMeX loss becomes a tie at ×8; Kvasir-x1 and VQA-Med losses hold. Source: {RVSRC}.")])

    # fixed / harmed
    fr = []
    for c in D.OPEN8:
        e = EXN["cells"][c]
        fr.append(f"<tr><td>{D.SHORT[c]}</td><td class='n'>{e['n']:,}</td><td class='n win'>{e['fixed']:,}</td><td class='n loss'>{e['harmed']:,}</td>"
                  f"<td class='n'>{e['fix_rate_of_recoverable']:.2f}</td><td class='n'>{e['harm_rate_of_greedy_right']:.3f}</td></tr>")
    t = EXN["total"]
    vm = EXN["cells"]["vqamed_open"]
    pdg = RV["pick_differs_from_greedy_string"]
    d.add("fixharm", "Fixes vs harms: what the selector changes, question by question",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 720px"><table class="d"><tr><th>benchmark</th><th class="n">questions</th><th class="n">fixed</th><th class="n">harmed</th><th class="n">fixed ÷ recov.</th><th class="n">harmed ÷ greedy-right</th></tr>
 {''.join(fr)}<tr class="hi"><td><b>total</b></td><td class="n">{t['n']:,}</td><td class="n win"><b>{t['fixed']:,}</b></td><td class="n loss"><b>{t['harmed']:,}</b></td><td class="n">{t['fixed']/t['recoverable']:.2f}</td><td class="n">{t['harmed']/t['greedy_right']:.3f}</td></tr></table></div>
 <div class="col">
  <div class="term"><b class="k">fixed / harmed</b><span>greedy wrong → pick right / greedy right → pick wrong (Hu 2026)</span></div>
  <div class="term w"><b class="k">VQA-Med: selection failure</b><span>{vm['recoverable']} recoverable · {vm['fixed']} fixed · {vm['harmed']} broken · answer changed on {pdg['vqamed_open']:.0%}</span></div>
 </div>
</div>""", "V · Results", src="extra_numbers.json (from final_eval_FINAL_cap320_T07_records.json, judge); answer-changed rate: numbers_review_part3.json; accounting: Hu 2026 arXiv:2607.17531 Eq. 1–6",
          minutes=1.5,
          look="the fixed and harmed columns, then the VQA-Med row.",
          say=f"""# fixed / harmed
- A net gain hides two movements, so here they are separately, using the accounting from Hu's 2026 paper.
- "Fixed": greedy was wrong and our pick is right.
- "Harmed": greedy was right and our pick is wrong.
- "Recoverable": greedy was wrong but at least one of the eight samples was right.
# total
- Over all 36 thousand questions, the selector fixes {t['fixed']:,} questions that greedy got wrong.
- It breaks {t['harmed']:,} that greedy got right.
- {t['harmed_unavoidable_no_correct_candidate']} of those could not have been avoided by any selector, because none of the eight samples was correct.
# harmed ÷ greedy-right
- On most benchmarks harms are a few percent of greedy's correct answers.
# VQA-Med: selection failure
- VQA-Med is the exception: {vm['recoverable']} questions where greedy is wrong but a correct sample exists.
- Yet the probe fixes only {vm['fixed']} and breaks {vm['harmed']} — {vm['harm_rate_of_greedy_right']:.0%} of greedy's correct answers.
- That is not a shortage of correct samples; the probe cannot tell them apart there.""",
          qa=[("Does the method ever make things worse?",
               f"Yes, per question always some, and on VQA-Med net: {vm['fixed']} fixed vs {vm['harmed']} broken, with a significant lenient-EM loss "
               f"{pm(D.cell('vqamed_open')['pick_minus_greedy_em'])} {ci(*D.cell('vqamed_open')['ci_pick_minus_greedy_em'])}. There is no per-benchmark guarantee of never being worse than greedy.")])

    # where it helps: greedy/pick/oracle per benchmark
    cats = [D.SHORT[c] for c in D.OPEN8]
    ser = [("greedy", C["grey"], [D.cell(c)["greedy_judge"] for c in D.OPEN8]),
           ("probe pick", C["teal"], [D.cell(c)["pick_judge"] for c in D.OPEN8]),
           ("oracle@8", C["indigo_l"], [D.cell(c)["oracle_judge"] for c in D.OPEN8])]
    hc = " · ".join(f"{D.SHORT[c]} {cap[c]:.2f}".replace("-", "−") for c in D.OPEN8)
    d.add("helps", "Gain follows headroom; selection fails on VQA-RAD, VQA-Med",
          f"""{bars(cats, ser, 0, 1.0, 0.2, w=1180, h=310, value_fmt=lambda v: f"{v:.2f}", cat_fs=12.5)}
<div class="row" style="margin-top:6px">
 <div class="col"><div class="term t"><b class="k">headroom captured = (pick − greedy) ÷ (oracle@8 − greedy)</b><span>{hc}. Macro <b>{cap['macro']:.2f}</b>.</span></div></div>
 <div class="col" style="flex:0 0 440px"><div class="term"><b class="k">pattern</b><span>gain tracks headroom (r = {RV['judge.corr_gain_vs_headroom_8_benchmarks']:.2f}) far more than the oracle level (r = {RV['judge.corr_gain_vs_oracle_8_benchmarks']:.2f}; 8 benchmarks, descriptive). VQA-Med has headroom {hdr['vqamed_open']:.3f} and captures none of it.</span></div></div>
</div>""", "V · Results", tall=True, src="final_eval_FINAL_cap320_T07.json cells.*.all.{greedy,pick,oracle}_judge; headroom / correlations: " + RVSRC,
          minutes=1.5,
          look="the gap between the grey and teal bars, then between teal and blue, benchmark by benchmark.",
          say=f"""# greedy · probe pick · oracle@8
- This chart puts greedy, our pick, and the oracle side by side for each benchmark.
- The gap between the grey and teal bars is our gain.
- The gap between the teal and blue bars is what a perfect selector would still add.
# headroom captured = (pick − greedy) ÷ (oracle@8 − greedy)
- The fair way to score a selector: the share of the available headroom it captures — the gain divided by oracle minus greedy.
- Averaged over benchmarks we capture {cap['macro']:.2f} of the headroom.
  - {cap['omnimed_open']:.2f} on OmniMedVQA; about two-thirds on RadImageNet, Kvasir and GEMeX.
  - About a third on PathVQA and SLAKE; nothing on VQA-RAD and VQA-Med.
- Those two are not out of reach: VQA-Med still has {hdr['vqamed_open']:.3f} of headroom.
- The probe just cannot pick the right sample there.
# pattern
- Across benchmarks the gain follows the headroom closely, much more than it follows how high the oracle is.""",
          qa=[("You said selection efficiency is 0.81 — which is it, 0.81 or 0.44?",
               f"Different quantities. pick ÷ oracle@8, averaged over benchmarks, is {D.SELEFF:.2f}; it is flattered because greedy alone already reaches "
               f"most of the oracle (VQA-RAD scores 0.78 with zero gain). The share of the headroom captured, (pick − greedy)/(oracle − greedy), is {cap['macro']:.2f}. "
               "The second is the honest measure of the selector."),
              ("Why does it fail on VQA-RAD and VQA-Med?",
               "Not for lack of correct samples. On VQA-RAD 17 fixed vs 17 broken; on VQA-Med 139 vs 153. We have no diagnosis yet; the within-question "
               "AUROC (slide on gaps) is the measurement that would show whether the probe ranks at random there. VQA-Med is also mostly the 2019 training split, "
               "which Lingshu was fine-tuned on (FINAL_METHOD_2026-10-16.md §6.4).")])

    # does it learn an answer prior?
    psh = RVA["judge.answer_prior_share_per_benchmark_sept_halfA"]
    shr = " · ".join(f"{D.SHORT[c]} {psh[c]['prior_share_of_probe_gain']:.0%}" for c in ["radimagenet_open", "omnimed_open", "gemex_open", "kvasir_x1_open"])
    d.add("memo", "Does the probe just learn which answers are common?",
          f"""
<div>
 <div>
  <table class="d"><tr><th>question / gold seen in training?</th><th class="n">n</th><th class="n">judge gain</th><th class="n">lenient-EM gain</th></tr>
  <tr><td>question template seen</td><td class="n">11,630</td><td class="n">+0.108</td><td class="n">+0.018</td></tr>
  <tr><td>question template <b>never seen</b></td><td class="n">6,822</td><td class="n">+0.116</td><td class="n loss">−0.015</td></tr>
  <tr><td>gold answer seen</td><td class="n">11,764</td><td class="n">+0.122</td><td class="n">+0.031</td></tr>
  <tr><td>gold answer <b>never seen</b></td><td class="n">6,688</td><td class="n">+0.092 [+0.082, +0.101]</td><td class="n loss">−0.039 [−0.046, −0.032]</td></tr>
  <tr><td>within PathVQA: template seen / never seen</td><td class="n">1,285 / 338</td><td class="n">+0.067 / +0.018 [−0.021, +0.056]</td><td class="n">+0.037 / −0.027</td></tr></table>
  <div class="note">September probe (genframe_head_pooled_ens_v2), image half A; first four rows pooled over benchmarks</div>
 </div>
</div>
<div class="row" style="margin-top:14px">
 <div class="col" style="flex:1.6">
  <div class="term w"><b class="k">answer-frequency prior: share of the probe's judge gain</b><span>macro ~25% · per benchmark: {shr} · below greedy on PathVQA, SLAKE, VQA-Med</span></div>
  <div class="term"><b class="k">so</b><span>the exact-match macro win rests on RadImageNet and OmniMedVQA — where the prior is strongest</span></div>
 </div>
 <div class="col">
  <div class="term t"><b class="k">claim</b><span>not <i>only</i> an answer prior · “no memorisation” not claimed (image-shuffle control owed)</span></div>
 </div>
</div>""", "V · Results", src="template_stratified_gain_2026-09-20.json pooled_micro, cells.pathvqa_open.{repeat,novel}; audit_stats_baselines_2026-09-20.json summary_by_currency.judge and per_benchmark.*.judge (September probe, half A; shares: numbers_review_part3.json, r2_part3_extra.py)",
          minutes=2.0,
          look="the prior's share per benchmark (top-right box), then the 'never seen' rows of the table.",
          say=f"""# Does the probe just learn which answers are common?
- A fair question is whether the probe simply learned which answers are common. Two checks.
# answer-frequency prior: share of the probe's judge gain
- First, a baseline that scores each answer only by how often that exact string was correct in training.
- Over all benchmarks it recovers about a quarter of the probe's gain. But per benchmark it is uneven.
- On OmniMedVQA and RadImageNet many questions ask for a modality or an organ from a small vocabulary.
  - There the prior alone gets about two-thirds of the way — {psh['omnimed_open']['prior_share_of_probe_gain']:.0%} and {psh['radimagenet_open']['prior_share_of_probe_gain']:.0%} of the probe's gain.
- So on those two, a large part of the gain is knowing which answers are usually right, and the probe adds the rest.
- On GEMeX the prior gets less than half, on Kvasir about a quarter, and on PathVQA, SLAKE and VQA-Med it is below greedy.
# so
- Those are exactly the two benchmarks the exact-match macro win rests on.
# question template never seen
- Second, pooled over all benchmarks, the judge gain is just as large on question templates the probe never saw.
# gold answer never seen
- And still clearly positive on gold answers it never saw.
# within PathVQA: template seen / never seen
- But careful: those groups come mostly from different benchmarks.
- Inside PathVQA alone the gain on unseen templates is not significant.
# claim
- What the evidence supports is "not only an answer prior", not "no memorisation".
- The test that would settle it — swapping in a wrong image — has not been run yet.
# lenient-EM gain
- On unseen answers exact match is a loss, partly because those strata are the sentence-gold benchmarks where exact match is unreliable.""",
          qa=[("Is OmniMedVQA / RadImageNet just an answer prior, or a modality classifier?",
               f"Largely, on those two: the leave-one-out answer prior reaches {psh['omnimed_open']['prior_share_of_probe_gain']:.0%} / "
               f"{psh['radimagenet_open']['prior_share_of_probe_gain']:.0%} of the probe's judge gain (prior {psh['omnimed_open']['prior_minus_greedy']:+.4f} vs probe "
               f"{psh['omnimed_open']['probe_minus_greedy']:+.4f} on OmniMedVQA; {psh['radimagenet_open']['prior_minus_greedy']:+.4f} vs {psh['radimagenet_open']['probe_minus_greedy']:+.4f} "
               "on RadImageNet). September probe, image half A, audit_stats_baselines_2026-09-20.json per_benchmark; the prior is leave-one-out and optimistic. "
               "The probe adds the rest; the image-shuffle control that would separate image use from answer frequency is owed."),
              ("Why does the exact-match gain turn negative on unseen golds?",
               "Those strata are dominated by GEMeX and Kvasir-x1, whose golds are LLM-written sentences where string matching rejects paraphrases; "
               "so it is not clean evidence either way. The decisive controls (image-shuffle; Hewitt–Liang control task) are on the gap list.")])

    # second recipe on the base model
    d.add("repl", "A second training recipe of the same language model: Qwen2.5-VL-7B",
          """
<div class="row">
 <div class="col">
  <table class="d"><tr><th>Qwen2.5-VL-7B</th><th class="n">probe − greedy</th><th class="n" style="padding-right:16px">sig.</th></tr>
  <tr><td>judge</td><td class="n win">+0.0814 [+0.0708, +0.0924]</td><td class="n" style="padding-right:16px">6 / 8</td></tr>
  <tr><td>lenient EM</td><td class="n win">+0.0276 [+0.0168, +0.0384]</td><td class="n" style="padding-right:16px">3 / 8</td></tr>
  <tr><td>strict EM</td><td class="n win">+0.0187 [+0.0076, +0.0298]</td><td class="n" style="padding-right:16px">4 / 8</td></tr>
  <tr><td>token-F1</td><td class="n win">+0.0317 [+0.0217, +0.0425]</td><td class="n" style="padding-right:16px">6 / 8</td></tr></table>
  <div class="note">same pipeline, September recipe: single layer 20, 5 seeds, image half A only, rank readout, judge = MedVLThinker-32B. Qwen greedy macro judge 0.303. sig. = benchmarks with a significant gain.</div>
 </div>
 <div class="col" style="flex:0 0 470px">
  <div class="term w"><b class="k">same family</b><span>Qwen2.5-VL-7B is Lingshu's own base model, so this is a second training recipe on the same language model — not an independent replication.</span></div>
  <div class="term r"><b class="k">withdrawn</b><span>a MedGemma-4b replication was withdrawn: its sampled answers ran to the length cap with template garbage (58.5%).</span></div>
  <div class="term"><b class="k">owed</b><span>a properly decoded non-Qwen generator.</span></div>
 </div>
</div>""", "V · Results", src="replication_currency_2026-09-20.json qwen.macro.probe_minus_greedy_{judge,em,strict_em,token_f1}; AUDIT_2026-09-18.md §3",
          minutes=1.0,
          say="""# Qwen2.5-VL-7B
- Does it work on another model? On Qwen2.5-VL-7B the same pipeline, in its September form, gains eight points under the judge.
- It is significant under every metric, including strict exact match.
# same family
- But Qwen is Lingshu's own base model: the same language model with different training, not an independent replication.
- And it uses the same judge.
# withdrawn
- An attempt on MedGemma-4b was withdrawn because its sampled answers were broken.
# owed
- A replication on a genuinely different model family is still owed.""",
          qa=[("Is the Qwen result an independent replication?",
               "No: Lingshu's base model, the September recipe (single layer, 5 seeds, half A), the same judge. The +0.0814 here is the single-layer refit; "
               "the pooled ensemble in head_final_stack_qwen_2026-09-13.json gives +0.0820.")])

    # ================================================================== PART VI
    d.part = "VI · Comparison"
    d.add("p6", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART VI</div>
<h1>Against published work</h1><p>Three levels of comparability: same candidates · same metric code and test split · different protocol (context only).</p></div>""",
          raw=True, minutes=0.2, say="# Against published work\n- Part six: how this compares with published work.")

    d.add("tiers", "How to compare honestly",
          """
<table class="d">
<tr><th>tier</th><th>what is held fixed</th><th>comparisons</th><th>what may be claimed</th></tr>
<tr class="hl"><td><b>1 · same candidates</b></td><td>images, questions, the 8 samples, graders — only the selector differs</td><td>WED (ran the authors' code on our pools)</td><td>which selector is better on these candidates, with CIs</td></tr>
<tr><td><b>2 · same metric code / test split</b></td><td>the official test questions and the scoring code</td><td>LLaVA-Med recall & closed accuracy; ImageCLEF VQA-Med 2019 test</td><td>where our numbers sit — not a ranking (different generators, training data)</td></tr>
<tr><td><b>3 · different protocol</b></td><td>nothing</td><td>Lingshu's published numbers; Best-of-Evidence; MedProb App. H; Verification Mirage</td><td>context only — never a ranking</td></tr>
</table>
<div class="term w" style="margin-top:16px"><b class="k">why this matters</b><span>the same Lingshu-7B scores 61.9 on PathVQA in its own paper (open: GPT-4.1 judge; closed: rule-scored) and 71.9 in an independent benchmark (step-by-step \\boxed{} prompt, option-matched scoring, item set not stated): evaluation choices alone move the same model by ~10 points.</span></div>""",
          "VI · Comparison", src="Lingshu Table 6 p.17, §4.2 p.16; Liu et al. 2025 Table 1 p.4 (Lingshu-7B PathVQA 0.7192) — PAPER_NOTES_GH G4",
          minutes=1.0,
          say="""# How to compare honestly
- Before the numbers, how to compare honestly.
# 1 · same candidates
- The strongest comparison holds everything fixed except the selector: same questions, same eight samples, same graders.
- We have one of those, against WED.
# 2 · same metric code / test split
- The second level uses the same official test split and the same scoring code.
- That tells you where our numbers sit, but the generators differ, so it is not a ranking.
# 3 · different protocol
- The third level is everything else, which is context only.
# why this matters
- The very same Lingshu-7B model is reported at 61.9 on PathVQA in its own paper and 71.9 in an independent benchmark.
- The prompt, the grader, the answer format and possibly the question set all differ.
- Evaluation choices alone move the same model by about ten points.""")

    # WED
    w = D.wed["cells"]

    def dci(e, key):
        v = e[key]
        return (v["delta"], v.get("ci")) if isinstance(v, dict) else (v, None)

    def cls_of(c_):
        return "win" if c_ and c_[0] > 0 else ("loss" if c_ and c_[1] < 0 else "tie")
    wrows = ""
    for c in ["pathvqa_open", "slake_open", "vqa_rad_open"]:
        e = w[c]
        for k_, (cur, lab) in enumerate([("judge", "judge"), ("em", "lenient EM")]):
            dv, dc = dci(e, f"probe_minus_wed_{cur}")
            wv, wc = dci(e, f"wed_minus_greedy_{cur}")
            wrows += (f"<tr><td>{D.SHORT[c] if k_ == 0 else ''}</td><td>{lab}</td><td class='n'>{e[f'greedy_{cur}']:.3f}</td><td class='n'>{e[f'wed_{cur}']:.3f}</td>"
                      f"<td class='n'><b>{e[f'probe_{cur}']:.3f}</b></td><td class='n {cls_of(dc)}'>{pm(dv)} {ci(*dc) if dc else ''}</td>"
                      f"<td class='n {cls_of(wc)}'>{pm(wv)}</td></tr>")
    pw = w["pathvqa_open"]["wed_minus_greedy_judge"]
    act = " · ".join(f"{D.SHORT[c]} {pm(wa[c]['probe_minus_wed_judge_active']['pt'])} {ci(*wa[c]['probe_minus_wed_judge_active']['ci'])} (n = {wa[c]['game_active_scored']:,})"
                     for c in ["pathvqa_open", "slake_open", "vqa_rad_open"])
    wv7 = RVA["wed.verdict_per_grader"]
    gname = {"judge": "judge", "xjudge": "MedGemma", "em": "lenient EM", "wed_em": "WED's EM", "f1": "token-F1", "wed_f1": "WED's F1", "llava_recall": "LLaVA recall"}
    other = "; ".join(f"{D.SHORT[c]} " + ", ".join(f"{gname[g]} {pm(dci(w[c], f'probe_minus_wed_{g}')[0])} {ci(*dci(w[c], f'probe_minus_wed_{g}')[1])}"
                                                   for g in ["xjudge", "wed_em"])
                      for c in ["pathvqa_open", "slake_open", "vqa_rad_open"])
    d.add("wedh2h", "Tier 1 — head-to-head with WED on identical candidates",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 720px"><table class="d"><tr><th>benchmark</th><th>metric</th><th class="n">greedy</th><th class="n">WED</th><th class="n">probe</th><th class="n">probe − WED</th><th class="n">WED − greedy</th></tr>{wrows}</table>
 <div class="note">2 of the 7 metrics shown; all 7 in the script and in wed_vs_probe_2026-10-16.json</div></div>
 <div class="col">
  <div class="term g"><b class="k">result (7 metrics)</b><span>PathVQA 7 wins · SLAKE 6 wins, lenient EM ties · VQA-RAD 7 ties (n = 200)</span></div>
  <div class="term r"><b class="k">WED vs greedy</b><span>below greedy on PathVQA ({pm(pw['delta'])} judge)</span></div>
  <div class="term w"><b class="k">asymmetries</b><span>WED's own sampling loop not run · WED read cap320 images, the probe's features the 1-megapixel cap · WED needs no labels</span></div>
 </div>
</div>""", "VI · Comparison", src="wed_vs_probe_2026-10-16.json cells.*.{probe_minus_wed,wed_minus_greedy}_* (T = 0.7 pools; authors' code, src/exp_2026_10/wed/wed_select.py); game-active subset: numbers_review_part3.json; WED Table 1 p.8",
          minutes=2.0,
          look="the probe − WED column, PathVQA first; then the red box.",
          say=f"""# Tier 1 — head-to-head with WED on identical candidates
- The controlled comparison: we ran WED's own code on our candidates, so only the way the answer is chosen differs.
- We score it under seven metrics; the table shows two, the judge and lenient exact match.
# probe − WED
- On PathVQA the probe beats WED under all seven, by about eight points under the judge.
# result (7 metrics)
- On SLAKE it wins under six of seven; under our lenient exact match it ties.
- On VQA-RAD everything ties — with 200 questions that means we cannot tell, not that they are equal.
# WED vs greedy
- One result needs explaining: on PathVQA, WED is significantly worse than greedy decoding.
- Its own paper reports gains there.
# asymmetries
- Two asymmetries matter.
- First: WED normally runs its own sampling loop until it has eight distinct answers; we gave it our eight samples instead.
  - After its clean-up step it often had only one candidate left — on {wa['slake_open']['wed_le1_candidate']} of 645 SLAKE questions and {wa['pathvqa_open']['wed_le1_candidate']:,} of 3,357 PathVQA questions.
  - So its game never ran there.
- Second: WED read the image at our generation budget, cap320, while our probe's features used the higher one-megapixel cap.
- Check: restrict to the questions where WED's game actually runs. There the probe's margin is larger, not smaller.
  - {wa['pathvqa_open']['probe_minus_wed_judge_active']['pt']*100:.1f} points on PathVQA and {wa['slake_open']['probe_minus_wed_judge_active']['pt']*100:.1f} on SLAKE under the judge.
- Finally, in fairness: WED needs no training, ours needs labelled data.""",
          qa=[("What are the seven metrics, and the other numbers?",
               "judge, MedGemma, lenient EM, WED's own EM, token-F1, WED's F1, LLaVA recall (wed_vs_probe_2026-10-16.json; verdicts in numbers_review_part3.json "
               f"wed.verdict_per_grader). MedGemma and WED's EM, probe − WED: {other}."),
              ("Where WED's game actually runs?",
               f"Restricting to questions where WED kept ≥ 2 candidates, probe − WED under the judge: {act} (image-clustered bootstrap, numbers_review_part3.json)."),
              ("Did you beat WED, or a crippled WED?",
               "Same 8 candidates, the authors' code for scoring, game and stopping. Where WED's game runs (≥ 2 candidates) the probe's margin is larger "
               f"({act}). None of WED's four internal variants (BDG-W, BDG-classic, verifier-only, generator-only) beats greedy on PathVQA "
               "(point estimates 0.270–0.286 vs greedy 0.314; review/scr/wed_variants.py — no CIs, not on a slide). A 'WED as published' arm with its own "
               "sampling loop is owed."),
              ("Why is WED below greedy on PathVQA when its paper reports gains?",
               "Different operating regime (our fixed 8 samples, often 1–2 distinct candidates after its canonicalisation), a different generator (Lingshu-7B "
               "vs its 2–8B general VLMs) and lower image resolution in its scoring pass. We report it as measured and do not claim WED is broken in general.")])

    # LLaVA-Med protocol
    fos = fin["full_official_splits"]

    def r(sp):
        a = fos[sp]
        return (a["greedy_llava_open_recall"] * 100, a["pick_llava_open_recall"] * 100, a["greedy_llava_closed_acc"] * 100, a["pick_llava_closed_acc"] * 100)
    rad, slk, pth = r("vqa_rad"), r("slake_en"), r("pathvqa")

    def rc(c):
        a = D.cell(c)
        return f"{pm(a['pick_minus_greedy_llava_recall'])} {ci(*a['ci_pick_minus_greedy_llava_recall'])}"
    pc = rows_tr["fold0"]["per_cell"]["pathvqa_closed"], rows_tr["fold1"]["per_cell"]["pathvqa_closed"]
    d.add("llavamed", "Tier 2 — LLaVA-Med's metric on the full official test splits",
          f"""
<table class="d sm"><tr><th>model</th><th class="n">VQA-RAD open / closed</th><th class="n">SLAKE open / closed</th><th class="n">PathVQA open / closed</th><th>trained on the benchmark?</th></tr>
<tr><td>LLaVA-Med zero-shot (Table 4b range)</td><td class="n">28.2–37.8 / 60.7–61.4</td><td class="n">37.7–39.7 / 49.8–54.3</td><td class="n">11.3–12.3 / 49.1–54.1</td><td>no</td></tr>
<tr><td>LLaVA-Med fine-tuned per dataset (Table 4a)</td><td class="n">61.5–64.8 / 83.1–84.2</td><td class="n">83.1–87.1 / 85.3–86.8</td><td class="n">38.0–39.6 / 91.1–91.2</td><td>yes, each train split</td></tr>
<tr><td>Lingshu-7B greedy (ours)</td><td class="n">{rad[0]:.2f} / {rad[2]:.2f}</td><td class="n">{slk[0]:.2f} / {slk[2]:.2f}</td><td class="n">{pth[0]:.2f} / {pth[2]:.2f}</td><td>in Lingshu's training mix</td></tr>
<tr class="hl"><td><b>+ probe (ours)</b></td><td class="n"><b>{rad[1]:.2f} / {rad[3]:.2f}</b></td><td class="n"><b>{slk[1]:.2f} / {slk[3]:.2f}</b></td><td class="n"><b>{pth[1]:.2f} / {pth[3]:.2f}</b></td><td>+ probe trained on the other image half, open <i>and</i> closed</td></tr></table>
<div class="row" style="margin-top:14px">
 <div class="col"><div class="term t"><b class="k">claim</b><span>open recall: PathVQA {rc('pathvqa_open')} · SLAKE {rc('slake_open')} · VQA-RAD tie<br>PathVQA closed +0.0297 [+0.0197, +0.0397] — <b>in-domain</b> (trained on its other half)</span></div></div>
 <div class="col"><div class="term w"><b class="k">not a ranking</b><span>LLaVA-Med fine-tuned is higher on all six columns · test counts differ except SLAKE-English (645/416)</span></div></div>
</div>""", "VI · Comparison", src="final_eval_FINAL_cap320_T07.json full_official_splits.*, cells.*.all.ci_pick_minus_greedy_llava_recall, cells.pathvqa_closed; training rows: " + RVSRC.split(" (")[0] + " training_rows_by_source; LLaVA-Med arXiv:2306.00890 Tables 3–4; LLaVA-Med v1.0.0 eval code imported verbatim",
          minutes=1.5,
          look="the two bottom rows of the table (greedy vs + probe), then the claim box.",
          say=f"""# Tier 2 — LLaVA-Med's metric on the full official test splits
- Second level: LLaVA-Med's own metric, computed with their published code, on the full official test splits.
# + probe (ours)
- Under their metric our method improves open-question recall significantly on PathVQA and SLAKE; on VQA-RAD it ties.
# claim
- Closed yes/no accuracy rises significantly on PathVQA.
- But to be precise: the probe's training data included the other image half of the PathVQA closed questions.
  - {pc[0]:,} and {pc[1]:,} rows in the two folds.
  - So this is an in-domain gain, not transfer from open questions.
- SLAKE closed dips slightly, a tie.
# not a ranking
- Where do we sit? LLaVA-Med fine-tuned separately on each dataset is higher on all six columns.
- So this is not a ranking — different generator, different training.
- The test counts also differ: their VQA-RAD has 179 open and 272 closed questions against our 200 and 251.
- Their PathVQA 6,761 against our 6,719. SLAKE English is the one split that matches theirs exactly.""",
          qa=[("Did the probe train on closed questions?",
               f"Yes. Each fold's training set includes the other image half of VQA-RAD-, PathVQA- and SLAKE-closed and of the VQA-Med 2019 test: "
               f"{rows_tr['fold0']['closed3_plus_vqamed2019_other_half']:,} rows (fold 0) and {rows_tr['fold1']['closed3_plus_vqamed2019_other_half']:,} (fold 1) "
               f"of {rows_tr['fold0']['total']:,} / {rows_tr['fold1']['total']:,}. Image-disjoint from what is scored, so no leakage, but not transfer.")])

    # VQA-Med 2019 + Lingshu published
    vq = D.cell("vqamed2019_test")
    d.add("vqamed19", "Tier 2/3 — ImageCLEF VQA-Med 2019 and Lingshu's published numbers",
          f"""
<div class="row">
 <div class="col">
  <h3 style="font-family:'Roboto Slab';font-size:18px;margin-bottom:8px">VQA-Med 2019 official test (500)</h3>
  <table class="d sm"><tr><th></th><th class="n">strict accuracy</th><th class="n">judge</th></tr>
  <tr><td>2019 best team</td><td class="n">0.624</td><td class="n">—</td></tr>
  <tr><td>Lingshu-7B greedy</td><td class="n">{vq['greedy_imageclef']:.3f}</td><td class="n">{vq['greedy_judge']:.3f}</td></tr>
  <tr class="hl"><td>+ probe</td><td class="n">{vq['pick_imageclef']:.3f}</td><td class="n">{vq['pick_judge']:.3f}</td></tr></table>
  <div class="note">both deltas are ties; strict metric re-implemented, not the official evaluator; 2019 systems trained on the category vocabularies; our probe trained on the other image half of this test set ({rows_tr['fold0']['per_cell']['vqamed2019_test']} / {rows_tr['fold1']['per_cell']['vqamed2019_test']} rows).</div>
 </div>
 <div class="col">
  <h3 style="font-family:'Roboto Slab';font-size:18px;margin-bottom:8px">Lingshu's own Table 6 vs our protocol (open + closed, judge)</h3>
  <table class="d sm"><tr><th></th><th class="n">VQA-RAD</th><th class="n">SLAKE</th><th class="n">PathVQA</th></tr>
  <tr><td>ours, greedy</td><td class="n">{fos['vqa_rad']['greedy_judge']*100:.1f}</td><td class="n">{fos['slake_en']['greedy_judge']*100:.1f}</td><td class="n">{fos['pathvqa']['greedy_judge']*100:.1f}</td></tr>
  <tr class="hl"><td>ours, + probe</td><td class="n">{fos['vqa_rad']['pick_judge']*100:.1f}</td><td class="n">{fos['slake_en']['pick_judge']*100:.1f}</td><td class="n">{fos['pathvqa']['pick_judge']*100:.1f}</td></tr>
  <tr><td>Lingshu-7B as published</td><td class="n">67.9</td><td class="n">83.1</td><td class="n">61.9</td></tr>
  <tr><td>Lingshu-32B as published</td><td class="n">76.5</td><td class="n">89.2</td><td class="n">65.9</td></tr></table>
  <div class="term w"><b class="k">not comparable</b><span>their open questions are graded by GPT-4.1 and closed ones rule-scored (ours: our judge for both); their SLAKE includes the Chinese half (2,094 vs our 1,061 English); default image resolution. Only our two rows compare.</span></div>
 </div>
</div>""", "VI · Comparison", src="final_eval_FINAL_cap320_T07.json cells.vqamed2019_test, full_official_splits; VQA-Med 2019 overview (CEUR Vol-2380) p.1, p.6 (best team 0.624); Lingshu arXiv:2506.07044v4 Table 6 p.17 (= v5 Table I), §4.2 p.16",
          minutes=1.5,
          say="""# VQA-Med 2019 official test (500)
- Two more anchors. On the official 2019 VQA-Med test, the probe neither helps nor hurts — both changes are ties.
- Strict string accuracy is far below the best 2019 team.
  - Those systems were trained to output the challenge's fixed vocabulary; our model answers in free text.
# Lingshu's own Table 6 vs our protocol (open + closed, judge)
- On the right, Lingshu's own published scores. It is tempting to put our numbers next to theirs.
# not comparable
- But they grade open questions with GPT-4.1 and closed ones by rule.
- Their SLAKE includes the Chinese questions, and they use a different image resolution.
- So only our two rows are comparable to each other.
- I will not claim we match the 32-billion model.""")

    # medical TTC context
    pv = EXN["probe_minus_vote_judge"]
    SC = D.sc_macro("judge")      # majority vote − greedy, judge, macro (final_eval_FINAL_cap320_T07.json)
    d.add("context", "Tier 3 — medical test-time methods, each in its own units",
          f"""
<table class="d sm"><tr><th>method</th><th>domain · benchmarks</th><th>generator · N · grader</th><th><b>baseline</b></th><th>reported gain</th></tr>
<tr><td>Best-of-Evidence 2026</td><td>medical VQA · VQA-Med (train partition), PathVQA open, PMC-VQA, MedXpertQA-MM</td><td>Qwen3-VL-30B · 16 · 235B judge</td><td><b>majority vote</b></td><td>+0.26 to +0.58 pts (2 of 4 significant)</td></tr>
<tr class="hl"><td>ours, against the same baseline</td><td>open medical VQA · our 8 benchmarks</td><td>Lingshu-7B · 8 · MedVLThinker-32B</td><td><b>majority vote</b></td><td>+{pv['macro']*100:.1f} [{pv['ci'][0]*100:+.1f}, {pv['ci'][1]*100:+.1f}] pts judge — our vote is itself {-SC[0]*100:.1f} pts <b>below</b> greedy</td></tr>
<tr><td>WED 2026 (its paper)</td><td>open medical VQA · VQA-RAD, PathVQA open (no yes/no)</td><td>2–8B VLMs · ≤ 8 distinct (≤ 16 draws) · Grok-4.1-Fast</td><td><b>greedy</b></td><td>−1.2 to +5.0 pts (9 of 10 cells positive)</td></tr>
<tr><td>MedProb App. H 2026</td><td>medical VQA · OmniMedVQA, 100 items</td><td>Qwen2.5-VL-7B · not stated · not stated</td><td><b>one sample</b></td><td>0.09 → 0.13 on 100 questions</td></tr>
<tr class="hl"><td>ours, against greedy</td><td>open medical VQA · our 8 benchmarks</td><td>Lingshu-7B · 8 · MedGemma-27B (cross-family)</td><td><b>greedy</b></td><td>+{X[0]*100:.1f} [{X[1]*100:+.1f}, {X[2]*100:+.1f}] pts (+{J[0]*100:.1f} training judge)</td></tr>
<tr><td>Verification Mirage 2026</td><td>medical VQA · VQA-RAD, PathVQA, SLAKE, PMC-VQA, MedXpertQA</td><td>6 VLMs incl. Lingshu · — · LLM grader</td><td>—</td><td>self-verification accepts 62–73% of Lingshu's wrong answers</td></tr>
</table>
<div class="term w" style="margin-top:14px"><b class="k">context, not a ranking</b><span>different generators, graders and N; each row is in its own units. Compare only rows with the same baseline. The one controlled comparison is the WED head-to-head.</span></div>""",
          "VI · Comparison", src="BoE Table 2 p.9; WED p.5, Table 1 p.8; MedProb Table 14 p.27; Mirage Tables 11–15; ours final_eval_FINAL_cap320_T07.json and extra_numbers.json (probe − vote, paired image-clustered bootstrap)",
          minutes=1.5,
          say=f"""# Tier 3 — medical test-time methods, each in its own units
- For context, other medical test-time methods in their own units, grouped by baseline.
# Best-of-Evidence 2026
- Best-of-Evidence, with a 235-billion-parameter judge and sixteen samples, gains under one point over majority vote.
# ours, against the same baseline
- The matching number for us, also over majority vote, is {pv['macro']*100:.1f} points under our judge.
- But on our short free-text answers majority vote is already {-SC[0]*100:.1f} points below greedy.
- So this delta mostly shows how weak voting is here, not that we beat Best-of-Evidence.
# WED 2026 (its paper)
- WED's own paper reports between minus one and plus five points over greedy on small models.
# MedProb App. H 2026
- MedProb's appendix test is a hundred questions.
# ours, against greedy
- Over greedy we gain {X[0]*100:.1f} points under the cross-family judge.
# context, not a ranking
- These use different models, graders and question sets, so this is context, not a ranking.
- The only controlled comparison is the WED head-to-head.""")

    # ================================================================== PART VII
    d.part = "VII · Limits"
    d.add("p7", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART VII</div>
<h1>Limits, gaps, and next steps</h1><p>What the evidence does not support, what was never tested, and what comes next.</p></div>""",
          raw=True, minutes=0.2, say="# Limits, gaps, and next steps\n- Finally, what we cannot claim, and what comes next.")

    d.add("limits", "What the evidence does not support",
          """
<div class="row">
 <div class="col">
  <div class="term r"><b class="k">not a new mechanism</b><span>hidden-state probe verifiers exist (SWIFT, HSRM, CASE, MedProb)</span></div>
  <div class="term r"><b class="k">not domain-general</b><span>each benchmark needs labelled questions</span></div>
  <div class="term r"><b class="k">judge-sensitive</b><span>main judge = label source; lenient EM small, loses on 3</span></div>
  <div class="term r"><b class="k">no human evaluation</b><span>nobody has rated this method's picks</span></div>
  <div class="term r"><b class="k">one model family</b><span>Lingshu and its base model only</span></div>
 </div>
 <div class="col">
  <div class="term"><b class="k">cost, as run</b><span>sampling 1.13× FLOPs (vLLM) · re-read 6.90× (HF) · end to end not measured</span></div>
  <div class="term w"><b class="k">harms where selection fails</b><span>VQA-Med: 153 broken for 139 fixed</span></div>
  <div class="term"><b class="k">data exposure</b><span>VQA-Med cell mostly Lingshu's training split; effect on Δ not measured</span></div>
  <div class="term w"><b class="k">no claim against bigger models</b><span>vs Lingshu-32B greedy: judge 1 loss · 2 ties · 1 win</span></div>
 </div>
</div>""", "VII · Limits", src="literature/DOMAIN_GUIDE_2026-09-30.md §2.5 (branch worktree-lit-domain-package); AUDIT_2026-09-18.md; pathvqa_judge_audit.json; free_head_2026-08-16.json; bestofn_vllm_2026-09-16.json; extra_numbers.json; audit_stats_baselines_2026-09-20.json bigger_model_reference_32B",
          minutes=2.5,
          look="left column (what we cannot claim), then right column (costs and risks).",
          say="""# What the evidence does not support
- What the evidence does not support — left column first.
# not a new mechanism
- The mechanism is not ours: probes on hidden states used as best-of-N verifiers already exist in text models.
- MedProb showed one on a hundred medical questions.
# not domain-general
- Every one of our eight benchmarks contributed labelled training questions.
- A benchmark left out of training gains no more than a probe trained on the four-domain set.
# judge-sensitive
- The gains are much larger under the judges than under exact match.
- The main judge is also where the training labels came from — which is why I lead with the cross-family judge.
- Exact match loses on Kvasir, GEMeX and VQA-Med.
# no human evaluation
- No human has rated this method's answers.
- Earlier checks: an LLM adjudicator on SLAKE and VQA-RAD and a small hand-checked sample on PathVQA, all on older July outputs.
- Nothing on the other five benchmarks.
# one model family
- Lingshu and its own base model; a MedGemma-4b attempt was withdrawn.
# cost, as run
- Cost, as run, comes from two different measurements that I do not add together.
- Sampling eight answers with vLLM's shared prefill costs 1.13 times greedy in compute.
- The measured latency and energy, 2.74 and 3.59 times, come from twenty short VQA-RAD questions, not from the sentence-answer benchmarks.
- Separately, the re-read of each distinct answer at the one-megapixel cap, in the HuggingFace pipeline, costs about seven greedy passes.
- So "no extra forward pass" is true by design, not as we ran it; end-to-end cost and memory were not measured.
# harms where selection fails
- Where selection fails, on VQA-Med, the method breaks more answers than it fixes.
# data exposure
- Our VQA-Med set is mostly data the generator was fine-tuned on; whether that changes the delta is not measured.
# no claim against bigger models
- Against Lingshu-32B greedy we do not claim to match it.
- One loss, two ties and one win under the judge, on four benchmarks, with an earlier probe.""",
          qa=[("Has any human checked these answers?",
               f"Not on this method's picks. Earlier checks (July pools): an LLM adjudicator (claude_judge/verdicts_lingshu32b/) on SLAKE and VQA-RAD; "
               f"on PathVQA, 50 judge decisions on direct (short) answers, 2 wrong: 0.04, Wilson 95% [{RVA['judge_audit.wilson95_2_of_50'][0]:.3f}, "
               f"{RVA['judge_audit.wilson95_2_of_50'][1]:.3f}]; the same audit found more judge errors on long reasoning-mode answers (up to 0.28 on rejections) "
               "and a 90-item disagreement sample (clear judge error 7.8% [0.0382, 0.1519]) — pathvqa_judge_audit.json; who labelled those is not recorded. "
               "A stratified human rating is owed."),
              ("How often do your two judges agree?",
               "On September training candidates (112,770 rows) the two judges agree on 92.9% of labels, Cohen's κ = 0.79 (judge_2x2_2026-09-20.json train_label_agreement_AB and train_label_pos_rate; κ from the marginals, numbers_review_part2.json judge_agreement). MedGemma says 'correct' more often (23.9% vs 19.4%). Not yet measured on the October final pools. On the final method the per-benchmark deltas agree in sign on 7 of 8 benchmarks (VQA-RAD: +0.0000 vs +0.0100)."),
              ("What does it cost compared with greedy?",
               "Not measured end to end, and the two parts come from different harnesses, so I do not add them. Sampling 8 with vLLM's shared prefill: 1.13× FLOP-eq; "
               "2.74× latency and 3.59× energy measured on 20 VQA-RAD open questions, ≤ 32 new tokens, one request at a time (bestofn_vllm_2026-09-16.json) — "
               "not measured on the sentence-answer benchmarks. The HF re-read: 3.81 passes × 1.81 = 6.90 greedy-pass equivalents (free_head_2026-08-16.json, "
               "3 benchmarks). The probe itself is negligible. Capturing states during generation would remove the re-read."),
              ("What does 'full resolution' mean here?",
               "Not native resolution: a 1-megapixel cap (max_pixels 1,003,520, ≤ 1,280 image tokens) — 'full resolution' is only our file name. "
               "cap320 is max_pixels 250,880 (≤ 320 tokens). GEMeX images have a median of 7.8 MP, so both caps downsample them.")])

    # how far the statistics reach
    srows = ""
    for lab, cu in [("cross-family judge", "xjudge"), ("judge (label source)", "judge"), ("lenient EM", "em")]:
        mm = D.macro(cu); mc = {"pt": mm[0], "ci": (mm[1], mm[2])}; bu = RV[f"{cu}.benchmark_as_unit_t_interval"]; mi = RV[f"{cu}.micro_question_weighted"]
        lo = RV[f"{cu}.leave_one_benchmark_out"]
        lpt = [v["pt"] for v in lo.values()]
        nsig = sum(1 for v in lo.values() if v["ci"][0] > 0 or v["ci"][1] < 0)
        bcls = "win" if bu["ci"][0] > 0 else "tie"
        srows += (f"<tr><td>{lab}</td><td class='n'>{pm(mc['pt'])} {ci(*mc['ci'])}</td><td class='n {bcls}'>{ci(*bu['ci'])}</td>"
                  f"<td class='n'>{min(lpt):+.4f} … {max(lpt):+.4f} ({nsig}/8 significant)</td><td class='n'>{pm(mi['pt'])} {ci(*mi['ci'])}</td></tr>")
    d.add("stats", "How far the statistics reach",
          f"""
<table class="d"><tr><th>metric</th><th class="n">macro, images resampled</th><th class="n">benchmarks as the unit (t, 7 df)</th><th class="n">leave one benchmark out</th><th class="n">question-weighted (micro)</th></tr>{srows}</table>
<div class="row" style="margin-top:16px">
 <div class="col"><div class="term"><b class="k">benchmarks as the unit</b><span>judge gains hold; lenient EM does not (rests on OmniMedVQA, RadImageNet) — each benchmark had labelled training data</span></div></div>
 <div class="col"><div class="term w"><b class="k">held fixed</b><span>the trained probes and the image split · not clustered by template</span></div></div>
 <div class="col"><div class="term t"><b class="k">tuning vs test</b><span>half A {RV['judge.fold0_macro_pt']:+.4f} · half B {RV['judge.fold1_macro_pt']:+.4f} (judge)</span></div></div>
</div>""", "VII · Limits", src=RVSRC + "; refit floor AUDIT_2026-09-18.md §5",
          minutes=2.0,
          look="the 'benchmarks as the unit' column: the two judge rows exclude zero, the exact-match row does not.",
          say=f"""# macro, images resampled
- How far do the statistics reach?
- Our main intervals resample images inside each of the eight benchmarks, so they say the gain is real on these eight benchmarks.
# benchmarks as the unit (t, 7 df)
- A stricter question: if we drew a new benchmark of the same kind and labelled part of it for training, would the gain hold?
- Treating the eight benchmarks themselves as the sample, both judge gains stay positive, but the exact-match interval crosses zero.
# benchmarks as the unit
- That assumes each new benchmark comes with labelled training questions, as all eight of ours did.
- Without labels from the new benchmark, the leave-one-out result applies — no better than a probe trained on the four-domain set.
# leave one benchmark out
- The exact-match gain also depends on two benchmarks: drop either OmniMedVQA or RadImageNet and it becomes a tie.
# held fixed
- What our intervals hold fixed: the trained probes and the image split.
- Refitting the probes with other seeds or thread counts moves the macro by up to about 0.003.
- Resampling the training set or the split was not measured.
- And they cluster by image, not by question template.
# tuning vs test
- The September choices were made on image half A; half B alone gives a slightly larger gain.
# question-weighted (micro)
- Weighting by question instead of by benchmark makes the judge gains larger and the exact-match gain slightly smaller.""",
          qa=[("Your CIs hold the trained probes fixed. What is the real uncertainty?",
               f"Refitting with other seeds/thread counts moves macros by up to ~0.003 (AUDIT_2026-09-18.md §5). Resampling the training set or the image split was not measured. "
               f"Fold to fold (different probes, different questions): {RV['judge.fold0_macro_pt']:+.4f} vs {RV['judge.fold1_macro_pt']:+.4f}, the same order as the CI width."),
              ("Could the 4 extra training domains leak test images?",
               f"Exact pixel-md5 overlaps exist only between VQA-RAD-train and VQA-Med / VQA-RAD-closed / VQA-Med-2019-test, and are removed "
               f"({rows_tr['fold0']['extra_dropped_image_overlap']} / {rows_tr['fold1']['extra_dropped_image_overlap']} rows per fold). Near-duplicates by perceptual hash were flagged "
               "(e.g. 100 SLAKE-train candidates, AUDIT appendix DI-5) and not resolved — owed.")])

    d.add("untested", "Never tested — the honest gap list",
          """
<div class="row">
 <div class="col">
  <h3 style="font-family:'Roboto Slab';font-size:17px;color:#283593;margin-bottom:8px">selectors not run on our pools</h3>
  <ul class="b sm"><li>log-probability ranking; probe + likelihood (rerankers since Collins 2005) — log-probs exist for the October 1-megapixel pools, 7 of 8 benchmarks</li>
  <li>USC, embedding agreement (EBA), MBR, semantic-cluster vote</li><li>image-contrast scores (CEBaG, VASE)</li>
  <li>a LoRA verifier trained on the same pooled data</li><li>WED with its own sampling loop (“as published”)</li></ul>
 </div>
 <div class="col">
  <h3 style="font-family:'Roboto Slab';font-size:17px;color:#283593;margin-bottom:8px">design axes not tested</h3>
  <ul class="b sm"><li>ranking loss on the final recipe; learned (gated) pooling; LayerNorm in the probe; learning rate for BCE</li>
  <li>per-question or per-modality temperature; adaptive N</li><li>temperature grid finer than 0.2 / 0.4 / 0.7 / 1.0; top-p, top-k with the probe</li></ul>
 </div>
 <div class="col">
  <h3 style="font-family:'Roboto Slab';font-size:17px;color:#283593;margin-bottom:8px">diagnostics not run</h3>
  <ul class="b sm"><li>control task / selectivity (Hewitt & Liang)</li><li>within-question AUROC (CASE, HSRM)</li><li>image-shuffle control (MedProb)</li>
  <li>probe-error vs generator-error coupling (Mirage)</li><li>human rating of the picks — none on the shipped method; no independent check at all on 5 of 8 benchmarks</li>
  <li>variance over training resamples / image-split salt; template-clustered CIs</li><li>judge agreement on the final pools (κ = 0.79 is on September labels)</li><li>perceptual-hash near-duplicates resolved</li></ul>
 </div>
</div>""", "VII · Limits", src="DESIGN_EVIDENCE.md §9 (21 gaps); PAPER_NOTES_* 'honest gaps' sections; review/R_results.md, R_papers.md",
          minutes=1.5,
          say="""# Never tested — the honest gap list
- And the list of what was never tested.
# selectors not run on our pools
- Several selectors we have not run on our data.
- Most importantly ranking by the model's own likelihood, and combining that with the probe, which classical rerankers rely on.
- The likelihoods were not stored for the shipped cap320 pools.
- But they were stored for the October one-megapixel pools on seven of the eight benchmarks, so this baseline can be run on the CPU there.
- Also WED with its own sampling procedure.
# design axes not tested
- Several design axes, including a ranking loss on the final recipe and learned pooling.
# diagnostics not run
- A control task with random labels, and the within-question AUROC that CASE introduced.
- A test where the image is swapped, a human rating of our picks, and the variance from retraining.
- Most of these are cheap to run on cached features, and they are first on the list.""")

    d.add("next", "Next steps",
          """
<div class="row">
 <div class="col">
  <div class="term t"><b class="k">done since: cap320 features</b><span>probes refit on states read at the generation cap: judge −0.0012 [−0.0060, +0.0033] (tie); every metric ties</span></div>
  <div class="term"><b class="k">cheap, CPU-only</b><span>likelihood ranking on the 1-MP pools · within-question AUROC · control task · image-shuffle · judge κ on final pools (probe-count sweep: done — 9 probes tie 24)</span></div>
  <div class="term"><b class="k">temperature without the dataset label</b><span>per-modality T · per-question T or N from the probe's scores</span></div>
 </div>
 <div class="col">
  <div class="term"><b class="k">baselines owed</b><span>USC · LoRA verifier on pooled data · WED as published · 32B and fine-tuning baselines</span></div>
  <div class="term"><b class="k">evaluation owed</b><span>stratified human rating · template-clustered CIs · refit variance</span></div>
  <div class="term"><b class="k">generality</b><span>a non-Qwen generator · capture states inside vLLM sampling</span></div>
 </div>
</div>""", "VII · Limits", src="DESIGN_EVIDENCE.md §9; this deck's limits and gap slides; log-probs: src/labeling/run_openvqa.py cumlogprob_all (October 1-MP pools, RadImageNet missing)", minutes=1.5,
          say="""# done since: cap320 features
- All features were re-extracted at the image budget the answers were generated at, cap320, and the probes refit with the same protocol.
- Result: a tie in every metric — judge minus 0.0012, interval minus 0.0060 to plus 0.0033.
- So the higher-resolution re-read is not what produces the gain; what is still owed is capturing the states inside the sampling pass itself.
# cheap, CPU-only
- After that, the cheap experiments on the CPU.
- First, ranking by the model's own likelihood, and likelihood plus probe.
  - The log probabilities were stored for the October one-megapixel pools on seven of eight benchmarks, so this needs no GPU.
- Then the within-question AUROC, which should tell us why VQA-RAD and VQA-Med fail.
- Then the control task, the image-swap test, and the agreement between our two judges.
# temperature without the dataset label
- Then the temperature question you raised: can the model pick its temperature without knowing the dataset?
- By modality — although GEMeX and VQA-RAD preferring opposite temperatures suggests modality is not the driver.
- Or per question, from the probe's own scores.
# baselines owed
- And the remaining baselines.
# evaluation owed
- A human rating.
# generality
- And the second model family that a reviewer would ask for.""")

    lob = RV["em.leave_one_benchmark_out"]
    d.add("summary", "Summary — the claim, stated precisely",
          f"""
<div class="row">
 <div class="col"><div class="kpi"><div class="l">cross-family judge</div><div class="big">{pm(X[0])}</div><div class="s">{ci(X[1], X[2])} · MedGemma-27B, never used in training</div></div></div>
 <div class="col"><div class="kpi"><div class="l">training judge</div><div class="big">{pm(J[0])}</div><div class="s">{ci(J[1], J[2])} · MedVLThinker-32B, also the label source</div></div></div>
 <div class="col"><div class="kpi"><div class="l">lenient exact match</div><div class="big">{pm(E[0])}</div><div class="s">{ci(E[1], E[2])} · a tie with benchmarks as the unit</div></div></div>
</div>
<ul class="b" style="margin-top:22px">
 <li>8 benchmarks · 36,869 questions · each scored by probes that never saw its image</li>
 <li>judge: 6 significant wins; 2 ties (VQA-RAD, VQA-Med), both selection failures — the right answer was often in the pool</li>
 <li>vs WED on identical candidates: wins on PathVQA and SLAKE, ties on VQA-RAD</li>
 <li>mechanism not new; each benchmark needs labelled training questions</li>
</ul>
<div class="row" style="margin-top:14px">
 <div class="col"><div class="term r"><b class="k">do not say</b><span>invented · generalises to unlabelled benchmarks · transfers to closed questions · matches the 32B · free as run · human-validated · judge-only numbers</span></div></div>
</div>""", "VII · Limits", src="final_eval_FINAL_cap320_T07.json macro_over_open8.all; wed_vs_probe_2026-10-16.json; " + RVSRC.split(" (")[0],
          minutes=1.5,
          look="the three numbers left to right, then the four lines.",
          say=f"""# Summary — the claim, stated precisely
- To summarise: a 918-thousand-parameter probe reading Lingshu-7B's frozen internal states.
- It chooses among eight sampled answers at temperature 0.7.
- It improves open-ended medical VQA over the model's own greedy answer on eight benchmarks.
# cross-family judge
- By {X[0]:.3f} under a judge from another model family that it never trained on.
# training judge
- By {J[0]:.3f} under the judge that supplied its labels.
# lenient exact match
- And {E[0]:.3f} in lenient exact match — all significant on these benchmarks.
- With benchmarks as the unit, the judge gains stay significant and the exact-match gain does not.
- And that assumes each new benchmark comes with labelled training data.
# judge: 6 significant wins; 2 ties (VQA-RAD, VQA-Med), both selection failures — the right answer was often in the pool
- Under the judge it wins on six of eight benchmarks and ties on two.
- On those two the right answer is among the samples but the probe fails to pick it.
# vs WED on identical candidates: wins on PathVQA and SLAKE, ties on VQA-RAD
- On identical candidates it beats the training-free selector WED on PathVQA and SLAKE, and ties on VQA-RAD.
# mechanism not new; each benchmark needs labelled training questions
- The mechanism is not new; the setting, the breadth and the care of the evaluation are.
- Thank you — I am happy to go into any part in detail; there are backup slides on every component.""",
          qa=[("What generalises beyond this project?",
               "Training, not prompting, is what makes a selector work here: a trained 7B selector ties a zero-shot 32B verifier (+0.005 [−0.023, +0.032], "
               "n = 600, July pools, EM/substring; verifier_32b_gpu.json) and beats asking the model to verify itself; and selection gains follow the headroom "
               "only where the probe can find the right sample.")])

    # ================================================================== APPENDIX
    d.part = "Backup"
    d.add("pA", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">BACKUP</div>
<h1>Backup slides</h1><p>Full numbers, recipe, the per-benchmark-temperature variant, data exposure, references.</p></div>""",
          raw=True, minutes=0, say="# Backup slides\n- (Backup — only if asked.)")

    full = ""
    for c in D.OPEN8:
        a = D.cell(c)
        full += (f"<tr><td>{D.SHORT[c]}</td><td class='n'>{a['n']:,}</td><td class='n'>{a['greedy_judge']:.3f}</td><td class='n'>{a['pick_judge']:.3f}</td>"
                 f"<td class='n'>{a['oracle_judge']:.3f}</td><td class='n'>{f'{cap[c]:.2f}'.replace('-', '−')}</td><td class='n'>{pm(a['pick_minus_greedy_judge'],4)}</td><td class='n'>{pm(a['pick_minus_greedy_xjudge'],4)}</td>"
                 f"<td class='n'>{pm(a['pick_minus_greedy_em'],4)}</td><td class='n'>{pm(a['pick_minus_greedy_f1'],4)}</td><td class='n'>{pm(a['pick_minus_greedy_llava_recall'],4)}</td>"
                 f"<td class='n'>{pm(a['sc_minus_greedy_judge'],4)}</td></tr>")
    d.add("bk_full", "Backup — every benchmark, every metric (shipped method)",
          f"""<table class="d"><tr><th>benchmark</th><th class="n">n</th><th class="n">greedy J</th><th class="n">pick J</th><th class="n">oracle@8</th><th class="n">headroom captured</th><th class="n">Δ judge</th>
<th class="n">Δ MedGemma</th><th class="n">Δ lenient EM</th><th class="n">Δ F1</th><th class="n">Δ LLaVA</th><th class="n">vote − greedy</th></tr>{full}</table>
<div class="note" style="margin-top:10px">question-weighted (micro): judge {pm(RV['judge.micro_question_weighted']['pt'])} {ci(*RV['judge.micro_question_weighted']['ci'])} · MedGemma {pm(RV['xjudge.micro_question_weighted']['pt'])} {ci(*RV['xjudge.micro_question_weighted']['ci'])} · lenient EM {pm(RV['em.micro_question_weighted']['pt'])} {ci(*RV['em.micro_question_weighted']['ci'])}. Macro is the conservative choice for the judges and slightly favourable for exact match.</div>""",
          "Backup", src="final_eval_FINAL_cap320_T07.json cells.*.all; headroom captured and micro: " + RVSRC, minutes=0)

    tv = D.tmap_vs["macro"]["fold1_out_of_sample"]
    d.add("bk_tmap", "Backup — per-benchmark temperature (adopted, reversed)",
          f"""
<table class="d"><tr><th>frozen map</th><th>GEMeX, OmniMedVQA, RadImageNet 1.0 · VQA-RAD 0.2 · VQA-Med 0.4 · others 0.7</th></tr>
<tr><td>vs T = 0.7, image half B (out of sample)</td><td>judge {pm(tv['judge']['macro'])} {ci(*tv['judge']['ci'])} · MedGemma {pm(tv['xjudge']['macro'])} {ci(*tv['xjudge']['ci'])} · lenient EM {pm(tv['em']['macro'])} (tie)</td></tr>
<tr><td>whole method, both folds</td><td>judge +0.0897 · MedGemma +0.0783 · lenient EM +0.0165 (final_eval_FINAL_cap320_Tmap.json)</td></tr></table>
<div class="row" style="margin-top:14px">
 <div class="col"><div class="term w"><b class="k">history</b><span>adopted as the final method in FINAL_METHOD_2026-10-16.md; reversed on 2026-10-18 in favour of a single T = 0.7.</span></div></div>
 <div class="col"><div class="term w"><b class="k">why not shipped</b><span>the map needs each question's dataset identity at test time — known on a benchmark, not in deployment. The pooled probe itself needs no dataset label at inference; the map would.</span></div></div>
</div>""",
          "Backup", src="tmap_vs_t07_2026-10-16.json macro.fold1_out_of_sample; frozen_temperature_map_2026-10-15.json; final_eval_FINAL_cap320_Tmap.json; FINAL_METHOD_2026-10-16.md", minutes=0,
          qa=[("Why ship the lower number?",
               "Because the +0.0091 comes from knowing which benchmark a question belongs to. The shipped method must answer an unlabelled question; choosing "
               "temperature from the question itself (modality, or the probe's own scores) is a next step.")])

    f0, f1_ = rows_tr["fold0"], rows_tr["fold1"]
    d.add("bk_recipe", "Backup — the full recipe",
          f"""<table class="d xs">
<tr><th>component</th><th>value</th><th>source</th></tr>
<tr><td>generator</td><td>Lingshu-7B (HF snapshot b98aecd4…), frozen; vLLM 0.9.0.1</td><td>run_openvqa.py; runners</td></tr>
<tr><td>prompt</td><td>“You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain.”</td><td>SYS_GEN</td></tr>
<tr><td>sampling</td><td>N = 8, T = 0.7, max_pixels 250,880 (cap320), min_pixels 3,136</td><td>run_openvqa.py --cap cap320</td></tr>
<tr><td>distinct answers</td><td>samples merged after lower-casing and trimming whitespace; the first-sampled spelling is re-read</td><td>extract_generator_hidden.py norm()</td></tr>
<tr><td>features</td><td>teacher-forced HF pass at the 1-megapixel cap (max_pixels 1,003,520); h̄ = mean over the answer's tokens; output_hidden_states[18, 20, 22]</td><td>extract_generator_hidden.py</td></tr>
<tr><td>probe</td><td>z-score → Linear(3584, 256) → GELU → Linear(256, 1); 918,017 params</td><td>head_sweep.MLP</td></tr>
<tr><td>training</td><td><b>class-weighted</b> BCE, pos_weight = (1 − π)/π, π = share of correct rows in the fold's training set; AdamW lr 1e-3 wd 1e-2, 30 epochs, batch 256, 8 seeds × 3 layers, CPU, 4 threads</td><td>head_sweep.fit (bce_step)</td></tr>
<tr><td>training rows (fold 0 / 1)</td><td>{f0['total']:,} / {f1_['total']:,}: other image half of the 8 open benchmarks ({f0['open8_other_half']:,} / {f1_['open8_other_half']:,}) + other half of the 3 closed splits and VQA-Med 2019 test ({f0['closed3_plus_vqamed2019_other_half']:,} / {f1_['closed3_plus_vqamed2019_other_half']:,}) + 4 extra domains ({f0['extra_domains_kept']:,} / {f1_['extra_domains_kept']:,} of 31,498 rows; image overlaps removed)</td><td>numbers_review_part3.json training_rows_by_source</td></tr>
<tr><td>labels</td><td>MedVLThinker-32B judge (text-only, sees gold); MedGemma-27B-it for evaluation only</td><td>run_judge.py</td></tr>
<tr><td>readout</td><td>mean within-question rank (ties averaged, ÷ (n−1)); ties → vote count</td><td>rank_vote</td></tr>
<tr><td>probe sets</td><td>evaluated as two half-data sets of 24 (ckpts/train/final_probe_2026-10-16/fold0, fold1); a deployed set would be refit on all images (not done)</td><td>final_method_eval.py --save_probes</td></tr>
<tr><td>evaluation</td><td>two-fold image cross-fitting; paired bootstrap, 10,000 resamples of images within benchmark; macro</td><td>final_eval_FINAL_cap320_T07.json</td></tr>
</table>""", "Backup", src="src/training_methods/head_sweep.py:170–175; src/exp_2026_10/final_method_eval.py; src/labeling/run_openvqa.py:51; training rows " + RVSRC.split(" (")[0], minutes=0)

    d.add("bk_exposure", "Backup — data exposure and contamination",
          """<table class="d sm"><tr><th>source</th><th>fact</th></tr>
<tr><td>Lingshu Table 5 p.12</td><td>supervised stage 3 includes training splits of PathVQA, PMC-VQA, SLAKE, VQA-Med-2019, VQA-RAD; Kvasir-VQA only in the separate Lingshu-RL</td></tr>
<tr><td>Lingshu §2.3 p.9</td><td>deduplicated against the MedEvalKit <i>test</i> sets (not VQA-Med 2019)</td></tr>
<tr><td>Lingshu Table 10 p.20</td><td>removing its medical multimodal data: PathVQA 61.9 → 45.2, SLAKE 83.1 → 69.0</td></tr>
<tr><td>our VQA-Med cell</td><td>3,663 questions, mostly the 2019 <i>training</i> split — data Lingshu was fine-tuned on (FINAL_METHOD_2026-10-16.md §6.4)</td></tr>
<tr><td>Contamination audit (arXiv 2606.10066)</td><td>VQA-RAD clean; SLAKE image-source overlap 4.2% (19.8% sensitivity backbone), no effect on answer likelihood; PathVQA signal is an ordering artefact; OmniMedVQA public mirror shows text exposure for all 5 VLMs incl. Qwen2.5-VL</td></tr>
<tr><td>near-duplicates (ours)</td><td>exact pixel-md5 overlaps removed; perceptual-hash candidates (e.g. SLAKE-train 100) flagged, not resolved (AUDIT_2026-09-18 appendix DI-5)</td></tr>
</table>
<div class="term t" style="margin-top:12px"><b class="k">why the deltas are less exposed than the levels</b><span>every comparison is the same model on the same question (greedy vs pick), so exposure enters both arms. Whether memorised items change the <i>delta</i> (peaked sampling distributions) is not measured.</span></div>""",
          "Backup", src="Lingshu arXiv:2506.07044v4; Xu, Wu, Ryu arXiv:2606.10066; FINAL_METHOD_2026-10-16.md §6.4; AUDIT_2026-09-18 appendix DI-5", minutes=0)

    d.add("bk_refs", "Backup — references",
          """<style>#s-bk_refs p{break-inside:avoid;margin-bottom:6px}</style><div style="columns:2;column-gap:36px;font-size:12.5px;line-height:1.45">
<p><b>Test-time compute & verifiers.</b> Cobbe et al. 2021 arXiv:2110.14168 · Brown et al. 2024 arXiv:2407.21787 · Snell et al. 2024 arXiv:2408.03314 ·
Zhang et al. GenRM, ICLR 2025, arXiv:2408.15240 · Sammani et al. 2026 arXiv:2606.28864 · Nakano et al. 2021 (WebGPT) arXiv:2112.09332 · Uesato et al. 2022 arXiv:2211.14275 · Lightman et al. 2023 arXiv:2305.20050 · Gao et al. 2022 (reward over-optimisation) arXiv:2210.10760</p>
<p><b>Selection & its limits.</b> Wang et al., ICLR 2023 (self-consistency) · Chen et al. 2023 (USC) arXiv:2311.17311 · Ontalvilla et al. 2026 (EBA) arXiv:2606.12003 ·
Bay & Yearick 2026 arXiv:2606.28661 · Hu 2026 arXiv:2607.17531 · Stroebl et al. 2024 arXiv:2411.17501 · Kang et al. 2025 (self-certainty) arXiv:2502.18581 · Jacob et al. 2023 (Consensus Game) arXiv:2310.09139 · Zhang, Zang & Kainz 2024 (BDG) arXiv:2410.01064</p>
<p><b>Hidden-state verifiers & probing.</b> Alain & Bengio 2016 arXiv:1610.01644 · Hewitt & Liang, EMNLP 2019 · Belinkov, Comp. Ling. 48(1) 2022 · Q-Probe, ICML 2024, arXiv:2402.14688 ·
SWIFT, KDD 2026, arXiv:2505.12225 · LiLaVe arXiv:2504.16760 · ReProbe, ACL 2026, arXiv:2511.06209 · HSRM arXiv:2608.30841 · CASE arXiv:2608.17124 · Kadavath et al. 2022 arXiv:2207.05221 · Burns et al., ICLR 2023 (CCS) arXiv:2212.03827 · Azaria & Mitchell 2023 arXiv:2304.13734 · Orgad et al. 2024 arXiv:2410.02707 · Kossen et al. 2024 (SEP) arXiv:2406.15927 · Bucciarelli et al. 2026 (VHS) arXiv:2603.22492 · frozen-VLA probe selector arXiv:2605.28527</p>
<p><b>Uncertainty.</b> Farquhar et al., Nature 630 (2024) · Geng et al., NAACL 2024 · Byun et al. arXiv:2604.02543 · Khanmohammadi et al. arXiv:2606.15910</p>
<p><b>Medical VLMs & selection.</b> Qwen2.5-VL arXiv:2502.13923 · Lingshu arXiv:2506.07044 · MedGemma arXiv:2507.05201 · Verification Mirage arXiv:2605.10850 ·
MedProb arXiv:2609.04336 · DualRead arXiv:2609.06419 · WED arXiv:2605.18313 · Best-of-Evidence arXiv:2607.20950 · LLaVA-Med arXiv:2306.00890</p>
<p><b>Benchmarks & evaluation.</b> PathVQA arXiv:2003.10286 · SLAKE arXiv:2102.09542 · VQA-RAD, Sci. Data 2018 · RadImageNet-VQA arXiv:2512.17396 · Kvasir-VQA-x1 arXiv:2506.09958 ·
OmniMedVQA arXiv:2402.09181 · VQA-Med 2019 (CEUR Vol-2380) · GEMeX arXiv:2411.16778 · MT-Bench arXiv:2306.05685 · contamination audit arXiv:2606.10066</p>
<p><b>Lineage.</b> Collins & Koo, Comp. Ling. 31(1) 2005 · Li et al. (CEM) arXiv:2010.11428 · Li et al. (R-EBM) arXiv:2103.14152 · Jumper et al., Nature 596 (2021)</p>
</div>""", "Backup", minutes=0)
