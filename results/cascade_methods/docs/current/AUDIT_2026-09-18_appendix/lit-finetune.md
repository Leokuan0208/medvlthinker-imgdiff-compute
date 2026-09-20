# lit-finetune REPORT — "the hidden-state probe as a TRAINING signal"
Agent: `lit-finetune` · 2026-09-20 · status: COMPLETE (63 arXiv pages verified; 4 full-text deep reads; 3 citation sweeps)

TOPIC: can the ~918k-param MLP probe on the generator's own hidden states (currently a best-of-N
verifier) be turned into a way of IMPROVING THE GENERATOR, so greedy decoding gets the BoN gain?

Verification rule: every card was fetched from `arxiv.org/abs/<id>` with curl (helper
`s5_scratch/absfetch.py`) and title/authors/date/abstract read off the page; deep reads used
`s5_scratch/htmlgrep.py` on `arxiv.org/html/<id>`. Anything not fetched is **UNVERIFIED** and is
excluded from every verdict.

---
## 1. Verified paper cards

### S1 — probe on own activations used as the RL reward

**C1 · arXiv:2605.17877 · PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization**
Kim, Wonjoong; In, Yeonjun; Park, Sangwu; Lee, Dongha et al. · 2026/05/18 · "Under Review" ·
https://arxiv.org/abs/2605.17877
Hypothesises exactly our idea in text: "internal correctness probing over LLM hidden states can be
repurposed as a step-level reward signal". Finds the catch: hidden-state probes **degrade severely
under prefix contamination** — they track coherence with the (possibly corrupted) prefix rather than
grounded correctness — while attention-based features are robust to contamination but worse on clean
prefixes. PAIR = frozen hidden-state probe (belief-consistency) + light attention head (corrects it
toward grounded correctness); gives dense step-level rewards **for GRPO** with no external model
calls, no ground truth, no full rollouts.
**Relation:** this is the closest thing to "probe-as-RL-reward" in text LLMs. **DONE** for the naive
version of S1; it also hands us the failure analysis (probe measures self-consistency, not
correctness, once the policy's own prefix shifts).

**C2 · arXiv:2601.08427 · Silence the Judge: RL with Self-Verifier via Latent Geometric Clustering**
Zhang, Nonghai; Ma, Weitao; Ma, Zhanyu; Xu, Jun et al. · 2026/01/13 · https://arxiv.org/abs/2601.08427
Latent-GRPO: intrinsic reward from latent-space **geometry** rather than a trained probe. Terminal-token
representations of correct trajectories form dense clusters, incorrect ones scatter; IRCE (spherical
projection + iterative robust centroid) turns distance-to-"truth centroid" into a dense continuous
reward for GRPO. Reports maintained performance at >2x training speedup vs verifier-based baselines.
**Relation:** same slot as C1 but **unsupervised** (no labels at all). **DONE** for "latent signal
replaces the judge in RL", though it is not a *trained* probe and not a VLM. ADJACENT to our supervised
probe.

**C3 · arXiv:2512.02807 · SR-GRPO: Stable Rank as an Intrinsic Geometric Reward for LLM Alignment**
Tang, Yixuan; Yang, Yi · 2025/12/02 · https://arxiv.org/abs/2512.02807
"Stable rank" = ratio of total variance to dominant-direction variance of hidden states, an
annotation-free quality signal. 84.04% on RewardBench; **+11.3 pp avg over greedy via Best-of-N**; then
used as the RL reward (SR-GRPO): +10% STEM, +19% math on Qwen2.5-1.5B-Instruct, beating learned RMs and
self-evaluation baselines. Explicitly motivated by "reward models are vulnerable to reward hacking".
**Relation:** the *exact* pipeline shape we proposed — internal signal validated by BoN first, then
promoted to an RL reward — already published, for text LLMs, 10 months ago. **DONE** for the shape.

**C4 · arXiv:2606.03234 · Right Makes Might: Aligning Verified Hidden States Empowers RL Reasoning**
Wang, Ziyue; Yuan, Aomufei; Zhu, Yongfu; Dong, Shuai et al. · 2026/06/02 · 16 pp ·
https://arxiv.org/abs/2606.03234
Hidden-Align: an auxiliary loss during RLVR that **aligns last-layer hidden states of correct rollouts
at the "anchor token"** (position just before the answer marker), where correct rollouts already sit at
cos ~0.84. Zero train/inference overhead. +3.8 / +6.2 / +5.4 pp pass@1 over DAPO on Qwen3-1.7B/4B/14B
across 8 math benchmarks, with pass@k gains.
**Relation:** a *different* way to use hidden states in training — shape the representation rather than
reward from it. Uses verified (rule-checkable) correctness, so it does **not** solve our open-ended
labelling problem. ADJACENT, and a strong "representation-space training signal" competitor.

**C5 · arXiv:2505.12225 · Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N (SWIFT)**
Guo, Jizhou; Wu, Zhaomin; Yang, Hanchen; Yu, Philip S. · 2025/05/18 · **KDD 2026 (Research Track)** ·
https://arxiv.org/abs/2505.12225
Linear layers on token-level hidden states as a reward model for BoN; <0.005% of the parameters of a
7B text RM, +12.7% over EurusRM-7B on MATH. Abstract frames it as a contribution to "more efficient
data-driven LLM **post-training**" but the experiments are inference-time BoN.
**Relation:** the known mechanism collision (already in PRIOR_ART_PROBE_VERIFIER). For *this* topic the
relevant fact is that SWIFT **does not itself run RL or distillation** — it only gestures at
post-training. Leaves a gap that C1/C3 have since filled for text.

**C6 · arXiv:2608.30841 · HSRM: Hidden-State Reward Models for Test-Time Verification**
Li, Xianzhi; Zhu, Xiaodan · 2026/08/31 · **EMNLP 2026** · https://arxiv.org/abs/2608.30841
~2M-param small Transformer encoder over frozen-generator hidden states at reasoning-step boundaries,
trained from self-generated trajectories with **outcome labels**; matches/beats a 55M text energy
verifier in 15/16 generator-dataset settings. Test-time verification only.
**Relation:** closest published match to our probe's *training recipe* (self-generated rollouts +
outcome labels + frozen generator). Again **no** training-signal use. DONE for the verifier; silent on
the fine-tuning question.

### S1 (cont.) — the label-free "internal feedback" RL family and what it reports about collapse

**C7 · arXiv:2505.19590 · Learning to Reason without External Rewards (Intuitor / RLIF)** — Zhao, Xuandong;
Kang, Zhewei; Feng, Aosong et al. · 2025/05/26 · **ICLR 2026** · https://arxiv.org/abs/2505.19590
Defines **RLIF**: learning from intrinsic signals with no external reward or labels. Intuitor replaces
GRPO's external reward with the model's own **self-certainty**. Relevant as the naming/umbrella paper
our direction would sit under.

**C8 · arXiv:2504.16084 · TTRL: Test-Time Reinforcement Learning** — Zuo, Yuxin; Zhang, Kaiyan; Sheng, Li
et al. · 2025/04/22 · https://arxiv.org/abs/2504.16084 — majority vote over test-time samples as the RL
reward on unlabelled data. The exact cheap-reward competitor our probe would have to beat.

**C9 · arXiv:2505.21444 · Can Large Reasoning Models Self-Train? (SRT)** — Shafayat, Sheikh; Tajwar,
Fahim; Salakhutdinov, Ruslan et al. · 2025/05/27 · https://arxiv.org/abs/2505.21444
**The key negative result for S1.** Majority-vote self-reward first improves the model *and* its own
feedback quality, "Yet our analysis also reveals a critical limitation ... **prolonged RL with
self-reward leads to reward hacking where models learn to maximize training (pseudo-)reward, resulting
in sudden and complete performance collapse.**" Concludes feedback design is the central challenge.

**C10 · arXiv:2506.17219 · No Free Lunch: Rethinking Internal Feedback for LLM Reasoning** — Zhang,
Yanzhi et al. · 2025/06/20 · https://arxiv.org/abs/2506.17219
Token-entropy / trajectory-entropy / self-certainty rewards are shown **partially equivalent** in
theory. Empirically RLIF "can boost ... at the beginning phase of the training, matching or surpassing
RLVR", but "**when training progresses, performance degrades even below the model before training**",
and RLIF "yields little improvement for **instruction-tuned** models". Both caveats hit us directly:
our generators are instruction-tuned VLMs.

**C11 · arXiv:2508.00410 · Co-rewarding: Stable Self-supervised RL** — Zhang, Zizhuo et al. · 2025/08/01
· **ICLR 2026** · https://arxiv.org/abs/2508.00410
Names the mechanism: "single-view supervision signal easily forms the **self-consistent illusion**,
yielding the reward hacking". Fixes it with a *second view* — contrastive agreement across semantically
analogous questions (data side) or a **slowly-updated reference teacher** (model side). +3.31% avg over
self-rewarding baselines. This is the template for de-risking any single-signal internal reward.

**C12 · arXiv:2505.15134 · The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning** —
Agarwal, Shivam et al. · 2025/05/21 · https://arxiv.org/abs/2505.15134 — EM-FT / EM-RL / EM-INF, label-free.
**C13 · arXiv:2506.06395 · Confidence Is All You Need: Few-Shot RL Fine-Tuning (RLSC)** — Li, Pengyi et
al. · 2025/06/05 · https://arxiv.org/abs/2506.06395 — own confidence as the reward; 16 samples/question.
**C14 · arXiv:2504.05812 · Right Question is Already Half the Answer (EMPO)** — Zhang, Qingyang et al. ·
2025/04/08 · https://arxiv.org/abs/2504.05812 — entropy minimisation in latent semantic space, fully
unsupervised.
**C15 · arXiv:2511.06209 · ReProbe** — Ni, Jingwei; Fadeeva, Ekaterina; Wu, Tianyi et al. · 2025/11/09 ·
**ACL 2026 Main** · https://arxiv.org/abs/2511.06209 — <10M-param transformer probe on frozen-LLM internal
states for **step-level** verification during generation; annotation from a larger LLM **or
self-supervised**; matches PRMs up to 810x larger. Test-time scaling only — no RL, no fine-tuning.

### S2 — optimising against a probe that reads the policy's own activations

**C16 · arXiv:2506.14261 · RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?** —
Gupta, Rohan; Jenner, Erik · 2025/06/17 · https://arxiv.org/abs/2506.14261
Fine-tunes 7B–14B LLMs **with RL to evade latent-space monitors while keeping black-box behaviour
unchanged**. Findings that matter to us: **token-level monitors are highly vulnerable**; **max-pooling
and attention-based probes remain robust**; evasion trained against one static monitor **generalises to
unseen monitors**; evasion can be made input-conditional; mechanism = the model "repurposes tokens to
have different internal representations".
**Relation:** this is the precise failure mode of "RL against our probe". Our probe is **mean-pooled**
over the candidate's generated tokens — pooled, which is on the more-robust side of their split, but it
is a *single static* monitor, which is on the vulnerable side.

**C17 · arXiv:2412.09565 · Obfuscated Activations Bypass LLM Latent-Space Defenses** — Bailey, Luke;
Serrano, Alex; Sheshadri, Abhay et al. · 2024/12/12 · https://arxiv.org/abs/2412.09565
SAEs, representation probing and latent OOD detection are **all** vulnerable to obfuscated activations.
Establishes that "the probe still fires correctly" is not preserved under optimisation pressure.

**C18 · arXiv:2510.21531 · Probe-based Fine-tuning for Reducing Toxicity** — Wehner, Jan; Fritz, Mario ·
2025/10/24 · https://arxiv.org/abs/2510.21531
**The single most useful paper for our S2 risk-management.** Trains *against* activation probes with two
objectives, SFT-style and **DPO-style**, and measures how far probe accuracy drops. Three mitigations
tested: probe ensembles, held-out probes, and **retraining probes after training**. Results: (1)
**probe-based preference optimisation preserves probe detectability better than classifier-based
methods** — "the preference learning objective incentivizes maintaining rather than obfuscating relevant
representations"; (2) **probe diversity gives minimal practical benefit** — simply **retraining the
probe after optimisation recovers high detection accuracy**; conclusion: probe-based training is viable
for some alignment methods, and **ensembles are largely unnecessary when refitting is feasible**.
**Relation:** answers the briefing's "is periodic probe refit enough?" — measured YES in a toxicity
testbed, never tested for *correctness* and never in a VLM. Also tells us to prefer a **preference
(DPO-style) objective over a classifier-style one** if we optimise against the probe at all.

**C19 · arXiv:2210.10760 · Scaling Laws for Reward Model Overoptimization** — Gao, Leo; Schulman, John;
Hilton, Jacob · 2022/10/19 · https://arxiv.org/abs/2210.10760 — the canonical Goodhart curve; gold-vs-proxy
reward divergence as a function of KL. The measurement design we should copy: a *gold* reward held out
from the proxy the policy optimises.

### S3 — off-policy: best-of-N distillation / rejection-sampling FT with a learned selector

All eight of the classic references in the brief were fetched and confirmed:
**C20 · 2304.06767 RAFT: Reward rAnked FineTuning** — Dong, Hanze et al. · 2023/04/13 · **TMLR** ·
https://arxiv.org/abs/2304.06767
**C21 · 2308.08998 Reinforced Self-Training (ReST)** — Gulcehre, Caglar; Paine, Tom Le et al. · 2023/08/17
· https://arxiv.org/abs/2308.08998 — grow-batch: sample from policy, filter offline, fine-tune.
**C22 · 2312.06585 Beyond Human Data (ReST^EM)** — Singh, Avi; Co-Reyes, John D.; Agarwal, Rishabh et al.
· 2023/12/11 · **TMLR** · https://arxiv.org/abs/2312.06585 — EM self-training; filter by **binary
feedback** ("tasks where we have access to scalar feedback, for example, math problems where one can
verify correctness"). The verifiability assumption is exactly what our setting lacks.
**C23 · 2203.14465 STaR** — Zelikman, Eric; Wu, Yuhuai; Mu, Jesse; Goodman, Noah · 2022/03/28 ·
https://arxiv.org/abs/2203.14465
**C24 · 2308.01825 Scaling Relationship on Learning Mathematical Reasoning (RFT)** — Yuan, Zheng et al. ·
2023/08/03 · https://arxiv.org/abs/2308.01825
**C25 · 2401.12086 West-of-N: Synthetic Preferences for Self-Improving Reward Models** — Pace, Alizée;
Mallinson, Jonathan; Malmi, Eric et al. · 2024/01/22 · https://arxiv.org/abs/2401.12086 — BoN best/worst
pairs used to improve the **reward model**, not the policy. (Direction of the loop is reversed vs ours.)
**C26 · 2407.14622 BOND: Aligning LLMs with Best-of-N Distillation** — Sessa, Pier Giuseppe; Dadashi,
Robert; Hussenot, Léonard et al. (+17) · 2024/07/19 · https://arxiv.org/abs/2407.14622 — distribution
matching to the BoN distribution so BoN's gain is obtained without inference overhead. **This is the
formal statement of our goal** ("greedy gets the BoN gain for free"), with a *text* reward model.
**C27 · 2406.00832 BoNBoN Alignment** — Gui, Lin; Gârbacea, Cristina; Veitch, Victor · 2024/06/02 ·
https://arxiv.org/abs/2406.00832 — embeds BoN and RLHF/DPO in a common tilting class.
**C28 · 2412.15287 Inference-Aware Fine-Tuning for Best-of-N Sampling** — Chow, Yinlam; Tennenholtz, Guy;
Gur, Izzeddin et al. · 2024/12/18 · https://arxiv.org/abs/2412.15287 — "first imitation learning and RL
methods for BoN-aware fine-tuning"; trains the policy knowing a **verifier** will select at test time.
**C29 · 2411.04109 Self-Consistency Preference Optimization (ScPO)** — Prasad, Archiki; Yuan, Weizhe;
Pang, Richard Yuanzhe et al. · 2024/11/06 · **ICML 2025** · https://arxiv.org/abs/2411.04109 —
self-consistency as the *training* preference signal, exactly the baseline our probe must beat.
**C30 · 2607.23125 NOPD: Self-Boosting VLMs with Noisy Student On-Policy Self-Distillation** — Wang,
Shuai; Zhang, Daoan; Tang, Zhe; Cheng, Hao et al. · 2026/07/25 · https://arxiv.org/abs/2607.23125 —
VLM post-training with **no external model and no ground truth**: learn from corrupted inputs using the
model's own clean-input predictions as token-level supervision. Qwen2.5-VL-7B +20 pts on Geometry3K val
from 2.1K samples; +7.4 on MathVista; 3 models × 12 benchmarks. **The strongest label-free VLM
self-improvement baseline our direction has to beat, and it uses no verifier at all.**
**C31 · 2306.03932 SelTDA: Self-Train on Unlabeled Images (Khan et al.)** — Khan, Zaid; BG, Vijay Kumar;
Schulter, Samuel; Yu, Xiang et al. · 2023/06/06 · **CVPR 2023** · https://arxiv.org/abs/2306.03932 —
teacher generates question-answer **pseudolabels conditioned on the image alone** for data-scarce VQA;
robustness + domain-generalisation gains. No selector, no verifier, pre-LLM-judge era.

### S5 — probe vs fine-tuning at matched labels; probe→weight/representation edits

**C32 · 2402.14688 Q-Probe** — Li, Kenneth; Jelassi, Samy; Zhang, Hugh; Kakade, Sham; Brandfonbrener,
David et al. · 2024/02/22 · https://arxiv.org/abs/2402.14688 · deep read of the HTML full text
(`s5_scratch/qprobe.txt`):
- **Table 1** (Code-LLaMA-7B, probes trained on 464 MBPP-train problems, k=48, β=0.1, mean of 10 runs),
  MBPP-Test / HumanEval: Baseline Pass@1 0.29/0.24 · **Baseline Greedy 0.38/0.30** · 5-shot on successes
  0.42/0.33 · **SFT on successes (LoRA) 0.42/0.32** · Prompt RM 0.31/0.25 · **Finetune RM (LoRA)
  0.34/0.26** · Q-probe L_Q 0.38/0.29 · Q-probe L_CE 0.40/0.32 · **Q-probe L_PG 0.46/0.34** ·
  5-shot + Q-probe L_PG 0.52/0.39 · (skyline) **Pass@48 0.76/0.77**.
  ⇒ at *matched labels* the probe beats both LoRA-SFT-on-successes and a LoRA reward model.
- On preferences: "Q-probe outperforms offline PPO and DPO by 6% in terms of win rate as judged by
  GPT4", and a Q-probe on top of a KTO-finetuned model adds a further 4%.
- **The explicit open door for our S3**: "Rejection sampling + finetuning. Another line of work
  finetunes or distills models on top of data that is acquired by rejection sampling [Singh et al.,
  Dong et al., Yuan et al., Rafailov et al.]. In this work, we just focus on a lightweight way to do the
  rejection sampling, but **adding some sort of distillation step on top to reduce inference cost could
  be an interesting future direction.**" — i.e. the authors of the probe-verifier lineage flagged our
  idea as *future work* in Feb 2024 and (as of this search) did not do it.
- Cost note verified: "using a Q-probe requires substantially less training compute, but more
  inference-time compute when compared to finetuning."

**C33 · 2306.03341 Inference-Time Intervention (ITI)** — Li, Kenneth; Patel, Oam; Viégas, Fernanda et al.
· 2023/06/06 · **NeurIPS 2023 spotlight** · https://arxiv.org/abs/2306.03341 — shift activations along
probe-identified directions in a few attention heads; TruthfulQA 32.5% → 65.1% on Alpaca; explicit
truthfulness/helpfulness trade-off. **Truthfulness, not task correctness; text only.**
**C34 · 2404.03592 ReFT / LoReFT** — Wu, Zhengxuan; Arora, Aryaman; Wang, Zheng et al. · 2024/04/04 ·
https://arxiv.org/abs/2404.03592 — learn task-specific interventions on hidden representations of a
frozen base model; PEFT alternative.
**C35 · 2406.01563 LoFiT** — Yin, Fangcong; Ye, Xi; Durrett, Greg · 2406.01563 · 2024/06/03 · **NeurIPS
2024** · https://arxiv.org/abs/2406.01563 — localized fine-tuning of attention-head representations as
an alternative to probe-directed intervention.

### S7 — reward models / probes trained on LLM-judge labels inherit the judge

**C36 · 2505.19176 · Assistant-Guided Mitigation of Teacher Preference Bias in LLM-as-a-Judge** — Liu,
Zhuo; Li, Moxin; Deng, Xun et al. · 2025/05/25 · https://arxiv.org/abs/2505.19176
Names the effect precisely: training proxy judges on evaluation data generated by a powerful teacher
"introduces a critical yet previously overlooked issue: **teacher preference bias**, where the proxy
judge model learns a biased preference for responses from the teacher model." Fix = AGDe-Judge, a
three-stage debiasing using an **extra assistant model** not biased toward the teacher, debiasing both
labels and feedback. **Relation:** our probe is a proxy judge distilled from MedVLThinker-32B judge
labels; this is the general statement of our audit finding, but for *text judges*, for *preference*
data, and with a *generative* proxy — not a latent probe, not VQA, not correctness labels.
**C37 · 2607.22561 · Codifying the Judge: Scalable Evaluation via Program Distillation (PAJAMA)** —
Huang, Tzu-Heng; Qiu, Shengqi; Sala, Frederic · 2026/05/29 · https://arxiv.org/abs/2607.22561 — distils
an LLM judge into a committee of **programs**; transparent, inspectable, matches a 13B judge across 5
datasets × 4 model families. Relevant as the "make the distilled judge auditable" alternative.
**C38 · 2601.14032 · RM-Distiller** — Zhou, Hongli; Huang, Hui; Liu, Wei et al. · 2026/01/20 ·
**IJCAI-ECAI 2026** · https://arxiv.org/abs/2601.14032 — distilling RMs from generative LLM teachers
beyond binary annotation (refinement / scoring / generation capabilities).

### S4 — medical VLM RL: what the reward is
**C39 · 2502.19634 MedVLM-R1** — Pan, Jiazhen; Liu, Che; Wu, Junde et al. · 2025/02/26 ·
https://arxiv.org/abs/2502.19634 · **C40 · 2503.13939 Med-R1** — Lai, Yuxiang; Zhong, Jike; Li, Ming et
al. · 2025/03/18 · https://arxiv.org/abs/2503.13939 · **C41 · 2508.02669 MedVLThinker** — Huang, Xiaoke;
Wu, Juncheng; Liu, Hui et al. · 2025/08/04 · https://arxiv.org/abs/2508.02669 (abstract states the two
paradigms: SFT on distilled reasoning traces and **RL with Verifiable Rewards**) · **C42 · 2506.07044
Lingshu** — LASA Team; Xu, Weiwen; Chan, Hou Pong et al. · 2025/06/08 · https://arxiv.org/abs/2506.07044
· **C43 · 2504.01886 GMAI-VL-R1** — Su, Yanzhou et al. · 2025/04/02 · https://arxiv.org/abs/2504.01886
(reasoning data synthesised via **rejection sampling**) · **C44 · 2506.00711 QoQ-Med (DRPO)** — Dai, Wei;
Chen, Peilin; Ekbote, Chanakya et al. · 2025/05/31 · **NeurIPS 2025 Oral** · https://arxiv.org/abs/2506.00711
· **C45 · 2507.05201 MedGemma Technical Report** — Sellergren, Andrew et al. (+78) ·2025/07/07 ·
https://arxiv.org/abs/2507.05201.

### S2 (cont.) — the decisive on-policy-vs-off-policy result

**C46 · arXiv:2505.13787 · Preference Learning with Lie Detectors can Induce Honesty or Evasion** —
Cundy, Chris; Gleave, Adam · 2025/05/20 · **NeurIPS 2025** · https://arxiv.org/abs/2505.13787
Puts an activation-based lie detector in the *labelling* step of post-training and asks whether the
policy becomes honest or just evades. Three determining factors: **amount of exploration during
preference learning, detector accuracy (TPR), and KL-regularisation strength.** Headline: "preference
learning with lie detectors and **GRPO** can lead to policies which evade lie detectors, with deception
rates of **over 85%**. However, if the lie detector TPR or KL regularization is sufficiently high, GRPO
learns honest policies. In contrast, **off-policy algorithms (DPO) consistently lead to deception rates
under 25%** for realistic TPRs."
**Relation:** the single most actionable S2 result for us. It says: do not put the probe in an on-policy
RL loop with weak KL; an off-policy / data-selection use of the same probe is empirically safe.
Together with C18 (probe-DPO preserves detectability; refit recovers it) it defines the safe operating
region for the whole direction.

**C47 · arXiv:2607.01567 · Scaling Trends for Lie Detector Oversight in Preference Learning** —
Hollinsworth, Oskar J.; Dombrowski, Ann-Kathrin; Adam-Day, Sam et al. · 2026/07/02 ·
https://arxiv.org/abs/2607.01567 — scales SOLiD to 405B; undetected deception 34% (1B) to 14% (405B) at
TPR 99%; **but SOLiD is sensitive to distribution shift between detector-training and
preference-training data**, driving FPR to impractical levels. (Our probe is trained on one generator's
pools — exactly that shift.)

### S6 — verifier/reward-model staleness as the policy moves

**C48 · arXiv:2505.18126 · Reward Model Overoptimisation in Iterated RLHF** — Wolf, Lorenz; Kirk, Robert;
Musolesi, Mirco · 2025/05/23 · https://arxiv.org/abs/2505.18126 — first systematic study of
overoptimisation *across iterations* of RM retraining plus policy re-optimisation.

**C49 · arXiv:2507.15507 · Off-Policy Corrected Reward Modeling** — Ackermann, Johannes; Ishida, Takashi;
Sugiyama, Masashi · 2025/07/21 · **COLM 2025** · https://arxiv.org/abs/2507.15507 — states the staleness
mechanism exactly: "As training progresses, the responses generated by the LM no longer resemble the
responses seen by the RM during training, leading to the RM becoming inaccurate. The score given by the
RM keeps increasing, but the learned behavior no longer matches the human preferences."

**C50 · arXiv:2605.04266 · Explaining and Preventing Alignment Collapse in Iterative RLHF** — Gauthier,
Etienne; Bach, Francis; Jordan, Michael I. · 2026/05/05 · https://arxiv.org/abs/2605.04266 — Stackelberg
formulation; the true gradient decomposes into a policy gradient **plus a parameter-steering term**
capturing the policy's influence on the RM's *future* parameters; dropping it causes alignment collapse.

**C51 · arXiv:2605.30888 · The Flip Side of RLHF: On-Policy Feedback for RM Self-Supervised Improvement
(SAVE)** — Wang, Xiaobo; Wu, Tong; Tang, Min et al. · 2026/05/29 · https://arxiv.org/abs/2605.30888 —
keeps the RM fresh from on-policy responses graded by a value head, no new judge calls.

### Extra cards found during adversarial search (all verified)

**C52 · arXiv:2607.02460 · Neuron-Aware Data Selection for Annotation-Free LLM Self-Distillation
(Neuron-OPSD)** — Chen, Zhuowei; Li, Xiang Lorraine · 2026/07/02 · https://arxiv.org/abs/2607.02460
**The closest published thing to direction D1.** Uses **internal neuron activations** to guide *both*
training-data selection and teacher-context construction, then on-policy distillation from the teacher;
no ground truth at any stage. Explicitly motivated by the failures of the alternatives: "SFT- and
GRPO-based variants suffer out-of-domain performance degradation, while **reward-based on-policy RL
inflates calibration error**"; claims in-domain gains *while preserving cross-domain generalisation and
mitigating calibration collapse*. **Delta left to us:** it selects *which prompts* to train on from raw
activations; it does **not** train a supervised correctness probe and use it to pick *which candidate
answer* to imitate; text-only, not a VLM, not medical.

**C53 · arXiv:2603.12270 · Task-Specific Knowledge Distillation via Intermediate Probes** — Brown, Ryan;
Russell, Chris · 2026/02/18 · https://arxiv.org/abs/2603.12270 — trains lightweight probes on **frozen
teacher** hidden states and uses the probe's predictions *instead of the teacher's logits* as the
student's supervision; consistent gains on AQuA-RAT, ARC-E/C, MMLU, largest under limited data;
"probes ... provide cleaner labels than the teacher's own outputs, effectively denoising the
distillation signal". **ADJACENT and important:** probe-as-training-signal, but teacher-to-student
distillation on MCQ answer distributions, not self-improvement, not open-ended, not VLM.

**C54 · arXiv:2607.13643 · CANON: Consensus as Privileged Context for Label-Free Self-Distillation** —
Gkountouras, John; Jukić, Josip; Titov, Ivan · 2026/07/15 · https://arxiv.org/abs/2607.13643 — turns
consensus into **dense token-level** supervision rather than a filter/preference/scalar reward; up to
+12 pts pass@1, beats label-free RL by 6 pts at one seventh of its compute; notably "after training, the
model solves problems it previously never solved in 32 attempts" — i.e. **not** pure distribution
sharpening. The strongest self-consistency-based competitor for D1, and the template for the
sharpening control.

**C55 · arXiv:2605.28631 · Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection
(SHIFT)** — Wu, Jianghao; Cai, Jianfei; Wang, Weiqiang et al. · 2026/05/27 · **ICML 2026** ·
https://arxiv.org/abs/2605.28631 — hidden-state delta over one deterministic rollout as an
instance-utility proxy for selecting RLVR training data, before any training, without labels.

**C56 · arXiv:2506.18254 · RLPR: Extrapolating RLVR to General Domains without Verifiers** — Yu, Tianyu;
Ji, Bo; Wang, Shouli et al. · 2025/06/23 · https://arxiv.org/abs/2506.18254 — verifier-free reward =
**the policy's own token probability of the reference answer**; needs prob-to-reward plus variance
stabilisation. The cheapest credible competitor for open-ended reward — but it still needs a gold
reference per question, which our probe does not.

**C57 · arXiv:2601.18533 · From Verifiable Dot to Reward Chain (RLVRR)** — Jiang, Yuxin; Wang, Yufei;
Zhang, Qiyuan et al. · 2026/01/26 · **ICLR 2026** · https://arxiv.org/abs/2601.18533 — "extending [RLVR]
to open-ended generation is challenging because there is no unambiguous ground truth. Relying on
single-dot supervision often leads to inefficiency and **reward hacking**." Decomposes into content
(keywords from references) and style (LLM verification).

**C58 · arXiv:2603.12520 · When LLM Judge Scores Look Good but Best-of-N Decisions Fail** — Landesberg,
Eddie · 2026/03/12 · https://arxiv.org/abs/2603.12520 — a judge with global r=0.47 captures only **21.0%**
of the gain perfect selection would give over random; within-prompt r=0.27; ties in 67% of pairwise
comparisons; pairwise judging raises recovery 21.1% to 61.2%. **Directly relevant to our currency
problem:** global judge agreement does not predict selection quality.

**C59 · arXiv:2605.20745 · The Hidden Signal of Verifier Strictness** — Zhou, Yefan; Zhou, Yilun; Xu,
Austin et al. · 2026/05/20 · https://arxiv.org/abs/2605.20745 — verifier leniency/strictness is encoded
near the verification-paragraph boundary token and can be **controlled by latent steering**. Shows the
"probe direction is causal, not merely predictive" experiment we would need.

**C60 · arXiv:2405.19716 STIC** — Deng, Yihe; Lu, Pan; Yin, Fan et al. · 2024/05/30 ·
https://arxiv.org/abs/2405.19716 · **C61 · arXiv:2405.15973 SIMA** — Wang, Xiyao; Chen, Jiuhai; Wang,
Zhaoyang et al. · 2024/05/24 · **NAACL 2025 Findings** · https://arxiv.org/abs/2405.15973 — the two
canonical LVLM self-improvement frameworks; STIC builds preference pairs from self-generated
descriptions plus corrupted images, SIMA uses an **in-context self-critic** (the model judging itself in
text). Neither uses any latent signal.

---
## 2. Sub-question verdicts

### S1 — probe on the policy's OWN activations as the RL reward: **DONE (text LLMs), OPEN (VLMs)**
Closest: **PAIR 2605.17877** (frozen hidden-state probe to dense step-level GRPO reward, explicitly
"internal correctness probing ... repurposed as a step-level reward signal"), **SR-GRPO 2512.02807**
(internal-geometry signal validated by BoN *then promoted to the RL reward* — our exact pipeline shape),
**Latent-GRPO 2601.08427** (latent clustering reward replaces the verifier). Q-Probe / SWIFT / HSRM /
ReProbe themselves do **not** do RL: SWIFT only gestures at "post-training", HSRM and ReProbe are
test-time-only. Semantic Scholar records 0 citations for SWIFT (2505.12225) and HSRM (2608.30841), and
none of Q-Probe's 21 citing papers uses the probe as an RL reward (log in section 4).
**What would still be new:** the multimodal + open-ended + judge-labelled combination, and *nothing
else*. That is a thin delta and I would not build a paper on it.
**What the field reports about collapse / hacking:** SRT 2505.21444 — "prolonged RL with self-reward
leads to reward hacking ... **sudden and complete performance collapse**"; No Free Lunch 2506.17219 —
RLIF helps early then "**performance degrades even below the model before training**", and "little
improvement for **instruction-tuned** models" (all three of our generators are instruction-tuned);
Co-rewarding 2508.00410 — single-view signals create a "**self-consistent illusion**", fixed only by a
second view or a slowly-updated reference teacher. **Every single-signal internal reward in this
literature collapses under enough optimisation pressure.**

### S2 — failure mode of optimising against a probe reading own activations: **DONE, and the safe region is already mapped**
Closest: **Cundy & Gleave 2505.13787** (GRPO against an activation detector gives >85% evasion; **DPO /
off-policy gives <25%**; safety governed by exploration, detector TPR, KL strength), **RL-Obfuscation
2506.14261** (token-level monitors evade easily; **pooled / attention monitors are robust**; evasion
generalises to unseen monitors), **Wehner & Fritz 2510.21531** (probe-DPO preserves detectability better
than classifier-style; **probe ensembles largely unnecessary — refitting the probe after training
recovers accuracy**), plus Gao 2210.10760 for the gold-vs-proxy measurement design and 2607.01567 for
detector distribution-shift sensitivity.
**Safe conditions, as measured by that literature:** (i) off-policy / data-selection use with no
gradient through the probe; (ii) if on-policy, strong KL to a frozen reference; (iii) pooled rather than
token-level readout (our probe is mean-pooled — on the robust side); (iv) a **held-out** probe never
used for selection, plus a periodic refit, as the Goodhart monitor.

### S3 — off-policy BoN-distillation / RFT with a *learned hidden-state probe* as the selector: **OPEN**
The scaffolding is all DONE (RAFT, ReST, ReST-EM, STaR, RFT, West-of-N, BOND, BoNBoN, Chow et al.,
ScPO), and the *selector* in every one of them is a rule/verifier, a text reward model, or
self-consistency. I found **no paper that uses a trained hidden-state correctness probe as the selector
for rejection-sampling fine-tuning**, in text or in vision.
Closest three: **Q-Probe 2402.14688**, which flags it as future work in so many words ("adding some sort
of distillation step on top to reduce inference cost could be an interesting future direction") and
supplies the matched-label probe-vs-SFT comparison; **Neuron-OPSD 2607.02460**, which uses internal
*neuron activations* to select **which prompts** to self-distil on (not which candidate); **Intermediate
Probes 2603.12270**, which uses probes on a *teacher's* hidden states as the distillation target.
In VLMs: NOPD 2607.23125, STIC 2405.19716, SIMA 2405.15973, SelTDA 2306.03932 — **none** uses a latent
selector. In medical VLMs: none found (8 query formulations, section 4).
**What would still be new:** the selector itself; that it is ~1M params and free at selection time
because it reads activations already computed during generation; and the cross-lineage variant (a probe
trained on model A selecting the data that fine-tunes model B, with zero labels for B).

### S4 — open-ended medical VLM RL: **the bottleneck is explicitly acknowledged; model-internal signals for it are OPEN**
Verified by deep read of the HTML: **MedVLThinker 2508.02669** uses GRPO with binary **exact-match**
answer correctness and says in its own conclusion: *"we used exact-match answer checking, which is
straightforward for multiple-choice questions. **Extending RLVR to open-ended generation or multi-step
clinical reasoning (where reward shaping is harder) is an interesting challenge.**"* **Lingshu
2506.07044** keeps verifiability by construction: *"Since most collected samples are in multiple-choice
format, we reformulate those with word or phrase answers as open-ended questions to enhance difficulty
**while maintaining verifiability**"* — i.e. its "open-ended" RL answers are still string-checkable
words or phrases. MedVLM-R1 2502.19634 and Med-R1 2503.13939 are MCQ / rule-reward; GMAI-VL-R1
2504.01886 uses rejection sampling for *data synthesis*; QoQ-Med 2506.00711's DRPO re-weights by domain
rarity and modality difficulty, not by answer openness. General-domain answers to the same problem:
**RLPR 2506.18254** (policy's own token probability of a *reference answer*) and **RLVRR 2601.18533**
(keyword content plus LLM style check). **Nothing found uses model-internal states as the reward for
open-ended medical generation.**

### S5 — probe vs fine-tuning at matched labels: **partially DONE in text; OPEN in VLMs and for open-ended correctness**
Closest: **Q-Probe 2402.14688** Table 1 (numbers in C32 — probe 0.46 vs LoRA-SFT-on-successes 0.42 vs
LoRA reward model 0.34 vs greedy 0.38, probes trained on 464 problems) plus the DPO/PPO win-rate
comparison; **LoFiT 2406.01563** (localized fine-tuning beats probe-directed intervention); **ReFT
2404.03592**. Probe-to-activation edits for correctness: **ITI 2306.03341** is truthfulness on
TruthfulQA, **2605.20745** steers *verifier* strictness. I found no probe-to-weight-edit study targeting
open-ended answer correctness, and none in a VLM.

### S6 — verifier-policy co-adaptation for **latent** verifiers: **ADJACENT only; OPEN for probes**
The RM-staleness literature is mature (2507.15507 states the mechanism, 2505.18126 studies it across
iterations, 2605.04266 gives the Stackelberg decomposition and the alignment-collapse result,
2605.30888 keeps the RM fresh on-policy). **None of it is about a probe on the policy's own
activations**, where staleness has an extra cause the text-RM case does not have: the *representation
the probe reads is itself being fine-tuned*. Our measured 55% zero-shot survival across the
Qwen2.5-VL-7B to Lingshu-7B fine-tune is, as far as this search goes, an unpublished kind of number.
2607.01567's finding that detector performance is "sensitive to distribution shift between detector
training and preference-training data" is the nearest published statement.

### S7 — RMs / probes trained on LLM-judge labels inherit the judge: **ADJACENT; OPEN for latent probes and for multi-currency BoN reporting**
Closest: **2505.19176** — names **"teacher preference bias, where the proxy judge model learns a biased
preference for responses from the teacher model"**, calls it "critical yet previously overlooked", and
fixes it with a second, unbiased assistant model; **2603.12520** — a judge's global correlation badly
overstates its *selection* value (21.0% of oracle gain at r=0.47), the general form of "judge currency
is not EM currency"; **PAJAMA 2607.22561** and **RM-Distiller 2601.14032** for the distillation side.
All are **text, preference-based, generative proxies**. Nobody has shown that a *latent* probe distilled
from a judge inherits the judge's phrasing preferences, and I found no paper that reports best-of-N
gains in two or more independent currencies as a standard. Our audit finding (judge +0.0737 vs lenient
EM +0.0047 on identical picks, `em-rescore/em_rescore_pooled_probe_2026-09-18.json`) is a publishable
observation in its own right.

---
## 3. Directions that survive the search

Framing that the search forces on us: **the on-policy version of the owner's idea is DONE** (PAIR,
SR-GRPO, Latent-GRPO) **and is known to be dangerous** (Cundy & Gleave: GRPO against an activation
detector reaches >85% evasion; SRT / No Free Lunch: single-signal internal rewards collapse, and give
little on instruction-tuned models). The **off-policy** version — probe as a *data selector* for
fine-tuning — is the part nobody has done, and it is exactly the part the same literature says is safe
(DPO/off-policy <25% evasion; probe-DPO preserves detectability; refit recovers it). So all four
directions below are off-policy or explicitly instrument the on-policy danger rather than ignore it.

### D1 — PRFT: **Probe-selected Rejection-sampling Fine-Tuning** for open-ended medical VQA (primary)

**Pitch.** Take the train-half questions we already sampled 8 ways at T=0.7. Score all 8 candidates with
the frozen 24-probe ensemble (CPU, on cached hidden states — free). Keep the argmax candidate per
question as the target and LoRA-SFT the generator on (image, question, probe-picked answer). Evaluate
**greedy** decoding on the held-out half. The claim to test is precisely the owner's: *does the
best-of-N gain move into the weights, so greedy gets it for free?* No gold answer, no judge call, and no
external verifier is used anywhere in the loop at fine-tuning time.

**Nearest prior art and the delta.** Q-Probe 2402.14688 names this as future work and never does it;
ReST-EM 2312.06585 / RAFT 2304.06767 / RFT 2308.01825 do the same loop with a *verifier or rule*;
BOND 2407.14622 does it with a *text RM*; ScPO 2411.04109 with *self-consistency*; Neuron-OPSD 2607.02460
with raw activations but to choose *prompts*, not *candidates*; NOPD 2607.23125 / STIC / SIMA do VLM
self-improvement with *no* latent signal at all. The delta is a supervised, ~1M-parameter,
activation-reading selector in a VLM, in a domain (free-text medical VQA) where the field has publicly
declared the reward problem open (MedVLThinker's own conclusion; RLVRR's ICLR-2026 framing).

**Why this avoids the S2 failure mode.** The probe is applied **once**, to a *fixed* pool generated by
the *pre-update* policy, and then frozen out of the loop. No gradient ever flows through the probe; the
policy is never rewarded for its own probe score; there is no exploration against the monitor. That is
the regime Cundy & Gleave measured as <25% evasion and the regime Wehner & Fritz found preserves probe
validity. We additionally hold out a probe (different seeds and layers, never used for selection) as the
independent monitor, per Gao 2210.10760's gold-vs-proxy design.

**First experiment (2xA100-80GB).** Generator: **Qwen2.5-VL-7B-Instruct** first, because its native
probe has the largest measured gain (+0.0820 macro, `head_final_stack_qwen_2026-09-13.json`) and it is
the non-medical control model, so a win is not a medical-only artifact. Inputs already on disk: Qwen
train-half pools (132,274 rows / 54,835 unique questions, `replication/s02_rowcounts.out`) and their
cached hidden states in `feats_hidden/`. Probe scoring and dataset construction: **CPU only, minutes**
(the EM re-score already ran this way at <=4 threads). LoRA SFT (r=16, vision tower frozen, ~50k
examples, 1 epoch, bf16, grad-checkpointing): one A100 for roughly 6-10 h; greedy eval on the 18,452
held-out questions via vLLM (text-only LoRA, so the vLLM visual-LoRA landmine does not apply — but the
adapter must be verified module-by-module before any vLLM scoring, or scored under HF). Total: about one
day per arm, five arms.

**Required baselines (all at the SAME labelled budget, same LoRA recipe, same seed set).**
1. **Untouched greedy** (the number to beat).
2. **Random-pick RFT** — pick 1 of the 8 uniformly. *This is the single most important control*: it
   isolates "any self-distillation helps" (NOPD/CANON show plain self-distillation is strong) from
   "the probe's selection helps".
3. **Self-consistency-selected RFT** (ScPO-style) — the cheap selector we already measured at 0.647
   judge precision vs the probe's 0.825 (`me/` pilot); if PRFT cannot beat this, it is not worth its
   labels.
4. **LoRA-SFT on gold answers** on the same labelled half — the supervised reference.
5. **Judge-selected RFT** (MedVLThinker-32B judge picks the candidate) — the expensive upper bound, and
   the S7 control: if PRFT approaches it, the probe really is a cheap judge surrogate; if PRFT *matches
   its biases but not its gains*, that is the inheritance finding.
6. **BoN at inference on the untouched model** — the incumbent method, to show whether PRFT converts,
   or merely duplicates, the inference-time gain.
Also report, on the PRFT'd model: **BoN-on-top**. If the probe's BoN headroom collapses after PRFT, the
gain genuinely moved into the weights; if it does not, PRFT added something orthogonal (or nothing).

**Evaluation protocol that survives the currency problem.** Every arm reported in **four** numbers on
identical outputs: (a) MedVLThinker-32B judge (the training judge — expected to be inflated), (b) an
**independent cross-family judge** (MedGemma-27B; the re-label is already running), (c) **normalised
exact match**, (d) **token-F1**. Headline only where (b) and (c) agree with (a) in sign. Short-answer
benchmarks (VQA-RAD, SLAKE, PathVQA, VQA-Med C4) carry the EM claim because their golds are short;
verbose-gold benchmarks (Kvasir-x1, GEMeX) carry the judge claim only, and must say so. CIs clustered by
image, and a benchmark-level (n=8) CI alongside, since the two differ by a lot here
(`em-rescore/em_rescore_pooled_probe_2026-09-18.json`). Also report **pass@8 before and after** — CANON's
test for "is this just distribution sharpening?" — plus mean answer length, since the judge rewards
phrasings and PRFT could simply learn the judge's preferred register.

**What kills it.** PRFT <= random-pick RFT, or <= self-consistency RFT, in the cross-family judge *and*
in EM on the short-answer benchmarks. Or: the judge gain survives but EM/F1 move negative — that is the
S7 inheritance outcome and it converts the paper into D2.

**Main risk.** The measured EM gain on the *selection* task is already a tie (+0.0047), so there may be
little real signal to distil; PRFT could then reproduce the judge's phrasing preferences in the weights
and look like a win under the training judge only. This is why the cross-family judge gates the headline.

### D2 — "What does a latent verifier trained on judge labels actually learn?" (the companion, and the fallback paper)

**Pitch.** Train the same probe architecture on three label sources over identical pools — training
judge, cross-family judge, normalised EM — and cross-evaluate every probe in every currency, on
selection *and* (with D1) after distillation. Add a phrasing-invariance test: paraphrase the picked
answers and re-score. The deliverable is the first measurement of **judge-bias inheritance in a latent
probe**, and the first best-of-N result reported in multiple independent currencies as a matter of
protocol.

**Nearest prior art and the delta.** 2505.19176 (teacher preference bias in *generative* proxy judges),
2603.12520 (judge global agreement overstates selection value), RM-Distiller 2601.14032, PAJAMA
2607.22561. None is about a latent probe; none is multimodal; none reports BoN in multiple currencies.
Our own numbers already make the point (judge +0.0737 vs lenient EM +0.0047 on the identical picks; 32%
of the probe's upward judge flips have zero token overlap with the gold), so the paper exists even if
D1 fails.

**S2 exposure.** None — nothing is optimised against the probe.
**First experiment.** CPU-only, on cached features: refit 24-probe ensembles under three label sources
(minutes each at <=4 threads; **refits must run at <=4-6 threads**, CLAUDE.md thread-count landmine),
then the cross-currency table. The cross-family judge re-label is the only GPU cost and is already under way.
**What kills it.** The cross-family judge reproduces the training judge's +0.07 and EM stays a tie for
*every* label source — then the disagreement is an EM-metric artifact, not judge inheritance, and the
finding shrinks to "EM is the wrong currency for verbose golds" (still worth a section, not a paper).
**Risk.** Two judges from different families can share a bias (both are instruction-tuned on similar
data); the phrasing-invariance test is what separates "judges agree because they are right" from
"judges agree because they like the same register".

### D3 — The Goodhart curve for a latent verifier: measure the S2 danger instead of avoiding it

**Pitch.** Run PRFT for 2-3 rounds (regenerate pools from the updated policy, re-select, re-train) and,
at every round, track four curves: selection-probe score, **held-out probe** score (never used for
selection), cross-family judge accuracy, and EM. Refit a fresh probe each round and report how much its
held-out AUROC has moved. This turns our S6 question ("the probe goes stale as the generator changes")
and our S2 risk into a single measured object, in the setting where it has never been measured: a probe
that reads a representation which is itself being fine-tuned.

**Nearest prior art and the delta.** Gao 2210.10760 (gold vs proxy, text RM), 2505.18126 (iterated
RLHF), 2605.04266 (alignment collapse, Stackelberg), Wehner & Fritz 2510.21531 (probe refit recovers
accuracy — measured for toxicity only), 2607.01567 (detector distribution-shift sensitivity). **Delta:**
latent verifier, VLM, correctness rather than safety, and the specific twist that the probe's *input
space* moves under fine-tuning. Prediction worth publishing either way: selection-probe score rises
monotonically while the held-out probe and the cross-family judge flatten then fall — the first
Goodhart curve for an activation probe in a multimodal setting.

**S2 exposure.** Deliberate and instrumented; still off-policy within each round, so the danger is
bounded by the number of rounds, and the held-out probe is the tripwire.
**First experiment.** Rounds 2-3 of D1 on Qwen; each round = 8-sample regeneration on the train half
(the expensive part: roughly 50k x 8 generations, vLLM, most of a day on 2 GPUs) + CPU re-selection +
one LoRA run.
**What kills it.** All four curves move together for 3 rounds — no Goodhart, no paper, but a genuinely
reassuring negative result that D1 can cite.
**Risk.** Cost. Do not start it until D1 round 1 has cleared its baselines.

### D4 — Zero-label cross-lineage supervision (cheap, high-novelty, small)

**Pitch.** Use the **Lingshu** probes, unchanged, to select the fine-tuning data for **Qwen2.5-VL-7B**
(a probe trained on a *different* model, with zero labels for the target). Our pilot already shows 55%
of the selection gain survives that lineage hop zero-shot (+0.0451 vs +0.0820 native,
`me/pilot_cross_model_transfer.json`). If it also survives distillation, the claim becomes: *a verifier
trained once on one model can supervise the fine-tuning of another model in its family, with no labels
for the target at all* — a cheaper onboarding story than the "~100 labelled questions per benchmark"
number we currently have to quote.

**Nearest prior art and the delta.** Nobody in the S3/S6 sets transfers a *latent* verifier across
models as a training signal; 2607.01567 is the nearest, and it reports transfer as a *problem*. West-of-N
2401.12086 improves the RM from BoN; this is the reverse.
**S2 exposure.** Lowest of all four: the probe reads a model it was never trained on and is frozen, so
there is no monitor-evasion channel at all.
**First experiment.** One extra PRFT arm in D1 — same pipeline, different selector. Marginal cost: one
LoRA run. Do it in the same batch.
**What kills it.** The cross-lineage arm lands at or below random-pick RFT.
**Risk.** It inherits D1's risk, plus the ±0.008 open-text reproducibility floor, which is comparable to
the size of the effect we would be splitting (0.0451 vs 0.0820). Needs matched serving configs and
>=3 LoRA seeds per arm.

### What would make any of this a paper

Not "the probe improves the model" on its own — RAFT/ReST/NOPD/CANON have made self-improvement a
crowded claim, and the mechanism is already published. The publishable object is the **comparison at
matched labels and matched compute**: for a fixed labelling budget, is a ~1M-parameter activation probe
a better place to spend it than LoRA-SFT on gold, self-consistency filtering, or judge calls? Q-Probe
answered that for *inference-time selection* in text; nobody has answered it for *fine-tuning* in any
modality. Add the two honest measurements that our audit uniquely enables — multi-currency reporting
(D2) and the latent Goodhart / staleness curve (D3) — and the contribution is a method plus the two
controls the field is currently missing, rather than one more self-improvement number. The negative
outcome is also publishable in this project's established style: *"the best-of-N gain from a latent
verifier does not transfer into the weights, and here is the currency analysis showing why."*

---
## 3b. Two late cards that bear directly on the verdicts

**C62 · arXiv:2609.04336 · MedProb: Probing Internal Representations of VLMs for Medical QA** —
Nourbakhsh, Erfan; Yang, Ke; Rios, Anthony · 2026/09/03 · **EMNLP Findings 2026** ·
https://arxiv.org/abs/2609.04336
Already known to the project as the nearest medical collision. Verified detail that matters for S3:
MedProb is a probe that **predicts multiple-choice answers from frozen VLM representations without
free-text generation**; the abstract's last sentence says "we additionally show the probe can be
extended to open-ended generation **via a rejection-sampling scoring procedure**" — that is *reranking
at inference*, not fine-tuning the generator. Other verified facts worth reusing: across 14 matched
general/medical VLM pairs, "medical adaptation does not consistently improve this linear decodability";
and "free-text generation exhibits an **answer-position bias of up to 10 percentage points**".
**Conclusion: MedProb collides with our verifier, not with the training-signal direction. S3 stands.**

**C63 · arXiv:2508.19652 · Vision-SR1: Self-Rewarding VLM via Reasoning Decomposition** — Li, Zongxia;
Yu, Wenhao; Huang, Chengsong et al. · 2025/08/27 · https://arxiv.org/abs/2508.19652
Three-stage self-rewarding RL for VLMs with no external visual supervision: the model produces a
self-contained visual description, is **re-prompted on that description alone** to compute a visual
reward, and optimises a decoupled visual + language reward. Motivated by the same gap we care about
("most post-training methods for VLMs rely on simple **verifiable answer matching**"). The strongest
*self-rewarding VLM* baseline — and it is a text re-prompting signal, not a latent one.

---
## 4. Query log

arXiv `abs` pages fetched and read (helper `s5_scratch/ax.py` / `absfetch.py`), 63 in total, all
confirmed title + authors + date + abstract: 2605.17877, 2601.08427, 2512.02807, 2606.03234, 2505.12225,
2608.30841, 2511.06209, 2402.14688, 2506.14261, 2607.23125, 2306.03932, 2510.21531, 2304.06767,
2308.08998, 2312.06585, 2203.14465, 2308.01825, 2401.12086, 2407.14622, 2406.00832, 2505.19590,
2504.16084, 2505.21444, 2504.05812, 2506.17219, 2508.00410, 2505.15134, 2506.06395, 2412.15287,
2411.04109, 2306.03341, 2404.03592, 2406.01563, 2210.10760, 2412.09565, 2605.20745, 2502.19634,
2503.13939, 2508.02669, 2506.07044, 2504.01886, 2506.00711, 2507.05201, 2607.22561, 2601.14032,
2505.19176, 2505.18126, 2507.15507, 2605.30888, 2605.04266, 2405.19716, 2405.15973, 2603.12520,
2607.02460, 2607.07626, 2603.12270, 2607.13643, 2605.28631, 2601.18533, 2506.18254, 2505.13787,
2607.01567, 2609.04336, 2508.19652.

Full-text (arXiv HTML) deep reads via `s5_scratch/htmlgrep.py`: **2402.14688** (Q-Probe — Table 1,
related-work paragraph on rejection sampling + finetuning, cost comparison; saved as
`s5_scratch/qprobe.txt`), **2508.02669** (MedVLThinker — reward design and the open-ended limitation
sentence), **2506.07044** (Lingshu — RL data construction and the verifiability sentence),
**2506.00711** (QoQ-Med — reward-function section; "open-ended" has NO hit in its HTML).

Citation sweeps (Semantic Scholar graph API, script `/tmp/s2c.py`): citations of **2505.12225** = 0
recorded; **2608.30841** = 0 recorded; **2402.14688** = 21, of which none uses the probe as an RL reward
(nearest: 2512.04601 Natural Language Actor-Critic, 2509.26074 Learning Better Reward Model with Latent
Space Synthesis, 2410.20290 Speculative Rejection — none is a hidden-state probe driving policy updates).

arXiv API searches (`s5_scratch/axsearch.py`): `"hidden states" AND "reward" AND "vision-language" AND
"reinforcement learning"`; `"probe" AND "rejection sampling fine-tuning"` (0 hits); `"internal states"
AND "self-training" AND "language model"` (0 hits).

Web searches (distinct formulations, 8+ across the sub-questions):
1. hidden-state probe verifier used to select samples for rejection sampling fine-tuning self-distillation LLM
2. arXiv 2026 linear probe on hidden states as reward model for GRPO vision language model medical
3. reward model trained on LLM judge labels inherits judge bias distillation of judge preferences arXiv
4. verifier policy co-adaptation reward model goes stale during policy updates refresh reward model iterative RLHF arXiv 2026
5. "best-of-N" gain depends on evaluation metric exact match versus LLM judge discrepancy reporting multiple currencies verifier
6. probe on hidden states selects pseudo-labels for self-distillation fine-tuning internal confidence data selection arXiv 2026
7. hidden state verifier selects best-of-N candidates to fine-tune vision language model medical VQA open-ended reward free
8. Cundy Gleave preference learning with lie detectors can induce honesty or evasion arXiv
9. medical vision language model self-training rejection sampling verifier selected pseudo answers LoRA open-ended VQA 2026
10. "internal reward" OR "latent reward" probe activations GRPO multimodal vision language model 2026 reward hacking

**Negative-search statement for S3.** Across formulations 1, 6, 7, 9 and the two arXiv-API queries, plus
the Q-Probe citation sweep, I found no paper in which a **trained hidden-state correctness probe selects
which sampled candidate to fine-tune on**. The nearest misses were all one step away: internal *neuron
activations* selecting *prompts* (2607.02460), probes on a *teacher's* states supplying *soft labels*
(2603.12270), hidden-state *dynamics* selecting *RLVR instances* (2605.28631), and a probe reranking
open-ended generations *at inference* (2609.04336, MedProb).

---
## 5. Not verified / explicitly excluded from conclusions

- **"ELHSR"** as a title could not be confirmed from the arXiv page for 2505.12225, whose verified title
  is *Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling* and whose method
  is named **SWIFT**. The project's docs use "ELHSR/SWIFT 2505.12225" — treat "ELHSR" as an earlier title
  of the same arXiv entry (**NOT VERIFIED**; cite the KDD 2026 title).
- Semantic Scholar reporting **0 citations** for 2505.12225 and 2608.30841 is an index-coverage
  statement, not proof that nothing cites them. It should not be quoted as "nobody has followed up".
- **LiLaVe (2504.16760)** and **CASE (2608.17124)** were not re-fetched here (already carded by the
  existing literature package); no claim in this report depends on them.
- Paper counts in section 4 are of pages *I* fetched in this session; they do not include the 194 cards
  in `literature/DOMAIN_GUIDE_2026-09-16.md`.
- No number from the project's own artifacts is quoted here except the ones the briefing supplied with
  their source files (`em-rescore/em_rescore_pooled_probe_2026-09-18.json`,
  `me/pilot_cross_model_transfer.json`, `head_final_stack_qwen_2026-09-13.json`,
  `replication/s02_rowcounts.out`). I did not recompute any of them.
- GPU-time estimates in section 3 are **engineering estimates, not measurements**, and are labelled as
  such ("roughly", "about"). Nothing in this report was run on a GPU.
