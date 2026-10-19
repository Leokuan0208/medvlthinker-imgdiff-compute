# Figure style notes — how the probe / verifier literature draws its method figures

Source PDFs: `.claude/worktrees/lit-domain-package/literature/papers/`. Each figure below was rendered and viewed;
page renders and zoomed crops are in `/home/jamesyang/.claude/jobs/61c9eff9/tmp/deck/figstudy/`
(`<paper>_p<N>.png`, `zoom_*.png`). Hex colours marked "sampled" were read from pixels of the zoomed crops
(dominant saturated colours, quantised to steps of 8), so they are approximate.

---

## 1. Figure-by-figure study (12 figures)

### 1.1 HSRM — `C_li2026hsrm.pdf`, Fig. 1, p.3 (the closest analogue to our method)
**Depicts:** a frozen LLM; generated tokens T_{s,n} and their last-layer hidden states H_{s,n}; HSRM taking the
hidden states at step boundaries vs a text verifier re-tokenising and re-encoding.
Caption: *"During decoding, a frozen LLM produces generated tokens T_{s,n} together with their last-layer hidden
states H_{s,n}. HSRM with 2M parameters extracts the hidden states at reasoning-step boundaries to form a compact
step-level sequence [S]. Text-based verifiers instead take the sampled tokens, re-tokenize and re-embed them, and
run a forward pass over the full sequence [S × n] …"*
**Visual grammar:**
- **The LLM is a single wide grey rounded rectangle with a drop shadow** ("Large Language Model", white sans text).
  No layer stack is drawn at all.
- **Tokens and hidden states are two parallel rows of small rounded squares**: top row = hidden states (pale
  cream, sampled `#F8E8D0`), bottom row = generated tokens (lilac, sampled `#D8C8F8`). Each square is subscripted
  (H_{1,1}, H_{1,2}, …, H_{s,n}); "…" separates segments. The two rows are joined by short grey dashed down-arrows
  labelled "Sampling".
- **Selection of positions is shown by colour saturation**: the selected step-boundary squares in the top row are a
  darker amber (`#F8D8A0`/`#F0C878`); thin amber elbow connectors lift them into a separate bracketed group "[S]"
  (H_{s1}, H_{s2}, H_{s3}). This is exactly the device we need for "which tokens are pooled".
- **Row labels sit at the far left in grey serif bold** ("Last layer hidden state", "Sampling", "Generated
  tokens"), joined to the row by a faint dashed leader line — a swim-lane layout.
- Two **colour-coded lanes** (amber = ours, purple = baseline) each end in a "[score]" label; all inter-stage
  arrows are **dashed with filled triangular heads**, in the lane colour.
- Dimension annotations in square brackets in the lane colour: "[S]", "[S × n]".
- Method equations (Sec. 3) use h_t^ℓ ∈ ℝ^{d_gen} for *"the residual-stream representation at token position t
  and layer ℓ"*, z = LN((1/S)Σ_s z_s), f_φ(x,y) = wᵀz + b.

### 1.2 SWIFT / ELHSR — `C_guo2025elhsr.pdf`, Fig. 1, p.2
**Depicts:** "An illustration comparing traditional reward model and SWIFT."
**Visual grammar:**
- **Layer stack drawn as a vertical column of rounded grey blocks** ("Transformer Block", "⋯", "Transformer
  Block") between a light-blue "Input Text" box (bottom) and "Output Text" box (top), inside a grey panel titled
  "Task-performing LLM" in bold serif. Data flows **bottom-to-top** with short black arrows.
- **Taps:** a thin black line leaves *each* block to the right, the lines merge with a "Concatenate" label and enter
  the green "SWIFT (Ours) Linear-based" box. This is the canonical "tap from several layers" drawing.
- **Background tinting to split comparison regions:** salmon panel (traditional RM, top) vs pale-green panel
  (ours, bottom), each with a check/box list ("✓ Parameter Count ~10⁵", "□ Data-hungry"); the exponent is coloured
  red (10⁹) or green (10⁵).
- The head is decomposed into two child boxes ("Token-level Linear Gating", "Token-level Linear Reward") joined by
  a bracket — how they show head internals without a neuron diagram.
- Equations: concatenated h_t = [h_t^1; …; h_t^L] ∈ ℝ^{Ld}; trained with *"binary cross-entropy with logits loss"*.

### 1.3 Q-Probe — `C_li2024qprobe.pdf`, Fig. 1, p.2
Caption: *"An illustration of the Q-probe inference procedure. Given a prompt, we use the language model to
generate k = 3 completions … and the respective embeddings of the k prompt-completion pairs. Then the linear
Q-probe maps the features into the logits of a softmax distribution."*
**Visual grammar:**
- LLM = one khaki rounded box in the centre; prompt (grey code box) enters from left; **three candidate outputs fan
  out above** (Program 1/2/3, each a rounded box tinted green / blue / red), and **three feature vectors fan out
  below**.
- **Feature vectors are horizontal strips of 5 small squares**, each strip tinted to match its candidate — colour
  identity links candidate ↔ embedding. This is the best "N candidates, each with its own vector" device.
- The **probe is drawn as a vertical stack of small squares (a column vector, purple)**; the three feature strips
  arc into it with long curved arrows. Probe → "Softmax" labelled arrow → a mini bar chart "Probabilities over
  programs" (bars in the candidates' colours) → "∼" → sampled program.
- LaTeX/Computer-Modern serif labels throughout, small caps-free.

### 1.4 MedProb — `C_nourbakhsh2026medprob.pdf`, Fig. 1, p.1 (medical VLM probe, same domain as ours)
Caption: *"Prior methods (left) give a medical image and question to a VLM or multi-agent system … MedProb (right)
uses the same input, but keeps the VLM frozen, extracts hidden representations from each layer, and predicts the
multiple-choice answer with a linear classifier."*
**Visual grammar:**
- Two-panel "Previous Works | Our Work" split by a **vertical dotted divider**, each panel headed by a bright-blue
  pill (sampled `#18A0E0`) with white bold serif title.
- Real chest X-ray thumbnail + question text in a grey rounded card at the top of each panel.
- **VLM drawn as a coloured neural-net node icon** (circles + edges) in a peach card (`#F8E0C8`).
- **Frozen / trained marked with icons: ❄ snowflake (blue) on the VLM and on each hidden-representation column;
  🔥 flame (orange) on the "Linear Classifier"** — the dominant convention in 2025–26 VLM papers (Lingshu uses the
  same flame).
- **Hidden representations drawn as tall thin pale-blue vertical bars** (one per layer) inside a dashed-outline
  box "Internal Hidden Representations", with a few crossing lines between them — vectors as thin bars.
- Outcome rows: red card with ✗ and a red-filled wrong option vs green card with ✓ and green-filled correct option.
- Fig. 2 (p.5) repeats the same icons and adds a **prompt-composition stack**: "System Prompt + Image + Question +
  Options [A][B][C][D]" as stacked blue chips joined by "+" — a useful template for our token-sequence figure.
- Method text: *"we extract the hidden state of the last input token"*, h_i^{(ℓ)} = ψ_ℓ(f, p_i) ∈ ℝ^{d_ℓ}
  — parenthesised layer superscript.

### 1.5 Cobbe et al. (GSM8K verifiers) — `A_cobbe2021verifiers.pdf`, Fig. 4, p.7 and Fig. 12, p.20
Fig. 4 caption: *"A diagram of the verification training pipeline."* Fig. 12: *"Visualization of the joint
training objective. We mask out tokens in the question and only consider the loss corresponding to tokens in the
solution."*
**Visual grammar (the classic, minimal style):**
- Numbered circled steps ①②③ above three columns ("Train generator", "Generate and label 100 solutions/problem",
  "Train Verifier"); models are **plain white rectangles with thin black outline**, no fill, no icons.
- **Token-type colour coding with a legend at the left**: Q_i questions = blue (sampled `#B0C0D8`, border
  `#3860A8`), S_i solutions = orange (`#E8C8B0`), Y_i labels = green (`#B0D8B0`). Math-italic labels with
  superscript sample index S_i^1 … S_i^{100}, Y_i^{100}.
- **One tall Q_i block spanning all the S_i^j rows** — compact way to show "one question, N candidates".
- Fig. 12: token row of small squares over a wide grey "Verifier" bar; **masked positions are covered by a
  dark-grey translucent overlay**, legend "masked out (does not contribute to loss)". This is the right device for
  showing "pooled only over answer tokens" (grey-out the prompt).

### 1.6 Qwen2.5-VL — `G_bai2025qwen25vl.pdf`, Fig. 1, p.3 (our backbone)
Caption: *"The Qwen2.5-VL framework demonstrates the integration of a vision encoder and a language model decoder
… Images of varying sizes and video frames … are dynamically mapped to token sequences of varying lengths."*
**Visual grammar:**
- **Decoder = one long beige rounded bar** ("Qwen2.5 LM Decoder", italic purple sans) with a **row of small square
  tokens below (input) and above (output)**. Text tokens are lavender (`#C0C8F0`); each image's tokens get their own
  colour (salmon, yellow, orange, teal) with a **dimension bracket underneath** ("11427 tokens / Picture 1").
  Output positions that are not predicted are **dashed-outline squares**.
- Vision Encoder = a second, shorter beige bar below; a single up-arrow "Native Resolution Input".
- Right-hand **dashed call-out box** expands the ViT block (RMSNorm → Window/Full Attention → FFN with SwiGLU, ×M,
  ×1) — the standard "zoom into one block" device with a dashed leader line.
- This is the figure our token-sequence panel should echo (same bar + token-row grammar) so the audience recognises
  the backbone.

### 1.7 Lingshu — `G_xu2025lingshu.pdf`, Fig. 6, p.11 (our generator)
Caption: *"The training pipeline of Lingshu, which consists of four stages: medical shallow alignment, medical deep
alignment, medical instruction tuning, and medical-oriented reinforcement learning."*
**Visual grammar:** chevron (arrow-pentagon) stage blocks in a blue gradient, numbered ①–④; under each, a cream
card listing trained modules with a **🔥 flame** ("🔥 Projector, LLM"); data cards with image thumbnails and a token
count bar ("4.1M"). Confirms the flame = trained convention in exactly the model family we use.

### 1.8 R-EBM — `I_li2021rebm.pdf`, Fig. 1, p.2 (residual energy-based reranker; closest classical ancestor)
Caption: *"Schematic of a residual energy-based model (R-EBM) for an end-to-end ASR model. The baseline
auto-regressive model is fixed during R-EBM training."*
**Visual grammar (signal-processing / IEEE style):** small light-blue rounded boxes (Encoder, Decoder) stacked
bottom-up with **bold X at the input**; trained module in **salmon** ("Residual EBM"); **operators drawn as circled
glyphs** (⊗ product, ∫-like curve for exp(·), sigmoid curve in a circle labelled "sigmoid(·)"); outputs labelled
in math (P_θ(y|X), Confidence(y), −E_θ(X, y)). A bulleted plain-text list "R-EBM input features: – decoder hidden
state – …" sits beside the diagram instead of drawing them. Text: *"hidden representations for each output token
are gathered by mean pooling before being passed to the output layer that uses sigmoid activation"* — the same
pool → sigmoid recipe as ours.

### 1.9 CEM — `I_li2020cem.pdf`, Fig. 2, p.2
Caption: *"Confidence estimation module (CEM) for attention-based sequence-to-sequence models."*
**Visual grammar:** two stacked outlined regions — **frozen backbone in a teal-outlined panel, trained head in a
red/pink-outlined panel above it** (colour of panel outline = frozen vs trained; no icons). Modules are small
rounded boxes in **small-caps serif** (ENCODER, ATTENTION, DECODER, SOFTMAX, FC, SIGMOID); **bold vectors**
(**a**_t, **d**_t, **e**_1…**e**_L) label the arrows; taps are thin orthogonal (Manhattan-routed) lines from the
backbone up into "FC". Output p_t at top. Loss: L(c,p) = −(1/T) Σ_t (c_t log p_t + (1−c_t) log(1−p_t)).
This is the cleanest model for our "probe internals" figure.

### 1.10 Snell et al. — `A_snell2024scaling.pdf`, Fig. 2, p.8
Caption: *"Comparing different PRM search methods. Left: Best-of-N samples N full answers and then selects the best
answer according to the PRM final score."*
**Visual grammar:** blue "Question" pill at top; **N curved spline paths** fan down to N circular nodes at the
bottom; each node sits in a **dashed lavender rounded square = "Apply Verifier"**; nodes are filled **green
(selected by verifier) or red (rejected)**, the path to the selected one is also green. **Key/legend as a single
horizontal strip under all panels**. Top-right dashed note box explains the panel in a sentence.

### 1.11 Brown et al. (Large Language Monkeys) — `A_brown2024monkeys.pdf`, Fig. 1, p.2
Caption: *"The repeated sampling procedure … 1) We generate many independent candidate solutions … 2) We use a
domain-specific verifier … to select a final answer."*
**Visual grammar:** bold step headers above ("**Step 1:** Generate many candidate solutions."); candidates as
three short **dashed-outline pill boxes** — red dashed for wrong, green dashed for right — joined by thick black
arrows; the final answer is a solid-green pill; a "Problem 1 (coverage)" / "Problem 2 (precision)" question sits
under each stage. Good model for the coverage/selection split.

### 1.12 GenRM — `B_zhang2024genrm.pdf`, Fig. 3, p.3; and WED — `E_hagen2026wed.pdf`, Fig. 1, p.3
GenRM: rounded input chips ("Problem", "Solution") → large rounded verifier box with gradient fill → **small
diamond node** → "Yes 0.4 → (r)" with the reward as a **circled r in gold**; N CoT bars (orange gradient) and an
"Average" **dashed blue rounded bracket** around the per-sample scores. WED (our medical head-to-head paper): CT
thumbnail + question; four candidate answers in colour (red/blue/green/olive), each passing through **solid boxes
(generator rank) then dashed boxes (verifier rank)**, then a line plot of rank trajectories — solid vs dashed lines
encode which player.

(Also looked at: Best-of-Evidence Fig. 1 p.1 — busy multi-panel pipeline with pastel panels, an example of what
*not* to imitate for a slide; Bay et al. Fig. 1–2 p.2 — clean Tufte-style plots, coverage curve green
`#3F7F5A`-ish vs selection navy, shaded "identifiability gap"; Farquhar Nature Fig. 1 p.2 — dotted-ruled table
layout with coloured left-edge cluster bars.)

---

## 2. House style recommendation

### 2.1 What the literature converges on
1. **Backbone = one long rounded bar (or a stack of thin blocks), neutral fill; the trained part is the only
   saturated colour.** (HSRM, Qwen2.5-VL, SWIFT, CEM, R-EBM all do this.)
2. **Tokens = rows of small squares; vectors = thin bars or strips of squares; colour = token type.**
3. **Frozen vs trained:** either ❄/🔥 icons (MedProb, Lingshu — the current VLM convention) or panel outline
   colour (CEM). Use icons *and* a fill difference, so it reads in grayscale.
4. **Dashed = "applied/derived/not computed", solid = main data path.** Dashed boxes for "apply verifier"
   (Snell), dashed tokens for un-predicted positions (Qwen), dashed outlines for candidate answers (Brown).
5. **Correct = green, wrong = red** for candidates; selected path highlighted in the "ours" colour.
6. **Math labels in serif italic** (Computer Modern / STIX), module labels in sans or small caps.
7. **Single legend strip** at the bottom (Snell) or left (Cobbe); never per-element legends.

### 2.2 Palette for an indigo/teal deck
| role | hex | notes |
|---|---|---|
| ink (text, arrows, outlines) | `#1F2340` | near-black indigo |
| frozen backbone fill | `#ECEEF6` | very pale indigo-grey; outline `#B8BDD6` |
| frozen accent / ❄ icon | `#5B6BBF` | indigo |
| **trained probe (ours)** | `#0F9D8A` fill `#D5F2EC` | teal = the only saturated module colour; 🔥 icon `#E8743B` |
| prompt / system tokens | `#C9CDE0` | grey-indigo (de-emphasised, will be masked) |
| image tokens | `#F2C38B` | warm sand (echoes Qwen's warm image tokens) |
| question tokens | `#A8B8E8` | light indigo (Qwen-like lavender) |
| **answer tokens (pooled)** | `#2BB3A0` | teal, saturated = "these are read" |
| masked / not pooled overlay | `#1F2340` at 35 % opacity | Cobbe Fig. 12 device |
| correct candidate | `#2E9E5B` (fill `#DDF2E4`) | |
| wrong candidate | `#D2554B` (fill `#F8DEDB`) | |
| layer-tap highlight (ℓ = 18, 20, 22) | `#7A5CC7` | violet, distinct from teal and indigo |
| secondary / annotation grey | `#8A8FA8` | dimension brackets, "…" |

Check grayscale legibility: frozen (pale) vs trained (mid teal) vs answer tokens (mid-dark teal) differ in
lightness, not hue only.

### 2.3 Lines and type
- Module outlines 1.25 pt; main data arrows 1.5 pt solid with small filled triangular heads; tap lines 1.0 pt in
  the tap colour; derived/optional arrows 1.0 pt dashed (4–3 pt dash). Corner radius ≈ 6 px on boxes, 2 px on
  token squares. No drop shadows (HSRM's shadow reads as clip-art on slides).
- Module names: sans, medium weight (deck font). Math: serif italic (KaTeX/MathJax or STIX), bold for vectors
  if you follow CEM; be consistent.
- Dimension annotations in grey brackets beside the element: `[T_a × 3584]`, `ℝ^{3584}`, `×28`.

### 2.4 Notation (conventions seen in these papers; pick one and use it on every slide)
- Hidden state of token *t* at layer ℓ: **h_t^{(ℓ)} ∈ ℝ^{3584}** (MedProb writes h_i^{(ℓ)} with parenthesised
  layer; HSRM writes h_t^ℓ and calls it the *residual-stream representation*; SWIFT h_t^i). Parenthesised is
  safest (avoids confusion with powers).
- Candidate answers: y_1, …, y_N with N = 8 (Snell/Cobbe/HSRM use N; Q-probe uses k). Prompt: x (image + question).
  Answer-token positions: 𝒜(y_i), |𝒜| = T_i.
- Mean pool: **h̄_i^{(ℓ)} = (1/|𝒜(y_i)|) Σ_{t∈𝒜(y_i)} h_t^{(ℓ)}** (HSRM: z = (1/S)Σ_s z_s; R-EBM: "mean pooling").
- Probe: **s_i^{(ℓ,k)} = w₂ᵀ GELU(W₁ h̄_i^{(ℓ)} + b₁) + b₂**, W₁ ∈ ℝ^{256×3584}, w₂ ∈ ℝ^{256};
  p = σ(s). (If features are standardised before the probe, show it as ĥ = (h̄ − μ)/σ_feat — check the code; MedProb
  standardises with training-set mean/variance.)
- Loss: **ℒ = −[c log σ(s) + (1−c) log(1−σ(s))]**, c ∈ {0,1} = correctness label (CEM eq. 6 form; SWIFT calls it
  "binary cross-entropy with logits loss").
- Readout: r_i = (1/24) Σ_{ℓ,k} rank_{ℓ,k}(i); ŷ = argmax_i r_i, ties broken by vote count v_i (number of candidates
  with the same normalised answer). Write the tie-break explicitly; do not call it Borda unless it is one (see §3).

### 2.5 Six figure specs

**(a) Whole-pipeline figure (landscape, the "Fig. 1").** Left→right swim-lane like HSRM.
Image thumbnail + question card (MedProb style) → frozen VLM bar (❄) → **8 candidate pills fanning out**
(Brown/Snell grammar; neutral outline, no colour yet) → bracket "teacher-forced re-read: [x ; y_i]" back through
the same frozen VLM bar (draw the bar once with a looping arrow, or a second ghosted copy labelled "same weights, ❄")
→ three violet tap lines at ℓ = 18, 20, 22 (SWIFT tap device) → mean-pool glyph (a small funnel or "1/T Σ") over the
teal answer-token squares only → **24 small teal probe chips arranged as a 3 × 8 grid** (rows = layer, cols = seed;
🔥 once on the grid) → per-candidate score columns → "rank-average + vote tie-break" box → selected candidate pill
turns green (others grey). Single legend strip at the bottom: ❄ frozen · 🔥 trained · token colours · dashed = re-read.
Step numbers ①–④ above the stages (Cobbe/Lingshu). Note on the slide whether generation and feature extraction run
at the same image resolution (CLAUDE.md §0a says features came from a separate teacher-forced pass at a different
`max_pixels` than generation; check the FINAL_METHOD doc before drawing it as one pass).

**(b) Probe internals (CEM/R-EBM style).** Vertical, bottom-up: thin vertical bar "h̄_i^{(ℓ)} ∈ ℝ^{3584}" (draw as a
tall strip of squares, Q-probe style) → box "Linear 3584→256" → circled GELU glyph → "Linear 256→1" → circled σ
glyph → p_i. Annotate parameter count (3584·256 + 256 + 256 + 1 — compute the number in the deck, don't eyeball it)
beside the box in grey brackets. Put the BCE formula to the right, and the frozen backbone as a pale panel underneath
with a teal-outlined panel for the probe (CEM's two-panel outline colour) plus ❄/🔥.

**(c) Token sequence / pooling mask (Qwen + Cobbe Fig. 12 grammar).** One long row of small squares under a pale
"Lingshu-7B decoder (28 layers) ❄" bar: [system prompt] grey · [vision tokens] sand with a bracket "image tokens
(count depends on max_pixels; e.g. 320 at cap320 = 250,880 / 28²)" · [question] light indigo · [answer y_i] teal.
Overlay a dark translucent mask over everything except the answer tokens, legend "not pooled". Above the bar, three
violet lines rising from the answer squares at layers 18/20/22 into a "mean" glyph. Optional inset (dashed call-out,
Qwen style): image → ViT → 2×2 merger/projector → vision tokens (verify merger details in Qwen2.5-VL §2.1 before
labelling).

**(d) Ensemble and readout.** 3 × 8 grid of probes (rows ℓ = 18, 20, 22; columns seeds 1–8) → for one question,
an 8-row × 24-column heat-strip of ranks (candidates × probes) → row-mean → bar per candidate → highest bar
selected; if tied, a small "vote count" chip decides (show a tie being broken, drawn as two equal bars with a vote
badge). Snell-style green/red node colouring only at the end.

**(e) Cross-fitting / evaluation.** Two horizontal bars "half A" / "half B" of the question set (split by image, if
that is how the FINAL_METHOD doc defines the halves — confirm). Two rows: "train probes on A → score B", "train on B →
score A" with teal arrows; merged predictions → metrics in four labelled currencies (judge, cross-family judge, EM,
recall). Cobbe Fig. 4 grammar: three numbered columns, plain outlined boxes, colour only on data blocks.

**(f) Layer depth.** A vertical stack of 28 thin horizontal slabs (side view, SWIFT-style column but finer), ViT +
merger drawn as a separate shorter block beneath; slabs 18, 20, 22 filled violet, others pale; right-hand axis
"relative depth ℓ/28" with ticks 0, 0.25, 0.5, 0.75, 1 (18/28 ≈ 0.64, 20/28 ≈ 0.71, 22/28 ≈ 0.79 — **state the
indexing convention**, since HF `hidden_states[0]` is the embedding output, so "layer 18" may mean the output of
block 18 or 19). If a per-layer accuracy/AUROC curve exists in the artifacts, place it beside the stack sharing the
vertical axis (HSRM Fig. 3, MedProb Fig. 5 plot accuracy vs layer and find upper-middle layers strongest — useful
framing, cite them, don't import their numbers).

---

## 3. Terminology check

| component | conventional name(s) | who uses it |
|---|---|---|
| our classifier on hidden states | **probe**; *linear probe* (logistic regression); *MLP probe* when non-linear | Q-probe ("linear Q-probe", "small probe which is the same dimension as the residual stream"); MedProb ("linear probing", "linear logistic probe and the nonlinear MLP probe", App. L); MedGemma (linear probe on embeddings) |
| same thing framed as a reward model | **hidden-state reward model / verifier**, "lightweight hidden-state verifier" | HSRM ("lightweight hidden-state verifier for best-of-N"); SWIFT ("token-level reward model that directly mines LLM hidden states") |
| scalar output layer on an LM | **scalar head** | Cobbe ("a small scalar head that outputs predictions on a per-token basis"); "reward head" appears once in HSRM |
| whole-answer correctness scorer | **outcome reward model (ORM)** vs process reward model (PRM) | Snell (ORM vs PRM), HSRM ("Early outcome reward models rerank complete solutions using final-answer supervision (Cobbe et al., 2021)") |
| picker over N samples | **verifier** (Cobbe, Brown, Snell, GenRM, HSRM); *discriminative RM* vs *generative verifier* (GenRM); *reranker* (Collins 2005, R-EBM "n-best rescoring") |
| the procedure | **best-of-N** (Snell, GenRM, HSRM, SWIFT, Bay); *repeated sampling* (Brown); *rejection sampling* (Q-probe); *n-best reranking* (Collins, R-EBM) |
| upper bound | **oracle / pass@N / coverage** (Brown "coverage", Snell, HSRM "pass@8 upper bound", Bay "identifiability gap") |
| the vectors read | **hidden states**; **residual stream** (Q-probe, HSRM); *internal representations* (MedProb); *embeddings* (Q-probe, informal) |
| re-reading the answer with the prompt | **teacher forcing** — only CEM uses the word (for training); none of the probe papers draws a re-read pass, because HSRM/SWIFT read states produced *during* generation. If ours is a separate forward pass over [x; y_i], say "teacher-forced re-encoding" and draw it, since HSRM's caption explicitly contrasts itself with re-encoding |
| pooling | **mean pooling** (R-EBM, HSRM "mean-pools the encoded step representations"); SWIFT uses *gated weighted average*; MedProb uses *last input token* (no pooling) |
| frozen backbone | **frozen** (HSRM 8×, MedProb 25×, Q-probe) |
| loss | **binary cross-entropy (with logits)** (SWIFT, HSRM, CEM, R-EBM) |
| combining ranks | **none of the 34 papers uses "Borda count" or "rank aggregation"** (text search: 0 hits). SWIFT has *rank selection* and *scaled averaging* for combining with external RMs. "Rank averaging" is accurate for ours; Borda count is equivalent only if every probe ranks all 8 and ties are handled the same way — say "mean rank (Borda-equivalent)" only after checking that. |
| tie-break by agreement | **majority voting / self-consistency** (Wang 2023, Brown, Snell); "best-of-N weighted" (Snell: sums verifier scores per distinct answer) is the closest named relative of a vote-aware readout |

Uncertainties to flag: (1) the 28-layer index convention for "layers 18/20/22"; (2) whether feature extraction and
generation share image resolution (affects figure (a)); (3) the Qwen merger details (2×2) — read §2.1 of the Qwen
paper before labelling; (4) colour hexes are my proposal, not drawn from any deck file.

## Coordinator-verified facts (2026-10-18)
- Layer index: `extract_generator_hidden.py:23` — layers are indices into `output_hidden_states`, 0 = embeddings,
  so "layer 18" = residual stream after decoder block 18 of 28 (relative depth 0.643 / 0.714 / 0.786 for 18/20/22).
- Two passes: generation under vLLM at cap320 (max_pixels 250,880); features from a separate teacher-forced HF
  pass at max_pixels 1,003,520 (AUDIT_2026-09-18 §4). Draw the re-read pass explicitly. (If the cap320-feature
  arm is adopted, the re-read is at the generation resolution — update then.)
- Capture-inside-generation test (earlier probe, Aug 2026): 24 / 2,345 picks changed, accuracy 0.0000 [−0.0021, +0.0021]
  (`free_head_2026-08-16.json`, head_only::cap320_ar vs cap320_tf).
