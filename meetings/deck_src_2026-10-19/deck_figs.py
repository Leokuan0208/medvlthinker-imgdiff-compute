"""deck_figs.py — research-paper-style method figures (hand-built SVG) for the professor deck.
Visual grammar follows FIGURE_STYLE_NOTES.md: frozen = pale fill + snowflake, trained = teal + flame,
taps = violet, pooled answer tokens = teal, masked positions = dark translucent overlay, math in serif italic."""
import base64, json
from deck_lib import C, esc

D = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19"
EX = json.load(open(f"{D}/worked_example_12231.json"))
IMG64 = base64.b64encode(open(f"{D}/ex12231.jpg", "rb").read()).decode()

INK, MUT = C["ink"], C["muted"]
SNOW = "❄"
FIRE = "\U0001F525"


_SVG_N = [0]


def svg(w, h, body, extra_defs=""):
    _SVG_N[0] += 1
    u = f"f{_SVG_N[0]}"
    for mk in ("ahv", "aht", "ah", "hatch"):          # longest first, so "ah" does not eat "ahv"/"aht"
        body = body.replace(f"url(#{mk})", f"url(#{mk}{u})")
    return _svg_raw(w, h, body, extra_defs).replace('id="ahv"', f'id="ahv{u}"').replace('id="aht"', f'id="aht{u}"') \
        .replace('id="ah"', f'id="ah{u}"').replace('id="hatch"', f'id="hatch{u}"')


def _svg_raw(w, h, body, extra_defs=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
            f'<marker id="ahv" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{C["tap"]}"/></marker>'
            f'<marker id="aht" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{C["teal"]}"/></marker>'
            f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{C["frozen_o"]}" stroke-width="2"/></pattern>{extra_defs}</defs>'
            f'{body}</svg>')


def t(x, y, s, fs=13, fill=INK, anchor="start", weight="400", cls="", italic=False, family=""):
    st = ' font-style="italic"' if italic else ""
    fam = f' font-family="{family}"' if family else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x}" y="{y}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{st}{fam}{c}>{s}</text>'


def m(x, y, s, fs=15, fill=INK, anchor="start"):
    """math label in serif italic (KaTeX fonts are loaded by the page)."""
    return (f'<text x="{x}" y="{y}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" '
            f'font-family="KaTeX_Math, KaTeX_Main, Times New Roman, serif" font-style="italic">{s}</text>')


def box(x, y, w, h, fill="#fff", stroke=INK, sw=1.25, r=6, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def arrow(x1, y1, x2, y2, col=INK, sw=1.5, dash="", mk="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}"{d} marker-end="url(#{mk})"/>'


def path(d, col=INK, sw=1.5, dash="", mk="ah", fill="none"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    me = f' marker-end="url(#{mk})"' if mk else ""
    return f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}"{da}{me}/>'


def step(x, y, n, label):
    return (f'<circle cx="{x}" cy="{y}" r="11" fill="{C["indigo_d"]}"/>' + t(x, y + 4.5, n, 12.5, "#fff", "middle", "700")
            + t(x + 18, y + 5, label, 14, C["indigo_d"], "start", "700"))


def tokens(x, y, n, col, size=9, gap=2, stroke="none"):
    return "".join(f'<rect x="{x + k*(size+gap)}" y="{y}" width="{size}" height="{size}" rx="2" fill="{col}" stroke="{stroke}"/>' for k in range(n))


def frozen_bar(x, y, w, h, label, sub=""):
    s = box(x, y, w, h, C["frozen"], C["frozen_o"], 1.25, 8)
    s += t(x + w / 2, y + h / 2 + (0 if sub else 5), label, 14.5, INK, "middle", "600")
    if sub:
        s += t(x + w / 2, y + h / 2 + 17, sub, 11.5, MUT, "middle")
    s += t(x + w - 16, y + 19, SNOW, 15, C["indigo_l"], "middle")
    return s


# ============================================================================ Fig. 1 — whole pipeline
def fig_pipeline():
    W, H = 1192, 452
    c = EX["candidates"]                    # Large Bowel / Small Bowel / Spleen (stored order)
    order = sorted(range(len(c)), key=lambda k: -c[k]["votes"])
    s = []
    # input
    s.append(f'<image href="data:image/jpeg;base64,{IMG64}" x="8" y="56" width="150" height="150" preserveAspectRatio="xMidYMid slice"/>')
    s.append(box(8, 56, 150, 150, "none", C["line_d"], 1, 4))
    s.append(t(8, 224, "SLAKE · CT abdomen", 11.5, MUT))
    s.append(box(8, 232, 150, 74, "#f6f7fb", C["line_d"], 1, 6))
    s.append(t(16, 251, "“What organ is the gray", 12.2, INK))
    s.append(t(16, 266, "part on the left of the", 12.2, INK))
    s.append(t(16, 281, "image?”", 12.2, INK))
    s.append(t(16, 299, "gold: Large Bowel", 11.5, C["good"], weight="600"))
    s.append(m(83, 334, "x", 16, INK, "middle") + t(92, 334, " = (image, question)", 12, MUT))
    # stage 1: sample
    s.append(step(196, 26, "1", "Sample N = 8 answers  (T = 0.7)"))
    s.append(arrow(162, 180, 192, 180))
    s.append(frozen_bar(196, 120, 150, 120, "Lingshu-7B", "frozen VLM"))
    s.append(t(271, 258, "vLLM · cap320", 11.5, MUT, "middle"))
    s.append(t(271, 272, "image encoded once", 11.5, MUT, "middle"))
    samples = EX["samples_T07"]
    for k, a in enumerate(samples):
        yy = 52 + k * 31
        ok = a.rstrip(".") == "Large Bowel"
        col = C["good"] if ok else C["bad"]
        s.append(path(f"M346,180 C 362,180 360,{yy+12} 378,{yy+12}", C["line_d"], 1, mk=""))
        s.append(box(380, yy, 112, 24, "#fff", col, 1.2, 12, "4 3"))
        s.append(t(436, yy + 16, esc(a.rstrip(".")), 12, INK, "middle"))
        s.append(m(372, yy + 9, f"y<tspan font-size='9' dy='3'>{k+1}</tspan>", 11, MUT, "end"))
    # dedupe
    s.append(t(560, 70, "distinct answers", 11.5, MUT, "middle"))
    s.append(t(560, 84, "+ vote count", 11.5, MUT, "middle"))
    dy = {}
    for j, k in enumerate(order):
        yy = 110 + j * 56
        dy[k] = yy
        a = c[k]["answer"].rstrip(".")
        ok = c[k]["judge_label"] == 1
        s.append(box(508, yy, 104, 30, "#fff", C["good"] if ok else C["bad"], 1.4, 15))
        s.append(t(560, yy + 19, esc(a), 12.5, INK, "middle", "600"))
        s.append(f'<circle cx="612" cy="{yy}" r="11" fill="{C["indigo_d"]}"/>' + t(612, yy + 4, f"×{c[k]['votes']}", 10.5, "#fff", "middle", "700"))
    s.append(arrow(494, 180, 504, 180, C["line_d"]))
    # stage 2: re-read + taps
    s.append(step(640, 26, "2", "Re-read each answer, tap layers 18 / 20 / 22"))
    s.append(box(640, 100, 190, 200, C["frozen"], C["frozen_o"], 1.25, 8, "5 4"))
    s.append(t(735, 120, "same weights", 12, MUT, "middle") + t(812, 122, SNOW, 15, C["indigo_l"], "middle"))
    # mini layer stack
    for L in range(28):
        yy = 286 - L * 5.6
        on = (L + 1) in (18, 20, 22)
        s.append(f'<rect x="660" y="{yy:.1f}" width="80" height="4" rx="1" fill="{C["tap"] if on else "#d6d9e8"}"/>')
    for L in (18, 20, 22):
        yy = 286 - (L - 1) * 5.6 + 2
        s.append(arrow(742, yy, 780, yy, C["tap"], 1.2, mk="ahv"))
        s.append(m(784, yy + 4, f"h̄<tspan font-size='9' dy='-7'>({L})</tspan>", 13, C["tap"]))
    s.append(t(700, 314, "28 decoder blocks", 11.5, MUT, "middle"))
    s.append(t(735, 334, "teacher-forced on", 11.5, MUT, "middle"))
    s.append(m(735, 350, "[x ; y<tspan font-size='9' dy='3'>i</tspan><tspan dy='-3'>]</tspan>, mean over answer tokens", 12, MUT, "middle"))
    s.append(arrow(626, 180, 636, 180, C["line_d"]))
    # stage 3: 24 probes
    s.append(step(862, 26, "3", "24 probes score each answer"))
    gx, gy = 866, 112
    for r, L in enumerate((18, 20, 22)):
        s.append(t(gx - 6, gy + r * 36 + 17, f"ℓ={L}", 11, C["tap"], "end", "600"))
        for k in range(8):
            s.append(box(gx + k * 17, gy + r * 36, 14, 26, C["probe_bg"], C["probe"], 1.1, 3))
    s.append(t(gx + 68, gy + 126, "8 seeds per layer", 11.5, MUT, "middle"))
    s.append(t(gx + 146, gy - 6, FIRE, 13, anchor="middle"))
    s.append(t(gx + 68, gy + 146, "MLP 3584→256→1, BCE", 11.5, MUT, "middle"))
    s.append(arrow(840, 180, 858, 180, C["tap"], 1.2, mk="ahv"))
    # stage 4: select
    s.append(step(1032, 26, "4", "Rank-average, pick"))
    s.append(t(1034, 64, "mean rank over 24 probes", 11.5, MUT))
    bx = 1036
    for j, k in enumerate(order):
        yy = 84 + j * 52
        mr = c[k]["mean_rank"]
        ok = c[k]["judge_label"] == 1
        s.append(t(bx, yy + 14, esc(c[k]["answer"].rstrip(".")), 12, INK))
        s.append(box(bx, yy + 20, 130, 12, "#f0f2f8", "none", 0, 3))
        s.append(box(bx, yy + 20, max(2, 130 * mr), 12, C["teal"] if ok else C["grey"], "none", 0, 3))
        s.append(t(bx + 136, yy + 31, f"{mr:.3f}", 11.5, INK, cls="mono"))
    s.append(arrow(1012, 180, 1028, 180, C["teal"], 1.4, mk="aht"))
    s.append(box(1036, 252, 150, 46, C["good_bg"], C["good"], 1.6, 23))
    s.append(t(1111, 272, "answer:", 11.5, C["good"], "middle"))
    s.append(t(1111, 290, esc(EX["pick"].rstrip(".")), 15, C["good"], "middle", "700"))
    s.append(t(1111, 318, f"majority vote would say “{esc(EX['majority_vote'].rstrip('.'))}”", 11, C["bad"], "middle"))
    s.append(t(1111, 333, "ties in mean rank → more votes wins", 11, MUT, "middle"))
    # legend strip
    ly = 410
    s.append(f'<line x1="0" y1="{ly-22}" x2="{W}" y2="{ly-22}" stroke="{C["line"]}"/>')
    lg = [(SNOW + "  frozen (not trained by us)", C["indigo_l"]), (FIRE + "  trained by us", INK),
          ("— violet: hidden-state taps", C["tap"]), ("pill outline: green = judged correct, red = wrong", MUT),
          ("numbers: real, question SLAKE 12231, probes of the fold that never saw this image", MUT)]
    lx = 0
    for txt, col in lg:
        s.append(t(lx, ly, txt, 12, col))
        lx += 9 + 6.4 * len(txt)
    return svg(W, H, "".join(s))


# ============================================================================ token sequence + pooling mask
def fig_tokens():
    W, H = 1192, 360
    s = []
    # decoder bar
    s.append(box(30, 40, 1130, 150, C["frozen"], C["frozen_o"], 1.25, 8))
    s.append(t(44, 62, "Lingshu-7B language model (Qwen2.5-7B decoder): 28 blocks, hidden size 3584", 14, INK, weight="600"))
    s.append(t(1146, 62, SNOW, 16, C["indigo_l"], "end"))
    # residual stream lanes (block outputs)
    for L in range(29):
        yy = 176 - L * 3.9
        on = L in (18, 20, 22)
        s.append(f'<line x1="44" y1="{yy:.1f}" x2="1146" y2="{yy:.1f}" stroke="{C["tap"] if on else "#dcdfeb"}" stroke-width="{1.6 if on else 0.8}"/>')
    for L in (18, 20, 22):
        yy = 176 - L * 3.9
        s.append(t(40, yy + 3.5, f"ℓ={L}", 10.5, C["tap"], "end", "600"))
    s.append(t(40, 179, "ℓ=0", 10, MUT, "end"))
    s.append(t(40, 70, "ℓ=28", 10, MUT, "end"))
    # token row
    y = 214
    groups = [("system prompt", C["sys"], 14, "≈ 20"), ("image tokens", C["img"], 46, "≈ 274 at cap320 · ≈ 520 in the feature pass"),
              ("question", C["q"], 12, "≈ 15"), ("answer  y", C["ans"], 6, "≈ 5")]
    x = 60
    spans = {}
    for name, col, n, cnt in groups:
        x0 = x
        s.append(tokens(x, y, n, col, 13, 3))
        x += n * 16 + 18
        spans[name] = (x0, x - 18)
        s.append(f'<path d="M{x0},{y+22} v6 h{x-18-x0} v-6" fill="none" stroke="{MUT}" stroke-width="1"/>')
        s.append(t((x0 + x - 18) / 2, y + 44, name, 13, INK, "middle", "600"))
        s.append(t((x0 + x - 18) / 2, y + 60, cnt + " tokens", 11.5, MUT, "middle"))
    # mask over non-answer tokens inside the decoder region
    a0, a1 = spans["answer  y"]
    s.append(f'<rect x="54" y="72" width="{a0-60}" height="112" fill="{INK}" fill-opacity="0.08"/>')
    s.append(t((54 + a0) / 2, 128, "states computed here are not read by the probe", 12.5, MUT, "middle", italic=True))
    # taps from answer columns
    for L in (18, 20, 22):
        yy = 176 - L * 3.9
        s.append(f'<rect x="{a0-2}" y="{yy-3:.1f}" width="{a1-a0+4}" height="6" rx="2" fill="{C["tap"]}" fill-opacity="0.35"/>')
    s.append(box(a0 - 6, 70, a1 - a0 + 12, 116, "none", C["ans"], 1.5, 4, "4 3"))
    # pooling to the right
    px = a1 + 40
    for j, L in enumerate((22, 20, 18)):
        yy = 176 - L * 3.9
        s.append(path(f"M{a1+6},{yy:.1f} C {a1+24},{yy:.1f} {px-14},{90+j*30} {px},{90+j*30}", C["tap"], 1.3, mk="ahv"))
        s.append(m(px + 4, 95 + j * 30, f"h̄<tspan font-size='10' dy='-7'>({L})</tspan><tspan dy='7'> = </tspan>"
                   f"<tspan font-size='12'>1/|𝒜|</tspan><tspan> Σ</tspan><tspan font-size='10' dy='4'>t∈𝒜</tspan>"
                   f"<tspan dy='-4'> h</tspan><tspan font-size='10' dy='-7'>({L})</tspan><tspan font-size='10' dy='11'>t</tspan>"
                   f"<tspan dy='-4'>  ∈ ℝ</tspan><tspan font-size='10' dy='-7'>3584</tspan>", 15, INK))
    # notes
    s.append(t(60, 318, "𝒜 = positions of the candidate's own answer tokens. One forward pass per (image, question, candidate); "
                        "the states at all other positions are computed but ignored.", 13, MUT))
    s.append(t(60, 338, "Token counts: generation geometry measured at cap320, mean over 8,943 candidates (free_head_2026-08-16.json, "
                        "DELIVERABLE_1/3); system and question counts approximate.", 11.5, C["faint"]))
    return svg(W, H, "".join(s))


# ============================================================================ layer depth
def fig_depth(layer_gain=None):
    W, H = 560, 470
    s = []
    x0, y0 = 120, 30
    s.append(t(x0 + 70, y0 - 10, "Lingshu-7B", 13, INK, "middle", "600"))
    for L in range(28, 0, -1):
        yy = y0 + (28 - L) * 13
        on = L in (18, 20, 22)
        s.append(box(x0, yy, 140, 10, C["tap"] if on else C["frozen"], C["tap"] if on else C["frozen_o"], 0.8, 2))
        if L in (1, 7, 14, 18, 20, 22, 28):
            s.append(t(x0 - 8, yy + 9, f"block {L}", 10.5, C["tap"] if on else MUT, "end", "600" if on else "400"))
    s.append(box(x0, y0 + 28 * 13 + 4, 140, 16, "#e6e9f3", C["frozen_o"], 0.8, 3))
    s.append(t(x0 + 70, y0 + 28 * 13 + 16, "token embeddings (ℓ = 0)", 10.5, MUT, "middle"))
    s.append(box(x0 + 10, y0 + 28 * 13 + 26, 120, 36, "#f3ece0", C["img"], 1, 4))
    s.append(t(x0 + 70, y0 + 28 * 13 + 41, "ViT (32 layers)", 10.5, INK, "middle"))
    s.append(t(x0 + 70, y0 + 28 * 13 + 55, "+ 2×2 merger", 10.5, INK, "middle"))
    # relative depth axis
    ax = x0 + 160
    s.append(f'<line x1="{ax}" y1="{y0}" x2="{ax}" y2="{y0+28*13}" stroke="{MUT}"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 + 28 * 13 - v * 28 * 13
        s.append(f'<line x1="{ax}" y1="{yy:.1f}" x2="{ax+5}" y2="{yy:.1f}" stroke="{MUT}"/>' + t(ax + 9, yy + 4, f"{v:g}", 10.5, MUT))
    s.append(t(ax + 6, y0 - 10, "ℓ / 28", 11, MUT))
    for L in (18, 20, 22):
        yy = y0 + (28 - L) * 13 + 5
        s.append(t(ax + 44, yy + 4, f"{L}/28 = {L/28:.3f}", 11.5, C["tap"], weight="600"))
    if layer_gain:
        bx = ax + 150
        s.append(t(bx, y0 - 10, "probe gain vs greedy", 11, MUT))
        mx = max(v for _, v in layer_gain)
        for L, v in layer_gain:
            if isinstance(L, int):
                yy = y0 + (28 - L) * 13
                s.append(box(bx, yy, 90 * v / mx, 10, C["tap"] if L in (18, 20, 22) else "#cfd2e3", "none", 0, 2))
                s.append(t(bx + 90 * v / mx + 4, yy + 9, f"+{v:.4f}", 10, INK, cls="mono"))
    return svg(W, H, "".join(s))


# ============================================================================ probe internals
def fig_probe():
    W, H = 560, 486
    s = []
    cx = 150
    # input strip
    s.append(t(cx, 478, "input  h̄ ∈ ℝ³⁵⁸⁴  (one layer, one answer)", 12.5, MUT, "middle"))
    for k in range(26):
        s.append(f'<rect x="{cx-130+k*10}" y="440" width="8" height="18" rx="1.5" fill="{C["ans"]}" fill-opacity="{0.35+0.65*((k*37)%11)/10:.2f}"/>')
    blocks = [("standardise", "h̃ = (h̄ − μ) ⊘ (s + ε), train-set stats", 372, "#f6f7fb", C["line_d"], False),
              ("Linear 3584 → 256", "W₁, b₁ · 917,760 params", 300, C["probe_bg"], C["probe"], True),
              ("GELU", "Gaussian-error linear unit", 228, "#fff", C["probe"], False),
              ("Linear 256 → 1", "w₂, b₂ · 257 params", 156, C["probe_bg"], C["probe"], True)]
    for name, sub, y, fill, st, trained in blocks:
        s.append(box(cx - 130, y, 260, 46, fill, st, 1.3, 6))
        s.append(t(cx, y + 20, name, 14.5, INK, "middle", "600"))
        s.append(t(cx, y + 37, sub, 12, MUT, "middle"))
        s.append(arrow(cx, y + 72 if y < 372 else 438, cx, y + 48))
        if trained:
            s.append(t(cx + 146, y + 30, FIRE, 16, anchor="middle"))
    s.append(arrow(cx, 156, cx, 70))
    s.append(m(cx, 56, "f<tspan font-size='11' dy='4'>ℓ,k</tspan><tspan dy='-4'>(h̃</tspan><tspan font-size='11' dy='4'>i</tspan><tspan dy='-4'>) ∈ ℝ</tspan>", 18, INK, "middle"))
    s.append(t(cx, 30, "score (uncalibrated; only its order is used)", 12.5, MUT, "middle"))
    # training-only side branch: sigmoid inside the loss
    s.append(path(f"M{cx},112 H306", C["grey_d"], 1.3, "4 3"))
    s.append(box(310, 88, 236, 50, "#fff", C["grey_d"], 1.2, 6, "5 4"))
    s.append(t(428, 108, "σ(·) → class-weighted BCE", 13.5, INK, "middle", "600"))
    s.append(t(428, 126, "training only — not used at inference", 12, MUT, "middle"))
    # total
    s.append(box(310, 236, 236, 70, "#fbfcfe", C["line_d"], 1, 8))
    s.append(t(428, 260, "918,017 parameters per probe", 13.5, INK, "middle", "600"))
    s.append(t(428, 279, "× 24 probes = 22.03 M", 12.5, MUT, "middle"))
    s.append(t(428, 296, "(dropout = identity, p = 0)", 11.5, C["faint"], "middle"))
    s.append(t(320, 340, FIRE + " = trained (both Linear layers)", 12, INK))
    return svg(W, H, "".join(s))


# ============================================================================ readout heat-strip (real ranks)
def fig_readout():
    c = EX["candidates"]
    order = sorted(range(len(c)), key=lambda k: -c[k]["mean_rank"])
    W, H = 1192, 300
    s = []
    x0, y0, cw, ch = 150, 50, 26, 40
    s.append(t(x0, 24, "within-question rank given by each probe  (0 = worst, 1 = best;  ranks = position / (n − 1), ties averaged)", 12.5, MUT))
    for g, L in enumerate((18, 20, 22)):
        s.append(t(x0 + g * 8 * cw + 4 * cw + g * 10, y0 - 8, f"layer {L}  ·  seeds 0–7", 11.5, C["tap"], "middle", "600"))
    for r, k in enumerate(order):
        yy = y0 + r * (ch + 8)
        ok = c[k]["judge_label"] == 1
        s.append(t(x0 - 12, yy + 25, esc(c[k]["answer"].rstrip(".")), 14, C["good"] if ok else C["bad"], "end", "600"))
        for p, rk in enumerate(c[k]["probe_ranks"]):
            g = p // 8
            xx = x0 + p * cw + g * 10
            col = f'rgba(0,137,123,{0.08 + 0.92*rk:.2f})'
            s.append(f'<rect x="{xx}" y="{yy}" width="{cw-3}" height="{ch}" rx="3" fill="{col}" stroke="{C["line"]}"/>')
            s.append(t(xx + (cw - 3) / 2, yy + 25, f"{rk:g}", 10.5, "#fff" if rk > 0.6 else INK, "middle"))
        # mean
        mx = x0 + 24 * cw + 2 * 10 + 30
        s.append(t(mx, yy + 18, "mean", 10.5, MUT))
        s.append(t(mx, yy + 34, f"{c[k]['mean_rank']:.3f}", 15, INK, weight="700", cls="mono"))
        s.append(t(mx + 80, yy + 18, "votes", 10.5, MUT))
        s.append(t(mx + 80, yy + 34, f"{c[k]['votes']} / 8", 15, INK, weight="700", cls="mono"))
        if r == 0:
            s.append(box(mx + 140, yy + 4, 120, 32, C["good_bg"], C["good"], 1.5, 16))
            s.append(t(mx + 200, yy + 25, "selected", 13, C["good"], "middle", "700"))
    s.append(t(x0, H - 12, f"Real values for SLAKE question {EX['idx']} (CT abdomen), scored by the fold-{EX['scored_by_fold_probes']} probes "
                           f"(trained without this image). Majority vote picks “{EX['majority_vote'].rstrip('.')}” (5 of 8 samples) — judged wrong.", 12, C["faint"]))
    return svg(W, H, "".join(s))


# ============================================================================ probes vs number of samples
def fig_probes_vs_n():
    W, H = 1192, 344
    s = []
    for pi, (n, label) in enumerate([(4, "N = 4 (illustration, not measured)"), (8, "N = 8 (shipped)"), (16, "N = 16 (measured)")]):
        ox = 20 + pi * 395
        s.append(t(ox + 175, 22, label, 15, C["indigo_d"], "middle", "700"))
        # candidates column
        nn = min(n, 16)
        hh = 220 / nn
        for k in range(nn):
            s.append(box(ox, 40 + k * hh, 80, max(4, hh - 2), "#fff", C["line_d"], 1, 3))
        s.append(t(ox + 40, 278, f"{n} samples →", 11.5, MUT, "middle"))
        s.append(t(ox + 40, 292, f"n ≤ {n} distinct", 11.5, MUT, "middle"))
        # probes grid (always 24)
        gx = ox + 220
        for r in range(3):
            for k in range(8):
                s.append(box(gx + k * 17, 90 + r * 40, 13, 30, C["probe_bg"], C["probe"], 1, 3))
        s.append(t(gx + 66, 228, "the same 24 probes", 12, C["teal_d"], "middle", "600"))
        # fan lines: every candidate to the grid
        for k in range(nn):
            s.append(f'<line x1="{ox+80}" y1="{40+k*hh+hh/2:.1f}" x2="{gx-4}" y2="150" stroke="{C["teal"]}" stroke-opacity="0.35" stroke-width="1"/>')
        s.append(t(gx + 66, 248, "each probe scores every", 11.5, MUT, "middle"))
        s.append(t(gx + 66, 262, "distinct answer separately", 11.5, MUT, "middle"))
    s.append(t(20, 320, "A probe is pointwise: its score for an answer depends only on that answer's own hidden state (question and image in context); identical samples are merged first.", 13, INK))
    s.append(t(20, 338, "Changing N changes how many answers are scored, never which probes are used.", 13, INK, weight="600"))
    return svg(W, H, "".join(s))


# ============================================================================ cross-fitting
def fig_crossfit():
    W, H = 1192, 330
    s = []
    s.append(t(20, 26, "every image is hashed into half A (= 0) or half B (= 1):  half = md5(\"nd\" + pixel_md5) mod 2   — all questions about one image stay together", 13, MUT))
    for f, (train, score, col) in enumerate([("B", "A", C["indigo"]), ("A", "B", C["teal"])]):
        y = 52 + f * 118
        s.append(t(20, y + 18, f"fold {f}", 15, col, weight="700"))
        s.append(box(90, y, 300, 70, "#fff", col, 1.4, 8))
        s.append(t(240, y + 24, f"train 24 probes on", 13, INK, "middle"))
        s.append(t(240, y + 44, f"half {train} of all 8 benchmarks", 13.5, INK, "middle", "600"))
        s.append(t(240, y + 61, "+ comparison sets + 4 extra domains", 12, MUT, "middle"))
        s.append(arrow(392, y + 35, 450, y + 35, col, 1.6))
        s.append(box(452, y, 300, 70, C["frozen"] if f == 0 else C["teal_bg"], col, 1.4, 8))
        s.append(t(602, y + 30, f"score every question in half {score}", 13.5, INK, "middle", "600"))
        s.append(t(602, y + 50, "(rows whose image is in this half are", 12, MUT, "middle"))
        s.append(t(602, y + 64, "removed from training everywhere)", 12, MUT, "middle"))
        s.append(arrow(754, y + 35, 826, 150, col, 1.6))
    s.append(box(830, 112, 330, 76, C["good_bg"], C["good"], 1.5, 10))
    s.append(t(995, 140, "all 36,869 questions scored", 14, INK, "middle", "700"))
    s.append(t(995, 160, "each by probes that never saw its image", 12.5, INK, "middle"))
    s.append(t(995, 177, "(exact pixel match; near-duplicates not checked)", 12, MUT, "middle"))
    s.append(t(20, 300, "Intervals: paired bootstrap, 10,000 resamples of images within each benchmark (questions sharing an image move together); "
                        "macro = equal weight per benchmark.", 12.5, INK))
    s.append(t(20, 318, "Image half B (= 1) alone is out of sample for the September design choices, the tie-break and the temperature map (made on half A); the cap320 check used both.", 12.5, MUT))
    return svg(W, H, "".join(s))


# ============================================================================ best-of-N concept with real macro numbers
def fig_bestofn(greedy, pick, oracle):
    W, H = 640, 330
    s = []
    x0, y0, w, hh = 70, 30, 420, 230
    def Y(v): return y0 + hh - v * hh
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+hh}" stroke="{MUT}"/>')
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0+w}" y2="{Y(v):.1f}" stroke="{C["line"]}"/>' + t(x0 - 6, Y(v) + 4, f"{v:g}", 11, MUT, "end"))
    bars = [("greedy", greedy, C["grey"]), ("probe pick", pick, C["teal"]), ("oracle@8", oracle, C["indigo_l"])]
    for k, (lab, v, col) in enumerate(bars):
        bx = x0 + 30 + k * 130
        s.append(box(bx, Y(v), 80, Y(0) - Y(v), col, "none", 0, 3))
        s.append(t(bx + 40, Y(v) - 6, f"{v:.3f}", 13, INK, "middle", "700", cls="mono"))
        s.append(t(bx + 40, Y(0) + 18, lab, 13, INK, "middle", "600"))
    # brackets
    bx = x0 + 30 + 2 * 130 + 92
    s.append(f'<path d="M{bx},{Y(1):.1f} h8 V{Y(oracle):.1f} h-8" fill="none" stroke="{C["bad"]}" stroke-width="1.4"/>')
    s.append(t(bx + 12, (Y(1) + Y(oracle)) / 2, "coverage", 11.5, C["bad"]) + t(bx + 12, (Y(1) + Y(oracle)) / 2 + 14, f"wall {1-oracle:.3f}", 11.5, C["bad"]))
    s.append(f'<path d="M{bx},{Y(oracle):.1f} h8 V{Y(pick):.1f} h-8" fill="none" stroke="{C["warn"]}" stroke-width="1.4"/>')
    s.append(t(bx + 12, (Y(oracle) + Y(pick)) / 2 + 4, f"selection gap {oracle-pick:.3f}", 11.5, C["warn"]))
    s.append(t(x0, H - 22, "Macro over the 8 open benchmarks, judge accuracy, shipped method (single T = 0.7).", 11.5, MUT))
    s.append(t(x0, H - 8, "final_eval_FINAL_cap320_T07.json  cells.*.all.{greedy,pick,oracle}_judge", 10.5, C["faint"]))
    return svg(W, H, "".join(s))


# ============================================================================ VLM anatomy (Qwen2.5-VL as used by Lingshu-7B)
def fig_vlm(spec):
    """spec: dict with verified architecture facts (filled from PAPER_NOTES_GH)."""
    W, H = 1192, 300
    s = []
    s.append(f'<image href="data:image/jpeg;base64,{IMG64}" x="10" y="60" width="120" height="120"/>')
    s.append(t(70, 200, "image", 12, MUT, "middle"))
    s.append(t(70, 217, spec.get("img_note", ""), 11.5, C["faint"], "middle"))
    # patches
    for r in range(4):
        for k in range(4):
            s.append(box(150 + k * 13, 82 + r * 13, 11, 11, C["img"], "none", 0, 2))
    _pl = spec.get("patch", "14×14 patches").rsplit(" ", 1)
    s.append(t(176, 152, _pl[0], 12, MUT, "middle") + t(176, 167, _pl[1] if len(_pl) > 1 else "", 12, MUT, "middle"))
    s.append(arrow(208, 108, 236, 108))
    vit = spec.get("vit", "")
    v1, v2 = (vit.rsplit(" · ", 1) if vit.count(" · ") >= 2 else (vit, ""))
    s.append(frozen_bar(240, 62, 180, 94, "Vision encoder (ViT)", v1))
    if v2:
        s.append(t(330, 140, v2, 12, MUT, "middle"))
    s.append(arrow(422, 108, 436, 108))
    s.append(box(438, 72, 150, 74, "#f3ece0", C["img"], 1.2, 6))
    s.append(t(513, 96, "merger / projector", 13, INK, "middle", "600"))
    s.append(t(513, 115, spec.get("merger", "2×2 patches → 1 token"), 12, MUT, "middle"))
    s.append(t(513, 132, spec.get("merger2", "MLP → 3584-d"), 12, MUT, "middle"))
    s.append(arrow(590, 108, 606, 108))
    # token sequence
    s.append(tokens(610, 100, 5, C["sys"], 11, 2) + tokens(677, 100, 10, C["img"], 11, 2) + tokens(810, 100, 5, C["q"], 11, 2)
             + tokens(877, 100, 3, C["ans"], 11, 2, C["teal_d"]))
    s.append(t(636, 131, "system", 12, MUT, "middle") + t(740, 131, "image tokens", 12, MUT, "middle")
             + t(842, 131, "question", 12, MUT, "middle") + t(902, 131, "answer", 12, C["teal_d"], "middle"))
    s.append(t(780, 158, spec.get("tokens_note", ""), 12, C["muted"], "middle"))
    s.append(arrow(780, 170, 780, 200))
    s.append(frozen_bar(560, 202, 440, 70, "Language model decoder (Qwen2.5-7B)", spec.get("llm", "")))
    s.append(arrow(1002, 237, 1040, 237))
    s.append(t(1046, 232, "next-token", 12.5, INK) + t(1046, 248, "distribution", 12.5, INK))
    s.append(t(1046, 266, "(sampled at T)", 11, MUT))
    return svg(W, H, "".join(s))
