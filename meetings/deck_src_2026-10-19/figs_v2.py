"""figs_v2.py — re-laid-out versions of the three figures whose first drafts overflowed (pipeline, token
sequence, layer depth). Same visual grammar as deck_figs.py; coordinates planned for a 1192-px content width."""
from deck_lib import C, esc
from deck_figs import svg, t, m, box, arrow, path, tokens, frozen_bar, EX, IMG64, SNOW, FIRE, INK, MUT


def head(x, y, n, label, sub=""):
    s = f'<circle cx="{x+11}" cy="{y}" r="11" fill="{C["indigo_d"]}"/>' + t(x + 11, y + 4.5, n, 12.5, "#fff", "middle", "700")
    s += t(x + 28, y + 5, label, 15, C["indigo_d"], weight="700")
    if sub:
        s += t(x + 28, y + 22, sub, 11.5, MUT)
    return s


def fig_pipeline():
    W, H = 1192, 470
    c = EX["candidates"]
    order = sorted(range(len(c)), key=lambda k: -c[k]["votes"])
    s = []
    # ---------------- input
    s.append(f'<image href="data:image/jpeg;base64,{IMG64}" x="0" y="60" width="140" height="140" preserveAspectRatio="xMidYMid slice"/>')
    s.append(box(0, 60, 140, 140, "none", C["line_d"], 1, 4))
    s.append(t(0, 216, "SLAKE · CT, abdomen", 11, MUT))
    s.append(box(0, 224, 140, 92, "#f6f7fb", C["line_d"], 1, 6))
    for k, ln in enumerate(["“What organ is the", "gray part on the", "left of the image?”"]):
        s.append(t(8, 243 + k * 15, ln, 12.2, INK))
    s.append(t(8, 304, "gold: Large Bowel", 11.5, C["good"], weight="600"))
    s.append(t(2, 338, "input  x = (image, question)", 12, MUT))
    # the re-read needs the image and the question again
    s.append(path("M70,344 V356 H676 V300", C["line_d"], 1.1, "4 3"))
    s.append(t(380, 371, "image + question go into the re-read with every answer", 12, MUT, "middle"))
    # ---------------- stage 1: sample
    s.append(head(166, 18, "1", "Sample", "8 answers · T = 0.7 · cap320"))
    s.append(arrow(142, 140, 164, 140))
    s.append(frozen_bar(166, 86, 124, 108, "Lingshu-7B", "frozen VLM"))
    s.append(t(228, 213, "vLLM: image encoded", 12, MUT, "middle") + t(228, 228, "once, 8 continuations", 12, MUT, "middle"))
    for k, a in enumerate(EX["samples_T07"]):
        yy = 50 + k * 31
        ok = a.rstrip(".") == "Large Bowel"
        col = C["good"] if ok else C["bad"]
        s.append(path(f"M290,140 C 304,140 302,{yy+12} 318,{yy+12}", C["line_d"], 1, mk=""))
        s.append(box(320, yy, 100, 24, "#fff", col, 1.2, 12, "4 3"))
        s.append(t(370, yy + 16, esc(a.rstrip(".")), 11.8, INK, "middle"))
        s.append(t(434, yy + 16, f"y{chr(0x2080 + k + 1)}", 12.5, MUT))
    # dedupe
    s.append(t(508, 62, "distinct answers", 12, MUT, "middle") + t(508, 77, "(vote count)", 12, MUT, "middle"))
    for j, k in enumerate(order):
        yy = 100 + j * 58
        ok = c[k]["judge_label"] == 1
        s.append(box(458, yy, 100, 30, "#fff", C["good"] if ok else C["bad"], 1.5, 15))
        s.append(t(508, yy + 19, esc(c[k]["answer"].rstrip(".")), 12, INK, "middle", "600"))
        s.append(f'<circle cx="558" cy="{yy}" r="11" fill="{C["indigo_d"]}"/>' + t(558, yy + 4, f"×{c[k]['votes']}", 10.5, "#fff", "middle", "700"))
    s.append(arrow(446, 160, 456, 160, C["grey"]))
    # ---------------- stage 2: re-read + taps
    s.append(head(592, 18, "2", "Re-read & tap", "teacher-forced, one pass per answer"))
    s.append(box(592, 86, 168, 208, C["frozen"], C["frozen_o"], 1.25, 8, "5 4"))
    s.append(t(676, 104, "same weights  " + SNOW, 11.5, MUT, "middle"))
    for L in range(1, 29):
        yy = 278 - L * 5.9
        on = L in (18, 20, 22)
        s.append(f'<rect x="604" y="{yy:.1f}" width="74" height="4" rx="1" fill="{C["tap"] if on else "#d6d9e8"}"/>')
    s.append(t(641, 292, "blocks 1–28", 12, MUT, "middle"))
    for j, L in enumerate((22, 20, 18)):
        y0 = 278 - L * 5.9 + 2
        y1 = 126 + j * 32
        s.append(path(f"M680,{y0:.1f} C 694,{y0:.1f} 694,{y1} 708,{y1}", C["tap"], 1.2, mk=""))
        s.append(m(712, y1 + 5, f"h̄<tspan font-size='10' dy='-7'>({L})</tspan>", 15, C["tap"]))
    s.append(t(676, 236, "mean over the", 12, MUT, "middle") + t(676, 251, "answer's own tokens", 12, MUT, "middle"))
    s.append(arrow(570, 160, 590, 160, C["grey"]))
    # ---------------- stage 3: probes
    s.append(head(790, 18, "3", "Score", "24 probes = 3 layers × 8 seeds"))
    gx, gy = 812, 96
    for r, L in enumerate((22, 20, 18)):
        s.append(t(gx - 4, gy + r * 36 + 18, f"ℓ{L}", 12.5, C["tap"], "end", "600"))
        for k in range(8):
            s.append(box(gx + k * 16, gy + r * 36, 13, 26, C["probe_bg"], C["probe"], 1.1, 3))
    s.append(t(gx + 64, gy + 128, FIRE + "  MLP 3584→256→1", 11.5, INK, "middle"))
    s.append(t(gx + 64, gy + 145, "class-weighted BCE", 12, MUT, "middle"))
    s.append(arrow(760, 160, 786, 160, C["tap"], 1.3, mk="ahv"))
    # ---------------- stage 4: select
    s.append(head(966, 18, "4", "Select", "mean rank, ties → more votes"))
    for j, k in enumerate(sorted(range(len(c)), key=lambda k: -c[k]["mean_rank"])):
        yy = 76 + j * 50
        ok = c[k]["judge_label"] == 1
        mr = c[k]["mean_rank"]
        s.append(t(970, yy + 12, esc(c[k]["answer"].rstrip(".")), 12, INK))
        s.append(box(970, yy + 18, 150, 12, "#f0f2f8", "none", 0, 3))
        s.append(box(970, yy + 18, max(2, 150 * mr), 12, C["teal"] if ok else C["grey"], "none", 0, 3))
        s.append(t(1126, yy + 29, f"{mr:.3f}", 11.5, INK, cls="mono"))
    s.append(arrow(946, 160, 962, 160, C["teal"], 1.4, mk="aht"))
    s.append(box(970, 236, 168, 48, C["good_bg"], C["good"], 1.6, 24))
    s.append(t(1054, 255, "returned answer", 12, C["good"], "middle"))
    s.append(t(1054, 275, esc(EX["pick"].rstrip(".")), 15, C["good"], "middle", "700"))
    s.append(t(1054, 304, f"majority vote: “{esc(EX['majority_vote'].rstrip('.'))}” ✗", 11.5, C["bad"], "middle"))
    # ---------------- legend
    ly = 398
    s.append(f'<line x1="0" y1="{ly-22}" x2="{W}" y2="{ly-22}" stroke="{C["line"]}"/>')
    s.append(t(0, ly, SNOW + " frozen (not trained by us)", 12, C["indigo_l"]))
    s.append(t(215, ly, FIRE + " trained by us", 12, INK))
    s.append(t(350, ly, "violet = hidden-state taps (layers 18 / 20 / 22)", 12, C["tap"]))
    s.append(t(680, ly, "pill outline: green = judged correct · red = wrong", 12, MUT))
    s.append(t(0, ly + 20, f"Real example: SLAKE question {EX['idx']}, scored by the probes of the fold that never saw this image "
                           f"(worked_example_12231.json). Generation and re-read are two passes over the same frozen weights.", 11.5, C["faint"]))
    return svg(W, H, "".join(s))


def fig_tokens():
    W, H = 1192, 384
    s = []
    bx, by, bw, bh = 54, 26, 950, 210
    s.append(box(bx, by, bw, bh, C["frozen"], C["frozen_o"], 1.25, 8))
    s.append(t(bx + 14, by + 20, "Lingshu-7B language model (Qwen2.5-7B decoder) · 28 blocks · hidden size 3584 · "
                                 "each line = residual stream after one block", 13, INK, weight="600"))
    s.append(t(bx + bw - 14, by + 20, SNOW, 15, C["indigo_l"], "end"))
    lane0 = by + bh - 14
    sp = 5.9

    def LY(L):
        return lane0 - L * sp
    for L in range(29):
        on = L in (18, 20, 22)
        s.append(f'<line x1="{bx+12}" y1="{LY(L):.1f}" x2="{bx+bw-12}" y2="{LY(L):.1f}" stroke="{C["tap"] if on else "#dcdfeb"}" stroke-width="{1.7 if on else 0.8}"/>')
    for L in (0, 28):
        s.append(t(bx - 6, LY(L) + 4, f"ℓ={L}", 12, MUT, "end"))
    for j, L in enumerate((22, 20, 18)):
        ly = by + 50 + j * 16
        s.append(path(f"M{bx-6},{ly} L{bx+2},{LY(L):.1f}", C["tap"], 0.8, mk=""))
        s.append(t(bx - 8, ly + 4, f"ℓ={L}", 12, C["tap"], "end", "600"))
    # token row
    y = by + bh + 22
    groups = [("system prompt", C["sys"], 8, "≈ 20 tokens"),
              ("image tokens", C["img"], 27, "≤ max_pixels/784: ≤ 320 at cap320 (generation), ≤ 1,280 in the feature pass"),
              ("question", C["q"], 8, "13 here"), ("template", C["sys"], 3, "5"), ("answer  𝒜(y)", C["ans"], 5, "4 here")]
    x = bx + 14
    spans = {}
    for name, col, n, cnt in groups:
        x0 = x
        s.append(tokens(x, y, n, col, 13, 3))
        x += n * 16 + 18
        spans[name] = (x0, x - 21)
        s.append(f'<path d="M{x0},{y+20} v5 h{x-21-x0} v-5" fill="none" stroke="{MUT}" stroke-width="1"/>')
        s.append(t((x0 + x - 21) / 2, y + 40, name, 13, INK, "middle", "600"))
    s.append(t(spans["image tokens"][0], y + 76, groups[1][3], 12, MUT))
    s.append(t(spans["system prompt"][0], y + 59, "30 incl. chat template", 12, MUT))
    s.append(t(spans["question"][0], y + 59, "13 here", 12, MUT))
    s.append(t(spans["template"][0], y + 59, "5", 12, MUT))
    s.append(t(spans["answer  𝒜(y)"][0], y + 59, "4 here", 12, C["teal_d"]))
    a0, a1 = spans["answer  𝒜(y)"]
    # mask over everything but the answer
    s.append(f'<rect x="{bx+12}" y="{by+30}" width="{a0-bx-18}" height="{bh-38}" fill="{INK}" fill-opacity="0.07"/>')
    s.append(t((bx + a0) / 2, by + 118, "states at these positions are computed, but the probe does not read them", 12.5, MUT,
               "middle", italic=True))
    for L in (18, 20, 22):
        s.append(f'<rect x="{a0-3}" y="{LY(L)-3.5:.1f}" width="{a1-a0+6}" height="7" rx="2" fill="{C["tap"]}" fill-opacity="0.45"/>')
    s.append(box(a0 - 7, by + 28, a1 - a0 + 14, bh - 34, "none", C["ans"], 1.6, 4, "4 3"))
    # pooled vectors on the right
    vx = bx + bw + 26
    for j, L in enumerate((22, 20, 18)):
        vy = by + 40 + j * 58
        s.append(path(f"M{a1+8},{LY(L):.1f} C {a1+60},{LY(L):.1f} {vx-30},{vy+8} {vx-6},{vy+8}", C["tap"], 1.2, mk="ahv"))
        for k in range(9):
            s.append(f'<rect x="{vx+k*10}" y="{vy}" width="8" height="16" rx="1.5" fill="{C["ans"]}" fill-opacity="{0.35+0.65*((k*29+L)%9)/8:.2f}"/>')
        s.append(m(vx, vy + 34, f"h̄<tspan font-size='10' dy='-7'>({L})</tspan><tspan dy='7'> ∈ ℝ</tspan><tspan font-size='10' dy='-7'>3584</tspan>", 14, C["tap"]))
    s.append(t(bx + 14, H - 8, "Counts for SLAKE 12231 (Lingshu tokenizer). 𝒜(y) = the answer's own tokens, incl. a trailing period; no end-of-turn token.",
               12, C["muted"]))
    return svg(W, H, "".join(s))


def fig_depth(layer_gain=None):
    W, H = 600, 520
    s = []
    x0, y0, rh = 120, 26, 14
    s.append(t(x0 + 70, y0 - 8, "Lingshu-7B decoder", 12.5, INK, "middle", "600"))
    for L in range(28, 0, -1):
        yy = y0 + (28 - L) * rh
        on = L in (18, 20, 22)
        s.append(box(x0, yy, 140, rh - 3, C["tap"] if on else C["frozen"], C["tap"] if on else C["frozen_o"], 0.8, 2))
        if L in (1, 7, 14, 18, 20, 22, 28):
            s.append(t(x0 - 8, yy + 10, f"block {L}" + ("*" if L == 28 else ""), 12, C["tap"] if on else MUT, "end", "600" if on else "400"))
    s.append(box(x0, y0 + 28 * rh + 2, 140, 15, "#e6e9f3", C["frozen_o"], 0.8, 3))
    s.append(t(x0 + 70, y0 + 28 * rh + 13, "embeddings  [0]", 11, MUT, "middle"))
    s.append(box(x0 - 20, y0 + 28 * rh + 24, 180, 42, "#f3ece0", C["img"], 1, 4))
    s.append(t(x0 + 70, y0 + 28 * rh + 41, "vision encoder (ViT, 32 layers)", 11.5, INK, "middle"))
    s.append(t(x0 + 70, y0 + 28 * rh + 58, "+ 2×2 merger → 3584-d tokens", 11.5, INK, "middle"))
    s.append(t(0, H - 6, "* hidden_states[28] is taken after the final RMSNorm; 18 / 20 / 22 are raw block outputs.", 11.5, C["muted"]))
    ax = x0 + 158
    s.append(f'<line x1="{ax}" y1="{y0}" x2="{ax}" y2="{y0+28*rh}" stroke="{MUT}"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 + 28 * rh - v * 28 * rh
        s.append(f'<line x1="{ax}" y1="{yy:.1f}" x2="{ax+5}" y2="{yy:.1f}" stroke="{MUT}"/>' + t(ax + 8, yy + 4, f"{v:g}", 11.5, MUT))
    s.append(t(ax + 2, y0 - 8, "depth ℓ / 28", 11.5, MUT))
    for L in (18, 20, 22):
        yy = y0 + (28 - L) * rh + 5
        s.append(t(ax + 38, yy + 5, f"{L}/28 = {L/28:.2f}", 12, C["tap"], weight="600"))
    if layer_gain:
        bx = ax + 150
        s.append(t(bx - 40, y0 - 8, "probe − greedy, judge (5 benchmarks)", 11.5, MUT))
        mx = max(v for _, v in layer_gain if isinstance(_, int))
        for L, v in layer_gain:
            if isinstance(L, int):
                yy = y0 + (28 - L) * rh
                # dotted leader from the decoder row to its bar, so every bar sits on its own block
                s.append(f'<line x1="{ax+112}" y1="{yy+5.5:.1f}" x2="{bx-2}" y2="{yy+5.5:.1f}" stroke="{C["line_d"]}" stroke-dasharray="2 3"/>')
                s.append(box(bx, yy, 60 * v / mx, rh - 3, C["tap"] if L in (18, 20, 22) else "#cfd2e3", "none", 0, 2))
                s.append(t(bx + 60 * v / mx + 4, yy + 10, f"L{L} +{v:.3f}", 12, INK, cls="mono"))
        ens = [v for L, v in layer_gain if not isinstance(L, int)]
        if ens:
            yy = y0 + (28 - 16) * rh + 8
            s.append(t(bx - 40, yy + 10, f"18+20+22 ensemble: +{ens[0]:.3f}", 12.5, C["tap"], weight="700"))
    return svg(W, H, "".join(s))
