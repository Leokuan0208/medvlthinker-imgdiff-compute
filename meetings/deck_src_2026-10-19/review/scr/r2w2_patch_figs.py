"""Round-2 worker 2: figure fixes in deck_figs.py and figs_v2.py (exact-string replacements, all-or-nothing)."""
D = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/"


def patch(fn, R):
    p = D + fn
    s = open(p).read()
    for a, b in R:
        assert s.count(a) == 1, (fn, a[:80])
        s = s.replace(a, b)
    open(p, "w").write(s)


patch("deck_figs.py", [
    # ---- unique marker ids per SVG: url(#ah) otherwise resolves to a marker inside a hidden slide
    ("""def svg(w, h, body, extra_defs=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'""",
     """_SVG_N = [0]


def svg(w, h, body, extra_defs=""):
    _SVG_N[0] += 1
    u = f"f{_SVG_N[0]}"
    for mk in ("ahv", "aht", "ah", "hatch"):          # longest first, so "ah" does not eat "ahv"/"aht"
        body = body.replace(f"url(#{mk})", f"url(#{mk}{u})")
    return _svg_raw(w, h, body, extra_defs).replace('id="ahv"', f'id="ahv{u}"').replace('id="aht"', f'id="aht{u}"') \\
        .replace('id="ah"', f'id="ah{u}"').replace('id="hatch"', f'id="hatch{u}"')


def _svg_raw(w, h, body, extra_defs=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'"""),
    # ---- fig_probe: both Linear layers trained, sigmoid as a dashed training-only side branch, params card in the gap
    ("""    W, H = 560, 470
    s = []
    cx = 150
    # input strip
    s.append(t(cx, 456, "input  h̄ ∈ ℝ³⁵⁸⁴  (one layer, one candidate)", 12, MUT, "middle"))
    for k in range(26):
        s.append(f'<rect x="{cx-130+k*10}" y="420" width="8" height="18" rx="1.5" fill="{C["ans"]}" fill-opacity="{0.35+0.65*((k*37)%11)/10:.2f}"/>')
    blocks = [("standardise", "h̃ = (h̄ − μ) ⊘ (s + ε)   (train-set mean / std, frozen)", 368, "#f6f7fb", C["line_d"]),
              ("Linear 3584 → 256", "W₁ ∈ ℝ²⁵⁶ˣ³⁵⁸⁴,  b₁ ∈ ℝ²⁵⁶   ·  917,760 params", 300, C["probe_bg"], C["probe"]),
              ("GELU", "Gaussian-error linear unit", 232, "#fff", C["probe"]),
              ("Linear 256 → 1", "w₂ ∈ ℝ²⁵⁶,  b₂ ∈ ℝ   ·  257 params", 164, C["probe_bg"], C["probe"]),
              ("σ(·) sigmoid", "training only, inside the class-weighted BCE", 96, "#fff", C["line_d"])]
    for name, sub, y, fill, st in blocks:
        s.append(box(cx - 120, y, 240, 42, fill, st, 1.3, 6))
        s.append(t(cx, y + 18, name, 14, INK, "middle", "600"))
        s.append(t(cx, y + 34, sub, 10.8, MUT, "middle"))
        s.append(arrow(cx, y + 64 if y < 368 else 418, cx, y + 44))
    s.append(arrow(cx, 96, cx, 62))
    s.append(m(cx, 50, "f<tspan font-size='10' dy='4'>ℓ,k</tspan><tspan dy='-4'>(h̃</tspan><tspan font-size='10' dy='4'>i</tspan><tspan dy='-4'>) ∈ ℝ</tspan>", 17, INK, "middle"))
    s.append(t(cx, 30, "score (uncalibrated logit; only its order is used)", 12, MUT, "middle"))
    s.append(t(cx + 132, 330, FIRE, 15, anchor="middle"))
    # total
    s.append(box(300, 380, 250, 60, "#fbfcfe", C["line_d"], 1, 8))
    s.append(t(425, 402, "918,017 parameters per probe", 13, INK, "middle", "600"))
    s.append(t(425, 420, "× 24 probes = 22.03 M", 12, MUT, "middle"))
    s.append(t(425, 434, "(dropout layer = identity, p = 0)", 10.5, C["faint"], "middle"))
    return svg(W, H, "".join(s))""",
     """    W, H = 560, 486
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
    return svg(W, H, "".join(s))"""),
    # ---- fig_crossfit: larger sub-labels (arrowheads come back with the unique marker ids)
    ("""        s.append(t(240, y + 61, "+ comparison sets + 4 extra domains", 11.5, MUT, "middle"))""",
     """        s.append(t(240, y + 61, "+ comparison sets + 4 extra domains", 12, MUT, "middle"))"""),
    ("""        s.append(t(602, y + 50, "(rows whose image is in this half are", 11.5, MUT, "middle"))
        s.append(t(602, y + 63, "removed from training everywhere)", 11.5, MUT, "middle"))""",
     """        s.append(t(602, y + 50, "(rows whose image is in this half are", 12, MUT, "middle"))
        s.append(t(602, y + 64, "removed from training everywhere)", 12, MUT, "middle"))"""),
    ("""    s.append(t(995, 176, "(exact pixel match; near-duplicates not checked)", 11.5, MUT, "middle"))""",
     """    s.append(t(995, 177, "(exact pixel match; near-duplicates not checked)", 12, MUT, "middle"))"""),
    # ---- fig_vlm: ViT subtitle on two lines, minor text >= 12 px, wider merger box
    ("""    s.append(t(70, 216, spec.get("img_note", ""), 10.5, C["faint"], "middle"))""",
     """    s.append(t(70, 217, spec.get("img_note", ""), 11.5, C["faint"], "middle"))"""),
    ("""    s.append(t(176, 150, _pl[0], 11, MUT, "middle") + t(176, 164, _pl[1] if len(_pl) > 1 else "", 11, MUT, "middle"))
    s.append(arrow(208, 108, 236, 108))
    s.append(frozen_bar(240, 66, 170, 86, "Vision encoder (ViT)", spec.get("vit", "")))
    s.append(arrow(412, 108, 440, 108))
    s.append(box(442, 76, 130, 66, "#f3ece0", C["img"], 1.2, 6))
    s.append(t(507, 100, "merger / projector", 12.5, INK, "middle", "600"))
    s.append(t(507, 117, spec.get("merger", "2×2 patches → 1 token"), 11, MUT, "middle"))
    s.append(t(507, 132, spec.get("merger2", "MLP → 3584-d"), 11, MUT, "middle"))
    s.append(arrow(574, 108, 606, 108))""",
     """    s.append(t(176, 152, _pl[0], 12, MUT, "middle") + t(176, 167, _pl[1] if len(_pl) > 1 else "", 12, MUT, "middle"))
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
    s.append(arrow(590, 108, 606, 108))"""),
    ("""    s.append(t(640, 130, "system", 10.5, MUT, "middle") + t(740, 130, "image tokens", 10.5, MUT, "middle")
             + t(842, 130, "question", 10.5, MUT, "middle") + t(895, 130, "answer", 10.5, C["teal_d"], "middle"))
    s.append(t(780, 158, spec.get("tokens_note", ""), 11, C["faint"], "middle"))""",
     """    s.append(t(636, 131, "system", 12, MUT, "middle") + t(740, 131, "image tokens", 12, MUT, "middle")
             + t(842, 131, "question", 12, MUT, "middle") + t(902, 131, "answer", 12, C["teal_d"], "middle"))
    s.append(t(780, 158, spec.get("tokens_note", ""), 12, C["muted"], "middle"))"""),
])

patch("figs_v2.py", [
    # ---- pipeline: drop the as-run annotation (s28/s35 carry it), labels >= 11.5-12 px
    ("""    s.append(t(620, 56, "as run: HF, 1,003,520 px (generation 250,880)", 10.5, C["warn"]))\n""", ""),
    ("""        s.append(t(434, yy + 16, f"y{chr(0x2080 + k + 1)}", 11, MUT))""",
     """        s.append(t(434, yy + 16, f"y{chr(0x2080 + k + 1)}", 12.5, MUT))"""),
    ("""    s.append(t(508, 62, "distinct answers", 11, MUT, "middle") + t(508, 76, "(vote count)", 11, MUT, "middle"))""",
     """    s.append(t(508, 62, "distinct answers", 12, MUT, "middle") + t(508, 77, "(vote count)", 12, MUT, "middle"))"""),
    ("""    s.append(t(228, 212, "vLLM: image encoded", 11, MUT, "middle") + t(228, 226, "once, 8 continuations", 11, MUT, "middle"))""",
     """    s.append(t(228, 213, "vLLM: image encoded", 12, MUT, "middle") + t(228, 228, "once, 8 continuations", 12, MUT, "middle"))"""),
    ("""    s.append(t(641, 291, "blocks 1–28", 10.5, MUT, "middle"))""",
     """    s.append(t(641, 292, "blocks 1–28", 12, MUT, "middle"))"""),
    ("""    s.append(t(676, 236, "mean over the", 11, MUT, "middle") + t(676, 250, "answer's own tokens", 11, MUT, "middle"))""",
     """    s.append(t(676, 236, "mean over the", 12, MUT, "middle") + t(676, 251, "answer's own tokens", 12, MUT, "middle"))"""),
    ("""        s.append(t(gx - 4, gy + r * 36 + 18, f"ℓ{L}", 10.5, C["tap"], "end", "600"))""",
     """        s.append(t(gx - 4, gy + r * 36 + 18, f"ℓ{L}", 12.5, C["tap"], "end", "600"))"""),
    ("""    s.append(t(gx + 64, gy + 144, "class-weighted BCE", 11, MUT, "middle"))""",
     """    s.append(t(gx + 64, gy + 145, "class-weighted BCE", 12, MUT, "middle"))"""),
    ("""    s.append(t(0, 338, "input  x = (image, question)", 11.5, MUT))""",
     """    s.append(t(2, 338, "input  x = (image, question)", 12, MUT))"""),
    ("""    s.append(t(380, 370, "image + question go into the re-read with every answer", 11, MUT, "middle"))""",
     """    s.append(t(380, 371, "image + question go into the re-read with every answer", 12, MUT, "middle"))"""),
    ("""    s.append(t(1054, 255, "returned answer", 11, C["good"], "middle"))""",
     """    s.append(t(1054, 255, "returned answer", 12, C["good"], "middle"))"""),
    # ---- tokens figure: tick and count labels >= 12 px, one footnote line (the rest is in the script)
    ("""def fig_tokens():
    W, H = 1192, 396""", """def fig_tokens():
    W, H = 1192, 384"""),
    ("""        s.append(t(bx - 6, LY(L) + 3.5, f"ℓ={L}", 10, MUT, "end"))""",
     """        s.append(t(bx - 6, LY(L) + 4, f"ℓ={L}", 12, MUT, "end"))"""),
    ("""        s.append(t(bx - 8, ly + 3.5, f"ℓ={L}", 10.5, C["tap"], "end", "600"))""",
     """        s.append(t(bx - 8, ly + 4, f"ℓ={L}", 12, C["tap"], "end", "600"))"""),
    ("""    s.append(t(spans["image tokens"][0], y + 74, groups[1][3], 11, MUT))
    s.append(t(spans["system prompt"][0], y + 58, "30 incl. chat template", 11, MUT))
    s.append(t(spans["question"][0], y + 58, "13 here", 11, MUT))
    s.append(t(spans["template"][0], y + 58, "5", 11, MUT))
    s.append(t(spans["answer  𝒜(y)"][0], y + 58, "4 here", 11, C["teal_d"]))""",
     """    s.append(t(spans["image tokens"][0], y + 76, groups[1][3], 12, MUT))
    s.append(t(spans["system prompt"][0], y + 59, "30 incl. chat template", 12, MUT))
    s.append(t(spans["question"][0], y + 59, "13 here", 12, MUT))
    s.append(t(spans["template"][0], y + 59, "5", 12, MUT))
    s.append(t(spans["answer  𝒜(y)"][0], y + 59, "4 here", 12, C["teal_d"]))"""),
    ("""    s.append(t(bx + 14, H - 22, "Token counts: image-token bound derived from the Qwen2.5-VL 2×2 merge of 14-px patches (one token per 28×28 px).",
               11, C["faint"]))
    s.append(t(bx + 14, H - 7, "𝒜(y) = the answer's own tokens incl. a trailing period, no end-of-turn token. Counts for SLAKE 12231 (Lingshu tokenizer); "
                               "mean image tokens at cap320 ≈ 274 per question (free_head_2026-08-16.json).", 11, C["faint"]))""",
     """    s.append(t(bx + 14, H - 8, "Counts for SLAKE 12231 (Lingshu tokenizer). 𝒜(y) = the answer's own tokens, incl. a trailing period; no end-of-turn token.",
               12, C["muted"]))"""),
    # ---- depth figure: rows 14 px, labels >= 12 px, leader lines from each tapped/tested row to its bar
    ("""    W, H = 600, 486
    s = []
    x0, y0, rh = 120, 26, 13""", """    W, H = 600, 520
    s = []
    x0, y0, rh = 120, 26, 14"""),
    ("""            s.append(t(x0 - 8, yy + 9, f"block {L}" + ("*" if L == 28 else ""), 10.5, C["tap"] if on else MUT, "end", "600" if on else "400"))""",
     """            s.append(t(x0 - 8, yy + 10, f"block {L}" + ("*" if L == 28 else ""), 12, C["tap"] if on else MUT, "end", "600" if on else "400"))"""),
    ("""    s.append(t(x0 + 70, y0 + 28 * rh + 13, "embeddings  (hidden_states[0])", 9.8, MUT, "middle"))
    s.append(box(x0 + 8, y0 + 28 * rh + 24, 124, 40, "#f3ece0", C["img"], 1, 4))
    s.append(t(x0 + 70, y0 + 28 * rh + 40, "vision encoder (ViT, 32 layers)", 9.8, INK, "middle"))
    s.append(t(x0 + 70, y0 + 28 * rh + 55, "+ 2×2 merger → 3584-d tokens", 9.8, INK, "middle"))
    s.append(t(0, H - 6, "* in HF, hidden_states[28] is taken after the final RMSNorm; 18 / 20 / 22 are raw block outputs.", 10, C["faint"]))""",
     """    s.append(t(x0 + 70, y0 + 28 * rh + 13, "embeddings  [0]", 11, MUT, "middle"))
    s.append(box(x0 - 20, y0 + 28 * rh + 24, 180, 42, "#f3ece0", C["img"], 1, 4))
    s.append(t(x0 + 70, y0 + 28 * rh + 41, "vision encoder (ViT, 32 layers)", 11.5, INK, "middle"))
    s.append(t(x0 + 70, y0 + 28 * rh + 58, "+ 2×2 merger → 3584-d tokens", 11.5, INK, "middle"))
    s.append(t(0, H - 6, "* hidden_states[28] is taken after the final RMSNorm; 18 / 20 / 22 are raw block outputs.", 11.5, C["muted"]))"""),
    ("""        s.append(f'<line x1="{ax}" y1="{yy:.1f}" x2="{ax+5}" y2="{yy:.1f}" stroke="{MUT}"/>' + t(ax + 8, yy + 4, f"{v:g}", 10.5, MUT))
    s.append(t(ax + 2, y0 - 8, "depth ℓ / 28", 10.5, MUT))
    for L in (18, 20, 22):
        yy = y0 + (28 - L) * rh + 5
        s.append(t(ax + 38, yy + 4, f"{L}/28 = {L/28:.3f}", 11.5, C["tap"], weight="600"))""",
     """        s.append(f'<line x1="{ax}" y1="{yy:.1f}" x2="{ax+5}" y2="{yy:.1f}" stroke="{MUT}"/>' + t(ax + 8, yy + 4, f"{v:g}", 11.5, MUT))
    s.append(t(ax + 2, y0 - 8, "depth ℓ / 28", 11.5, MUT))
    for L in (18, 20, 22):
        yy = y0 + (28 - L) * rh + 5
        s.append(t(ax + 38, yy + 5, f"{L}/28 = {L/28:.2f}", 12, C["tap"], weight="600"))"""),
    ("""        s.append(t(bx, y0 - 8, "probe − greedy (5 benchmarks)", 10.5, MUT))
        mx = max(v for _, v in layer_gain if isinstance(_, int))
        for L, v in layer_gain:
            if isinstance(L, int):
                yy = y0 + (28 - L) * rh
                s.append(box(bx, yy, 80 * v / mx, rh - 3, C["tap"] if L in (18, 20, 22) else "#cfd2e3", "none", 0, 2))
                s.append(t(bx + 80 * v / mx + 4, yy + 9, f"L{L} +{v:.4f}", 10, INK, cls="mono"))
        ens = [v for L, v in layer_gain if not isinstance(L, int)]
        if ens:
            yy = y0 + (28 - 17) * rh + 6
            s.append(t(bx, yy + 10, f"18+20+22 ensemble: +{ens[0]:.4f}", 11, C["tap"], weight="700"))""",
     """        s.append(t(bx - 40, y0 - 8, "probe − greedy, judge (5 benchmarks)", 11.5, MUT))
        mx = max(v for _, v in layer_gain if isinstance(_, int))
        for L, v in layer_gain:
            if isinstance(L, int):
                yy = y0 + (28 - L) * rh
                # dotted leader from the decoder row to its bar, so every bar sits on its own block
                s.append(f'<line x1="{x0+142}" y1="{yy+5.5:.1f}" x2="{bx-2}" y2="{yy+5.5:.1f}" stroke="{C["line_d"]}" stroke-dasharray="2 3"/>')
                s.append(box(bx, yy, 60 * v / mx, rh - 3, C["tap"] if L in (18, 20, 22) else "#cfd2e3", "none", 0, 2))
                s.append(t(bx + 60 * v / mx + 4, yy + 10, f"L{L} +{v:.3f}", 12, INK, cls="mono"))
        ens = [v for L, v in layer_gain if not isinstance(L, int)]
        if ens:
            yy = y0 + (28 - 16) * rh + 8
            s.append(t(bx - 40, yy + 10, f"18+20+22 ensemble: +{ens[0]:.3f}", 12.5, C["tap"], weight="700"))"""),
])
print("ok")
