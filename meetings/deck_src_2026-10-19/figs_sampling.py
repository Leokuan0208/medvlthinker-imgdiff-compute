"""figs_sampling.py — the temperature-sampling figure: one input, one frozen model, different answers on every run.
Real samples for SLAKE question 12231 at four temperatures (sampling_example_12231.json); the token-distribution bars in
the middle are a schematic (shape only), because per-token probabilities were not stored for these pools."""
import json, os
from deck_lib import C, esc
from deck_figs import svg, t, m, box, arrow, IMG64

INK, MUT = C["ink"], C["muted"]
HERE = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(f"{HERE}/sampling_example_12231.json"))
GOLD = "large bowel"


def fig_sampling():
    W, H = 1192, 410
    s = []
    # ---- input
    s.append(f'<image href="data:image/jpeg;base64,{IMG64}" x="0" y="34" width="118" height="118"/>')
    s.append(t(0, 22, "same input", 13, C["indigo_d"], weight="700"))
    s.append(t(0, 172, "“What organ is the gray", 12, INK) + t(0, 188, "part on the left of", 12, INK) + t(0, 204, "the image?”", 12, INK))
    s.append(t(0, 224, "gold: Large Bowel", 12, C["good"], weight="700"))
    s.append(arrow(124, 93, 150, 93))
    # ---- model
    s.append(box(152, 50, 128, 86, C["frozen"], C["frozen_o"], 1.25, 8))
    s.append(t(216, 84, "Lingshu-7B", 14, INK, "middle", "700") + t(216, 102, "frozen weights", 12, MUT, "middle")
             + t(216, 118, "run 8 times", 12, MUT, "middle"))
    s.append(arrow(282, 93, 308, 93))
    # ---- schematic next-token distribution, low vs high T
    s.append(t(310, 22, "each next token is drawn at random", 13, C["indigo_d"], weight="700"))
    s.append(m(310, 46, "p<tspan font-size='10' dy='4'>i</tspan><tspan dy='-4'> ∝ exp(u</tspan><tspan font-size='10' dy='4'>i</tspan><tspan dy='-4'> / T)</tspan>", 14, INK))
    labels = ["Small", "Large", "Spleen", "Right"]
    for row, (name, hs) in enumerate((("low T: sharp", [0.92, 0.18, 0.12, 0.08]), ("high T: flat", [0.62, 0.42, 0.34, 0.30]))):
        y0 = 70 + row * 92
        s.append(t(310, y0 + 10, name, 12, MUT, weight="600"))
        for k, (lab, h) in enumerate(zip(labels, hs)):
            x = 310 + k * 50
            s.append(box(x, y0 + 70 - 54 * h, 34, 54 * h, C["teal"] if k == 0 else C["teal_bg"], C["teal"], 1, 2))
            s.append(t(x + 17, y0 + 84, lab, 11, INK, "middle"))
    s.append(t(310, 268, "schematic: bar heights show the shape only", 11, C["faint"], italic=True))
    s.append(t(310, 282, "(token probabilities were not stored)", 11, C["faint"], italic=True))
    s.append(arrow(512, 140, 540, 140))
    # ---- real samples at four temperatures
    s.append(t(546, 22, "real samples for this question (8 per temperature)", 13, C["indigo_d"], weight="700"))
    cols = [("0.2", "T = 0.2"), ("0.4", "T = 0.4"), ("0.7", "T = 0.7 · shipped"), ("1.0", "T = 1.0")]
    cw = 158
    for ci, (key, head) in enumerate(cols):
        x = 546 + ci * (cw + 4)
        pool = EX["pools"][key]
        samples = [str(a).rstrip(".") for a in pool["samples"]]
        shipped = key == "0.7"
        if shipped:
            s.append(box(x - 4, 30, cw + 2, 300, C["teal_bg"], C["teal"], 1.2, 8))
        s.append(t(x + cw / 2 - 3, 48, head, 12.5, C["teal_d"] if shipped else INK, "middle", "700"))
        for k, a in enumerate(samples):
            y = 58 + k * 28
            ok = a.strip().lower() == GOLD
            s.append(box(x + 6, y, cw - 18, 22, "#fff", C["good"] if ok else C["bad"], 1.4, 11, "" if ok else "4 3"))
            s.append(t(x + cw / 2 - 3, y + 15, esc(a), 12, INK, "middle", "600" if ok else "400"))
        nd = len({a.lower() for a in samples})
        nc = sum(a.lower() == GOLD for a in samples)
        s.append(t(x + cw / 2 - 3, 296, f"{nd} distinct", 12, INK, "middle", "600"))
        s.append(t(x + cw / 2 - 3, 312, f"correct: {nc} of 8", 12, C["good"] if nc else C["bad"], "middle", "600"))
    # ---- greedy
    s.append(box(0, 346, W, 50, "#f6f7fb", C["line_d"], 1, 8))
    s.append(t(16, 367, "greedy decoding (always the most likely token): “Small Bowel” — wrong, and identical on every run.", 13, INK, weight="600"))
    s.append(t(16, 386, "The model gives that whole answer probability ≈ 0.76 (e^−0.2718, stored log-probability); "
                        "the right answer lives in the other ≈ 0.24, and only sampling reaches it.", 12.5, MUT))
    return svg(W, H, "".join(s))
