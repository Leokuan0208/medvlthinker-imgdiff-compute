#!/usr/bin/env python3
"""
Render the ATTACK-2 pre-generation-router COST-vs-ACCURACY FRONTIER as a self-contained HTML page.

Reads   results/cascade_methods/artifacts/pregen_router_2026-08-12.json   (verbatim -- every number on
the page is copied out of that file; nothing is recomputed here except pixel coordinates).
Writes  results/cascade_methods/artifacts/pregen_router_frontier_2026-08-12.html

Launch from the repo root:  python3 src/reporting/pregen_router_frontier_page.py
"""
import json, os, html

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
SRC = os.path.join(ART, "pregen_router_2026-08-12.json")
OUT = os.path.join(ART, "pregen_router_frontier_2026-08-12.html")

R = json.load(open(SRC))
TIE = R["tie_tolerance"]

# ---- plot geometry -------------------------------------------------------------------------------
W, H = 860, 560
ML, MR, MT, MB = 74, 26, 24, 62
PW, PH = W - ML - MR, H - MT - MB
X0, X1 = 0.15, 1.85          # compute ratio, x always-32B-direct
Y0, Y1 = 0.585, 0.675        # macro accuracy


def sx(v):
    return ML + (max(min(v, X1), X0) - X0) / (X1 - X0) * PW


def sy(v):
    return MT + PH - (max(min(v, Y1), Y0) - Y0) / (Y1 - Y0) * PH


def pts(cfg):
    rows = R["frontier"][cfg]["points"]
    rows = [r for r in rows if X0 <= r["ratio_as_charged"] <= X1]
    return sorted(rows, key=lambda r: r["ratio_as_charged"])


# series: (frontier key, label, css var, dashed)
SERIES = [
    ("R-B_pooled_gbm_dense", "R-B pooled router (deployable, prompt-only)", "--series-1", False),
    ("R-A_percell_gbm_dense", "R-A per-cell router (uses dataset identity)", "--series-2", False),
    ("PERMUTATION_NULL_pooled", "Within-cell permutation null", "--series-3", False),
    ("PURE_NOISE_ROUTER", "Pure-noise router = no-skill line (global null)", "--ink-muted", True),
]

# reference points pulled verbatim from the artifact
B = R["baselines"]
REFS = [
    (B["always_32b_direct"]["ratio_as_charged"], B["always_32b_direct"]["macro_acc"],
     "always-32B-direct (THE BAR)", "end", -10, -10),
    (B["always_7b"]["ratio_as_charged"], B["always_7b"]["macro_acc"], "always-7B", "start", 10, 4),
    (B["method_accuracy_max_veto"]["ratio_as_charged"], B["method_accuracy_max_veto"]["macro_acc"],
     "shipped cascade (accuracy-max)", "end", -10, 16),
    (B["method_compute_lean"]["ratio_as_charged"], B["method_compute_lean"]["macro_acc"],
     "shipped cascade (compute-lean)", "end", -10, 14),
    (R["external_comparators"]["cost_floor_crossfit_eps0"]["ratio_as_charged"],
     R["external_comparators"]["cost_floor_crossfit_eps0"]["macro_acc"],
     "cost-floor per-cell arm selection", "start", 10, -8),
]

svg = []
# feasible-cost band: cheaper than the bar
svg.append(f'<rect x="{sx(X0):.1f}" y="{MT}" width="{sx(1.0)-sx(X0):.1f}" height="{PH}" '
           f'fill="var(--band)"/>')
svg.append(f'<text x="{sx(1.0)-8:.1f}" y="{MT+14}" text-anchor="end" class="bandlab">'
           f'cheaper than the bar</text>')

# grid + axes
for i in range(7):
    v = Y0 + (Y1 - Y0) * i / 6
    y = sy(v)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" class="grid"/>')
    svg.append(f'<text x="{ML-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{v:.3f}</text>')
for v in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]:
    x = sx(v)
    svg.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+PH}" class="grid"/>')
    svg.append(f'<text x="{x:.1f}" y="{MT+PH+20}" text-anchor="middle" class="tick">{v:g}x</text>')
# the bar
svg.append(f'<line x1="{sx(1.0):.1f}" y1="{MT}" x2="{sx(1.0):.1f}" y2="{MT+PH}" class="bar"/>')
svg.append(f'<line x1="{ML}" y1="{sy(B["always_32b_direct"]["macro_acc"]):.1f}" x2="{ML+PW}" '
           f'y2="{sy(B["always_32b_direct"]["macro_acc"]):.1f}" class="bar"/>')

# series polylines + markers
for key, lab, var, dash in SERIES:
    if key not in R["frontier"]:
        continue
    P = pts(key)
    if not P:
        continue
    d = " ".join(f'{sx(r["ratio_as_charged"]):.1f},{sy(r["macro_acc"]):.1f}' for r in P)
    svg.append(f'<polyline points="{d}" fill="none" stroke="var({var})" stroke-width="2" '
               f'stroke-linejoin="round" stroke-linecap="round"'
               + (' stroke-dasharray="6 5"' if dash else '') + '/>')
    for r in P:
        cx, cy = sx(r["ratio_as_charged"]), sy(r["macro_acc"])
        feas = r["meets_constraint"]
        svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{4.5 if feas else 3}" '
            f'fill="{f"var({var})" if feas else "var(--surface-1)"}" stroke="var({var})" '
            f'stroke-width="2"><title>{html.escape(lab)}\n'
            f'compute {r["ratio_as_charged"]}x  ·  macro acc {r["macro_acc"]:.4f}\n'
            f'delta vs 32B-direct {r["delta_vs_32b_direct"]:+.4f} '
            f'[{r["ci_lo"]:+.4f}, {r["ci_hi"]:+.4f}]\n'
            f'32B fraction {r["macro_32b_fraction"]}  ·  '
            f'{"MEETS" if feas else "fails"} the non-inferiority constraint\n'
            f'gain over no-skill line {r["gain_over_no_skill_line"]:+.4f}</title></circle>')

# reference points
for x, y, lab, anch, dx, dy in REFS:
    svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="var(--ink)" '
               f'stroke="var(--surface-1)" stroke-width="2"><title>{html.escape(lab)}\n'
               f'{x}x  ·  {y:.4f}</title></circle>')
    svg.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" text-anchor="{anch}" '
               f'class="reflab">{html.escape(lab)}</text>')

svg.append(f'<text x="{ML+PW/2:.0f}" y="{H-14}" text-anchor="middle" class="axlab">'
           f'macro FLOP-eq, x one always-32B-direct pass (as charged, R32 = 4.57)</text>')
svg.append(f'<text transform="translate(18,{MT+PH/2:.0f}) rotate(-90)" text-anchor="middle" '
           f'class="axlab">macro accuracy (8 cells, 1/8 each)</text>')
SVG = "\n".join(svg)

legend = "\n".join(
    f'<span class="lg"><i style="background:var({v});{"opacity:.75" if d else ""}"></i>'
    f'{html.escape(l)}</span>' for k, l, v, d in SERIES if k in R["frontier"])


def row(r, name=None):
    lab = html.escape(name or r.get("label", ""))
    ok = r.get("meets_constraint")
    badge = ('<span class="ok">meets</span>' if ok else '<span class="no">fails</span>')
    ci = (f'{r["delta_vs_32b_direct"]:+.4f} <span class="ci">[{r["ci_lo"]:+.4f}, '
          f'{r["ci_hi"]:+.4f}]</span>') if "ci_lo" in r else "&mdash;"
    return (f'<tr><td>{lab}</td><td class="n">{r["macro_acc"]:.4f}</td><td class="n">{ci}</td>'
            f'<td class="n"><b>{r["ratio_as_charged"]:.3f}x</b></td>'
            f'<td class="n">{r.get("lat_par_ms","&mdash;")}</td>'
            f'<td class="n">{r.get("energy_j","&mdash;")}</td><td>{badge}</td></tr>')


HEAD = "\n".join(row(r) for r in R["HEADLINE_TABLE"]["rows"])

Z = R.get("ZERO_INFORMATION_BAR", {})
NB = Z.get("pure_noise_cheapest_feasible") or {}
RB = Z.get("real_router_cheapest_feasible") or {}
NEST = R["NESTED_HONEST_PRIMARY"]
HYB = R["hybrid_percell_arm_selection"]["rows"]["HYBRID_nested_cv"]
SKILL = R["permutation_null"]["WITHIN_CELL_SKILL_TEST"]

# the decomposition ladder: how cheap each level of information gets you, all at the same constraint
LADDER = [
    ("PURE_NOISE_ROUTER", "No information at all", "traces the no-skill line"),
    ("DIAG_R-B_BETWEEN_cell_only", "Which benchmark it is, only", "routes whole datasets"),
    ("PERMUTATION_NULL_pooled", "Within-cell labels shuffled", "keeps between-cell means"),
    ("DIAG_R-B_WITHIN_cell_only", "Which item it is, only", "between-cell levels removed"),
    ("R-B_pooled_gbm_dense", "The full prompt-only router", "both signals together"),
]
lad = "".join(
    f'<tr><td>{html.escape(lab)}<div class="dim">{html.escape(note)}</div></td>'
    f'<td class="n"><b>{R["frontier"][k]["cheapest_point_meeting_constraint"]["ratio_as_charged"]:.3f}x</b></td>'
    f'<td class="n">{R["frontier"][k]["cheapest_point_meeting_constraint"]["macro_acc"]:.4f}</td>'
    f'<td class="n">{R["frontier"][k]["cheapest_point_meeting_constraint"]["delta_vs_32b_direct"]:+.4f}'
    f' <span class="ci">[{R["frontier"][k]["cheapest_point_meeting_constraint"]["ci_lo"]:+.4f},'
    f' {R["frontier"][k]["cheapest_point_meeting_constraint"]["ci_hi"]:+.4f}]</span></td></tr>'
    for k, lab, note in LADDER if R["frontier"][k]["cheapest_point_meeting_constraint"])

skl = "".join(
    f'<tr><td class="n">{r["target_32b_fraction"]:.2f}</td><td class="n">{r["a_acc"]:.4f}</td>'
    f'<td class="n">{r["b_acc"]:.4f}</td><td class="n">{r["delta"]:+.4f} '
    f'<span class="ci">[{r["ci_lo"]:+.4f}, {r["ci_hi"]:+.4f}]</span></td>'
    f'<td>{"<span class=ok>yes</span>" if r["a_significantly_better"] else "<span class=no>no</span>"}</td></tr>'
    for r in SKILL["rows"])

page = f"""<title>Pre-Generation Router Frontier</title>
<style>
:root {{
  --surface-0:#f6f5f2; --surface-1:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --ink-muted:#84837d;
  --grid:#e6e5e1; --band:#f0efec; --rule:#dedcd7;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --good:#0ca30c; --crit:#d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --surface-0:#111110; --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --ink-muted:#8f8e86;
    --grid:#2e2e2b; --band:#232320; --rule:#33332f;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  }}
}}
:root[data-theme="dark"] {{
  --surface-0:#111110; --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --ink-muted:#8f8e86;
  --grid:#2e2e2b; --band:#232320; --rule:#33332f;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
}}
* {{ box-sizing:border-box; }}
:root {{
  --sans:"Optima","Palatino Sans",Avenir,"Segoe UI",system-ui,ui-sans-serif,sans-serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Roboto Mono",Menlo,Consolas,monospace;
}}
body {{ background:var(--surface-0); color:var(--ink); margin:0;
  font:15.5px/1.65 var(--sans); -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:940px; margin:0 auto; padding:44px 20px 76px; display:flex;
  flex-direction:column; gap:4px; }}
h1 {{ font-size:30px; line-height:1.18; margin:0; letter-spacing:-.018em; text-wrap:balance;
  font-weight:600; }}
.sub {{ color:var(--ink-2); margin:6px 0 26px; max-width:64ch; }}
h2 {{ font-family:var(--mono); font-size:12px; margin:38px 0 10px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-2); font-weight:600; }}
.dim {{ color:var(--ink-muted); font-size:12px; margin-top:2px; }}
.card {{ background:var(--surface-1); border:1px solid var(--rule); border-radius:12px;
  padding:18px; margin:16px 0; }}
svg {{ display:block; width:100%; height:auto; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.bar {{ stroke:var(--ink-muted); stroke-width:1.5; stroke-dasharray:3 4; }}
.tick {{ fill:var(--ink-muted); font-size:11px; }}
.axlab {{ fill:var(--ink-2); font-size:12px; }}
.reflab {{ fill:var(--ink-2); font-size:11px; font-weight:600; }}
.bandlab {{ fill:var(--ink-muted); font-size:11px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:12px; }}
.lg {{ display:flex; align-items:center; gap:7px; font-size:12.5px; color:var(--ink-2); }}
.lg i {{ width:14px; height:3px; border-radius:2px; display:inline-block; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; min-width:660px; }}
th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--rule); }}
th {{ font-family:var(--mono); color:var(--ink-2); font-weight:600; font-size:10.5px;
  text-transform:uppercase; letter-spacing:.07em; }}
td.n {{ text-align:right; font-family:var(--mono); font-size:12px;
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
.ci {{ color:var(--ink-muted); }}
.ok {{ color:var(--good); font-weight:700; font-size:12px; }}
.no {{ color:var(--crit); font-weight:700; font-size:12px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
.kpi {{ background:var(--surface-1); border:1px solid var(--rule); border-radius:10px; padding:15px 16px; }}
.kpi.lead {{ border-color:var(--series-1); box-shadow:inset 3px 0 0 var(--series-1); }}
.kpi .v {{ font-family:var(--mono); font-size:26px; font-weight:600; letter-spacing:-.03em;
  font-variant-numeric:tabular-nums; }}
.kpi .k {{ font-family:var(--mono); font-size:10.5px; color:var(--ink-2); text-transform:uppercase;
  letter-spacing:.07em; }}
.kpi .d {{ font-size:12.5px; color:var(--ink-muted); margin-top:4px; line-height:1.45; }}
.warn {{ border-left:3px solid var(--crit); padding-left:14px; }}
code {{ font-size:12px; background:var(--band); padding:1px 5px; border-radius:4px; }}
p {{ margin:10px 0; }}
</style>

<div class="wrap">
<h1>The pre-generation router: the frontier</h1>
<p class="sub">Decide from the prompt and the image <em>alone</em> which single model answers, and pay
only that one. Minimise macro FLOP-eq subject to macro accuracy staying non-inferior to
always-32B-direct (paired-bootstrap CI lower bound &ge; &minus;{TIE}).</p>

<div class="kpis">
  <div class="kpi lead"><div class="k">Best honest operating point</div>
    <div class="v">{HYB["ratio_as_charged"]:.3f}x</div>
    <div class="d">hybrid, nested CV &middot; {HYB["delta_vs_32b_direct"]:+.4f}
      [{HYB["ci_lo"]:+.4f}, {HYB["ci_hi"]:+.4f}] &mdash; cheaper <b>and</b> significantly
      more accurate</div></div>
  <div class="kpi"><div class="k">Zero-information bar</div>
    <div class="v">{NB.get("ratio_as_charged","&mdash;")}x</div>
    <div class="d">a <b>pure-noise</b> router cannot get below the bar at all, so every
      saving here is earned</div></div>
  <div class="kpi"><div class="k">Within-cell skill</div>
    <div class="v">{SKILL["n_targets_significant"]}</div>
    <div class="d">matched costs where the router beats its own permutation null</div></div>
  <div class="kpi"><div class="k">Prompt-only, honestly selected</div>
    <div class="v">{NEST["seeds_meeting_constraint"]}</div>
    <div class="d">seeds meeting the constraint at
      {NEST["ratio_as_charged_mean"]:.3f}x &mdash; the honest selector overshoots on cost</div></div>
</div>

<h2>Compute against macro accuracy</h2>
<div class="card">
<svg viewBox="0 0 {W} {H}" role="img" aria-label="Compute-accuracy frontier for pre-generation
routers against always-32B-direct">{SVG}</svg>
<div class="legend">{legend}
<span class="lg"><i style="background:var(--ink)"></i>fixed reference systems</span></div>
<p style="font-size:12.5px;color:var(--ink-2);margin-top:12px">Filled markers meet the
non-inferiority constraint; hollow markers fail it. The dashed cross-hairs are always-32B-direct on
both axes. Hover any marker for its accuracy, delta, CI and 32B fraction.</p>
</div>

<h2>Every operating point, side by side</h2>
<div class="card scroll">
<table><thead><tr><th>system</th><th class="n">macro acc</th>
<th class="n">&Delta; vs 32B-direct [95% CI]</th><th class="n">compute</th>
<th class="n">lat par ms</th><th class="n">energy J</th><th>constraint</th></tr></thead>
<tbody>{HEAD}</tbody></table>
</div>

<h2>Where the saving comes from</h2>
<p>Each row is the cheapest policy that still clears the constraint, given only that much
information. The gap between the first and last row is what the router's features actually buy.</p>
<div class="card scroll">
<table><thead><tr><th>what the router is allowed to know</th><th class="n">compute</th>
<th class="n">macro acc</th><th class="n">&Delta; vs 32B-direct [95% CI]</th></tr></thead>
<tbody>{lad}</tbody></table>
</div>
<p>Two things follow. The zero-information router cannot get below <b>1.000x</b> at all &mdash; so
unlike a wide-CI &ldquo;tie&rdquo;, nothing here is handed over by the tolerance. And routing whole
datasets gets only to <b>0.902x</b>, which is exactly the policy &ldquo;send PMC-VQA to the 7B and
everything else to the 32B&rdquo;: the between-cell component is one cell.</p>

<h2>Is there real within-item skill?</h2>
<p>The permutation null shuffles labels <em>inside</em> each cell, so it destroys within-cell signal
but keeps each cell's mean advantage. Comparing the router against it at matched cost isolates
within-item skill from dataset recognition.</p>
<div class="card scroll">
<table><thead><tr><th class="n">32B fraction</th><th class="n">router</th><th class="n">null</th>
<th class="n">difference [95% CI]</th><th>significant</th></tr></thead>
<tbody>{skl}</tbody></table>
</div>

<h2>What this does not show</h2>
<div class="card warn">
<p><b>Eval-selected frontier points are upper bounds.</b> Choosing the cheapest of 649 swept policies
by whether its bootstrap CI clears &minus;{TIE} is a multiple-comparison procedure, and the
feasibility flag is visibly noisy &mdash; 0.844x qualifies, 0.857x and 0.870x do not, 0.883x does
again. Read the frontier's shape, not its cheapest qualifying point.</p>
<p><b>The prompt-only router does not survive honest selection.</b> Choosing configuration and
threshold inside training folds only lands at {NEST["ratio_as_charged_mean"]:.3f}x with
{NEST["seeds_meeting_constraint"]} seeds meeting the constraint: that selector minimises cost against
a point-estimate screen, which does not control the CI at the operating point. The hybrid's nested
selection does hold, but it picks an arm per cell and therefore needs to know which benchmark a
question came from.</p>
<p><b>It saves compute, not memory.</b> A router that can send a question straight to the 32B still
needs the 32B resident, and the 7B as well &mdash; 46.95 GiB against the bar's 31.5 GiB.</p>
</div>

<p style="font-size:12px;color:var(--ink-muted);margin-top:28px">
Every figure copied verbatim from <code>results/cascade_methods/artifacts/pregen_router_2026-08-12.json</code>
&middot; pool: Variant B, 5 benchmarks / 8 cells / n = {R["pool"].split("n=")[1]} &middot;
paired item bootstrap, nboot = {R["n_bootstrap"]}, one shared resample stream &middot;
{R["n_seeds"]} seeds &middot; {R["kfold"]}-fold cross-fit &middot; no GPU.</p>
</div>
"""
open(OUT, "w").write(page)
print("wrote", OUT, f"({len(page)} bytes)")
