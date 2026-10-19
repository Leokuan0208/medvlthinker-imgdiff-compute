"""deck_lib.py — page shell (CSS/JS), and small SVG chart helpers for the professor deck.
Colour tokens are the indigo/teal scheme of meetings/progress_deck_2026-08-17.html; everything else is new."""
import html

# ----------------------------------------------------------------------------- colour tokens
C = dict(indigo="#3949ab", indigo_d="#283593", indigo_l="#5c6bc0", indigo_bg="#eaecf7",
         teal="#00897b", teal_d="#00695c", teal_l="#26a69a", teal_bg="#e2f1ef",
         ink="#1d2030", muted="#5b6472", faint="#8a93a3", line="#e3e7f0", line_d="#cdd4e3", bg="#eef1f8",
         good="#2e7d32", good_bg="#e7f3e8", bad="#c62828", bad_bg="#fbe9e9", warn="#ef6c00", amber="#f9a825",
         grey="#9aa1ad", grey_d="#6d7480",
         # figure palette (FIGURE_STYLE_NOTES §2.2), harmonised with the tokens above
         frozen="#eceef6", frozen_o="#b8bdd6", sys="#c9cdE0", img="#f2c38b", q="#a8b8e8", ans="#2bb3a0",
         tap="#7a5cc7", probe="#0f9d8a", probe_bg="#d5f2ec")

FONTS = ("https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,500;0,700;1,400"
         "&family=Roboto+Mono:wght@400;500;600&family=Roboto+Slab:wght@400;500;600;700&display=swap")
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"

CSS = """
:root{
  --indigo:#3949ab; --indigo-d:#283593; --indigo-l:#5c6bc0; --indigo-bg:#eaecf7;
  --teal:#00897b; --teal-d:#00695c; --teal-l:#26a69a; --teal-bg:#e2f1ef;
  --ink:#1d2030; --muted:#5b6472; --faint:#8a93a3; --line:#e3e7f0; --line-d:#cdd4e3; --bg:#eef1f8;
  --good:#2e7d32; --good-bg:#e7f3e8; --bad:#c62828; --bad-bg:#fbe9e9; --warn:#ef6c00; --amber:#f9a825;
  --violet:#7a5cc7; --card:#ffffff;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#1a1d2b}
body{font-family:'Roboto',system-ui,sans-serif;color:var(--ink);-webkit-font-smoothing:antialiased;overflow:hidden}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
#deck{width:1280px;height:720px;position:relative;transform-origin:center center}
.slide{position:absolute;inset:0;background:#fff;display:none;overflow:hidden;border-radius:6px}
.slide.on{display:block}
/* frame: a thin section rail on the left, title block, content area, footer */
.rail{position:absolute;left:0;top:0;bottom:0;width:10px;background:var(--indigo)}
.rail.t{background:var(--teal)} .rail.v{background:var(--violet)} .rail.g{background:var(--grey-d,#6d7480)}
.sec{position:absolute;left:44px;top:26px;font-family:'Roboto Mono',monospace;font-size:13px;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;color:var(--indigo)}
.sec.t{color:var(--teal-d)} .sec.v{color:var(--violet)}
h1.t{position:absolute;left:44px;right:44px;top:48px;font-family:'Roboto Slab',serif;font-weight:700;font-size:34px;
  line-height:1.15;letter-spacing:-.01em;color:var(--ink)}
h1.t small{display:block;font-family:'Roboto',sans-serif;font-weight:400;font-size:18px;color:var(--muted);margin-top:6px;letter-spacing:0}
.body{position:absolute;left:44px;right:44px;top:132px;bottom:46px}
.body.tall{top:110px}
.foot{position:absolute;left:44px;right:44px;bottom:14px;display:flex;justify-content:space-between;align-items:flex-end;
  font-size:11.5px;color:var(--faint);font-family:'Roboto Mono',monospace;gap:24px}
.foot .src{max-width:1060px;line-height:1.35}
.foot .n{color:var(--muted);font-weight:600;white-space:nowrap}
/* flex frame: title grows, body takes the rest, footer stays at the bottom */
.frame{position:absolute;left:44px;right:44px;top:24px;bottom:12px;display:flex;flex-direction:column}
.frame .sec{position:static}
.frame h1.t{position:static;margin-top:6px}
.frame .body{position:relative;left:auto;right:auto;top:auto;bottom:auto;flex:1;min-height:0;margin-top:20px}
.frame .body.tall{margin-top:12px}
.frame .foot{position:static;margin-top:8px}
/* building blocks */
.row{display:flex;gap:28px}
.row.full{height:100%}
.col{flex:1;min-width:0;display:flex;flex-direction:column;gap:14px}
.col.w2{flex:2} .col.w3{flex:3}
.center{display:flex;align-items:center;justify-content:center}
.term{border-left:4px solid var(--indigo);background:var(--indigo-bg);padding:10px 14px;border-radius:0 8px 8px 0}
.term.t{border-color:var(--teal);background:var(--teal-bg)} .term.v{border-color:var(--violet);background:#efeafa}
.term.w{border-color:var(--warn);background:#fff3e6} .term.r{border-color:var(--bad);background:var(--bad-bg)}
.term.g{border-color:var(--good);background:var(--good-bg)}
.term b.k{font-family:'Roboto Slab',serif;font-size:19px;display:block;margin-bottom:2px}
.term span{font-size:16.5px;line-height:1.4;color:#2b3045}
.term .alt{font-size:13px;color:var(--muted);font-family:'Roboto Mono',monospace;margin-top:4px;display:block}
.big{font-family:'Roboto Slab',serif;font-weight:700;font-size:44px;line-height:1;color:var(--teal-d)}
.big.i{color:var(--indigo-d)} .big.r{color:var(--bad)} .big.g{color:var(--good)}
.kpi{display:flex;flex-direction:column;gap:4px;padding:14px 16px;border:1px solid var(--line-d);border-radius:10px;background:#fbfcfe}
.kpi .l{font-size:13px;color:var(--muted);font-family:'Roboto Mono',monospace;letter-spacing:.04em;text-transform:uppercase}
.kpi .s{font-size:13.5px;color:var(--muted)}
.lead{font-size:20px;line-height:1.45;color:#2b3045}
.note{font-size:15px;color:var(--muted);line-height:1.45}
.tag{display:inline-block;font-family:'Roboto Mono',monospace;font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:10px;
  background:var(--indigo-bg);color:var(--indigo-d);letter-spacing:.04em;vertical-align:middle}
.tag.t{background:var(--teal-bg);color:var(--teal-d)} .tag.r{background:var(--bad-bg);color:var(--bad)}
.tag.g{background:var(--good-bg);color:var(--good)} .tag.w{background:#fff3e6;color:var(--warn)} .tag.v{background:#efeafa;color:var(--violet)}
ul.b{list-style:none;display:flex;flex-direction:column;gap:9px}
ul.b li{font-size:18px;line-height:1.4;padding-left:20px;position:relative;color:#2b3045}
ul.b li:before{content:"";position:absolute;left:4px;top:.6em;width:7px;height:7px;border-radius:50%;background:var(--indigo)}
ul.b.t li:before{background:var(--teal)} ul.b.sm li{font-size:16px}
table.d{border-collapse:collapse;width:100%;font-size:15.5px}
table.d th{font-family:'Roboto Mono',monospace;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  text-align:left;font-weight:600;border-bottom:2px solid var(--line-d);padding:6px 8px;vertical-align:bottom}
table.d td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;line-height:1.35}
table.d td.n,table.d th.n{text-align:right;font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1,"lnum" 1;white-space:nowrap}
table.d tr.hl td{background:var(--teal-bg)} table.d tr.hi td{background:var(--indigo-bg)}
table.d.sm{font-size:14px} table.d.sm td{padding:4px 7px} table.d.xs{font-size:13px} table.d.xs td{padding:3px 6px}
.win{color:var(--good);font-weight:600} .loss{color:var(--bad);font-weight:600} .tie{color:var(--muted)}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1,"lnum" 1}
.mono{font-family:'Roboto Mono',monospace}
.eq{font-size:19px;padding:6px 0}
.box{border:1px solid var(--line-d);border-radius:10px;padding:14px 16px;background:#fbfcfe}
.box h3{font-family:'Roboto Slab',serif;font-size:17px;margin-bottom:8px;color:var(--indigo-d)}
.box.t h3{color:var(--teal-d)}
.paper{border:1px solid var(--line-d);border-radius:10px;padding:12px 14px;background:#fff}
.paper .cite{font-family:'Roboto Mono',monospace;font-size:11.5px;color:var(--muted)}
.paper h3{font-family:'Roboto Slab',serif;font-size:17px;margin:2px 0 6px}
.divider{background:linear-gradient(110deg,var(--indigo-d),var(--indigo) 55%,var(--teal-d));color:#fff}
.divider .num{position:absolute;left:80px;top:220px;font-family:'Roboto Mono',monospace;font-size:20px;letter-spacing:.2em;opacity:.8}
.divider h1{position:absolute;left:80px;top:255px;right:80px;font-family:'Roboto Slab',serif;font-size:54px;line-height:1.1}
.divider p{position:absolute;left:82px;top:400px;right:120px;font-size:22px;line-height:1.5;opacity:.92}
.ask{position:absolute;right:44px;top:24px;font-family:'Roboto Mono',monospace;font-size:12px;color:var(--warn)}
svg text{font-family:'Roboto',sans-serif}
svg .m{font-family:'KaTeX_Main','Times New Roman',serif;font-style:italic}
svg .mono{font-family:'Roboto Mono',monospace}
#hud{position:fixed;right:14px;bottom:10px;color:#aab;font:12px 'Roboto Mono',monospace;user-select:none}
#hud button{background:none;border:1px solid #556;color:#ccd;border-radius:4px;padding:2px 9px;margin-left:4px;cursor:pointer}
#grid{position:fixed;inset:0;background:#1a1d2bf2;display:none;overflow:auto;padding:30px;z-index:50}
#grid.on{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;align-content:start}
#grid div{background:#fff;border-radius:6px;padding:10px 12px;cursor:pointer;font-size:13px;line-height:1.3}
#grid div b{font-family:'Roboto Mono',monospace;color:#3949ab;margin-right:6px}
@media print{ html,body{background:#fff;overflow:visible} #stage{position:static;display:block} #deck{transform:none!important}
  .slide{display:block!important;position:relative;page-break-after:always;border-radius:0;margin:0 auto} #hud{display:none} }
"""

JS = """
const S=[...document.querySelectorAll('.slide')]; let i=0;
function fit(){const d=document.getElementById('deck');const k=Math.min(innerWidth/1280,innerHeight/720)*0.985;d.style.transform='scale('+k+')';}
function go(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,j)=>s.classList.toggle('on',j===i));
  document.getElementById('pos').textContent=(i+1)+' / '+S.length; try{history.replaceState(null,'','#'+(i+1));}catch(e){} }
addEventListener('keydown',e=>{ if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' ')go(i+1);
  else if(e.key==='ArrowLeft'||e.key==='PageUp')go(i-1); else if(e.key==='Home')go(0); else if(e.key==='End')go(S.length-1);
  else if(e.key==='o'||e.key==='g')document.getElementById('grid').classList.toggle('on');
  else if(e.key==='Escape')document.getElementById('grid').classList.remove('on'); });
addEventListener('resize',fit);
const G=document.getElementById('grid');
S.forEach((s,j)=>{const t=s.getAttribute('data-title')||'';const d=document.createElement('div');d.innerHTML='<b>'+(j+1)+'</b>'+t;
  d.onclick=()=>{go(j);G.classList.remove('on')};G.appendChild(d)});
fit(); let h=parseInt((location.hash||'#1').slice(1)); go(isNaN(h)?0:h-1);
// scale each slide's content (0.85x to 1.25x) so it fills the body without spilling past its edges
function autofit(){ S.forEach(s=>{const b=s.querySelector('.body'),f=s.querySelector('.fit'); if(!b||!f)return;
  const was=s.style.display; s.style.display='block'; f.style.zoom=''; f.style.width='';
  const ok=()=>b.scrollHeight<=b.clientHeight+1 && b.scrollWidth<=b.clientWidth+1;
  const H=b.clientHeight, h=f.scrollHeight; let z=Math.max(0.85,Math.min(1.25,0.95*H/Math.max(h,1)));
  if(z>1.03||!ok()){ while(z>=0.85){ f.style.zoom=z; if(ok())break; z-=0.02; } }
  if(Math.abs(z-1)<=0.03&&ok()){f.style.zoom='';f.style.width='';}
  s.style.display=was; }); document.body.dataset.fitted='1'; }
addEventListener('load',()=>{ (document.fonts?document.fonts.ready:Promise.resolve()).then(()=>setTimeout(autofit,50)); });
document.addEventListener('DOMContentLoaded',()=>{ if(window.renderMathInElement) renderMathInElement(document.body,
  {delimiters:[{left:'$$',right:'$$',display:true},{left:'\\\\(',right:'\\\\)',display:false}],throwOnError:false}); });
"""


def page(title, slides_html):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONTS}" rel="stylesheet">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"></script>
<style>{CSS}</style></head>
<body><div id="stage"><div id="deck">
{slides_html}
</div></div>
<div id="hud"><span id="pos"></span><button onclick="go(i-1)">‹</button><button onclick="go(i+1)">›</button>
<button onclick="document.getElementById('grid').classList.toggle('on')">grid</button></div>
<div id="grid"></div>
<script>{JS}</script></body></html>"""


# ----------------------------------------------------------------------------- slide frame
class Deck:
    def __init__(self):
        self.slides = []          # (id, title, html)
        self.script = []          # per slide: dict(n, sid, title, part, minutes, look, say, qa)
        self.part = ""

    def add(self, sid, title, body, section="", tone="", sub="", src="", tall=False, raw=False, ask="",
            say="", look="", qa=(), minutes=1.0):
        n = len(self.slides) + 1
        import re as _re
        self.script.append(dict(n=n, sid=sid, title=_re.sub(r"<[^>]+>", "", html.unescape(title).replace("<br>", " ")),
                                part=self.part, minutes=minutes, look=look, say=say, qa=list(qa), src=src))
        if raw:
            inner = body + f'<div class="foot"><div class="src">{src}</div><div class="n">{n}</div></div>'
        else:
            inner = (f'<div class="rail {tone}"></div><div class="frame"><div class="sec {tone}">{section}</div>'
                     f'<h1 class="t">{title}{f"<small>{sub}</small>" if sub else ""}</h1>'
                     f'<div class="body{" tall" if tall else ""}"><div class="fit">{body}</div></div>'
                     f'<div class="foot"><div class="src">{src}</div><div class="n">{n}</div></div></div>')
            if ask:
                inner += f'<div class="ask">{ask}</div>'
        plain = html.unescape(title).replace("<br>", " ")
        import re
        plain = re.sub(r"<[^>]+>", "", plain)
        self.slides.append((sid, plain, f'<section class="slide" id="s-{sid}" data-title="{html.escape(plain)}">{inner}</section>'))
        return n

    def number_of(self, sid):
        for k, (s, _, _) in enumerate(self.slides):
            if s == sid:
                return k + 1
        raise KeyError(sid)

    def html(self, title):
        return page(title, "\n".join(h for _, _, h in self.slides))


# ----------------------------------------------------------------------------- SVG chart helpers
def esc(s):
    return html.escape(str(s))


def _ticks(lo, hi, step):
    out, v = [], round(lo / step) * step
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            out.append(round(v, 10))
        v += step
    return out


def fmt(v, nd=4, sign=True):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "−")


def forest(rows, xmin, xmax, step, w=560, rowh=34, label_w=170, series=None, title="", xlabel="", zero=True,
           value_fmt=lambda v: fmt(v, 3), show_vals=True, legend=True):
    """rows: list of (label, [ (est, lo, hi) or None per series ]). series: list of (name, colour)."""
    series = series or [("", C["teal"])]
    ns = len(series)
    has_leg = legend and ns > 1
    top = 52 if (has_leg and title) else 34 if (has_leg or title) else 12
    h = top + rowh * len(rows) + 40
    pw = w - label_w - (70 if show_vals else 16)

    def X(v):
        return label_w + (v - xmin) / (xmax - xmin) * pw
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
    if title:
        s.append(f'<text x="0" y="14" font-size="13" font-weight="600" fill="{C["ink"]}">{esc(title)}</text>')
    if legend and ns > 1:
        lx = label_w
        for name, col in series:
            s.append(f'<circle cx="{lx}" cy="{top-14}" r="5" fill="{col}"/><text x="{lx+9}" y="{top-10}" font-size="12" fill="{C["muted"]}">{esc(name)}</text>')
            lx += 16 + 7 * len(name) + 22
    for t in _ticks(xmin, xmax, step):
        x = X(t)
        s.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+rowh*len(rows)}" stroke="{C["line"]}" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{top+rowh*len(rows)+16}" font-size="11.5" fill="{C["muted"]}" text-anchor="middle">{fmt(t, 2 if step>=0.01 else 3)}</text>')
    if zero and xmin < 0 < xmax:
        s.append(f'<line x1="{X(0):.1f}" y1="{top-4}" x2="{X(0):.1f}" y2="{top+rowh*len(rows)+2}" stroke="{C["ink"]}" stroke-width="1.2"/>')
    if xlabel:
        s.append(f'<text x="{label_w+pw/2:.1f}" y="{h-4}" font-size="12" fill="{C["muted"]}" text-anchor="middle">{esc(xlabel)}</text>')
    for r, (lab, vals) in enumerate(rows):
        y0 = top + r * rowh
        s.append(f'<text x="{label_w-10}" y="{y0+rowh/2+4:.1f}" font-size="13.5" fill="{C["ink"]}" text-anchor="end">{lab}</text>')
        for k, v in enumerate(vals):
            if v is None:
                continue
            est, lo, hi = v
            col = series[k][1]
            yy = y0 + rowh / 2 + (k - (ns - 1) / 2) * min(10, rowh / (ns + 1))
            sig = (lo is not None and hi is not None and (lo > 0 or hi < 0))
            if lo is not None and hi is not None:
                s.append(f'<line x1="{X(max(lo,xmin)):.1f}" y1="{yy:.1f}" x2="{X(min(hi,xmax)):.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="2"/>')
                # an interval that runs past the axis gets an arrowhead at the edge, so clipping is visible
                if lo < xmin:
                    xe = X(xmin)
                    s.append(f'<path d="M{xe:.1f},{yy:.1f} l7,-4.5 v9 z" fill="{col}"/>')
                if hi > xmax:
                    xe = X(xmax)
                    s.append(f'<path d="M{xe:.1f},{yy:.1f} l-7,-4.5 v9 z" fill="{col}"/>')
            fill = col if sig or lo is None else "#fff"
            s.append(f'<circle cx="{X(est):.1f}" cy="{yy:.1f}" r="4.6" fill="{fill}" stroke="{col}" stroke-width="1.8"/>')
            if show_vals and k == 0:
                s.append(f'<text x="{w-2}" y="{y0+rowh/2+4:.1f}" font-size="12.5" fill="{C["ink"]}" text-anchor="end" class="mono">{value_fmt(est)}</text>')
    s.append("</svg>")
    return "".join(s)


def bars(cats, series, ymin, ymax, step, w=560, h=300, ylab="", value_fmt=lambda v: f"{v:.3f}", show_vals=True,
         highlight=None, cat_fs=12.5, legend=True, errs=None):
    """grouped vertical bars. series: list of (name, colour, [values per cat]). errs: same shape of (lo,hi) or None."""
    left, bottom, top = 48, 46, 26 if legend and len(series) > 1 else 10
    pw, ph = w - left - 8, h - bottom - top

    def Y(v):
        return top + ph - (v - ymin) / (ymax - ymin) * ph
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
    for t in _ticks(ymin, ymax, step):
        s.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{w-8}" y2="{Y(t):.1f}" stroke="{C["line"]}"/>'
                 f'<text x="{left-6}" y="{Y(t)+4:.1f}" font-size="11" fill="{C["muted"]}" text-anchor="end">{t:g}</text>')
    if ymin < 0 < ymax:
        s.append(f'<line x1="{left}" y1="{Y(0):.1f}" x2="{w-8}" y2="{Y(0):.1f}" stroke="{C["ink"]}" stroke-width="1.2"/>')
    if ylab:
        s.append(f'<text transform="translate(12,{top+ph/2}) rotate(-90)" font-size="12" fill="{C["muted"]}" text-anchor="middle">{esc(ylab)}</text>')
    if legend and len(series) > 1:
        lx = left
        for name, col, _ in series:
            s.append(f'<rect x="{lx}" y="4" width="11" height="11" rx="2" fill="{col}"/><text x="{lx+15}" y="14" font-size="12" fill="{C["muted"]}">{esc(name)}</text>')
            lx += 30 + 6.6 * len(name)
    gw = pw / len(cats)
    bw = min(34, gw * 0.8 / len(series))
    for i, cat in enumerate(cats):
        cx = left + gw * i + gw / 2
        lines = cat.split("\n")
        for li, ln in enumerate(lines):
            s.append(f'<text x="{cx:.1f}" y="{top+ph+16+li*14}" font-size="{cat_fs}" fill="{C["ink"]}" text-anchor="middle">{esc(ln)}</text>')
        for k, (name, col, vals) in enumerate(series):
            v = vals[i]
            if v is None:
                continue
            x = cx - bw * len(series) / 2 + k * bw
            y0, y1 = Y(max(v, 0) if ymin < 0 else v), Y(min(v, 0) if ymin < 0 else ymin)
            if ymin >= 0:
                y0, y1 = Y(v), Y(ymin)
            c2 = col if not (highlight and (i, k) in highlight) else highlight[(i, k)]
            s.append(f'<rect x="{x+1:.1f}" y="{min(y0,y1):.1f}" width="{bw-2:.1f}" height="{abs(y1-y0):.1f}" fill="{c2}" rx="2"/>')
            if errs and errs[k] and errs[k][i]:
                lo, hi = errs[k][i]
                s.append(f'<line x1="{x+bw/2:.1f}" y1="{Y(lo):.1f}" x2="{x+bw/2:.1f}" y2="{Y(hi):.1f}" stroke="{C["ink"]}" stroke-width="1.2"/>')
            if show_vals:
                ty = (min(y0, y1) - 4) if v >= 0 else (max(y0, y1) + 12)
                s.append(f'<text x="{x+bw/2:.1f}" y="{ty:.1f}" font-size="10.5" fill="{C["ink"]}" text-anchor="middle" class="mono">{value_fmt(v)}</text>')
    s.append("</svg>")
    return "".join(s)


def lines(xs, series, ymin, ymax, step, w=520, h=300, xlab="", ylab="", value_fmt=lambda v: f"{v:.3f}",
          label_last=True, xtick_fmt=lambda x: f"{x:g}", markers=True, extra=""):
    left, bottom, top, right = 52, 44, 14, 120 if label_last else 12
    pw, ph = w - left - right, h - bottom - top
    xmin, xmax = min(xs), max(xs)

    def X(x):
        return left + (x - xmin) / (xmax - xmin) * pw

    def Y(v):
        return top + ph - (v - ymin) / (ymax - ymin) * ph
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
    for t in _ticks(ymin, ymax, step):
        s.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{left+pw}" y2="{Y(t):.1f}" stroke="{C["line"]}"/>'
                 f'<text x="{left-6}" y="{Y(t)+4:.1f}" font-size="11" fill="{C["muted"]}" text-anchor="end">{t:g}</text>')
    if ymin < 0 < ymax:
        s.append(f'<line x1="{left}" y1="{Y(0):.1f}" x2="{left+pw}" y2="{Y(0):.1f}" stroke="{C["ink"]}" stroke-width="1.1"/>')
    for x in xs:
        s.append(f'<text x="{X(x):.1f}" y="{top+ph+17}" font-size="12" fill="{C["ink"]}" text-anchor="middle">{xtick_fmt(x)}</text>')
    if xlab:
        s.append(f'<text x="{left+pw/2:.1f}" y="{h-4}" font-size="12" fill="{C["muted"]}" text-anchor="middle">{esc(xlab)}</text>')
    if ylab:
        s.append(f'<text transform="translate(13,{top+ph/2}) rotate(-90)" font-size="12" fill="{C["muted"]}" text-anchor="middle">{esc(ylab)}</text>')
    for name, col, ys, *opt in series:
        dash = ' stroke-dasharray="5 4"' if opt and opt[0] == "dash" else ""
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys) if y is not None)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"{dash}/>')
        if markers:
            for x, y in zip(xs, ys):
                if y is not None:
                    s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.8" fill="#fff" stroke="{col}" stroke-width="2"/>')
        if label_last:
            yl = [y for y in ys if y is not None][-1]
            s.append(f'<text x="{left+pw+8}" y="{Y(yl)+4:.1f}" font-size="12.5" fill="{col}" font-weight="600">{esc(name)}</text>')
    s.append(extra.replace("__X__", "").replace("__Y__", ""))
    s.append("</svg>")
    return "".join(s)
