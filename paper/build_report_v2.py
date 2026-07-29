#!/usr/bin/env python3
"""Build the professor report v2 (research-loop, defined terms, peer baselines, data tables, TV-readable
figures). Reuses the weekly-update template's CSS + nav. Reads result JSONs at build time (fallbacks to
VERIFIED_FACTS numbers). Run AFTER analyze_clean_dump.py + figures exist."""
import os, json, base64, csv
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
def J(p, d=None):
    fp=os.path.join(ROOT,p)
    try: return json.load(open(fp))
    except Exception: return d
def img(path, cap, w="86%"):
    p=os.path.join(ROOT,path)
    if not os.path.exists(p): return f"<p class='caption'>[figure pending: {os.path.basename(path)}]</p>"
    b=base64.b64encode(open(p,"rb").read()).decode()
    return f"<img class='figimg' style='max-width:{w}' src='data:image/png;base64,{b}'><p class='caption' style='text-align:center'>{cap}</p>"

# ---- data (verified facts + computed) ----
peer=J("ckpts/train/lora_verifier_pooled4/peer_comparison.json",{})
ac=J("ckpts/train/lora_verifier_pooled4/accuracy_compute.json",{})
cg=J("ckpts/train/lora_verifier_pooled4/crossgen_result.json",{})
def fmt(x,n=3):
    return f"{x:.{n}f}" if isinstance(x,(int,float)) else "—"

# template head + nav
lines=open(os.path.join(ROOT,"meetings/report_template.html"),encoding="utf-8").read().split("\n")
head="\n".join(lines[:187])
mathjax='''<script>window.MathJax={tex:{inlineMath:[['$','$']]},svg:{fontCache:'global'}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>.eq{font-size:1.15rem;background:#f6f8fd;border-left:4px solid var(--indigo);border-radius:8px;padding:10px 16px;margin:10px 0;overflow-x:auto}
.callout{border-radius:12px;padding:14px 20px;margin:14px 0;font-size:1.18rem;line-height:1.55}
.callout.win{background:var(--good-bg);border:1px solid #bfe0c2}.callout.win b{color:var(--good)}
.callout.note{background:var(--indigo-bg);border:1px solid #cfd6f0}.callout.honest{background:#fff7e8;border:1px solid #f3e0b8}.callout.honest b{color:var(--warn)}
.callout.q{background:#eef1f8;border:1px dashed var(--indigo-l)}
.figimg{width:100%;display:block;margin:12px auto 4px;border:1px solid var(--line);border-radius:10px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}@media(max-width:900px){.two{grid-template-columns:1fr}}
ul.body{font-size:1.2rem;line-height:1.55;margin:8px 0 8px 22px}ul.body li{margin:6px 0}
dl.def{font-size:1.16rem}dl.def dt{font-weight:700;color:var(--indigo-d);margin-top:8px}dl.def dd{margin:2px 0 6px 0;color:var(--ink)}
.src{font-size:.95rem;color:var(--muted)}</style>'''
head=head.replace("</head>", mathjax+"\n</head>")
appbar='''<body>
<div class="appbar"><span class="mark">Med-VLM · CVGIP 2026</span>
<span class="title">Test-Time Compute for Medical VLMs — <b>What Actually Helps</b></span>
<span class="spacer"></span><span class="meta">progress report · continues from the ACC efficiency report</span></div>
<div class="progress"><div class="progress-bar" id="pbar"></div></div>
<div class="deck-viewport"><main class="deck" id="deck">
'''
def slide(inner): return f'<section class="slide"><div class="swrap"><div class="card">{inner}</div></div></section>\n'
S=[]
# slides are appended by build_slides() which is filled below; kept in a separate module section for clarity
exec(open(os.path.join(ROOT,"paper/_report_slides.py")).read())

close='''</main></div>
<button class="edge left" id="prev"><span class="chevbtn">&#8249;</span></button>
<button class="edge right" id="next"><span class="chevbtn">&#8250;</span></button>
<div class="counter" id="counter">1 / N</div>
<script>
const deck=document.getElementById('deck'),slides=[...document.querySelectorAll('.slide')],counter=document.getElementById('counter'),pbar=document.getElementById('pbar'),prev=document.getElementById('prev'),next=document.getElementById('next'),N=slides.length;let idx=0;
function render(){deck.style.transform='translateX(-'+(idx*100)+'vw)';counter.textContent=(idx+1)+' / '+N;pbar.style.width=(N>1?(idx/(N-1)*100):0)+'%';prev.disabled=(idx<=0);next.disabled=(idx>=N-1);}
function go(i){idx=Math.max(0,Math.min(N-1,i));if(slides[idx])slides[idx].scrollTop=0;render();}
prev.onclick=()=>go(idx-1);next.onclick=()=>go(idx+1);
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(idx+1);}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(idx-1);}else if(e.key==='Home')go(0);else if(e.key==='End')go(N-1);});
window.addEventListener('resize',render);render();
</script></body></html>'''
out=os.path.join(ROOT,"progress_report_professor.html")
open(out,"w",encoding="utf-8").write(head+"\n"+appbar+"".join(S)+close)
print(f"wrote {out} ({(len(''.join(S))+len(head))//1024} KiB, {len(S)} slides)")
