import re, sys, html
src = sys.argv[1]
out = sys.argv[2]
lines = open(src, encoding='utf-8').read().split('\n')
in_style = in_script = False
res = []
for i, ln in enumerate(lines, 1):
    s = ln
    if '<style' in s: in_style = True
    if '<script' in s: in_script = True
    skip = in_style or in_script
    if '</style>' in s: in_style = False
    if '</script>' in s: in_script = False
    if skip: continue
    marker = ''
    if re.search(r'<section|<div class="slide|<h1|<h2|<h3', s):
        marker = '## '
    t = re.sub(r'<[^>]+>', ' ', s)
    t = html.unescape(t)
    t = re.sub(r'\s+', ' ', t).strip()
    if t:
        res.append(f'{i:4d}: {marker}{t}')
open(out, 'w', encoding='utf-8').write('\n'.join(res))
print(len(res), 'lines')
