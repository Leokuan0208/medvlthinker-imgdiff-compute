import csv, io
import numpy as np
from PIL import Image
DIFF_CSV, OUT_CSV = "difficulty.csv", "complexity.csv"
def gray(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)
def c_entropy(g):
    h,_=np.histogram(g,bins=256,range=(0,255),density=True); h=h[h>0]
    return float(-(h*np.log(h)).sum())
def c_jpeg(p):
    im=Image.open(p).convert("RGB"); buf=io.BytesIO(); im.save(buf,format="JPEG",quality=90)
    return float(buf.tell()/(im.size[0]*im.size[1]*3))
def c_grad(g):
    gx,gy=np.gradient(g); return float(np.sqrt(gx**2+gy**2).mean())
def c_lap(g):
    lap=(-4*g+np.roll(g,1,0)+np.roll(g,-1,0)+np.roll(g,1,1)+np.roll(g,-1,1)); return float(lap.var())
def main():
    seen={}
    for r in csv.DictReader(open(DIFF_CSV)): seen.setdefault(r["image_id"], r["image_path"])
    out=[]
    for k,(iid,p) in enumerate(seen.items()):
        try:
            g=gray(p); out.append({"image_id":iid,"comp_entropy":round(c_entropy(g),5),
                "comp_jpeg":round(c_jpeg(p),5),"comp_grad":round(c_grad(g),5),"comp_lap":round(c_lap(g),5)})
        except Exception as e: print(f"  skip {iid}: {e}")
        if (k+1)%50==0: print(f"  {k+1}/{len(seen)} images")
    with open(OUT_CSV,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"wrote {OUT_CSV} ({len(out)} images)")
if __name__=="__main__": main()
