import json, csv, os, glob, random
SLAKE_ROOT = "/data/dan/dataset/slake"
IMG_SUBDIR = "imgs"
OUT_CSV = "subset.csv"
def load_splits(root):
    rows=[]
    for jp in glob.glob(os.path.join(root,"*.json")):
        if "__MACOSX" in jp: continue
        try: d=json.load(open(jp))
        except Exception: continue
        if isinstance(d,list): rows.extend(d)
    return rows
def main():
    rows=load_splits(SLAKE_ROOT); print(f"loaded {len(rows)} raw entries")
    out,kept=[],0
    for r in rows:
        if r.get("q_lang")!="en": continue
        if str(r.get("answer_type","")).upper()!="CLOSED": continue
        ans=str(r.get("answer","")).strip().lower().rstrip(".")
        if ans not in ("yes","no"): continue
        img=r.get("img_name","")
        p=os.path.join(SLAKE_ROOT,IMG_SUBDIR,img)
        if not os.path.exists(p): continue
        out.append({"qid":r.get("qid",kept),"image_id":img.split("/")[0],"image_path":p,
                    "question":r.get("question",""),"question_type":r.get("content_type","Unknown"),
                    "modality":r.get("modality","Unknown"),"options":"yes|no","gold":ans}); kept+=1
    if not out: print(">>> 0 rows — check SLAKE_ROOT"); return
    random.seed(42); random.shuffle(out)        # <-- mix strata so any --limit is representative
    with open(OUT_CSV,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"wrote {OUT_CSV}: {kept} yes/no closed questions (shuffled, seed=42)")
    from collections import Counter
    for k,v in sorted(Counter((r["question_type"],r["modality"]) for r in out).items(),key=lambda x:-x[1]):
        print(f"  {k}: {v}")
if __name__=="__main__": main()
