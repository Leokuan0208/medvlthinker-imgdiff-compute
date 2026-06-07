import sys, json, random, argparse, os
sys.path.append("/home/jamesyang/medvlthinker-imgdiff-compute/MedRAG/src")
from datasets import load_dataset
from utils import RetrievalSystem

ROOT="/data/dan/dataset/MedVLThinker-Eval"; OUT_DIR="/data/dan/retrieval_kb"
ap=argparse.ArgumentParser()
ap.add_argument("--n",type=int,default=400)
ap.add_argument("--k",type=int,default=5)
ap.add_argument("--corpus",default="Textbooks",
                choices=["Textbooks","StatPearls","PubMed","MedCorp"])
ap.add_argument("--datasets",nargs="+",default=["MedXpert-Reasoning","MedXpert-Understanding"])
A=ap.parse_args()

ds=load_dataset(ROOT); split="test" if "test" in ds else list(ds.keys())[0]; data=ds[split]

# --- MUST match gate_router.py slicing exactly (seed 42) ---
def subset(*keys):
    return [i for i,n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]
def mx_by_type(t):
    out=[]
    for i,n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc=data[i].get("misc")
        try: qt=json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt=""
        if qt.lower()==t: out.append(i)
    return out
DATASET_IDX={
    "MedXpert-Reasoning":     lambda: mx_by_type("reasoning"),
    "MedXpert-Understanding": lambda: mx_by_type("understanding"),
    "MedXpert-MM":            lambda: subset("medxpert"),
    "PMC-VQA":                lambda: subset("pmcvqa","pmc"),
    "SLAKE":                  lambda: subset("slake"),
    "VQA-RAD":                lambda: subset("vqarad","vqa_rad","rad"),
    "PathVQA":                lambda: subset("pathvqa","path"),
    "MMMU":                   lambda: subset("mmmu"),
}
def fixed_slice(idxs):
    rng=random.Random(42); s=idxs[:]; rng.shuffle(s); return s[:A.n]

print(f"init RetrievalSystem corpus={A.corpus} ...",flush=True)
rs=RetrievalSystem(retriever_name="MedCPT",corpus_name=A.corpus,db_dir=OUT_DIR)
for name in A.datasets:
    idxs=DATASET_IDX[name](); sel=fixed_slice(idxs)
    out=os.path.join(OUT_DIR,f"retrieved_{name}_{A.corpus}_n{A.n}.jsonl")
    print(f"--- {name}: pool={len(idxs)} n={len(sel)} k={A.k} corpus={A.corpus} -> {out}",flush=True)
    with open(out,"w") as fh:
        for j,i in enumerate(sel):
            try:
                texts,_=rs.retrieve(data[i]["question"],k=A.k)
                snips=[f'{t.get("title","").strip()}: {t.get("content","").strip()}' for t in texts]
            except Exception as e:
                snips=[]; print(f"   [{j+1}] idx={i} ERR {e!r}",flush=True)
            fh.write(json.dumps({"idx":i,"k":A.k,"snippets":snips})+"\n"); fh.flush()
            if (j+1)%50==0: print(f"   [{j+1}/{len(sel)}] ...",flush=True)
    print(f">> wrote {len(sel)} rows",flush=True)
