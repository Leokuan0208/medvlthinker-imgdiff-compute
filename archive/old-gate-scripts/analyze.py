import numpy as np, pandas as pd
from scipy import stats
DIFF_CSV, COMP_CSV = "difficulty.csv", "complexity.csv"
MIN_STRATUM = 15
def residualize(y, X):
    beta,*_=np.linalg.lstsq(X,y,rcond=None); return y - X@beta
def main():
    df=pd.read_csv(DIFF_CSV).merge(pd.read_csv(COMP_CSV),on="image_id")
    proxies=[c for c in df.columns if c.startswith("comp_")]
    print(f"{len(df)} cases | proxies: {proxies}\n")
    groups=[g["difficulty"].values for _,g in df.groupby("question_type") if len(g)>=5]
    if len(groups)>=2:
        H,p=stats.kruskal(*groups)
        print(f"[control] difficulty differs by question_type: H={H:.2f}, p={p:.4g}", "OK\n" if p<0.05 else " <-- WEAK\n")
    D=pd.get_dummies(df[["question_type","modality"]],drop_first=True).astype(float); D["_int"]=1.0
    X=D.values; dr_res=residualize(stats.rankdata(df["difficulty"].values),X)
    verdict="NO-GO"
    for pxy in proxies:
        rs,ns=[],[]
        for _,g in df.groupby(["question_type","modality"]):
            if len(g)>=MIN_STRATUM and g[pxy].std()>0:
                rho,_=stats.spearmanr(g["difficulty"],g[pxy])
                if not np.isnan(rho): rs.append(rho); ns.append(len(g))
        within=np.average(rs,weights=ns) if rs else float("nan")
        same_sign=(sum(r>0 for r in rs)>=2) or (sum(r<0 for r in rs)>=2)
        partial,pp=stats.pearsonr(dr_res, residualize(stats.rankdata(df[pxy].values),X))
        flag="GO" if (abs(partial)>=0.25 and pp<0.05 and same_sign) else ("REFINE" if abs(partial)>=0.10 else "weak")
        print(f"{pxy:18s} within-strata rho={within:+.3f} (n={len(rs)})  partial rho={partial:+.3f} p={pp:.3g} -> {flag}")
        if flag=="GO": verdict="GO"
        elif flag=="REFINE" and verdict!="GO": verdict="REFINE"
    print(f"\n==> VERDICT: {verdict}")
if __name__=="__main__": main()
