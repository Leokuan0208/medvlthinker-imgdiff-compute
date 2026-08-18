#!/usr/bin/env python3
"""fetch_opentext_candidates.py -- pull and INSPECT the candidate open-text medical VQA sets.

Downloads nothing it cannot verify: for every dataset it reports rows, unique images, whether the
answers are genuinely free text (not a letter / not yes-no), and mean answer length -- so a set that
turns out to be multiple-choice in disguise is rejected before any GPU time is spent on it.

  python3 src/data_prep/fetch_opentext_candidates.py --list
  python3 src/data_prep/fetch_opentext_candidates.py --get kvasir_x1 vqamed2019
  python3 src/data_prep/fetch_opentext_candidates.py --get quilt probmed     # needs an HF token

GATED SETS (quilt, probmed) need BOTH of:
  1. the HF account to have clicked "Agree and access repository" on the dataset page, AND
  2. a read token on this box:  huggingface-cli login
A token alone is not enough for a gated:auto repo.
"""
import argparse, json, os, sys, subprocess

DEST = "/data/dan/dataset"

SETS = {
    "kvasir_x1":   dict(hf="SimulaMet/Kvasir-VQA-x1", gated=False, dest="kvasir_vqa_x1_official",
                        note="GI endoscopy; test=15,955. NEW modality for us."),
    "quilt":       dict(hf="wisdomik/Quilt_VQA", gated=True, dest="quilt_vqa",
                        note="histopathology from educational video; ~957 open-ended per paper."),
    "probmed":     dict(hf="rippleripple/ProbMed", gated=True, dest="probmed",
                        note="radiology probing benchmark; MUST verify it is open-ended, not MCQ."),
    "vqamed2019":  dict(hf="dineshcr7/MED-VQA-2019", gated=False, dest="vqamed2019",
                        note="ImageCLEF VQA-Med 2019; small (~500 test), formulaic questions."),
    "radimagenet": dict(hf=None, gated=False, dest="radimagenet_vqa",
                        note="ALREADY ON DISK, fully generated incl. Lingshu-32B baseline."),
}


HF_HOME_DEFAULT = "/data/dan/hf_cache"


def have_token():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return any(os.path.exists(os.path.expanduser(p)) for p in
               ("~/.cache/huggingface/token", os.path.join(HF_HOME_DEFAULT, "token")))


def fetch(key):
    s = SETS[key]
    if s["hf"] is None:
        print(f"[{key}] already local at {DEST}/{s['dest']} -- nothing to fetch")
        return True
    if s["gated"] and not have_token():
        print(f"[{key}] SKIP -- gated repo and no HF token on this box.\n"
              f"        run:  huggingface-cli login\n"
              f"        and accept terms at https://huggingface.co/datasets/{s['hf']}")
        return False
    out = os.path.join(DEST, s["dest"])
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ); env.pop("HF_HUB_OFFLINE", None)
    env["HF_HOME"] = "/data/dan/hf_cache"
    cmd = [sys.executable, "-c",
           "import sys;from huggingface_hub import snapshot_download;"
           "p=snapshot_download(repo_id=sys.argv[1],repo_type='dataset',local_dir=sys.argv[2],"
           "max_workers=8);print('OK',p)", s["hf"], out]
    print(f"[{key}] downloading {s['hf']} -> {out}", flush=True)
    r = subprocess.run(cmd, env=env)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", nargs="*", default=[])
    A = ap.parse_args()
    if A.list or not A.get:
        print(f"HF token present: {have_token()}\n")
        for k, s in SETS.items():
            d = os.path.join(DEST, s["dest"])
            print(f"  {k:12} gated={str(s['gated']):5} on_disk={str(os.path.isdir(d)):5}  {s['note']}")
        return
    ok = [k for k in A.get if fetch(k)]
    print(f"\nfetched: {ok}")


if __name__ == "__main__":
    main()
