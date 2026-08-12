#!/usr/bin/env python3
"""
i8b_cheapleg_eval.py -- ATTACK A: evaluate Lingshu-I-8B as the cascade's cheap leg.

WHY THIS FILE EXISTS INSTEAD OF runners/run_cheapleg_mcq.sh
-----------------------------------------------------------
Lingshu-I-8B is the HF-native transformers InternVL port
(architectures=["InternVLForConditionalGeneration"], model_type="internvl").
vLLM 0.9.0.1 -- the version every published cell in this project was produced
with -- registers only "InternVLChatModel", the OpenGVLab remote-code
architecture.  Loading Lingshu-I-8B under it raises

    ValueError: `limit_mm_per_prompt` is only supported for multimodal models

(logs/i8b_vllm_try.log), i.e. vLLM does not see it as multimodal at all.
MedEvalKit's own HF InternVL wrapper uses AutoModel + trust_remote_code +
model.chat(), the OpenGVLab API, which this checkpoint does not implement.

MedEvalKit IS A PROTECTED DEPENDENCY AND IS NOT MODIFIED.  Instead this driver
IMPORTS MedEvalKit's dataset classes, prompt builders and cal_metrics verbatim
and hands them a model object satisfying MedEvalKit's duck-typed interface
(`generate_outputs(messages_list) -> list[str]` plus `last_meta`).  Every prompt
string, every item, every parsing rule and every metric is MedEvalKit's own.

CONSEQUENCE, AND IT IS MANDATORY: the serving stack changed (HF, not vLLM), so
NO STORED NUMBER MAY BE USED AS THE CONTROL.  This driver also runs Lingshu-7B,
in the same process shape, same batch size, same dtype, same decoding.  Every
reported delta is I-8B minus THAT arm.  The stored vLLM Lingshu-7B run is used
only as a NULL TEST of the driver's fidelity.

    python3 src/cascade_methods/i8b_cheapleg_eval.py --arm i8b   --datasets PATH_VQA,SLAKE,VQA_RAD,MedXpertQA-MM,PMC_VQA
    python3 src/cascade_methods/i8b_cheapleg_eval.py --arm base7b --datasets ...
"""
import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEK = os.path.join(REPO, "MedEvalKit")

I8B = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-I-8B/"
       "snapshots/b004bfc0554d90bd44baedf4de08c361e71ef017")
L7B = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
       "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")

ARMS = {
    # DEFAULT tiling: transformers' InternVLProcessor._defaults hard-codes crop_to_patches=True,
    # which OVERRIDES this checkpoint's own preprocessor_config.json ("crop_to_patches": false).
    # Real medical images therefore become 12 crops + 1 thumbnail = 3,340 prompt tokens.
    "i8b":       {"path": I8B, "family": "internvl"},
    # The checkpoint's OWN saved setting, forced back on: 1 crop, 268 prompt tokens.  Which of the
    # two the published model-card numbers correspond to is NOT knowable from the files, so both
    # are measured rather than one being assumed.  ~12x cheaper per pass.
    "i8b_1tile": {"path": I8B, "family": "internvl", "crop_to_patches": False},
    "base7b":    {"path": L7B, "family": "qwen2_5_vl"},
}


# --------------------------------------------------------------------------- model
class HFVLM:
    """MedEvalKit-compatible model backed by transformers (not vLLM).

    Implements exactly the two attributes MedEvalKit's BaseDataset.run touches:
    generate_outputs(messages_list) and last_meta.
    """

    def __init__(self, model_path, family, max_new_tokens=2048, batch_size=32, device="cuda:0",
                 crop_to_patches=None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        self.family = family
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.crop_to_patches = crop_to_patches

        # use_fast=True + device=cuda is a PURE THROUGHPUT fix, not a protocol change: on 32 real
        # PathVQA images the InternVL processor emits byte-identical output (3,340 tokens, 185
        # patches) at 47.4 img/s on the GPU versus 0.8 img/s on the CPU -- 59x.  Measured on CPU the
        # I-8B arm alone would have needed ~17 h of single-threaded preprocessing for the suite.
        # BOTH arms use their own model's default fast processor on the GPU, so the two arms stay
        # symmetric; only the comparison against the STORED vLLM run carries the difference, and
        # that is exactly what the null test measures.
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.last_meta = []

    # -- prompt construction ------------------------------------------------
    def _build(self, messages):
        """MedEvalKit hands {"prompt": str, "image": PIL} (or "images", or text-only)."""
        prompt = messages["prompt"]
        system = messages.get("system")
        imgs = []
        if "image" in messages:
            imgs = [messages["image"]]
        elif "images" in messages:
            imgs = list(messages["images"])

        content = []
        if self.family == "internvl":
            # native chat template emits <IMG_CONTEXT>, expanded to 256 tokens/image
            for _ in imgs:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
        else:  # qwen2_5_vl -- mirrors MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py
            for im in imgs:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": prompt})

        conv = []
        if system:
            conv.append({"role": "system", "content": system})
        conv.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        return text, imgs

    def _encode(self, batch):
        texts, images = [], []
        for m in batch:
            t, ims = self._build(m)
            texts.append(t)
            images.extend(ims)
        kw = {"text": texts, "return_tensors": "pt", "padding": True}
        if images:
            kw["images"] = images
            kw["device"] = self.device      # resize/normalise on the GPU (see __init__ note)
            if self.crop_to_patches is not None:
                kw["crop_to_patches"] = self.crop_to_patches
        try:
            inp = self.processor(**kw)
        except TypeError:                   # processor without a `device` kwarg
            kw.pop("device", None)
            inp = self.processor(**kw)
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        if "pixel_values" in inp and hasattr(inp["pixel_values"], "to"):
            # InternVL's fast image processor returns float32; the tower is bf16.
            inp["pixel_values"] = inp["pixel_values"].to(self.torch.bfloat16)
        return inp

    # -- generation ---------------------------------------------------------
    def generate_outputs(self, messages_list):
        torch = self.torch
        outs, metas = [], []
        bs = self.batch_size
        for s in range(0, len(messages_list), bs):
            batch = messages_list[s:s + bs]
            got = None
            for attempt, cur_bs in enumerate([bs, max(1, bs // 4), 1]):
                try:
                    got = self._gen_chunk(batch, cur_bs)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                except Exception:
                    traceback.print_exc()
                    break
            if got is None:  # per-item error guard: never lose the whole run
                got = ([""] * len(batch),
                       [{"margin": None, "conf": None, "cum_logprob": None,
                         "gen_toks": 0, "latency_s": None, "error": 1} for _ in batch])
            outs.extend(got[0])
            metas.extend(got[1])
        self.last_meta = metas
        return outs

    def _gen_chunk(self, batch, cur_bs):
        torch = self.torch
        texts, metas = [], []
        for s in range(0, len(batch), cur_bs):
            sub = batch[s:s + cur_bs]
            inp = self._encode(sub)
            n_in = inp["input_ids"].shape[1]
            # The cascade's gate signal is the FIRST generated token's top-2 probability gap
            # (MedEvalKit's vLLM wrappers take it from o.logprobs[0]).  return_dict_in_generate +
            # output_scores would retain every step's 151k-wide logits -- tens of GB at
            # max_new_tokens=2048 -- so capture step 0 only, via a pass-through LogitsProcessor.
            grab = _FirstStep()
            t0 = time.time()
            with torch.no_grad():
                seqs_full = self.model.generate(
                    **inp, max_new_tokens=self.max_new_tokens, do_sample=False,
                    logits_processor=[grab],
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            dt = (time.time() - t0) / max(1, len(sub))
            seqs = seqs_full[:, n_in:]
            dec = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
            top2 = torch.topk(torch.softmax(grab.scores.float(), dim=-1), k=2, dim=-1).values.cpu()
            eos = self.processor.tokenizer.eos_token_id
            for i, d in enumerate(dec):
                row = seqs[i]
                ntok = int((row != self.processor.tokenizer.pad_token_id).sum().item())
                if eos is not None:
                    ntok = int(min(ntok, (row != eos).sum().item() + 1))
                # n_prompt_tokens is the honest per-pass cost driver: Lingshu-I-8B spends a FIXED
                # 256 image tokens (crop_to_patches=false) where Qwen2.5-VL's count scales with
                # image resolution.  Parameter count alone (7.94B vs 8.29B) does not decide which
                # cheap leg is cheaper -- prompt length does.
                metas.append({"margin": float(top2[i, 0] - top2[i, 1]),
                              "conf": float(top2[i, 0]), "cum_logprob": None,
                              "gen_toks": ntok, "latency_s": dt, "n_prompt_tokens": int(n_in)})
                texts.append(d.strip())
            del seqs_full, inp, grab
        return texts, metas


class _FirstStep:
    """Pass-through logits processor that keeps a copy of step 0 only."""

    def __init__(self):
        self.scores = None

    def __call__(self, input_ids, scores):
        if self.scores is None:
            self.scores = scores.detach().clone()
        return scores


# --------------------------------------------------------------------------- resumable run
def resumable_run(dataset, model, jsonl_path):
    """MedEvalKit's BaseDataset.run, made resumable with per-item JSONL.

    Same semantics: model.generate_outputs on a batch, then attach response+meta
    to the sample.  Difference: results are flushed per batch and reloaded on
    restart, so a killed run loses at most one batch.
    """
    samples = dataset.samples
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["_i"]] = r
                except Exception:
                    pass  # truncated final line of a killed run
    print(f"[resume] {len(done)}/{len(samples)} already done", flush=True)

    todo = [i for i in range(len(samples)) if i not in done]
    B = 256
    fh = open(jsonl_path, "a")
    for s in range(0, len(todo), B):
        idxs = todo[s:s + B]
        msgs = [samples[i]["messages"] for i in idxs]
        t0 = time.time()
        try:
            resp = model.generate_outputs(msgs)
            metas = model.last_meta
        except Exception:
            traceback.print_exc()
            resp = [""] * len(idxs)
            metas = [{"error": 1}] * len(idxs)
        for k, i in enumerate(idxs):
            rec = {"_i": i, "response": resp[k]}
            if k < len(metas) and isinstance(metas[k], dict):
                rec.update(metas[k])
            done[i] = rec
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        rate = len(idxs) / max(1e-9, time.time() - t0)
        print(f"[gen] {s + len(idxs)}/{len(todo)}  {rate:.1f} it/s", flush=True)
    fh.close()

    out_samples = []
    for i, sm in enumerate(samples):
        sm = dict(sm)
        sm.pop("messages", None)
        r = done.get(i, {"response": ""})
        sm["response"] = r.get("response", "")
        for k in ("margin", "conf", "gen_toks", "latency_s"):
            if k in r:
                sm[k] = r[k]
        out_samples.append(sm)
    return out_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--datasets", default="PATH_VQA,SLAKE,VQA_RAD,MedXpertQA-MM,PMC_VQA")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only: first N items")
    ap.add_argument("--out_root", default="ckpts/i8b_cheapleg")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # The env vars MedEvalKit/eval.py sets from argparse before importing benchmarks.
    # Values copied verbatim from runners/run_cheapleg_mcq.sh, so the harness sees exactly
    # what the stored Lingshu-7B control run saw.  utils.py builds an openai_llm at import
    # time and requires api_key to EXIST (it is never called: use_llm_judge is "False").
    os.environ["REASONING"] = "False"          # the DIRECT prompt, MedEvalKit's own switch
    os.environ["datasets_path"] = "hf"
    os.environ["use_llm_judge"] = "False"
    os.environ["judge_model_type"] = "openai"
    os.environ["judge_model"] = "None"
    os.environ["api_key"] = "None"
    os.environ["base_url"] = "None"
    os.environ["use_vllm"] = "False"
    os.environ["max_image_num"] = "6"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sys.path.insert(0, MEK)
    os.chdir(MEK)                              # MedEvalKit resolves some paths relative to itself
    from benchmarks import prepare_benchmark

    spec = ARMS[args.arm]
    print(f"=== arm={args.arm} path={spec['path']} family={spec['family']}", flush=True)
    model = HFVLM(spec["path"], spec["family"],
                  max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                  crop_to_patches=spec.get("crop_to_patches"))

    out_root = os.path.join(REPO, args.out_root, args.arm)
    os.makedirs(out_root, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        outdir = os.path.join(out_root, ds)
        os.makedirs(outdir, exist_ok=True)
        mpath = os.path.join(outdir, "metrics.json")
        if os.path.exists(mpath) and not args.limit:
            print(f"[skip done] {ds}", flush=True)
            continue
        print(f"\n### {ds}", flush=True)
        t0 = time.time()
        dataset = prepare_benchmark(model, ds, None, outdir)
        dataset.load_data()
        if args.limit:
            dataset.samples = dataset.samples[:args.limit]
        print(f"[load] {len(dataset.samples)} samples in {time.time()-t0:.0f}s", flush=True)

        jsonl = os.path.join(outdir, "gen.jsonl" if not args.limit else "gen_smoke.jsonl")
        out_samples = resumable_run(dataset, model, jsonl)
        metrics, out_samples = dataset.cal_metrics(out_samples)
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(out_samples, f)
        with open(mpath if not args.limit else mpath + ".smoke", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[{ds}] {json.dumps(metrics)[:400]}", flush=True)

    print("I8B_CHEAPLEG_EVAL_DONE", args.arm, flush=True)


if __name__ == "__main__":
    main()
