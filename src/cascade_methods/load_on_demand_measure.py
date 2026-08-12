#!/usr/bin/env python3
"""load_on_demand_measure.py -- ATTACK 3 part 4.

Is "swap the 32B in from disk when it is needed" a deployable policy?  Measures the two
components of a cold load of Lingshu-32B from the HF cache:

  (1) DISK -> RAM.  Real sequential read of every safetensors shard, with
      posix_fadvise(POSIX_FADV_DONTNEED) issued on each file first so the page cache is
      dropped and the read is genuinely COLD.  A WARM repeat is measured on one shard so the
      best case is on the record too.
  (2) RAM -> GPU.  Pinned-memory H2D bandwidth, measured with a small buffer on whichever
      card has room.  SKIPPED (recorded as not-measured) if no card has headroom -- this
      round shares both A100s and must never race another agent's job.

The end-to-end swap-in time is then reported as a SUM OF MEASURED COMPONENTS, labelled as
such.  It is not itself an end-to-end measurement: a full from_pretrained().to('cuda') needs
62 GiB of free VRAM, which was not available.

    python3 src/cascade_methods/load_on_demand_measure.py
"""
import json, os, time, glob, sys

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
SNAP = glob.glob("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/*")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_sevenb_frontier_parts/load_on_demand.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CHUNK = 64 << 20


def drop_cache(path):
    """POSIX_FADV_DONTNEED on a clean file evicts its page-cache pages without root."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_file(path):
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            n += len(b)
    return n


def cached_kb():
    for l in open("/proc/meminfo"):
        if l.startswith("Cached:"):
            return int(l.split()[1])
    return -1


def main():
    assert SNAP, "no Lingshu-32B snapshot found"
    snap = SNAP[0]
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    assert shards, "no safetensors shards"
    real = [os.path.realpath(p) for p in shards]
    sizes = [os.path.getsize(p) for p in real]
    total = sum(sizes)

    out = {
        "title": "ATTACK 3 part 4 -- is load-on-demand of Lingshu-32B viable?",
        "date": "2026-08-12",
        "reproduce": "python3 src/cascade_methods/load_on_demand_measure.py",
        "snapshot": snap,
        "n_shards": len(shards),
        "total_bytes": total,
        "total_gib": round(total / 2**30, 4),
        "total_gb_decimal": round(total / 1e9, 4),
        "storage": "/data (see df -h; the HF cache blobs live there, not on the repo disk)",
        "no_fabricated_numbers": True,
    }

    # ---- COLD: drop each shard's page cache, then read the whole model sequentially ----
    for p in real:
        drop_cache(p)
    c0 = cached_kb()
    t0 = time.time()
    got = 0
    per_shard = []
    for p, s in zip(real, sizes):
        ts = time.time()
        got += read_file(p)
        te = time.time()
        per_shard.append({"bytes": s, "s": round(te - ts, 3),
                          "gib_s": round(s / 2**30 / max(te - ts, 1e-9), 3)})
    t1 = time.time()
    assert got == total, (got, total)
    cold_s = t1 - t0
    out["cold_read"] = {
        "what": "sequential read of all shards after posix_fadvise(POSIX_FADV_DONTNEED) on every file",
        "seconds": round(cold_s, 3),
        "gib_per_s": round(total / 2**30 / cold_s, 4),
        "gb_decimal_per_s": round(total / 1e9 / cold_s, 4),
        "meminfo_Cached_kB_before": c0,
        "meminfo_Cached_kB_after": cached_kb(),
        "per_shard": per_shard,
    }

    # ---- WARM: immediately re-read the largest shard (now in page cache) ----
    big = max(zip(real, sizes), key=lambda x: x[1])
    tw0 = time.time(); read_file(big[0]); tw1 = time.time()
    out["warm_read_one_shard"] = {
        "shard_bytes": big[1],
        "seconds": round(tw1 - tw0, 3),
        "gib_per_s": round(big[1] / 2**30 / max(tw1 - tw0, 1e-9), 4),
        "note": "page-cache-warm upper bound on read speed; a real swap-in is cold unless 63 GiB of RAM is dedicated to holding the model resident, which is a different (and larger) deployment cost.",
    }

    # ---- H2D bandwidth, only if a card genuinely has room ----
    h2d = {"measured": False, "reason": None}
    try:
        import torch
        best, bestfree = None, 0
        for i in range(torch.cuda.device_count()):
            free, tot = torch.cuda.mem_get_info(i)
            if free > bestfree:
                best, bestfree = i, free
        need = 4 << 30  # require 4 GiB free to allocate a 1 GiB buffer, with margin
        if best is None or bestfree < need:
            h2d["reason"] = (f"no card with >= {need/2**30:.0f} GiB free (best was cuda:{best} with "
                             f"{bestfree/2**30:.2f} GiB); both A100s are shared this round and a "
                             f"measurement is not worth racing another agent's job.")
        else:
            torch.cuda.set_device(best)
            nbytes = 1 << 30
            host = torch.empty(nbytes // 2, dtype=torch.bfloat16, pin_memory=True)
            dev = torch.empty_like(host, device=f"cuda:{best}")
            for _ in range(2):
                dev.copy_(host, non_blocking=True); torch.cuda.synchronize()
            reps, th0 = 5, time.time()
            for _ in range(reps):
                dev.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            th1 = time.time()
            bw = reps * nbytes / 2**30 / (th1 - th0)
            h2d = {"measured": True, "device": best, "buffer_gib": 1.0, "reps": reps,
                   "gib_per_s": round(bw, 3),
                   "free_gib_on_card_at_measure_time": round(bestfree / 2**30, 2),
                   "note": "pinned-memory H2D, bfloat16 buffer, non_blocking copy."}
            del dev, host
            torch.cuda.empty_cache()
    except Exception as e:  # per-item error guard
        h2d = {"measured": False, "reason": f"{type(e).__name__}: {e}"}
    out["h2d_bandwidth"] = h2d

    # ---- the composed swap-in estimate ----
    cold_gibs = total / 2**30 / cold_s
    comp = {
        "label": "SUM OF MEASURED COMPONENTS -- not an end-to-end measurement",
        "why_not_end_to_end": ("a real from_pretrained(Lingshu-32B).to('cuda') needs 62.31 GiB of free "
                               "VRAM (vram_testtime_2026-08-11.json a_weights_resident_gib) and neither "
                               "shared A100 had it during this round."),
        "disk_to_ram_s": round(total / 2**30 / cold_gibs, 3),
    }
    if h2d.get("measured"):
        comp["ram_to_gpu_s"] = round(total / 2**30 / h2d["gib_per_s"], 3)
        comp["total_cold_swap_in_s"] = round(comp["disk_to_ram_s"] + comp["ram_to_gpu_s"], 3)
        comp["total_warm_swap_in_s_page_cache_hot"] = round(
            total / 2**30 / out["warm_read_one_shard"]["gib_per_s"] + comp["ram_to_gpu_s"], 3)
    else:
        comp["ram_to_gpu_s"] = "not measured"
        comp["total_cold_swap_in_s"] = "lower-bounded by disk_to_ram_s; the H2D term is not measured"
    out["composed_swap_in"] = comp

    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "cold_read"}, indent=2)[:2000])
    print("cold read:", out["cold_read"]["seconds"], "s ->", out["cold_read"]["gib_per_s"], "GiB/s")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
