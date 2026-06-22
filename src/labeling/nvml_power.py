#!/usr/bin/env python3
"""nvml_power.py - lightweight GPU power sampler for batch-1 ENERGY measurement.
Background thread polls nvmlDeviceGetPowerUsage (summed over the GPUs the model uses) and integrates
power over a [t0,t1] window -> Joules. Used by the --batch1 paths of run_vlm_eval / run_peer_eval to
record per-sample energy_j alongside latency_s. NVML sees all physical GPUs regardless of
CUDA_VISIBLE_DEVICES, so pass the physical indices the model occupies."""
import threading, time, os
try:
    import pynvml; pynvml.nvmlInit(); _OK = True
except Exception: _OK = False

def gpu_indices(tp):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip(): return [int(x) for x in vis.split(",") if x.strip() != ""][:tp]
    return list(range(tp))

class PowerMonitor:
    def __init__(self, indices, interval=0.01):
        self.interval = interval; self.samples = []; self.flag = False; self.t = None; self.peak_mem = 0
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices] if _OK else []
        self.ok = _OK and bool(self.handles)
    def _run(self):
        while self.flag:
            try:
                p = sum(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0 for h in self.handles)  # total W
                m = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles)       # total bytes
                self.peak_mem = max(self.peak_mem, m)
            except Exception: p = 0.0
            self.samples.append((time.time(), p)); time.sleep(self.interval)
    def peak_mem_gb(self): return self.peak_mem / 1e9
    def start(self):
        if self.ok: self.flag = True; self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
    def stop(self):
        self.flag = False
        if self.t: self.t.join(timeout=1.0)
    def energy_between(self, t0, t1):  # trapezoidal integral of total power over [t0,t1] -> Joules
        s = [(t, p) for t, p in self.samples if t0 - self.interval <= t <= t1 + self.interval]
        if len(s) < 2: return None
        return sum(0.5 * (s[i][1] + s[i - 1][1]) * (s[i][0] - s[i - 1][0]) for i in range(1, len(s)))
    def trim(self, before):  # bound memory between samples
        self.samples = [(t, p) for t, p in self.samples if t >= before]
