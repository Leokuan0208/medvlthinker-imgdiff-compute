#!/usr/bin/env python3
"""pool_worker.py -- one worker per GPU over a shared job list (2026-10-22).

Replaces the static per-GPU chains: whichever GPU frees up first takes the next runnable job, so one slow job
no longer idles the other card. A job is runnable when every job it depends on has FINISHED -- produced or
failed -- so a failed producer never deadlocks its consumers (CLAUDE.md §0). Each job runs under `timeout`,
holding the GPU's flock (the same logs/.gpuG.lock every other runner uses) and passing that lock fd to the job,
so an orphaned child (e.g. a vLLM engine whose parent died) keeps the card marked busy instead of silently
sharing it. Two-GPU jobs ("gpus": 2) take the other card's lock as well.

  python3 runners/pool_worker.py G /abs/path/jobs.json      (one per GPU; state in <jobs>.state/)
"""
import fcntl, json, os, subprocess, sys, time

M = "/home/jamesyang/medvlthinker-imgdiff-compute"; L = f"{M}/logs"
G, Q = int(sys.argv[1]), sys.argv[2]
JOBS = json.load(open(Q)); SD = Q[:-len(".json")] + ".state"; os.makedirs(SD, exist_ok=True)
TAG = os.path.basename(Q)[:-len(".json")]
names = {j["name"] for j in JOBS}
assert len(names) == len(JOBS) and all(d in names for j in JOBS for d in j.get("deps", [])), "bad job list"
ENV = dict(os.environ, HF_HOME="/data/dan/hf_cache", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_XET="1")


def say(s):
    with open(f"{L}/{TAG}.log", "a") as fh:
        fh.write(f"[{time.strftime('%F %T')}] gpu{G} {s}\n")


def lock(g):
    fd = os.open(f"{L}/.gpu{g}.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


finished = lambda n: os.path.exists(f"{SD}/{n}.rc")
claimed = lambda n: os.path.exists(f"{SD}/{n}.claim")

while True:
    fd = lock(G)                                       # wait for MY card before looking for work
    todo = [j for j in JOBS if not claimed(j["name"])]
    if not todo:
        os.close(fd); say("no jobs left -- exit"); break
    job = None
    for j in todo:
        if all(finished(d) for d in j.get("deps", [])):
            try:
                os.close(os.open(f"{SD}/{j['name']}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                job = j; break
            except FileExistsError:
                continue
    if job is None:
        os.close(fd); time.sleep(60); continue
    fds = [fd]; devs = str(G)
    if job.get("gpus", 1) == 2:
        fds.append(lock(1 - G)); devs = "0,1"
    say(f"START {job['name']} (CUDA_VISIBLE_DEVICES={devs}, timeout {job['timeout']})")
    t0 = time.time()
    with open(f"{L}/{TAG}_{job['name']}.log", "a") as log:
        rc = subprocess.call(["timeout", "-k", "120", job["timeout"], "bash", "-c", job["cmd"]], cwd=M,
                             env=dict(ENV, CUDA_VISIBLE_DEVICES=devs), stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, pass_fds=tuple(fds))
    open(f"{SD}/{job['name']}.rc", "w").write(f"{rc}\n")
    say(f"END {job['name']} rc={rc} ({(time.time() - t0) / 3600:.2f} h)")
    for f in fds:
        os.close(f)
