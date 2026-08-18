#!/usr/bin/env python3
"""supervisor.py -- run a queue of long jobs so that NOTHING silently wastes hours.

WHY THIS EXISTS.  Two failures already cost real time on 2026-08-18: an OmniMedVQA draw spun in an
infinite loop for 4h34m with a log that simply stopped advancing, and a sweep pipeline marched past
three shards that had SIGSEGV'd because a crash is also an "exit".  Both were invisible until
somebody looked.  This supervisor makes both impossible:

  HARD TIMEOUT      every job gets a wall-clock cap; past it the process tree is killed.
  STALL DETECTION   if the job's log has not grown for `stall_s`, it is treated as hung and killed
                    -- an infinite loop produces no output, which is exactly the signature.
  ALWAYS ADVANCES   a failed, hung or timed-out job never blocks the queue; the outcome is recorded
                    and the next job starts.
  RESUMABLE         outcomes are journalled, so re-running the supervisor skips what already
                    succeeded rather than redoing hours of work.
  VERIFIABLE        a job may declare `expect` (a file that must exist and be non-empty) and/or
                    `expect_grep` (a string its log must contain).  Exit code 0 is not trusted on
                    its own, because a crashed shard can still leave a zero-status wrapper.

Usage: write a queue as JSON and run it.  Each entry:
    {"name": "...", "cmd": "...", "timeout_s": 7200, "stall_s": 900,
     "expect": "path/that/must/exist", "expect_grep": "DONE_MARKER"}

  python3 src/reporting/supervisor.py --queue runners/campaign.json
  python3 src/reporting/supervisor.py --queue runners/campaign.json --status
"""
import argparse, json, os, signal, subprocess, sys, time

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
JOURNAL = os.path.join(ROOT, "logs", "supervisor_journal.jsonl")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_journal():
    out = {}
    if os.path.exists(JOURNAL):
        for line in open(JOURNAL):
            try:
                r = json.loads(line)
                out[r["name"]] = r
            except Exception:
                pass
    return out


def record(rec):
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def kill_tree(p):
    """Kill the whole process group -- a bare kill leaves python children orphaned."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except Exception:
            pass
        for _ in range(20):
            if p.poll() is not None:
                return
            time.sleep(0.5)


def verify(job, logpath):
    """Exit status alone is not trusted; check the artefacts the job promised."""
    problems = []
    exp = job.get("expect")
    if exp:
        p = exp if os.path.isabs(exp) else os.path.join(ROOT, exp)
        if not os.path.exists(p):
            problems.append(f"expected file missing: {exp}")
        elif os.path.getsize(p) == 0:
            problems.append(f"expected file empty: {exp}")
    g = job.get("expect_grep")
    if g:
        try:
            if g not in open(logpath, errors="ignore").read():
                problems.append(f"log lacks marker {g!r}")
        except Exception as e:
            problems.append(f"log unreadable: {e}")
    return problems


def run_job(job, retries=1):
    name = job["name"]
    logpath = os.path.join(ROOT, "logs", f"sv_{name}.log")
    timeout_s = int(job.get("timeout_s", 7200))
    stall_s = int(job.get("stall_s", 900))

    for attempt in range(1, retries + 2):
        log(f"START {name} (attempt {attempt}, timeout {timeout_s}s, stall {stall_s}s)")
        t0 = time.time()
        with open(logpath, "a") as lf:
            lf.write(f"\n===== supervisor attempt {attempt} at {time.strftime('%H:%M:%S')} =====\n")
        with open(logpath, "a") as lf:
            p = subprocess.Popen(["bash", "-lc", job["cmd"]], cwd=ROOT, stdout=lf,
                                 stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        outcome, last_size, last_change = None, -1, time.time()
        while True:
            if p.poll() is not None:
                outcome = "exited"
                break
            now = time.time()
            if now - t0 > timeout_s:
                log(f"  TIMEOUT after {timeout_s}s -- killing {name}")
                kill_tree(p); outcome = "timeout"; break
            try:
                sz = os.path.getsize(logpath)
            except Exception:
                sz = last_size
            if sz != last_size:
                last_size, last_change = sz, now
            elif now - last_change > stall_s:
                log(f"  STALLED: log unchanged for {int(now-last_change)}s -- killing {name}")
                kill_tree(p); outcome = "stalled"; break
            time.sleep(10)

        rc = p.returncode
        problems = verify(job, logpath)
        ok = (outcome == "exited" and rc == 0 and not problems)
        rec = {"name": name, "attempt": attempt, "outcome": outcome, "rc": rc,
               "secs": round(time.time() - t0, 1), "problems": problems, "ok": ok,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        record(rec)
        if ok:
            log(f"  OK {name} in {rec['secs']}s")
            return True
        log(f"  FAIL {name}: outcome={outcome} rc={rc} problems={problems}")
        if attempt <= retries:
            log(f"  retrying {name}")
    log(f"  GIVING UP on {name} -- queue continues")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--force", nargs="*", default=[],
                    help="job names to re-run even if the journal says they succeeded")
    A = ap.parse_args()
    jobs = json.load(open(A.queue if os.path.isabs(A.queue) else os.path.join(ROOT, A.queue)))
    done = load_journal()

    if A.status:
        print(f"{'job':38} {'ok':>5} {'outcome':>9} {'secs':>9}  problems")
        for j in jobs:
            r = done.get(j["name"])
            if r:
                print(f"{j['name']:38} {str(r['ok']):>5} {r['outcome']:>9} {r['secs']:>9} "
                      f" {';'.join(r['problems'])}")
            else:
                print(f"{j['name']:38} {'-':>5} {'pending':>9} {'-':>9}")
        return

    log(f"queue: {len(jobs)} jobs")
    for j in jobs:
        prev = done.get(j["name"])
        if prev and prev.get("ok") and j["name"] not in A.force:
            log(f"SKIP {j['name']} (already ok in {prev['secs']}s)")
            continue
        run_job(j, retries=A.retries)
    log("QUEUE COMPLETE")


if __name__ == "__main__":
    main()
