#!/usr/bin/env python3
"""FQC chatbot — master benchmark runner.

Runs every benchmark built for this deployment, in the right order, and
writes one consolidated report at the end. Portable to different
hardware: every machine-specific path (backend URL, venv, CUDA libs,
model file, Mongo URI) is read from environment variables via
bench_config.py, not hardcoded here - see BENCHMARKING.md for exactly
which ones to set on a new box, and what each one defaults to.

Prerequisites (checked automatically, see --check-only):
  - MongoDB running and reachable at FQC_MONGO_URI.
  - The real production alert data seeded (chat/backend/seed_data.py).
  - Python deps installed in the venv at FQC_VENV_DIR.
  - The chat backend buildable/runnable from FQC_REPO_ROOT (this script
    starts and stops it itself for anything that needs a specific state -
    you do not need to have it running beforehand).

Usage:
    python run_all_benchmarks.py                  # fast + medium tiers
    python run_all_benchmarks.py --tier fast       # just the safety net
    python run_all_benchmarks.py --tier all        # everything, including
                                                    # the multi-minute/GB
                                                    # heavy tier
    python run_all_benchmarks.py --only regression,phrasing_battery
    python run_all_benchmarks.py --skip soak,scale
    python run_all_benchmarks.py --check-only      # just verify prereqs

Tiers:
    fast    - unit sanity suites + oracle regression + smoke. ~3 min.
              Safe to run after every code change; this is the suite
              CI/a pre-commit hook should run.
    medium  - phrasing/messy/rephrasing batteries, deep_bench (accuracy/
              groundedness/safety/refusal/quality/determinism), edge
              cases, self-fixing (natural + direct-injection probes),
              concurrency+ingestion, dashboard load. ~20-30 min, GPU-bound
              but no destructive/heavy setup.
    heavy   - scale-test data generation (31.5M synthetic rows, several
              GB of disk and ~3 min to build), scale/index comparison,
              DB-size-vs-speed (restarts the backend 3x), soak test
              (long-running by design), hardware/power profiling
              (restarts the backend for a cold-load measurement), raw
              per-token timing (loads its own model instance). Only run
              this tier when you have time and disk to spare, or after a
              hardware change specifically to re-baseline it.

Each individual script can also be run standalone exactly as before
(`python deep_bench.py` etc.) - this runner adds sequencing, prereq
checks, and a consolidated report on top, nothing more.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import bench_config as bc

TESTS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# The full catalogue - name, tier, script (relative to this file), what
# it needs, and where its own results end up.
# ---------------------------------------------------------------------
CATALOGUE = [
    # --- fast tier: the safety net, run this after every code change ---
    {"name": "unit_sanity", "tier": "fast", "kind": "glob",
     "pattern": "sanity_*.py", "needs_backend": False,
     "desc": "397 fine-grained checks across 17 files, one per repair function/fast-path (no HTTP, no model call)"},
    {"name": "regression", "tier": "fast", "kind": "module",
     "script": "regression_suite.py", "needs_backend": True,
     "desc": "36 oracle-verified questions across every category (counts, averages, breakdowns, reports, follow-ups)"},
    {"name": "smoke", "tier": "fast", "kind": "module",
     "script": "smoke_suite.py", "needs_backend": True,
     "desc": "16 representative questions end-to-end, checked for errors/empty answers"},

    # --- medium tier: accuracy/robustness, GPU-bound, ~20-30 min ---
    {"name": "phrasing_battery", "tier": "medium", "kind": "module",
     "script": "phrasing_battery.py", "needs_backend": True,
     "desc": "157 prompts - the same questions in 10+ natural-language phrasings each"},
    {"name": "messy_phrasing_battery", "tier": "medium", "kind": "module",
     "script": "messy_phrasing_battery.py", "needs_backend": True,
     "desc": "18 prompts across clean/typo-laden/programming-style registers"},
    {"name": "bugfix_rephrasing_battery", "tier": "medium", "kind": "module",
     "script": "bugfix_rephrasing_battery.py", "needs_backend": True,
     "desc": "107 prompts specifically targeting each fixed-bug pattern across many rewordings"},
    {"name": "hour_phrasing_battery", "tier": "medium", "kind": "module",
     "script": "hour_phrasing_battery.py", "needs_backend": True,
     "desc": "104 prompts targeting hour-range/time-of-day phrasing without the words between/from, plus qualitative morning/afternoon/evening/night"},
    {"name": "deep_bench", "tier": "medium", "kind": "module",
     "script": "deep_bench.py", "needs_backend": True,
     "desc": "254 questions + 120 repeat requests - execution accuracy, groundedness, safety, "
             "refusal correctness, 7-axis quality grading, determinism, latency percentiles"},
    {"name": "edge_cases", "tier": "medium", "kind": "module",
     "script": "edge_cases.py", "needs_backend": True,
     "desc": "111 tricky/adversarial/malformed inputs, checked for crashes/hangs, not just correctness"},
    {"name": "self_fixing_natural", "tier": "medium", "kind": "module",
     "script": "self_fixing_natural.py", "needs_backend": True,
     "desc": "195 genuinely hard questions, checking whether the self-correction retry fires unprompted"},
    {"name": "self_repair_probe", "tier": "medium", "kind": "module",
     "script": "self_repair_probe.py", "needs_backend": False,
     "desc": "15 deliberately-broken pipelines fed straight to the retry function, confirming it recovers"},
    {"name": "concurrency_ingestion", "tier": "medium", "kind": "module",
     "script": "concurrency_ingestion.py", "needs_backend": True,
     "desc": "1/2/4/8 simultaneous chat users + a background writer, watching for errors/lock contention"},
    {"name": "dashboard_load", "tier": "medium", "kind": "module",
     "script": "dashboard_load_test.py", "needs_backend": True,
     "desc": "YCSB-Workload-C-style read load (1->150 users) against the MongoDB-backed dashboard endpoints"},
    {"name": "latency", "tier": "medium", "kind": "module",
     "script": "latency_bench.py", "needs_backend": True,
     "desc": "19 representative questions, stage-by-stage timing (query-gen/DB/answer-gen), p50-p99"},

    # --- heavy tier: minutes-to-tens-of-minutes, restarts the backend,
    # needs real disk space for the scale-test data ---
    {"name": "gen_scale_data", "tier": "heavy", "kind": "module",
     "script": "gen_scale_collection.py", "needs_backend": False,
     "desc": "generates the 31,557,600-row (10yr @ 1/10s) scale-test collection - prereq for the next two"},
    {"name": "scale_index", "tier": "heavy", "kind": "module",
     "script": "scale_index_bench.py", "needs_backend": False,
     "desc": "query timing at 31.5M rows: no index vs two separate indexes vs the compound index fix"},
    {"name": "db_size_vs_speed", "tier": "heavy", "kind": "module",
     "script": "db_size_vs_speed.py", "needs_backend": False,  # manages its own restarts
     "desc": "the same 8 real questions at 3 real data volumes (current/3M/31.5M), restarting the backend for each"},
    {"name": "soak", "tier": "heavy", "kind": "module",
     "script": "soak_test.py", "needs_backend": True,
     "desc": "continuous load for several minutes, sampling backend RSS memory and latency drift"},
    {"name": "hardware_power", "tier": "heavy", "kind": "module",
     "script": "hardware_power_bench.py", "needs_backend": True,  # also restarts itself for cold-load timing
     "desc": "host CPU/GPU power+utilization during real load, plus a cold-start timing (kills+restarts the backend)"},
    {"name": "token_timing", "tier": "heavy", "kind": "module",
     "script": "token_timing_bench.py", "needs_backend": False,  # loads its own model instance
     "desc": "real per-token prefill/decode timing read from llama.cpp's own counters, bypassing HTTP entirely"},
]

TIER_ORDER = {"fast": 0, "medium": 1, "heavy": 2}


def run_glob(pattern, label):
    results = []
    for path in sorted(TESTS_DIR.glob(pattern)):
        t0 = time.time()
        proc = subprocess.run([bc.VENV_PYTHON, str(path)], cwd=TESTS_DIR,
                               env=bc.env_with_cuda(), capture_output=True, text=True)
        dt = time.time() - t0
        passed = "ALL PASS" in proc.stdout
        n_pass = proc.stdout.count("  PASS")
        n_fail = proc.stdout.count("  FAIL")
        status = "PASS" if proc.returncode == 0 and passed else "FAIL"
        print(f"  [{status}] {path.name:45s} {n_pass:3d} pass, {n_fail:2d} fail  ({dt:.1f}s)")
        if status == "FAIL":
            print("    " + "\n    ".join(proc.stdout.splitlines()[-15:]))
        results.append({"file": path.name, "status": status, "pass": n_pass, "fail": n_fail,
                         "duration_s": round(dt, 1)})
    return results


def run_module(script, label):
    t0 = time.time()
    proc = subprocess.run([bc.VENV_PYTHON, script], cwd=TESTS_DIR,
                           env=bc.env_with_cuda(), capture_output=True, text=True)
    dt = time.time() - t0
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"  [{status}] {script}  ({dt:.1f}s, exit={proc.returncode})")
    tail = proc.stdout.splitlines()[-25:]
    for line in tail:
        print("    " + line)
    if proc.returncode != 0 and proc.stderr:
        print("    STDERR:", proc.stderr.splitlines()[-10:])
    return {"script": script, "status": status, "duration_s": round(dt, 1),
            "returncode": proc.returncode, "tail": tail}


def check_prereqs(tier):
    print("=" * 70)
    print("Prerequisite checks")
    print("=" * 70)
    ok = True

    from pymongo import MongoClient
    try:
        client = MongoClient(bc.MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        n = client["slm_safety"]["alerts"].estimated_document_count()
        print(f"  [OK] MongoDB reachable at {bc.MONGO_URI} - slm_safety.alerts has {n:,} rows")
        if n == 0:
            print("  [WARN] alerts collection is empty - run chat/backend/seed_data.py first")
    except Exception as e:
        print(f"  [FAIL] MongoDB not reachable at {bc.MONGO_URI}: {e}")
        ok = False

    import os
    if os.path.exists(bc.VENV_PYTHON):
        print(f"  [OK] venv python found at {bc.VENV_PYTHON}")
    else:
        print(f"  [FAIL] venv python not found at {bc.VENV_PYTHON} - set FQC_VENV_DIR/FQC_VENV_PYTHON")
        ok = False

    if os.path.exists(bc.REPO_ROOT):
        print(f"  [OK] repo root found at {bc.REPO_ROOT}")
    else:
        print(f"  [FAIL] repo root not found at {bc.REPO_ROOT} - set FQC_REPO_ROOT")
        ok = False

    if tier in ("medium", "heavy", "all"):
        if os.path.exists(bc.MODEL_PATH):
            print(f"  [OK] model file found at {bc.MODEL_PATH}")
        else:
            print(f"  [WARN] model file not found at {bc.MODEL_PATH} - only needed for token_timing; "
                  "set FQC_MODEL_PATH if you'll run the heavy tier")

    print()
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["fast", "medium", "heavy", "all"], default="medium",
                   help="Run this tier and every tier below it (default: medium, i.e. fast+medium)")
    p.add_argument("--only", help="Comma-separated test names to run, ignoring --tier")
    p.add_argument("--skip", help="Comma-separated test names to skip")
    p.add_argument("--check-only", action="store_true", help="Just check prerequisites and exit")
    args = p.parse_args()

    if not check_prereqs(args.tier) and not args.check_only:
        print("Prerequisite checks failed - fix the above or pass --check-only to see details. Aborting.")
        sys.exit(1)
    if args.check_only:
        return

    if args.only:
        names = set(args.only.split(","))
        entries = [e for e in CATALOGUE if e["name"] in names]
    else:
        max_tier = TIER_ORDER["heavy"] if args.tier == "all" else TIER_ORDER[args.tier]
        entries = [e for e in CATALOGUE if TIER_ORDER[e["tier"]] <= max_tier]
    if args.skip:
        skip = set(args.skip.split(","))
        entries = [e for e in entries if e["name"] not in skip]

    print("=" * 70)
    print(f"Running {len(entries)} test group(s): " + ", ".join(e["name"] for e in entries))
    print("=" * 70)

    all_results = {}
    t_start = time.time()
    for entry in entries:
        print(f"\n--- {entry['name']} ---")
        print(f"    {entry['desc']}")
        if entry["needs_backend"]:
            if not bc.wait_for_health(timeout_s=5):
                print("    backend not up - starting it...")
                if not bc.restart_backend():
                    print("    FAILED to start backend - skipping this group")
                    all_results[entry["name"]] = {"status": "SKIPPED", "reason": "backend did not come up"}
                    continue
        if entry["kind"] == "glob":
            all_results[entry["name"]] = run_glob(entry["pattern"], entry["name"])
        else:
            all_results[entry["name"]] = run_module(entry["script"], entry["name"])

    total_dt = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"DONE in {total_dt/60:.1f} min")
    print("=" * 70)

    report_path = TESTS_DIR / "benchmark_run_report.json"
    json.dump({"tier": args.tier, "duration_s": round(total_dt, 1), "results": all_results},
              open(report_path, "w"), indent=2, default=str)
    print(f"Consolidated report written to {report_path}")

    # Individual scripts each already write their own detailed results
    # JSON to /tmp (deep_bench_results.json, edge_cases_results.json,
    # scale_index_bench_results.json, ...) - this report is the index,
    # not a replacement for those.


if __name__ == "__main__":
    main()
