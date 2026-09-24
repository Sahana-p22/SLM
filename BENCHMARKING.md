# FQC Chatbot — Benchmark Suite

Everything used to validate the FQC (factory quality-control alert log)
chatbot lives in `chat/backend/tests/`. It's hardware-portable: every
machine-specific path (backend URL/port, Python venv, CUDA library dirs,
model file, MongoDB URI) is read from environment variables via
`bench_config.py`, never hardcoded in an individual script. Moving to
new hardware means setting environment variables (or editing the
defaults in `bench_config.py`), not hunting through a dozen files.

This suite is FQC-specific only. It does not include MLPerf Edge
Agentic / BFCL (those benchmark the raw base model's generic tool-calling
ability, not this deployment, and were explicitly scoped out — see
`git log`/session notes if you need that thread again).

## Quick start

```bash
cd chat/backend/tests
python run_all_benchmarks.py --check-only      # verify prerequisites only
python run_all_benchmarks.py                   # fast + medium tiers (default)
python run_all_benchmarks.py --tier fast       # just the safety net, ~3 min
python run_all_benchmarks.py --tier all        # everything, including heavy (~40-60 min, GBs of disk)
python run_all_benchmarks.py --only regression,smoke
python run_all_benchmarks.py --skip soak,gen_scale_data,scale_index,db_size_vs_speed
```

`run_all_benchmarks.py` starts/stops the backend itself as needed and
writes one consolidated `benchmark_run_report.json` at the end. Every
script can also still be run standalone (`python regression_suite.py`),
exactly as before — the runner only adds sequencing, prerequisite
checks, and a summary on top.

## Porting to new hardware

Set whichever of these differ from the defaults (all in `bench_config.py`):

| Variable | Default | What it's for |
|---|---|---|
| `FQC_BACKEND_HOST` | `127.0.0.1` | Host the chat backend listens on |
| `FQC_BACKEND_PORT` | `8002` | Port the chat backend listens on |
| `FQC_API_BASE` | `http://{host}:{port}` | Full override, if host/port alone isn't enough |
| `FQC_REPO_ROOT` | `/home/wgtech/slm-llama3b` | Repo checkout location |
| `FQC_VENV_DIR` | `/home/wgtech/slm-main/.venv` | Python venv with all deps installed |
| `FQC_VENV_PYTHON` | `{venv}/bin/python` | Explicit interpreter override |
| `FQC_CUDA_LIB_DIRS` | pip-installed nvidia wheel paths under the venv | `LD_LIBRARY_PATH` additions llama-cpp-python's CUDA build needs. Set to `none` on CPU-only hardware or where CUDA is already on the system loader path |
| `FQC_MODEL_PATH` | `{repo_root}/models/.../Llama-3.2-3B-Instruct-Q8_0.gguf` | GGUF file, only needed for `token_timing_bench.py` (loads the model directly, no HTTP) |
| `FQC_MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `FQC_SCALE_TEST_DB` | `slm_safety_scale` | Separate DB name the heavy-tier scale tests write into, so they never touch production data |

The two repo-root convenience wrappers, `run_regression_8001.py` and
`run_regression_8002.py`, exist only to save a `cd`; the real, portable
entry point is `regression_suite.py` (or `run_all_benchmarks.py`) run
from `chat/backend/tests/` with `FQC_API_BASE` set if you're not on the
default port.

Also required, not env-controlled: MongoDB running and reachable, and
the real alert data seeded (`python -m chat.backend.seed_data`). Run
`--check-only` first on any new box — it verifies Mongo connectivity and
row count, venv python, repo root, and (for medium/heavy tiers) the
model file, and tells you exactly what's missing.

## Tiers

### `fast` (~3 min) — run after every code change
Deterministic, no LLM calls involved except where noted. This is what a
CI job or pre-commit hook should run.

- **`unit_sanity`** — `sanity_*.py` (17 files, 397 checks). Pure
  in-process unit tests of individual repair/fast-path functions in
  `llm_query.py`. No HTTP, no model, no Mongo. One file per bug class or
  fast-path found and fixed — e.g. `sanity_fqc_bugfixes_pipeline_shape.py`
  (inverted-filter, misclassified-total, chained-group repairs),
  `sanity_hour_range_phrasing.py` (explicit hour ranges + qualitative
  time-of-day words), `sanity_fast_path_exclude_gaps.py` (glued "3pm"
  word-boundary gap, "this afternoon" date resolution),
  `sanity_fast_path_hour_trigger.py` (leading day-phrase before
  "between"/"from", bare time-of-day words triggering the fast path
  directly instead of the LLM classifier).
- **`regression`** — `regression_suite.py`, 36 oracle-verified questions
  (talks to the live HTTP API) spanning counts, averages, type/day/week
  breakdowns, hour-of-day filters, comparisons, reports, and multi-turn
  follow-ups. Every number is checked against MongoDB computed
  independently, not against anything the pipeline itself claims.
- **`smoke`** — `smoke_suite.py`, 16 representative questions end-to-end,
  checked for errors/empty/malformed answers rather than exact values.

### `medium` (~20-30 min, GPU-bound) — accuracy & robustness
- **`phrasing_battery`** — 157 prompts, the same underlying questions in
  10+ natural phrasings each.
- **`messy_phrasing_battery`** — 18 prompts across clean / typo-laden /
  programming-style registers.
- **`bugfix_rephrasing_battery`** — 107 prompts specifically targeting
  each previously-fixed bug pattern, reworded many ways, so a fix can't
  silently regress just because the exact original wording changed.
- **`hour_phrasing_battery`** — 104 oracle-verified prompts crossing
  explicit hour ranges (6 pairs) × phrasing style (between/from/bare-to/
  bare-and/leading/trailing day-word) × day, plus qualitative
  morning/afternoon/evening/night/noon phrasing. Added specifically
  after a live bug report showed hour-range/time-of-day phrasing had
  never been tested at scale despite a 99.2% headline accuracy number —
  this battery is what closes that gap. Currently 104/104.
- **`deep_bench`** — 254 questions + 120 repeat-request determinism
  checks: execution accuracy, groundedness (numbers actually present in
  DB results, never invented), safety/refusal correctness, 7-axis answer
  quality grading, determinism, latency percentiles.
- **`edge_cases`** — 111 tricky/adversarial/malformed inputs, checked for
  crashes/hangs, not just correctness.
- **`self_fixing_natural`** — 195 genuinely hard questions, checking
  whether the self-correction retry (feeding a Mongo error back to the
  model once) fires unprompted when needed.
- **`self_repair_probe`** — 15 deliberately-broken pipelines fed straight
  to the retry function in-process, confirming it recovers without going
  through the LLM at all.
- **`concurrency_ingestion`** — 1/2/4/8 simultaneous chat users plus a
  background writer, watching for errors or lock contention (relevant
  because the underlying `llama_cpp.Llama` instance is not thread-safe —
  see the concurrency lock in `llm_query.py`'s streaming helper).
- **`dashboard_load`** — YCSB-Workload-C-style read load (1→150 users)
  against the MongoDB-backed dashboard endpoints.
- **`latency`** — 19 representative questions, stage-by-stage timing
  (query-generation / DB round trip / answer-generation), p50–p99.

### `heavy` (minutes-to-tens-of-minutes, restarts the backend, needs real disk)
Only run this tier when you have time/disk to spare, or specifically to
re-baseline after a hardware change.

- **`gen_scale_data`** — generates a 31,557,600-row (10 years @ 1 row/10s)
  synthetic collection in `FQC_SCALE_TEST_DB`, never touching production
  data. Prerequisite for the next two.
- **`scale_index`** — query timing at 31.5M rows: no index vs. two
  separate indexes vs. the compound index actually shipped.
- **`db_size_vs_speed`** — the same 8 real questions run against 3 real
  data volumes (current production size / 3M / 31.5M rows), restarting
  the backend and repointing it at each collection in turn.
- **`soak`** — continuous load for several minutes, sampling backend RSS
  memory and latency drift, watching for leaks/degradation.
- **`hardware_power`** — host CPU/GPU power + utilization under real
  load, plus a cold-start timing (kills and restarts the backend to
  measure load time from scratch).
- **`token_timing`** — real per-token prefill/decode timing read
  directly from llama.cpp's own counters, loading its own model instance
  and bypassing HTTP entirely (needs `FQC_MODEL_PATH`).

## What's deliberately NOT here

- **MLPerf Edge Agentic / BFCL** — benchmarks the raw base model's
  generic function-calling ability against an unrelated dataset. Not
  predictive of FQC chatbot quality; explicitly descoped after
  confirming with the model's actual FQC-specific failure modes instead.
- Anything that mutates production Mongo data outside of
  `FQC_SCALE_TEST_DB` — the heavy tier is careful to use a separate
  database for all generated/synthetic data.

## Adding a new test case

- A new deterministic bug fix in `llm_query.py` → add unit checks to an
  existing `sanity_*.py` file if it's the same bug family, or a new
  `sanity_<topic>.py` file otherwise (pattern: reproduce the exact live
  failure as an input, assert the fix, then assert at least one negative
  case that must NOT be touched by the fix). No HTTP/model calls in
  these — they call the repair function directly.
- A new phrasing/robustness concern → add prompts to whichever battery
  already covers that shape, or start a new `<topic>_battery.py`
  following `hour_phrasing_battery.py`'s pattern: oracle value computed
  directly from MongoDB via `pymongo`, independent of what the pipeline
  itself reports, compared against the live chatbot's actual answer
  text.
- Either way, register it in `CATALOGUE` in `run_all_benchmarks.py`
  (unit sanity files are auto-discovered via the `sanity_*.py` glob — no
  registration needed for those).
