"""1.13 continued: real per-token prefill/decode timing, measured wall-clock
around each call (not estimated from HTTP/SSE chunk arrival) and divided
by the exact prompt/completion token counts llama.cpp itself reports in
`usage`. This llama-cpp-python version doesn't expose a `Llama.timings()`
API (checked: no such method exists), so tok/s here is wall_s /
completion_tokens rather than read from llama.cpp's internal perf
counters directly - accurate for throughput, but doesn't split prefill
vs. decode time within a single call the way the raw counters would.
Loads its own short-lived model instance (released on exit) rather than
disturbing the live backend's.

15 query-generation-style calls (long system prompt + question, matching
the real app's actual query-gen prompt size) and 9 answer-generation-
style calls (short prompt, matching the real app's answer-phrasing call).
"""
import json
import statistics as st
import sys
import time

from bench_config import REPO_ROOT, MODEL_PATH
sys.path.insert(0, REPO_ROOT)
from llama_cpp import Llama
from chat.backend.llm_query import QUERY_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT

print("[token_timing] loading model...")
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False)
print(f"[token_timing] loaded in {time.time() - t0:.2f}s")

QUERY_QUESTIONS = [
    "How many alerts happened today?", "How many hand touch alerts this week?",
    "Break down alerts by type for the last 7 days.", "What's the average inspection time?",
    "Which day had the most missing cleaning alerts?", "Give me a quarterly report.",
    "How many alerts happened between 2pm and 4pm yesterday?", "Compare this week vs last week.",
    "How many distinct alert types are there?", "Give me a weekly report.",
    "How many fast inspection alerts happened at FQC Station 1?", "What's the busiest hour today?",
    "How many alerts happened this month?", "Give me a report for June.",
    "How many inspections took over 30 seconds?",
]

ANSWER_PROMPTS = [
    '{"total": 82}', '{"count": 386, "alert_type": "HAND_TOUCH"}',
    '[{"_id": "FAST_INSPECTION", "count": 1888}, {"_id": "HAND_TOUCH", "count": 1442}]',
    '{"avg": 11.09}', '{"_id": "2026-09-20", "count": 47}',
    '{"total_alerts": 18714, "by_type": [...]}', '{"total": 431}',
    '{"hour": 15, "count": 33}', '{"total": 4894}',
]


def measure(system_prompt, user_prompt, label):
    t0 = time.perf_counter()
    result = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        max_tokens=300, temperature=0.0,
    )
    wall = time.perf_counter() - t0
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "label": label, "wall_s": wall,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / wall if wall > 0 else None,
    }


print("\n=== Query-generation-style calls (15) ===")
query_results = []
for q in QUERY_QUESTIONS:
    r = measure(QUERY_SYSTEM_PROMPT, q, "query_gen")
    query_results.append(r)
    print(f"  prompt_tok={r['prompt_tokens']:5d} completion_tok={r['completion_tokens']:4d} "
          f"wall={r['wall_s']:.2f}s tok/s={r['tokens_per_s']:.1f}" if r['tokens_per_s'] else "  n/a")

print("\n=== Answer-generation-style calls (9) ===")
answer_results = []
for p in ANSWER_PROMPTS:
    r = measure(ANSWER_SYSTEM_PROMPT, p, "answer_gen")
    answer_results.append(r)
    print(f"  prompt_tok={r['prompt_tokens']:5d} completion_tok={r['completion_tokens']:4d} "
          f"wall={r['wall_s']:.2f}s tok/s={r['tokens_per_s']:.1f}" if r['tokens_per_s'] else "  n/a")


def summarize(results):
    toks = [r["tokens_per_s"] for r in results if r["tokens_per_s"]]
    return {
        "mean_prompt_tokens": st.mean(r["prompt_tokens"] for r in results),
        "mean_completion_tokens": st.mean(r["completion_tokens"] for r in results),
        "mean_wall_s": st.mean(r["wall_s"] for r in results),
        "mean_tokens_per_s": st.mean(toks) if toks else None,
    }

summary = {"query_gen": summarize(query_results), "answer_gen": summarize(answer_results)}
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))

json.dump({"summary": summary, "query_gen": query_results, "answer_gen": answer_results},
          open("/tmp/token_timing_results.json", "w"), indent=2)
