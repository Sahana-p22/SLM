# chat/backend/tests/regression_suite.py
#
# A fixed set of representative questions to run in one shot after any
# prompt/logic change in llm_query.py, so regressions get caught here
# instead of live in front of a user. Covers every category of bug this
# session actually hit: simple counts, averages, type/day/week
# breakdowns, hour-of-day filters, comparisons, reports, and multi-turn
# follow-ups (the hardest class — a fix in isolation can still break once
# real conversation history is involved).
#
# Each case is checked two ways where possible:
#   1. STRUCTURAL — does the pipeline shape make sense for the question
#      (aggregates instead of dumping raw docs, filters to the alert type
#      actually asked about, etc.)? These check the exact failure modes
#      found live this session.
#   2. VALUE — is the number the chatbot reports the same number you get
#      querying MongoDB directly and independently? This is the strongest
#      check: it doesn't trust anything llm_query.py computed internally.
#
# Run from the repo root, WITH THE BACKEND SERVER ALREADY RUNNING
# (uvicorn chat.backend.main:app — same one you use to test manually):
#   python -m chat.backend.tests.regression_suite
#
# Deliberately talks to the live HTTP API (POST /chat) rather than
# importing and calling answer_question() in-process. Importing it here
# would load a SECOND copy of the 3B model into the same 4GB GPU the
# already-running server has loaded — on this hardware that's a real
# CUDA-OOM risk, not a theoretical one. Going through HTTP also exercises
# the actual deployed path (main.py's request handling included), not
# just the inner function.
#
# Takes a while (each question is 1-2 LLM calls on a small local model —
# expect ~15-90s per case, more for follow-up chains and reports), so run
# it deliberately after a change, not on every keystroke. Add new cases
# as new bug classes get found; this list is meant to grow.

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests

from chat.backend.db import get_alerts_collection

from bench_config import API_BASE
ALERT_TYPES = {"FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"}
NO_DATA_RE = re.compile(r"no (matching )?data|no results|nothing (was )?found|\b0\b|zero", re.IGNORECASE)


def answer_question(question: str, history: list) -> dict:
    """Thin HTTP client for POST /chat, shaped to return the same dict
    shape the in-process answer_question() used to (question/intent/
    pipeline/explanation/result/row_count/answer) so none of the
    verifiers below need to know the difference."""
    resp = requests.post(
        f"{API_BASE}/chat",
        json={"question": question, "history": history},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


# =========================================================
# Small shared helpers for verifiers
# =========================================================

def _first_number(rows: list) -> float | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        for v in row.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
    return None


def _answer_contains_number(answer: str, n) -> bool:
    if n is None:
        return False
    candidates = {str(n)}
    if isinstance(n, float) and n == int(n):
        candidates.add(str(int(n)))
    if isinstance(n, (int, float)) and abs(n) >= 1000:
        candidates.add(f"{n:,}")
        if isinstance(n, float) and n == int(n):
            candidates.add(f"{int(n):,}")
    return any(re.search(rf"\b{re.escape(c)}\b", answer) for c in candidates)


def _pipeline_stage_names(pipeline: list) -> set:
    return {next(iter(s)) for s in (pipeline or []) if isinstance(s, dict) and len(s) == 1}


def _pipeline_filters_to_type(pipeline: list, alert_type: str) -> bool:
    for stage in pipeline or []:
        if not isinstance(stage, dict) or "$match" not in stage:
            continue
        cond = stage["$match"]
        if not isinstance(cond, dict):
            continue
        val = cond.get("alert_type")
        if val == alert_type:
            return True
        if isinstance(val, dict):
            in_list = val.get("$in") or val.get("$eq")
            if in_list == alert_type or in_list == [alert_type]:
                return True
    return False


def _today_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


# =========================================================
# Verifier builders — each returns a `verify(final, history) -> (ok, msg)`
# =========================================================

def is_aggregated(final, history=None):
    """Fails a pipeline that's just $match+$limit:200 (the default cap) —
    the exact shape that let raw, un-aggregated documents reach the
    answer step and produced a fabricated number (found live: "busiest
    day" invented from a 25-row preview of raw documents). A $sort +
    small $limit (e.g. $limit: 1 for "the single longest X") is a
    legitimate, different reduction — it isn't a raw dump, it's a
    correct top-N selection — so that shape is NOT flagged here."""
    pipeline = final.get("pipeline") or []
    stages = _pipeline_stage_names(pipeline)
    agg_stages = {"$group", "$bucket", "$bucketAuto", "$sortByCount", "$count", "$facet"}
    if stages & agg_stages:
        return True, "aggregated"
    limit_stages = [s["$limit"] for s in pipeline if isinstance(s, dict) and "$limit" in s]
    if "$sort" in stages and limit_stages and max(limit_stages) <= 20:
        return True, "sorted top-N reduction"
    return False, f"pipeline has no aggregation stage: {stages}"


def filters_to_type(alert_type: str):
    def verify(final, history=None):
        if not _pipeline_filters_to_type(final.get("pipeline") or [], alert_type):
            return False, f"pipeline never filters alert_type to {alert_type}: {final.get('pipeline')}"
        return True, f"filters to {alert_type}"
    return verify


def count_matches_db(filter_fn: Callable[[datetime], dict]):
    def verify(final, history=None):
        now = datetime.now(timezone.utc)
        expected = get_alerts_collection().count_documents(filter_fn(now))
        got = _first_number(final["result"])
        if expected == 0:
            # A MongoDB $count stage emits NO document at all when the
            # match is empty (not a {"total": 0} row) — an empty result
            # list is the CORRECT shape here, not a missing value.
            if got not in (None, 0):
                return False, f"DB says 0 matches, but pipeline returned {got}"
            if not NO_DATA_RE.search(final["answer"]):
                return False, f"DB says 0 matches, but answer doesn't say so: {final['answer']!r}"
            return True, "OK (0, correctly empty)"
        if got is None:
            return False, f"no numeric value in result: {final['result']}"
        if round(got) != expected:
            return False, f"pipeline returned {got}, DB says {expected}"
        if not _answer_contains_number(final["answer"], expected):
            return False, f"answer doesn't mention {expected}: {final['answer']!r}"
        return True, f"OK ({expected})"
    return verify


def avg_matches_db(filter_fn: Callable[[datetime], dict]):
    def verify(final, history=None):
        now = datetime.now(timezone.utc)
        rows = list(get_alerts_collection().aggregate([
            {"$match": filter_fn(now)},
            {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}},
        ]))
        if not rows:
            return False, "no matching alerts in DB to compare against"
        expected = round(rows[0]["avg"], 2)
        got = _first_number(final["result"])
        if got is None:
            return False, f"no numeric value in result: {final['result']}"
        if abs(got - expected) > 0.05:
            return False, f"pipeline returned {got}, DB says {expected}"
        if not _answer_contains_number(final["answer"], expected) and not _answer_contains_number(final["answer"], round(got, 1)):
            return False, f"answer doesn't mention {expected}: {final['answer']!r}"
        return True, f"OK ({expected}s)"
    return verify


def breakdown_matches_db(filter_fn: Callable[[datetime], dict]):
    """For 'break down by type' style questions — every type's count must
    both appear correctly in the raw rows AND be mentioned in the answer
    text with the right number (catches wrong-type/wrong-number
    mismatches, not just missing types)."""
    def verify(final, history=None):
        now = datetime.now(timezone.utc)
        rows = list(get_alerts_collection().aggregate([
            {"$match": filter_fn(now)},
            {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
        ]))
        expected = {r["_id"]: r["count"] for r in rows if r["_id"] in ALERT_TYPES}
        if not expected:
            return False, "no matching alerts in DB to compare against"
        for alert_type, count in expected.items():
            if not _answer_contains_number(final["answer"], count):
                return False, f"answer missing {alert_type}={count}: {final['answer']!r}"
        return True, f"OK ({expected})"
    return verify


def no_data_leak_across_turns(min_total: int = 1):
    """For follow-ups: the final turn's result shouldn't be empty/error —
    catches the class of bug where a follow-up silently lost its date
    scope or type filter and returned nothing or the wrong thing."""
    def verify(final, history=None):
        if final["intent"] == "error":
            return False, f"pipeline errored: {final.get('answer')}"
        if not final["result"]:
            return False, "empty result on final turn"
        return True, "has data"
    return verify


def smoke_check(final, history=None):
    """Generic fallback when no specific verifier is worth writing —
    just confirms the question was actually answered as a real query,
    not silently rejected or errored."""
    if final["intent"] in ("unsupported", "greeting"):
        return False, f"unexpected intent={final['intent']!r} for a real data question"
    if final["intent"] == "error":
        return False, f"pipeline errored: {final.get('answer')}"
    if not final.get("answer") or not final["answer"].strip():
        return False, "empty answer"
    return True, "answered"


def _report_scope_from_pipeline(pipeline):
    """Mirrors `_report_scope` in llm_query.py — the report's own date
    span decides which section layout it should have used."""
    for stage in pipeline or []:
        if not isinstance(stage, dict) or "$match" not in stage:
            continue
        ts = stage["$match"].get("timestamp")
        if not (isinstance(ts, dict) and ts.get("$gte") and ts.get("$lt")):
            continue
        try:
            gte = datetime.fromisoformat(ts["$gte"]) if isinstance(ts["$gte"], str) else ts["$gte"]
            lt = datetime.fromisoformat(ts["$lt"]) if isinstance(ts["$lt"], str) else ts["$lt"]
        except ValueError:
            return "quarter"
        span_days = (lt - gte).days
        if span_days <= 10:
            return "week"
        if span_days <= 40:
            return "month"
        return "quarter"
    return "quarter"


def report_structure_check(final, history=None):
    """Report-style questions use a scope-specific shape (see
    `_restructure_report` in llm_query.py): quarterly reports show
    total/type/month/week; monthly reports show total/type/week (each
    week broken down by type — no by_month); single-week reports show
    total/type/day (a true daily breakdown — no by_month/by_week)."""
    if not final["result"] or not isinstance(final["result"][0], dict):
        return False, "no facet result"
    facet = final["result"][0]
    scope = _report_scope_from_pipeline(final.get("pipeline"))

    base_sections = {"total_alerts", "by_type", "avg_inspection_time"}
    if scope == "week":
        expected, forbidden = base_sections | {"by_day"}, {"by_month", "by_week"}
    elif scope == "month":
        expected, forbidden = base_sections | {"by_week"}, {"by_month", "by_day"}
    else:
        expected, forbidden = base_sections | {"by_month", "by_week"}, {"by_day"}

    missing = expected - set(facet.keys())
    if missing:
        return False, f"report ({scope}) missing sections: {missing}"
    present_forbidden = forbidden & set(facet.keys())
    if present_forbidden:
        return False, f"report ({scope}) should not have: {present_forbidden}"
    return True, f"sections OK ({scope}): {sorted(facet.keys())}"


# =========================================================
# Test case definitions
# =========================================================

@dataclass
class TestCase:
    id: str
    category: str
    turns: list  # one question, or several for a follow-up chain
    verify: Callable = smoke_check
    extra_verify: list = field(default_factory=list)  # additional checks, all must pass


CASES: list[TestCase] = [

    # ---------------- counts ----------------
    TestCase("count_today", "counts", ["How many alerts happened today?"],
             count_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": _today_start(now)}}),
             [is_aggregated]),
    TestCase("count_yesterday_type", "counts", ["How many hand touch alerts happened yesterday?"],
             count_matches_db(lambda now: {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": _today_start(now) - timedelta(days=1), "$lt": _today_start(now)}}),
             [is_aggregated, filters_to_type("HAND_TOUCH")]),
    TestCase("count_last_7_days", "counts", ["How many alerts happened in the last 7 days?"],
             count_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": _today_start(now) - timedelta(days=6)}}),
             [is_aggregated]),
    TestCase("count_last_30_days_missing_cleaning", "counts", ["How many missing cleaning alerts happened in the last 30 days?"],
             count_matches_db(lambda now: {"alert_type": "MISSING_CLEANING", "timestamp": {"$gte": _today_start(now) - timedelta(days=29)}}),
             [is_aggregated, filters_to_type("MISSING_CLEANING")]),
    TestCase("count_this_month", "counts", ["How many alerts happened this month?"],
             count_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)}}),
             [is_aggregated]),

    # ---------------- averages ----------------
    TestCase("avg_all", "averages", ["Average inspection time for all alerts"],
             avg_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}})),
    TestCase("avg_fast_inspection", "averages", ["Average inspection time for fast inspection alerts"],
             avg_matches_db(lambda now: {"alert_type": "FAST_INSPECTION"}),
             [filters_to_type("FAST_INSPECTION")]),
    TestCase("avg_hand_touch_last_week", "averages", ["What's the average inspection time for hand touch alerts in the last 7 days?"],
             avg_matches_db(lambda now: {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": _today_start(now) - timedelta(days=6)}}),
             [filters_to_type("HAND_TOUCH")]),

    # ---------------- breakdowns ----------------
    TestCase("breakdown_by_type_all_time", "breakdowns", ["Break down all alerts by type"],
             breakdown_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}}),
             [is_aggregated]),
    TestCase("breakdown_by_type_last_week", "breakdowns", ["Break down alerts by type for the last 7 days"],
             breakdown_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": _today_start(now) - timedelta(days=6)}}),
             [is_aggregated]),
    TestCase("breakdown_by_type_this_month", "breakdowns", ["Break down this month's alerts by type"],
             breakdown_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)}}),
             [is_aggregated]),

    # ---------------- hour-of-day filters ----------------
    TestCase("peak_hour_this_week", "hour_filters", ["What's the peak hour for fast inspection alerts this week?"],
             smoke_check, [is_aggregated, filters_to_type("FAST_INSPECTION")]),
    TestCase("hour_range_2pm_4pm", "hour_filters", ["How many alerts happened between 2pm and 4pm in the last 30 days?"],
             smoke_check, [is_aggregated]),
    TestCase("hour_range_9am_11am_today", "hour_filters", ["How many missing cleaning alerts happened between 9am and 11am today?"],
             smoke_check, [is_aggregated, filters_to_type("MISSING_CLEANING")]),

    # ---------------- day-of-week / peak day ----------------
    TestCase("peak_day_hand_touch", "day_of_week", ["Which day had the most hand touch alerts?"],
             smoke_check, [is_aggregated, filters_to_type("HAND_TOUCH")]),
    TestCase("peak_day_fast_inspection_last_week", "day_of_week", ["Which day had the most fast inspection alerts last week?"],
             smoke_check, [is_aggregated, filters_to_type("FAST_INSPECTION")]),
    TestCase("busiest_weekday_missing_cleaning", "day_of_week", ["Which day of the week sees the most missing cleaning alerts?"],
             smoke_check, [is_aggregated, filters_to_type("MISSING_CLEANING")]),

    # ---------------- week grouping ----------------
    TestCase("busiest_week_this_quarter", "week_grouping", ["Which week had the most alerts this quarter?"],
             smoke_check, [is_aggregated]),
    TestCase("weekly_report_last_4_weeks", "week_grouping", ["Give me a weekly report for the last 4 weeks"],
             report_structure_check),

    # ---------------- comparisons ----------------
    TestCase("compare_types_this_week_vs_last_week", "comparisons",
             ["Compare fast inspection and hand touch alert counts for this week vs last week"],
             smoke_check, [is_aggregated]),
    TestCase("compare_avg_time_types", "comparisons",
             ["Compare average inspection time between fast inspection and missing cleaning alerts"],
             smoke_check, [is_aggregated]),
    TestCase("compare_today_vs_yesterday", "comparisons",
             ["How does today compare to yesterday in total alert count?"],
             smoke_check, [is_aggregated]),

    # ---------------- reports ----------------
    TestCase("quarterly_report", "reports", ["Give me a quarterly report"], report_structure_check),
    TestCase("monthly_report_named_month", "reports", ["Give me a report for June"], report_structure_check),
    TestCase("day_wise_report", "reports", ["Give me a day wise report for this month"], report_structure_check),

    # ---------------- follow-ups (multi-turn) ----------------
    TestCase("followup_count_then_breakdown", "follow_ups",
             ["How many hand touch alerts happened this week?", "Now break that down by day"],
             no_data_leak_across_turns()),
    TestCase("followup_report_then_narrow", "follow_ups",
             ["Give me a quarterly report", "Just show me the by_type part again but for last month instead"],
             no_data_leak_across_turns()),
    TestCase("followup_chain_3_turns", "follow_ups",
             ["How many alerts happened today?", "Break them down by type", "Which of those had the highest average inspection time?"],
             no_data_leak_across_turns()),
    TestCase("followup_peak_hour_then_day", "follow_ups",
             ["What's the peak hour for fast inspection alerts this week?", "Which day had the most instead?"],
             no_data_leak_across_turns()),
    TestCase("followup_greeting_then_question", "follow_ups",
             ["hi there", "How many alerts happened today?"],
             count_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": _today_start(now)}})),

    # ---------------- edge cases ----------------
    TestCase("last_15_minutes", "edge_cases", ["How many alerts happened in the last 15 minutes?"],
             count_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": now - timedelta(minutes=15)}}),
             [is_aggregated]),
    TestCase("last_hour", "edge_cases", ["How many alerts happened in the last hour?"],
             count_matches_db(lambda now: {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": now - timedelta(hours=1)}}),
             [is_aggregated]),
    TestCase("longest_inspection_time", "edge_cases", ["What's the single longest inspection time on record?"],
             smoke_check, [is_aggregated]),
    TestCase("threshold_over_30s", "edge_cases", ["How many hand touch alerts took longer than 30 seconds to inspect?"],
             smoke_check, [is_aggregated, filters_to_type("HAND_TOUCH")]),
    TestCase("gibberish_unsupported", "edge_cases", ["asdkjfh qwerty banana"],
             lambda final, history=None: (final["intent"] == "unsupported", f"intent={final['intent']!r}")),
    TestCase("greeting_only", "edge_cases", ["hello!"],
             lambda final, history=None: (final["intent"] == "greeting", f"intent={final['intent']!r}")),
]


# =========================================================
# Runner
# =========================================================

def _run_case(case: TestCase) -> dict:
    history = []
    turn_results = []
    final = None
    error = None

    try:
        for question in case.turns:
            final = answer_question(question, history)
            turn_results.append(final)
            history.append({
                "question": question,
                "answer": final["answer"],
                "pipeline": final.get("pipeline"),
            })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if error:
        return {"id": case.id, "category": case.category, "passed": False, "reason": error, "final": None}

    checks = [case.verify] + list(case.extra_verify)
    for check in checks:
        ok, msg = check(final, turn_results)
        if not ok:
            return {"id": case.id, "category": case.category, "passed": False, "reason": msg, "final": final}

    return {"id": case.id, "category": case.category, "passed": True, "reason": "ok", "final": final}


def main():
    try:
        requests.get(f"{API_BASE}/health", timeout=5).raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Backend not reachable at {API_BASE} ({exc}).")
        print("Start it first: uvicorn chat.backend.main:app --host 127.0.0.1 --port 8000")
        sys.exit(2)

    only_category = sys.argv[1] if len(sys.argv) > 1 else None
    cases = [c for c in CASES if only_category is None or c.category == only_category]

    print(f"Running {len(cases)} regression cases{f' (category={only_category})' if only_category else ''}...\n")

    results = []
    start = time.time()
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        result = _run_case(case)
        dt = time.time() - t0
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{i}/{len(cases)}] {status:4} {case.id:38} ({case.category:14}) {dt:5.1f}s  {result['reason']}")
        results.append(result)

    total_dt = time.time() - start
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'=' * 70}")
    print(f"{passed}/{len(results)} passed, {failed} failed  ({total_dt:.0f}s total)")

    if failed:
        print("\nFailed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['id']} ({r['category']}): {r['reason']}")

    report_path = Path(__file__).parent / "last_run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report written to {report_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
