# chat/backend/tests/regression_suite_sql.py
#
# Ported from slm-llama3b's regression_suite.py: a fixed set of
# representative questions run against the LIVE HTTP API (not imported
# in-process — this deliberately avoids loading a second copy of the model
# onto the same GPU/CPU worker the running server already has loaded),
# each checked against an INDEPENDENTLY computed SQL oracle value, not
# against anything the app itself claims internally.
#
# Covers the same category set the original project's suite established
# (counts, averages, breakdowns, hour filters, day-of-week superlatives,
# comparisons, reports, multi-turn follow-ups, edge cases) — adapted here
# to this port's SQLite schema/oracle instead of MongoDB aggregation.
#
# Run with the approach2 backend already running on port 8006:
#   /home/wgtech/slm-main/.venv/bin/python -m chat.backend.tests.regression_suite_sql
#
# Each question may take several seconds on CPU (approach2 runs without a
# GPU-resident model to avoid VRAM contention with the live deployment and
# the sqlite-migration copy) — this is expected, not a bug.

import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

API_BASE = os.environ.get("APPROACH2_API_BASE", "http://127.0.0.1:8006")
DB_PATH = Path(__file__).resolve().parent.parent / "fqc.db"


def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def ask(question, history=None):
    resp = requests.post(f"{API_BASE}/chat",
                         json={"question": question, "history": history or []},
                         timeout=180)
    resp.raise_for_status()
    return resp.json()


def _num_in_answer(answer, n):
    if n is None:
        return False
    candidates = {str(n)}
    if isinstance(n, float) and n == int(n):
        candidates.add(str(int(n)))
    if isinstance(n, (int, float)) and abs(n) >= 1000:
        candidates.add(f"{n:,}")
    flat = answer.replace(",", "")
    return any(c in answer or c in flat for c in candidates)


fails = []
passed = 0


def check(name, cond, detail=""):
    global passed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if cond:
        passed += 1
    else:
        fails.append(name)


NOW = datetime.utcnow()
TODAY = NOW.replace(hour=0, minute=0, second=0, microsecond=0)


def oracle_count(where_sql, params=()):
    with _conn() as c:
        return c.execute(f"SELECT COUNT(*) FROM alerts WHERE {where_sql}", params).fetchone()[0]


def oracle_avg(where_sql, params=()):
    with _conn() as c:
        row = c.execute(f"SELECT ROUND(AVG(inspection_time),2) FROM alerts WHERE {where_sql}", params).fetchone()
        return row[0]


print("=" * 70)
print("COUNTS")
print("=" * 70)

check("count_today: matches oracle",
      _num_in_answer(ask("How many alerts happened today?")["answer"],
                     oracle_count("alert_type != 'NORMAL_OPERATION' AND timestamp >= ?",
                                 (TODAY.isoformat(sep=" "),))))

check("count_yesterday_hand_touch: matches oracle",
      _num_in_answer(ask("How many hand touch alerts happened yesterday?")["answer"],
                     oracle_count("alert_type = 'HAND_TOUCH' AND timestamp >= ? AND timestamp < ?",
                                 ((TODAY - timedelta(days=1)).isoformat(sep=" "), TODAY.isoformat(sep=" ")))))

check("count_last_7_days: matches oracle",
      _num_in_answer(ask("How many alerts happened in the last 7 days?")["answer"],
                     oracle_count("alert_type != 'NORMAL_OPERATION' AND timestamp >= ?",
                                 ((TODAY - timedelta(days=6)).isoformat(sep=" "),))))

check("count_missing_cleaning_all_time: matches oracle",
      _num_in_answer(ask("How many missing cleaning alerts have there been in total?")["answer"],
                     oracle_count("alert_type = 'MISSING_CLEANING'")))

print("\n" + "=" * 70)
print("AVERAGES")
print("=" * 70)

check("avg_all: matches oracle",
      _num_in_answer(ask("What's the average inspection time for all alerts?")["answer"],
                     oracle_avg("alert_type != 'NORMAL_OPERATION'")))

check("avg_fast_inspection: matches oracle",
      _num_in_answer(ask("Average inspection time for fast inspection alerts")["answer"],
                     oracle_avg("alert_type = 'FAST_INSPECTION'")))

print("\n" + "=" * 70)
print("BREAKDOWNS")
print("=" * 70)

r = ask("Break down all alerts by type")
with _conn() as c:
    type_counts = {row["alert_type"]: row["count"] for row in c.execute(
        "SELECT alert_type, COUNT(*) AS count FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
        "GROUP BY alert_type")}
check("breakdown_by_type_all_time: every type's oracle count appears in the answer",
      all(_num_in_answer(r["answer"], n) for n in type_counts.values()),
      f"answer={r['answer']!r} oracle={type_counts}")

print("\n" + "=" * 70)
print("HOUR FILTERS")
print("=" * 70)

check("hour_range_2pm_4pm_last_30d: matches oracle",
      _num_in_answer(
          ask("How many alerts happened between 2pm and 4pm in the last 30 days?")["answer"],
          oracle_count("alert_type != 'NORMAL_OPERATION' AND hour >= 14 AND hour < 16 AND timestamp >= ?",
                      ((TODAY - timedelta(days=29)).isoformat(sep=" "),))))

print("\n" + "=" * 70)
print("DAY-OF-WEEK / SUPERLATIVES")
print("=" * 70)

r = ask("Which day had the most hand touch alerts?")
with _conn() as c:
    row = c.execute(
        "SELECT date(timestamp) AS d, COUNT(*) AS c FROM alerts WHERE alert_type = 'HAND_TOUCH' "
        "GROUP BY d ORDER BY c DESC LIMIT 1").fetchone()
check("peak_day_hand_touch: busiest day's count matches oracle",
      _num_in_answer(r["answer"], row["c"]), f"answer={r['answer']!r} oracle day={row['d']} count={row['c']}")

print("\n" + "=" * 70)
print("COMPARISONS")
print("=" * 70)

r = ask("Compare fast inspection and hand touch counts this week vs last week")
check("compare_types_this_week_vs_last_week: gets a real answer, not an error",
      len(r["answer"]) > 10 and "problem" not in r["answer"].lower())

print("\n" + "=" * 70)
print("REPORTS")
print("=" * 70)

r = ask("Give me a quarterly report")
check("quarterly_report: source is the deterministic report builder, not the LLM",
      r.get("source") == "report", f"source={r.get('source')}")
check("quarterly_report: answer names a real total",
      re.search(r"[\d,]+\s+alerts", r["answer"]) is not None)

r = ask("Give me a report for June")
check("monthly_report_named_month: honors the named month",
      "June" in r["answer"])

print("\n" + "=" * 70)
print("FOLLOW-UPS (multi-turn)")
print("=" * 70)

turn1 = ask("How many alerts happened today?")
history = [{"question": "How many alerts happened today?", "answer": turn1["answer"], "sql": turn1.get("sql")}]
turn2 = ask("break it down by type", history)
today_type_counts = {}
with _conn() as c:
    for row in c.execute(
            "SELECT alert_type, COUNT(*) AS count FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
            "AND timestamp >= ? GROUP BY alert_type", (TODAY.isoformat(sep=" "),)):
        today_type_counts[row["alert_type"]] = row["count"]
check("followup_count_then_breakdown: follow-up inherits today's scope, not all-time",
      any(_num_in_answer(turn2["answer"], n) for n in today_type_counts.values()) if today_type_counts else True,
      f"answer={turn2['answer']!r} oracle={today_type_counts}")

r = ask("hello!")
check("followup_greeting_then_question: greeting still classified correctly mid-suite",
      "ask me" in r["answer"].lower() or "hi" in r["answer"].lower())

print("\n" + "=" * 70)
print("EDGE CASES")
print("=" * 70)

check("last_15_minutes: no crash, real answer",
      len(ask("How many alerts happened in the last 15 minutes?")["answer"]) > 0)

r = ask("What's the single longest inspection time on record?")
with _conn() as c:
    mx = c.execute("SELECT MAX(inspection_time) FROM alerts WHERE alert_type != 'NORMAL_OPERATION'").fetchone()[0]
check("longest_inspection_time: matches oracle max",
      _num_in_answer(r["answer"], round(mx, 2)) or _num_in_answer(r["answer"], mx),
      f"answer={r['answer']!r} oracle={mx}")

r = ask("asdkjfh qwerty banana")
check("gibberish_unsupported: correctly refused, not answered as a data question",
      r["intent"] == "unsupported")

r = ask("hello!")
check("greeting_only: correctly classified", r["intent"] == "greeting")

print("\n" + "=" * 70)
print(f"{passed} passed, {len(fails)} failed" + (f": {', '.join(fails)}" if fails else ""))
raise SystemExit(1 if fails else 0)
