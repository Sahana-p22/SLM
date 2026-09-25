# chat/backend/tests/regression_suite_sql.py
#
# Oracle-verified regression suite for the SQLite/SQL-backed chat pipeline
# (llm_query_sql.py). Each question's oracle is an independently
# hand-written SQL query run directly against alerts.sqlite3 — the same
# principle as llm_query.py's own regression_suite.py, retargeted at SQL.
# Every answer is checked by confirming the real oracle value(s) appear
# in the model's phrased answer, not by string-matching the whole
# sentence (wording legitimately varies).
import os
import re
import sqlite3
import sys
import urllib.request
import json

BACKEND_URL = os.environ.get("FQC_BACKEND_URL", "http://127.0.0.1:8005")
SQLITE_PATH = os.environ.get(
    "ALERTS_SQLITE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "alerts.sqlite3"),
)


def ask(question: str) -> dict:
    req = urllib.request.Request(
        f"{BACKEND_URL}/chat",
        data=json.dumps({"question": question, "history": []}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def oracle(sql: str, params=()):
    conn = sqlite3.connect(SQLITE_PATH)
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return row


def answer_contains_number(answer: str, value) -> bool:
    if isinstance(value, float):
        value = round(value, 2)
        return str(value) in answer or f"{value:.1f}" in answer or str(int(value)) in answer
    return str(value) in answer.replace(",", "")


CASES = [
    (
        "how many alerts on september 19th 2023",
        lambda: oracle(
            "SELECT COUNT(*) FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
            "AND timestamp >= '2023-09-19T00:00:00' AND timestamp < '2023-09-20T00:00:00'"
        )[0],
    ),
    (
        "how many hand touch alerts on september 19th 2023",
        lambda: oracle(
            "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH' "
            "AND timestamp >= '2023-09-19T00:00:00' AND timestamp < '2023-09-20T00:00:00'"
        )[0],
    ),
    (
        "which day had the most alerts",
        lambda: oracle(
            "SELECT COUNT(*) as c FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
            "GROUP BY substr(timestamp,1,10) ORDER BY c DESC LIMIT 1"
        )[0],
    ),
    (
        "which days did we get 328 alerts",
        lambda: oracle(
            "SELECT COUNT(*) FROM (SELECT substr(timestamp,1,10) as day FROM alerts "
            "WHERE alert_type != 'NORMAL_OPERATION' GROUP BY day HAVING COUNT(*) = 328)"
        )[0],
    ),
    (
        "how many fast inspection alerts happened on 2026-09-01",
        lambda: oracle(
            "SELECT COUNT(*) FROM alerts WHERE alert_type = 'FAST_INSPECTION' "
            "AND timestamp >= '2026-09-01T00:00:00' AND timestamp < '2026-09-02T00:00:00'"
        )[0],
    ),
    (
        "how many missing cleaning alerts happened on august 2, 2023",
        lambda: oracle(
            "SELECT COUNT(*) FROM alerts WHERE alert_type = 'MISSING_CLEANING' "
            "AND timestamp >= '2023-08-02T00:00:00' AND timestamp < '2023-08-03T00:00:00'"
        )[0],
    ),
    (
        "what is the average inspection time for fast inspection alerts",
        lambda: round(oracle(
            "SELECT AVG(inspection_time) FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
            "AND alert_type = 'FAST_INSPECTION'"
        )[0], 2),
    ),
    (
        "how many alerts happened on 2021-09-01",
        lambda: oracle(
            "SELECT COUNT(*) FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
            "AND timestamp >= '2021-09-01T00:00:00' AND timestamp < '2021-09-02T00:00:00'"
        )[0],
    ),
]


def main() -> int:
    passed, failed = 0, 0
    for question, oracle_fn in CASES:
        try:
            expected = oracle_fn()
        except Exception as exc:
            print(f"[SKIP] {question!r} — oracle query failed: {exc}")
            continue
        try:
            result = ask(question)
        except Exception as exc:
            print(f"[FAIL] {question!r} — request failed: {exc}")
            failed += 1
            continue
        answer = result.get("answer", "")
        if answer_contains_number(answer, expected):
            print(f"[PASS] {question!r} -> expected {expected}, answer: {answer!r}")
            passed += 1
        else:
            print(f"[FAIL] {question!r} -> expected {expected}, got: {answer!r} (sql: {result.get('pipeline')})")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
