# chat/backend/tests_sql/sanity_sql_fixers.py
#
# Pure in-process unit tests for the deterministic SQL repair functions in
# llm_query_sql.py — no HTTP, no model, no database. Follows the naming/
# style convention of llm_query.py's own sanity_*.py files.
import sys
from datetime import datetime

sys.path.insert(0, "/home/wgtech/slm-llama3b-sqlite")

from chat.backend.llm_query_sql import (
    fix_date_range, strip_unwanted_date_filter, fix_missing_normal_operation_exclusion,
    fix_or_and_precedence, fix_missing_group_by, fix_missing_superlative_limit,
    fix_misclassified_count_target_ranking, fix_unaliased_order_by, extract_sql,
    canonicalize_enums, validate_answer, _apply_fixers,
)

NOW = datetime(2026, 9, 25, 12, 0, 0)

checks = []


def check(name, condition):
    checks.append((name, bool(condition)))


# --- fix_date_range ---
check(
    "fix_date_range injects missing WHERE for 'today'",
    "timestamp >= '2026-09-25T00:00:00'" in fix_date_range(
        "SELECT COUNT(*) FROM alerts;", "how many alerts today", NOW),
)
check(
    "fix_date_range corrects a wrong existing range",
    "2023-09-19" in fix_date_range(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2020-01-01T00:00:00' AND timestamp < '2020-01-02T00:00:00';",
        "how many alerts on september 19 2023", NOW),
)

# --- strip_unwanted_date_filter ---
check(
    "strip_unwanted_date_filter removes an unasked-for date clause",
    "timestamp" not in strip_unwanted_date_filter(
        "SELECT AVG(inspection_time) FROM alerts WHERE timestamp >= '2026-09-25T00:00:00' AND timestamp < '2026-09-26T00:00:00';",
        "what is the average inspection time", NOW),
)
check(
    "strip_unwanted_date_filter leaves a genuinely-dated question alone",
    "timestamp" in strip_unwanted_date_filter(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2026-09-25T00:00:00' AND timestamp < '2026-09-26T00:00:00';",
        "how many alerts today", NOW),
)

# --- fix_missing_normal_operation_exclusion ---
check(
    "fix_missing_normal_operation_exclusion adds exclusion",
    "NORMAL_OPERATION" in fix_missing_normal_operation_exclusion(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2026-01-01T00:00:00';", "how many alerts this year"),
)
check(
    "fix_missing_normal_operation_exclusion leaves a normal-operation question alone",
    "!=" not in fix_missing_normal_operation_exclusion(
        "SELECT COUNT(*) FROM alerts WHERE alert_type = 'NORMAL_OPERATION';", "how many normal operation events"),
)

# --- fix_or_and_precedence ---
check(
    "fix_or_and_precedence converts OR chain to IN",
    "IN (" in fix_or_and_precedence(
        "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH' OR alert_type = 'FAST_INSPECTION';"),
)

# --- fix_missing_group_by ---
check(
    "fix_missing_group_by adds GROUP BY when mixing agg + plain column",
    "GROUP BY" in fix_missing_group_by("SELECT alert_type, COUNT(*) FROM alerts;"),
)
check(
    "fix_missing_group_by is a no-op for a plain aggregate",
    "GROUP BY" not in fix_missing_group_by("SELECT COUNT(*) FROM alerts;"),
)

# --- fix_missing_superlative_limit ---
check(
    "fix_missing_superlative_limit adds ORDER BY + LIMIT for a real superlative",
    "LIMIT 1" in fix_missing_superlative_limit(
        "SELECT substr(timestamp,1,10) AS day, COUNT(*) AS c FROM alerts GROUP BY day;",
        "which day had the most alerts"),
)
check(
    "fix_missing_superlative_limit does NOT fire for a count-target question",
    "LIMIT" not in fix_missing_superlative_limit(
        "SELECT substr(timestamp,1,10) AS day, COUNT(*) AS c FROM alerts GROUP BY day;",
        "which days did we get 328 alerts"),
)

# --- fix_misclassified_count_target_ranking (the ported bugfix) ---
_wrong_shape = "SELECT substr(timestamp,1,10) AS day, COUNT(*) AS c FROM alerts WHERE alert_type != 'NORMAL_OPERATION' GROUP BY day ORDER BY c DESC LIMIT 1;"
_fixed = fix_misclassified_count_target_ranking(_wrong_shape, "which days did we get 328 alerts")
check("count-target fix removes ORDER BY/LIMIT busiest-day shape", "ORDER BY" not in _fixed and "LIMIT" not in _fixed)
check("count-target fix adds HAVING with the right target number", "HAVING" in _fixed and "328" in _fixed)
check(
    "count-target fix leaves a genuine superlative pipeline untouched",
    fix_misclassified_count_target_ranking(_wrong_shape, "which day had the most alerts") == _wrong_shape,
)

# --- fix_unaliased_order_by ---
check(
    "fix_unaliased_order_by adds a missing alias",
    "AS c" in fix_unaliased_order_by(
        "SELECT alert_type, COUNT(*) FROM alerts GROUP BY alert_type ORDER BY c DESC;"),
)
check(
    "fix_unaliased_order_by is a no-op when the alias already exists",
    fix_unaliased_order_by(
        "SELECT alert_type, COUNT(*) AS c FROM alerts GROUP BY alert_type ORDER BY c DESC;"
    ) == "SELECT alert_type, COUNT(*) AS c FROM alerts GROUP BY alert_type ORDER BY c DESC;",
)

# --- extract_sql / canonicalize_enums ---
check("extract_sql pulls SELECT out of prose/fences", extract_sql("```sql\nSELECT COUNT(*) FROM alerts;\n```") == "SELECT COUNT(*) FROM alerts;")
check("extract_sql returns None for UNSUPPORTED", extract_sql("UNSUPPORTED") is None)
check("canonicalize_enums fixes spaced enum values", "'HAND_TOUCH'" in canonicalize_enums("SELECT * FROM alerts WHERE alert_type = 'hand touch';"))

# --- validate_answer ---
check(
    "validate_answer accepts a correct single-value answer",
    validate_answer("There were 179 alerts.", ["c"], [(179,)])[0] == "There were 179 alerts.",
)
check(
    "validate_answer rejects a hallucinated number and falls back",
    validate_answer("There were 999 alerts.", ["c"], [(179,)])[1] is True,
)
check(
    "validate_answer catches singular/plural grammar mismatch",
    validate_answer("There were 1 alerts.", ["c"], [(1,)])[1] is True,
)

# --- end-to-end fixer chain, the exact reported bug scenario ---
_e2e = _apply_fixers(
    "SELECT substr(timestamp,1,10) AS day, COUNT(*) AS c FROM alerts GROUP BY day ORDER BY c DESC LIMIT 1;",
    "which days did we get 328 alerts", NOW,
)
check("end-to-end: count-target fix applies inside the full fixer chain", "HAVING" in _e2e and "LIMIT" not in _e2e)

passed = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n{passed}/{len(checks)} passed")
sys.exit(0 if passed == len(checks) else 1)
