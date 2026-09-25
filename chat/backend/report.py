"""Deterministic, multi-section alert reports for any period granularity —
day, week, month, quarter or year. Ported from the Retail deployment's
retail_llm/report.py (same technique: several targeted SQL queries composed
into one plain-Python summary, no model call needed for the numbers), but
producing the 5-section shape (total_alerts, by_type, by_month/week/day,
avg_inspection_time) that slm-llama3b's own MongoDB $facet report pipeline
established as this project's expected report content (see llm_query.py's
QUERY_SYSTEM_PROMPT worked example and _restructure_report/_report_scope).

Design note: slm-llama3b's Mongo version defaults an unqualified "quarter"
report to a trailing-89-day window with no period label. This port instead
uses Retail's calendar-aligned quarter/month/week boundaries with an explicit
label ("Q1 2026", "March 2026", ...) — clearer for the reader and just as
deterministic; a report that explicitly names a period ("report for June",
"last quarter's report") always uses THAT period regardless of this default.
"""
import re
from datetime import datetime, timedelta

from .dates import _add_month, _day_start, _month_start, _week_start, extract_range
from .db import run_readonly

_REPORT_RE = re.compile(r"\breport\b|\bsummary\b|\bday\s*wise\b", re.I)
_KIND_PATTERNS = [
    ("week", re.compile(r"\bweekly\b|\bper week\b|\bweek\b", re.I)),
    ("quarter", re.compile(r"\bquarterly\b|\bper quarter\b|\bquarter\b", re.I)),
    ("year", re.compile(r"\byearly\b|\bannual(ly)?\b|\bper year\b|\byear\b", re.I)),
    ("month", re.compile(r"\bmonthly\b|\bper month\b|\bmonth\b", re.I)),
]

_TYPE_LABELS = {
    "FAST_INSPECTION": "fast inspection",
    "HAND_TOUCH": "hand touch",
    "MISSING_CLEANING": "missing cleaning",
}


def is_report_request(question: str) -> bool:
    return bool(_REPORT_RE.search(question or ""))


def _kind_from_text(question: str) -> str | None:
    for name, rx in _KIND_PATTERNS:
        if rx.search(question):
            return name
    return None


def _kind_from_span(a: datetime, b: datetime) -> str:
    span_days = (b - a).days
    if span_days <= 10:
        return "week"
    if span_days <= 40:
        return "month"
    if span_days <= 120:
        return "quarter"
    return "year"


def _default_range(kind, now):
    if kind == "week":
        s = _week_start(now)
        return (s, s + timedelta(days=7))
    if kind == "quarter":
        q = (now.month - 1) // 3
        s = datetime(now.year, q * 3 + 1, 1)
        return (s, _add_month(s, 3))
    if kind == "year":
        return (datetime(now.year, 1, 1), datetime(now.year + 1, 1, 1))
    s = _month_start(now)
    return (s, _add_month(s, 1))


def _previous_range(rng, kind):
    s, e = rng
    if kind == "month":
        return (_add_month(s, -1), s)
    if kind == "quarter":
        return (_add_month(s, -3), s)
    if kind == "year":
        return (datetime(s.year - 1, 1, 1), datetime(s.year, 1, 1))
    dur = e - s
    return (s - dur, s)


def _period_label(kind, a, b):
    if kind == "year":
        return str(a.year)
    if kind == "month":
        return a.strftime("%B %Y")
    if kind == "quarter":
        q = (a.month - 1) // 3 + 1
        return f"Q{q} {a.year}"
    if kind == "week":
        return f"{a.strftime('%d %b %Y')} – {(b - timedelta(days=1)).strftime('%d %b %Y')}"
    return a.strftime("%d %b %Y")


def _week_label(year, month, week_num):
    # Matches slm-llama3b's own _week_label exactly: includes the day-range
    # suffix (e.g. "July Week 1 (07/01-07/07)"), clipped to the month's
    # real last day for a trailing partial week.
    from calendar import monthrange
    start_day = (week_num - 1) * 7 + 1
    end_day = min(start_day + 6, monthrange(year, month)[1])
    month_name = datetime(year, month, 1).strftime('%B')
    return f"{month_name} Week {week_num} ({month:02d}/{start_day:02d}-{month:02d}/{end_day:02d})"


def _one(sql, params=()):
    rows = run_readonly(sql, params)
    return rows[0] if rows else {}


def build(question: str, now):
    """-> dict(answer, sql, result, source), or None if not a report request."""
    if not is_report_request(question):
        return None

    rng = extract_range(question, now)
    kind = _kind_from_text(question)
    if rng and kind is None:
        kind = _kind_from_span(*rng)
    elif kind is None:
        kind = "quarter"
    if not rng:
        rng = _default_range(kind, now)
    a, b = rng
    prev_a, prev_b = _previous_range(rng, kind)
    a_iso, b_iso = a.isoformat(sep=" "), b.isoformat(sep=" ")
    pa_iso, pb_iso = prev_a.isoformat(sep=" "), prev_b.isoformat(sep=" ")

    total = _one(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
        "AND timestamp >= ? AND timestamp < ?", (a_iso, b_iso))
    prev_total = _one(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
        "AND timestamp >= ? AND timestamp < ?", (pa_iso, pb_iso))
    avg_row = _one(
        "SELECT ROUND(AVG(inspection_time), 2) AS avg FROM alerts "
        "WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ?",
        (a_iso, b_iso))

    by_type = run_readonly(
        "SELECT alert_type, COUNT(*) AS count FROM alerts "
        "WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ? "
        "GROUP BY alert_type ORDER BY count DESC", (a_iso, b_iso))

    by_month = by_week = by_day = []
    if kind in ("quarter", "year"):
        by_month = run_readonly(
            "SELECT strftime('%Y-%m', timestamp) AS month, COUNT(*) AS count FROM alerts "
            "WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ? "
            "GROUP BY month ORDER BY month", (a_iso, b_iso))
        day_rows = run_readonly(
            "SELECT date(timestamp) AS d, COUNT(*) AS count FROM alerts "
            "WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ? "
            "GROUP BY d", (a_iso, b_iso))
        by_week = _rollup_weeks(day_rows)
    elif kind == "month":
        day_type_rows = run_readonly(
            "SELECT date(timestamp) AS d, alert_type, COUNT(*) AS count FROM alerts "
            "WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ? "
            "GROUP BY d, alert_type", (a_iso, b_iso))
        by_week = _rollup_weeks_by_type(day_type_rows)
    else:  # week
        by_day = run_readonly(
            "SELECT date(timestamp) AS date, COUNT(*) AS count FROM alerts "
            "WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ? "
            "GROUP BY date ORDER BY date", (a_iso, b_iso))

    label = _period_label(kind, a, b)
    total_n = total.get("n") or 0
    prev_n = prev_total.get("n") or 0
    avg_time = avg_row.get("avg")

    growth = ""
    if prev_n:
        pct = (total_n - prev_n) / prev_n * 100
        direction = "up" if pct >= 0 else "down"
        growth = f" That is {direction} {abs(pct):.0f}% from the previous {kind}'s {prev_n:,} alerts."

    type_parts = []
    for row in by_type:
        n = row.get("count", 0)
        name = _TYPE_LABELS.get(row.get("alert_type"), str(row.get("alert_type")).lower())
        type_parts.append(f"{n:,} {name}")
    type_summary = ""
    if type_parts:
        if len(type_parts) == 1:
            type_summary = f" — {type_parts[0]}"
        else:
            type_summary = f" — {', '.join(type_parts[:-1])}, and {type_parts[-1]}"

    avg_summary = f", averaging {avg_time:.2f}s per inspection" if avg_time is not None else ""

    lines = [f"{label} had {total_n:,} alerts in total{type_summary}{avg_summary}.{growth}"]

    if kind == "week" and by_day:
        busiest = max(by_day, key=lambda r: r.get("count", 0))
        lines.append(f"Busiest day: {busiest['date']} ({busiest['count']:,} alerts).")
    if kind in ("month",) and by_week:
        week_totals: dict[str, int] = {}
        for r in by_week:
            week_totals[r["week"]] = week_totals.get(r["week"], 0) + r["count"]
        if week_totals:
            busiest_week = max(week_totals.items(), key=lambda kv: kv[1])
            lines.append(f"Busiest week: {busiest_week[0]} ({busiest_week[1]:,} alerts).")
    if kind in ("quarter", "year") and by_month:
        busiest_month = max(by_month, key=lambda r: r.get("count", 0))
        lines.append(f"Busiest month: {busiest_month['month']} ({busiest_month['count']:,} alerts).")

    # Matches slm-llama3b's own _restructure_report exactly: only the
    # sections that apply to this scope are present at all (no empty
    # by_month/by_week/by_day placeholders for scopes that don't use them).
    facets = {"total_alerts": total_n, "by_type": by_type}
    if kind == "week":
        facets["by_day"] = by_day
    elif kind == "month":
        facets["by_week"] = by_week
    else:
        facets["by_month"] = by_month
        facets["by_week"] = by_week
    facets["avg_inspection_time"] = avg_time
    return {
        "answer": "\n".join(lines),
        "sql": (f"-- report for {label}: total_alerts, by_type, "
                f"{'by_day' if kind == 'week' else 'by_week' if kind == 'month' else 'by_month + by_week'}, "
                f"avg_inspection_time"),
        "result": [facets],
        "source": "report",
    }


def _rollup_weeks(day_rows: list) -> list:
    """day rows {"d": "YYYY-MM-DD", "count": n} -> week totals, no type split."""
    totals: dict[tuple, int] = {}
    for row in day_rows:
        d_str, count = row.get("d"), row.get("count")
        if not d_str or count is None:
            continue
        d = datetime.strptime(d_str, "%Y-%m-%d")
        week_num = ((d.day - 1) // 7) + 1
        key = (d.year, d.month, week_num)
        totals[key] = totals.get(key, 0) + count
    return [{"week": _week_label(y, m, w), "count": c}
            for (y, m, w), c in sorted(totals.items())]


def _rollup_weeks_by_type(day_type_rows: list) -> list:
    """day+type rows -> one row per (week, type) pair, e.g. {"week": "...",
    "type": "FAST_INSPECTION", "count": 42} — mirrors slm-llama3b's Mongo
    _compute_week_type_rollup for a monthly report's per-week type split."""
    totals: dict[tuple, int] = {}
    for row in day_type_rows:
        d_str, alert_type, count = row.get("d"), row.get("alert_type"), row.get("count")
        if not d_str or not alert_type or count is None:
            continue
        d = datetime.strptime(d_str, "%Y-%m-%d")
        week_num = ((d.day - 1) // 7) + 1
        key = (d.year, d.month, week_num, alert_type)
        totals[key] = totals.get(key, 0) + count
    return [{"week": _week_label(y, m, w), "type": t, "count": c}
            for (y, m, w, t), c in sorted(totals.items())]
