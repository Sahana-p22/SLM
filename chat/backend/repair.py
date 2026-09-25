"""The reliability layer: turn a model's SQL string into something safe to
run, and flag pipelines that don't actually answer the question.

Ported approach from the Retail deployment's retail_llm/repair.py: everything
here is deterministic. Prompt instructions are advisory; code that inspects
and rewrites the output is not. This is the single biggest architectural
difference from slm-llama3b's original approach — that project only checked
for MongoDB execution errors and a fixed list of known bug patterns *after*
the fact; this module also runs a pre-execution semantic check (diagnose())
that catches "the SQL doesn't actually answer what was asked" (wrong table,
missing aggregation, wrong filter) BEFORE the query is ever run, and feeds a
precise, specific correction back to the model for a retry.
"""
import re

MAX_ROWS = 500

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|analyze)\b", re.I)
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max|group\s+by)\b", re.I)

ALERT_TYPES = ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING", "NORMAL_OPERATION"]
_ENUM_BY_NORM = {v.lower().replace("_", " "): v for v in ALERT_TYPES}

# keyword -> exact alert_type, for the "question names a type but SQL ignores it" check
TYPE_KEYWORDS = {
    "hand touch": "HAND_TOUCH", "hand-touch": "HAND_TOUCH",
    "fast inspection": "FAST_INSPECTION",
    "missing cleaning": "MISSING_CLEANING", "missing clean": "MISSING_CLEANING",
    "normal operation": "NORMAL_OPERATION",
}


def canonicalize_enums(sql: str) -> str:
    """Rewrite any quoted literal that matches a known alert_type up to case
    and underscore/space, the same guardrail as Retail's category
    canonicalisation — a small model mirrors the user's phrasing
    ("hand touch") instead of the real enum spelling ("HAND_TOUCH")."""
    def fix(m):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        norm = " ".join(raw.lower().replace("_", " ").split())
        real = _ENUM_BY_NORM.get(norm)
        return f"'{real}'" if real and real != raw else m.group(0)
    return re.sub(r"'([^']+)'|\"([^\"]+)\"", fix, sql)


class ValidationError(Exception):
    pass


def validate_sql(sql: str) -> str:
    """Raise ValidationError on anything not a single read-only SELECT.
    Returns the cleaned SQL (trailing ; stripped, LIMIT enforced)."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValidationError("empty SQL")
    if ";" in s:
        raise ValidationError("multiple statements")
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValidationError("not a SELECT")
    if _WRITE_KEYWORDS.search(s):
        raise ValidationError("write/DDL keyword present")
    if "alerts" not in low:
        raise ValidationError("query does not reference the alerts table")
    if not re.search(r"\blimit\s+\d+", low):
        if re.search(r"\bgroup\s+by\b", low) or not _AGG_RE.search(low):
            s = f"{s}\nLIMIT {MAX_ROWS}"
    else:
        s = re.sub(r"\blimit\s+(\d+)\b",
                   lambda m: f"LIMIT {min(int(m.group(1)), MAX_ROWS)}", s, flags=re.I)
    return s


_AGG_QUESTION_RE = re.compile(
    r"\b(how many|how much|total|average|avg|count|number of|per |each |"
    r"top \d+|most|least|highest|lowest|fastest|slowest|peak|busiest)\b", re.I)


def needs_aggregation(question: str, sql: str) -> bool:
    if not _AGG_QUESTION_RE.search(question):
        return False
    if re.search(r"\b(show me everything|list all|show all|history)\b", question, re.I):
        return False
    return not _AGG_RE.search(sql)


def missing_type_filter(question: str, sql: str):
    """The question names a specific alert type but the SQL never filters to it."""
    q, s = question.lower(), sql.lower()
    for kw, atype in TYPE_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", q) and atype.lower() not in s:
            return atype
    return None


def missing_normal_operation_exclusion(question: str, sql: str) -> bool:
    """A generic 'alerts' question with no specific type named must exclude
    NORMAL_OPERATION — this is the exact false-positive class the FQC
    benchmark's own dashboard/stats code already guards against; the LLM path
    needs the same guard, since a small model omits it more often than not."""
    q = question.lower()
    if re.search(r"\bnormal\s*operation\b", q):
        return False  # question is specifically about normal operation
    if not re.search(r"\balert", q):
        return False  # question isn't about "alerts" generically at all
    for kw in TYPE_KEYWORDS:
        if kw != "normal operation" and re.search(r"\b" + re.escape(kw) + r"\b", q):
            return False  # a specific non-normal type is named; that filter covers it
    low = sql.lower()
    return "normal_operation" not in low


_NUMBER_TARGET_RE = re.compile(
    r"\b(?:which|what|any)\s+day(?:s)?\b.{0,40}?\b(?:get|got|have|had|see|saw|"
    r"record(?:ed)?|reach(?:ed)?|hit)\b.{0,15}?\b(\d{1,7})\b.{0,10}\balerts?\b", re.I)
_SUPERLATIVE_RE = re.compile(
    r"\b(most|highest|busiest|maximum|max|top|least|lowest|fewest|minimum|min)\b", re.I)


def misclassified_count_target(question: str, sql: str):
    """The exact bug class fixed in slm-llama3b this session, ported here as
    a pre-execution check instead of a post-hoc pipeline rewrite: a question
    naming a specific target count ('which days did we get 328 alerts') must
    produce a HAVING COUNT(*) = N query, not a superlative ORDER BY/LIMIT 1
    one that silently ignores the number and returns the busiest day instead."""
    m = _NUMBER_TARGET_RE.search(question)
    if not m:
        return None
    if _SUPERLATIVE_RE.search(question):
        return None  # genuinely ambiguous phrasing; don't force it
    target = m.group(1)
    low = sql.lower()
    if f"= {target}" in low or f"={target}" in low:
        return None  # already has the right HAVING/WHERE comparison
    if re.search(r"\border\s+by\b.*\blimit\s+1\b", low, re.S):
        return (f"the question asks for the day(s) with EXACTLY {target} alerts (a specific "
                f"target count), but this SQL is a superlative 'busiest day' query (ORDER BY "
                f"... LIMIT 1) that ignores the number entirely — use GROUP BY date(timestamp) "
                f"HAVING COUNT(*) = {target} instead.")
    return None


def strip_unrequested_date_filter(sql: str) -> str:
    """Deterministic force-fix for unrequested_date_filter(): asking the
    model to 'drop that date filter' on retry was observed to NOT reliably
    work — it re-added the identical spurious filter across every retry
    attempt (e.g. 'how many hand touch alerts in total' kept getting a
    'today' filter appended, silently truncating an all-time total down to
    a single day's count). Once diagnose() has already confirmed the
    filter shouldn't be there, remove it in code rather than keep asking."""
    pattern = re.compile(
        r"\s+AND\s+timestamp\s*(?:>=|>|<=|<)\s*'[^']*'", re.I)
    fixed = pattern.sub("", sql)
    # a lone WHERE timestamp ... with nothing else (rare, alert_type filter
    # normally always present) -- fall back to removing the WHERE entirely.
    fixed = re.sub(r"\bWHERE\s+timestamp\s*(?:>=|>|<=|<)\s*'[^']*'\s*",
                   "", fixed, flags=re.I)
    return fixed


_RANGE_CMP_RE = re.compile(
    r"(timestamp\s*(>=|>))\s*'[^']*'|(timestamp\s*(<=|<))\s*'[^']*'", re.I)


def enforce_single_range(sql: str, start_iso: str, end_iso: str):
    """Deterministic force-fix: the model is given the correct date range as
    text ('Interpreted date range: ...') but was observed not reliably
    copying it verbatim -- e.g. asked for 'march 2024' (2024-03-01 ..
    2024-04-01) it sometimes wrote the end bound as 2024-04-02 instead,
    silently including an extra day. Rewrites the literal bounds of a
    single simple timestamp >= / < range to match the code-computed range
    exactly. Only touches SQL with exactly one lower-bound and one
    upper-bound comparison on `timestamp` (skips two-period UNION ALL
    comparisons, which use two different ranges on purpose and are left to
    the explicit dual-range prompting instead). Returns (sql, changed)."""
    lows = re.findall(r"timestamp\s*(>=|>)\s*'([^']*)'", sql, re.I)
    highs = re.findall(r"timestamp\s*(<=|<)\s*'([^']*)'", sql, re.I)
    if len(lows) != 1 or len(highs) != 1:
        return sql, False
    changed = False
    if lows[0][1] != start_iso:
        sql = re.sub(r"(timestamp\s*(?:>=|>)\s*)'[^']*'",
                     lambda m: f"{m.group(1)}'{start_iso}'", sql, count=1, flags=re.I)
        changed = True
    if highs[0][1] != end_iso:
        sql = re.sub(r"(timestamp\s*(?:<=|<)\s*)'[^']*'",
                     lambda m: f"{m.group(1)}'{end_iso}'", sql, count=1, flags=re.I)
        changed = True
    return sql, changed


def unrequested_date_filter(question: str, sql: str, has_range: bool):
    if has_range:
        return None
    q = question.lower()
    if re.search(r"\b(today|yesterday|this week|last week|this month|last month|"
                 r"this year|last year|this quarter|last quarter|since|between|"
                 r"from .+ to |on \d|in the last|past \d|\d{4})\b", q):
        return None
    if re.search(r"\btimestamp\s*(>=|<=|>|<|=)\s*'\d{4}-\d{2}-\d{2}", sql, re.I):
        return ("the question doesn't ask for a specific date/period, but the SQL "
                "filters by one anyway — drop that date filter.")
    return None


_HOUR_CMP_RE = re.compile(r"\bhour\s*(=|>=|>|<=|<)\s*(\d{1,2})\b", re.I)
_HOUR_BETWEEN_RE = re.compile(r"\bhour\s+between\s+(\d{1,2})\s+and\s+(\d{1,2})\b", re.I)


def _correct_between_bounds(start: int, end: int):
    """The BETWEEN-inclusive equivalent of this project's half-open
    hour convention (hour >= start AND hour < end): BETWEEN start AND
    end-1, for the ordinary (non-wraparound) case."""
    if start <= end:
        return (start, end - 1)
    return None  # overnight wraparound isn't expressible as a single BETWEEN


def wrong_hour_range(hour_range, sql: str):
    """question named an hour-of-day RANGE ('between 2pm and 4pm') but
    the SQL uses a single-hour equality or no hour filter at all -- found
    live: 'hour = 14' generated for '2pm and 4pm', silently dropping the
    '4pm' half of the range."""
    if not hour_range:
        return None
    start, end = hour_range
    low = sql.lower()
    if "hour" not in low:
        return (f"the question names an hour-of-day RANGE ({start}:00 to {end}:00), but the SQL "
                f"has no hour filter at all -- add hour >= {start} AND hour < {end} "
                f"(or, if {start} > {end}, an overnight wraparound: hour >= {start} OR hour < {end}).")
    cmps = _HOUR_CMP_RE.findall(sql)
    if len(cmps) == 1 and cmps[0][0] == "=":
        return (f"the question names an hour-of-day RANGE ({start}:00 to {end}:00), but the SQL "
                f"filters a single hour (hour = {cmps[0][1]}) instead of the whole range -- use "
                f"hour >= {start} AND hour < {end}.")
    bm = _HOUR_BETWEEN_RE.search(sql)
    if bm:
        got = (int(bm.group(1)), int(bm.group(2)))
        want = _correct_between_bounds(start, end)
        if want is not None and got != want:
            return (f"the question names an hour-of-day RANGE ({start}:00 to {end}:00, half-open -- "
                    f"i.e. up to but not including {end}:00), but the SQL says "
                    f"'hour BETWEEN {got[0]} AND {got[1]}' -- BETWEEN is inclusive on both ends, so this "
                    f"either misses or over-includes an hour. Use hour BETWEEN {want[0]} AND {want[1]}, "
                    f"or equivalently hour >= {start} AND hour < {end}.")
    return None


def enforce_hour_range(sql: str, start: int, end: int):
    """Deterministic force-fix, same rationale as enforce_single_range: a
    single 'hour = N' comparison is rewritten to the correct range in code
    rather than trusted to a retry. Only fires on exactly one hour
    comparison (skips anything more complex, e.g. an OR-based wraparound
    the model already wrote out itself). Returns (sql, changed)."""
    cmps = _HOUR_CMP_RE.findall(sql)
    replacement = (f"hour >= {start} AND hour < {end}" if start <= end
                  else f"(hour >= {start} OR hour < {end})")
    if len(cmps) == 1:
        return _HOUR_CMP_RE.sub(replacement, sql, count=1), True
    bm = _HOUR_BETWEEN_RE.search(sql)
    if bm:
        got = (int(bm.group(1)), int(bm.group(2)))
        want = _correct_between_bounds(start, end)
        if want is not None and got != want:
            return _HOUR_BETWEEN_RE.sub(replacement, sql, count=1), True
    return sql, False


def wrong_hour_grouping(question: str, sql: str):
    q = question.lower()
    low = sql.lower()
    if re.search(r"\bwhich hour\b|\bbusiest hour\b|\bpeak hour\b|\btime of day\b", q):
        if "hour" not in low:
            return "group by the existing `hour` column (0-23) — do not derive it with strftime."
    return None


_WEEKDAY_Q_RE = re.compile(
    r"\bday(?:s)?\s+of\s+(?:the\s+)?week\b|\bweekday(?:s)?\b", re.I)
_WEEKDAY_GROUP_OK_RE = re.compile(r"strftime\(\s*'%w'\s*,\s*timestamp\s*\)", re.I)
_DATE_GROUP_RE = re.compile(r"\bdate\(\s*timestamp\s*\)", re.I)
_WEEKDAY_CASE_EXPR = (
    "CASE strftime('%w', timestamp) WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' "
    "WHEN '2' THEN 'Tuesday' WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' "
    "WHEN '5' THEN 'Friday' ELSE 'Saturday' END"
)


def wrong_weekday_grouping(question: str, sql: str):
    """The question asks about the DAY OF THE WEEK (a weekday category --
    Monday/Tuesday/etc, aggregated across the whole dataset, 7 possible
    values) but the SQL groups by a specific calendar date instead --
    found live: 'which day of the week has the most hand touch alerts?'
    generated `GROUP BY date(timestamp)`, which returns one specific
    date's count, not an aggregate per weekday. Distinct from a genuine
    'which day' (a single calendar date) question, which correctly keeps
    grouping by date(timestamp) and is left untouched here."""
    if not _WEEKDAY_Q_RE.search(question):
        return None
    if _WEEKDAY_GROUP_OK_RE.search(sql):
        return None
    if _DATE_GROUP_RE.search(sql) or re.search(r"\bgroup\s+by\s+timestamp\b", sql, re.I):
        return ("the question asks about the DAY OF THE WEEK (a weekday category like Monday, "
                "aggregated across the whole dataset), not one specific calendar date -- group "
                "by strftime('%w', timestamp) (0=Sunday..6=Saturday), not date(timestamp).")
    return None


def enforce_weekday_grouping(question: str, sql: str):
    """Deterministic force-fix, same rationale as enforce_hour_range: when
    the question names the day of the week, a `GROUP BY date(timestamp)`
    (and its matching SELECT expression) is rewritten in code to a
    weekday-name CASE over strftime('%w', timestamp), so the query
    aggregates per weekday instead of returning a single specific date.
    Only fires when the question itself asks about the day of the week --
    a genuine 'which day' question is untouched. Returns (sql, changed)."""
    if not _WEEKDAY_Q_RE.search(question):
        return sql, False
    if not _DATE_GROUP_RE.search(sql):
        return sql, False
    return _DATE_GROUP_RE.sub(_WEEKDAY_CASE_EXPR, sql), True


def diagnose(question: str, sql: str, has_range: bool | None = None):
    """Return a human-readable problem string to feed back to the model, or
    None."""
    ct = missing_type_filter(question, sql)
    if ct:
        return f"The question is about '{ct}' alerts but the SQL never filters alert_type = '{ct}'."
    if missing_normal_operation_exclusion(question, sql):
        return ("The question asks about 'alerts' generically, but the SQL doesn't exclude "
                "NORMAL_OPERATION — add alert_type != 'NORMAL_OPERATION' (it's a compliant "
                "event, not an alert).")
    mc = misclassified_count_target(question, sql)
    if mc:
        return mc
    if needs_aggregation(question, sql):
        return ("This question asks for a computed figure (count/total/average/ranking) but "
                "the SQL only filters rows. Use COUNT/SUM/AVG and GROUP BY as needed.")
    hg = wrong_hour_grouping(question, sql)
    if hg:
        return hg
    wg = wrong_weekday_grouping(question, sql)
    if wg:
        return wg
    from .dates import extract_hour_range
    whr = wrong_hour_range(extract_hour_range(question), sql)
    if whr:
        return whr
    if has_range is None:
        from datetime import datetime, timezone
        from .dates import extract_range
        has_range = bool(extract_range(question, datetime.now(timezone.utc).replace(tzinfo=None)))
    udf = unrequested_date_filter(question, sql, has_range)
    if udf:
        return udf
    return None
