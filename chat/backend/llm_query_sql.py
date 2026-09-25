# chat/backend/llm_query_sql.py
#
# SQL-on-SQLite backend for the chat pipeline: Llama-3.2-3B-Instruct (GGUF,
# via llama.cpp, same model/runtime as llm_query.py) generating SQLite
# queries against the local `alerts.sqlite3` mirror (kept in sync with
# MongoDB by mongo_sqlite_sync.py) instead of the original MongoDB
# aggregation-pipeline path in llm_query.py. Wired into the same
# answer_question / answer_question_stream contract main.py expects, so
# main.py can import from here in place of llm_query with no other change.
#
# This module's SQL-generation prompt, fixers, and answer-validation
# machinery are adapted from a proven reference implementation
# (fqc-gpu/chat/backend/llm_query_sql.py, built for a different model/
# accelerator but the same alerts schema and question set), retargeted at
# this app's own model (Llama-3.2-3B-Instruct via llama.cpp) and its own
# persona/answer strings, and deliberately WITHOUT that reference's
# deterministic fast-path shortcuts — this app's Mongo version
# (llm_query.py) had all such shortcuts removed per an explicit prior user
# request ("every question now goes through the same LLM-generate ->
# validate -> repair pipeline uniformly, no shortcut bypasses it"), and
# this SQL version preserves that same design decision rather than quietly
# reintroducing them.
#
# Known, intentional scope reduction vs. llm_query.py's self-correction
# loop: the Mongo version chains up to two retries against a long list of
# Mongo-pipeline-JSON-specific heuristic checks (missing $group, $$NOW
# misuse, $first/$last-instead-of-group-by, etc.) — failure modes that are
# specific to writing a multi-stage aggregation pipeline as JSON. Since
# this version has the LLM write a single SQL SELECT statement instead,
# most of those failure shapes cannot occur structurally, so this module
# uses a single self-correction retry (feed the real SQLite error back,
# ask for a corrected query), matching the reference implementation's own
# proven design for a SQL target.
import os
import re
import threading
import time
import sqlite3
import difflib
from datetime import datetime, timedelta

from chat.backend.config import GGUF_MODEL_PATH

# --- Persona / canned answers (kept identical to llm_query.py's own —
# this migration changes the database engine and query language, not the
# app's voice) ---

UNSUPPORTED_ANSWER = (
    "I can only answer questions about the alert log — this data has alert_type, zone, cloth_detected, "
    "inspection_time, and timestamp for each alert. Could you rephrase your question in those terms?"
)

GAP_QUESTION_ANSWER = (
    "I can tell you counts and breakdowns for specific days, but I can't currently scan for gaps — "
    "days with zero alerts — across a whole date range. Try asking about a specific day instead, "
    "e.g. \"how many alerts on March 3rd?\""
)

GREETING_ANSWER = (
    "Hello! I'm your factory safety alert assistant — ask me anything about the alert log: counts, "
    "trends, breakdowns, top days, even a full report. E.g. \"which day had the most hand touch alerts?\" "
    "or \"give me a quarterly report\"."
)

ERROR_ANSWER = (
    "I wasn't able to answer that — the query it needed didn't run cleanly against the database. "
    "Could you try rephrasing the question, maybe with a simpler or more specific time range or category?"
)

FUTURE_DATE_TEMPLATE = "That's in the future — I don't have any data for {label} yet."

# --- Pre-LLM deterministic classification (ported verbatim from
# llm_query.py — pure text/regex logic, database-engine-agnostic) ---

_BARE_GREETING_RE = re.compile(
    r"^\s*(hi+|he+llo+|hey+a?|yo+|sup|howdy|good\s*(morning|afternoon|evening|day))"
    r"[\s,]*(deep\s*insight|assistant|there|bot)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _is_bare_greeting(question: str) -> bool:
    return bool(_BARE_GREETING_RE.match(question))


_DANGLING_TIME_PREP_RE = re.compile(r"\b(?:on|at|in|since|between|before|after|during)\s*\??\s*$", re.IGNORECASE)


def _is_incomplete_time_question(question: str) -> bool:
    return bool(_DANGLING_TIME_PREP_RE.search(question.strip()))


_DAY_WORD_RE = re.compile(r"\b(day|date)s?\b", re.IGNORECASE)
_ZERO_ALERTS_RE = re.compile(r"\b(no|zero|0)\s+alerts\b|\bwithout\s+(any\s+)?alerts\b", re.IGNORECASE)


def _is_gap_detection_question(question: str) -> bool:
    return bool(_DAY_WORD_RE.search(question) and _ZERO_ALERTS_RE.search(question))


_TYPO_VOCAB = [
    "yesterday", "today", "tomorrow", "morning", "afternoon", "evening", "night",
    "cleaning", "touch", "inspection", "missing", "normal", "operation",
    "weekly", "monthly", "yearly", "hourly", "quarterly", "quarter",
    "week", "month", "year", "hour", "minute", "day", "daily",
    "report", "average", "breakdown", "alerts", "alert", "compare",
    "distinct", "between", "zone", "station", "cloth", "detected",
]
_TYPO_VOCAB_SET = set(_TYPO_VOCAB)
_TYPO_WORD_RE = re.compile(r"[A-Za-z]+")
_spellchecker = None


def _get_spellchecker():
    global _spellchecker
    if _spellchecker is None:
        from spellchecker import SpellChecker
        _spellchecker = SpellChecker()
    return _spellchecker


def _normalize_common_typos(question: str) -> str:
    def fix_domain(m):
        word = m.group(0)
        lower = word.lower()
        if lower in _TYPO_VOCAB_SET or len(lower) < 4:
            return word
        matches = difflib.get_close_matches(lower, _TYPO_VOCAB, n=1, cutoff=0.8)
        if matches:
            corrected = matches[0]
            print(f"[llm_query_sql] typo-corrected {word!r} -> {corrected!r}")
            return corrected
        return word
    question = _TYPO_WORD_RE.sub(fix_domain, question)

    def fix_general(m):
        word = m.group(0)
        if not (word.islower() or (word[:1].isupper() and word[1:].islower())):
            return word
        if len(word) < 5:
            return word
        lower = word.lower()
        if lower in _TYPO_VOCAB_SET:
            return word
        try:
            sp = _get_spellchecker()
        except Exception:
            return word
        if lower in sp:
            return word
        correction = sp.correction(lower)
        if correction and correction != lower:
            fixed = correction.capitalize() if word[:1].isupper() else correction
            print(f"[llm_query_sql] general typo-corrected {word!r} -> {fixed!r}")
            return fixed
        return word
    return _TYPO_WORD_RE.sub(fix_general, question)


# --- Model loading (Llama-3.2-3B-Instruct via llama.cpp — ported from
# llm_query.py unchanged; this migration only changes what's asked of the
# model and what runs the resulting query, not the model or runtime) ---

_llm = None
_llm_lock = threading.Lock()


def _load_model():
    global _llm
    if _llm is not None:
        return
    with _llm_lock:
        if _llm is not None:
            return
        print("[llm_query_sql] Loading Llama-3.2-3B-Instruct (GGUF, Q8_0) via llama.cpp...")
        if hasattr(os, "add_dll_directory"):
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            os.add_dll_directory(torch_lib)
        from llama_cpp import Llama
        _llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        print("[llm_query_sql] Model ready.")


def _chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 500, sample: bool = False) -> str:
    _load_model()
    gen_kwargs = {"temperature": 0.4, "top_p": 0.9, "top_k": 50} if sample else {"temperature": 0.0}
    with _llm_lock:
        out = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            repeat_penalty=1.1,
            **gen_kwargs,
        )
    return (out["choices"][0]["message"]["content"] or "").strip()


def _chat_stream(system_prompt: str, user_prompt: str, max_new_tokens: int = 200):
    _load_model()
    with _llm_lock:
        stream = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            repeat_penalty=1.1,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                yield content


# --- SQLite connection (shared, matching the reference implementation's
# proven pattern — a single connection guarded only at creation time; read
# concurrency at this app's scale was measured error-free up to 8
# simultaneous users against the Mongo backend, and SQLite's own internal
# locking covers the read-only query pattern this module uses) ---

from chat.backend.db_sqlite import get_connection as _db


# --- SQL system prompt ---

def _sql_system_prompt(now: datetime) -> str:
    now_s = now.strftime("%Y-%m-%d")
    tomorrow_s = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    week_ago_s = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    return f"""You are a SQL query-writing assistant for a factory safety alert log.

Table: alerts
Columns: alert_type (TEXT, exactly one of: FAST_INSPECTION, HAND_TOUCH, MISSING_CLEANING, NORMAL_OPERATION),
timestamp (TEXT, ISO format like {now_s}T14:30:00), inspection_time (REAL, seconds),
zone (TEXT), cloth_detected (INTEGER 0 or 1).

Rules:
- alert_type values contain underscores, exactly as listed above. Never use spaces in them.
- Compare timestamps as strings using the ISO format. For "today"/a specific date/day, ALWAYS use a >= / < range covering that whole day — NEVER a single "=" comparison against one instant.
- Always add "alert_type != 'NORMAL_OPERATION'" for any general/total alert count or breakdown, unless the question explicitly asks about normal/compliant operation.
- "How many ..." always means COUNT(*) — never select raw rows/columns for a "how many" question.
- Only add GROUP BY if the question explicitly asks to break down/count/compare per some dimension. A plain total or filtered count needs NO GROUP BY.
- "Break down / count X by <dimension>" -> GROUP BY that exact dimension (alert_type, zone, or day), never a different one, and still keep the NORMAL_OPERATION exclusion from the rule above.
- "Which day/date had the most/least X" -> GROUP BY substr(timestamp,1,10), ORDER BY the count DESC (or ASC for least), LIMIT 1.
- "Which day(s) did we get/see/have/record/reach/hit N alerts" (a SPECIFIC number given) is NOT the same as "most/least" — GROUP BY substr(timestamp,1,10), then keep every group whose count equals N (HAVING COUNT(*) = N), do NOT sort by count and take only the top one.
- To match against MULTIPLE alert_type values (e.g. comparing two types), ALWAYS use "alert_type IN ('X', 'Y')" — NEVER chain "alert_type = 'X' OR alert_type = 'Y'". Mixing OR with an AND date filter without IN silently changes what the date filter applies to.
- "Compare X and Y counts" -> GROUP BY alert_type with alert_type IN ('X', 'Y'), so the result has one row per type, never a single combined COUNT(*).
- inspection_time is stored in SECONDS. Never convert it or imply another unit.
- If the question is unrelated to this alert log (small talk, unrelated topics), OR is too open-ended for one SQL query to answer (e.g. "summarize everything", "how are we doing"), output exactly: UNSUPPORTED
- Output ONLY one SQLite query, nothing else.

Example 1 (plain total, current date {now_s}):
Question: "how many alerts happened today"
Answer: SELECT COUNT(*) FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= '{now_s}T00:00:00' AND timestamp < '{tomorrow_s}T00:00:00';

Example 2 (breakdown by type):
Question: "break down alerts by type for the last 7 days"
Answer: SELECT alert_type, COUNT(*) FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= '{week_ago_s}T00:00:00' AND timestamp < '{tomorrow_s}T00:00:00' GROUP BY alert_type;

Example 3 (which day had the most):
Question: "which day had the most hand touch alerts"
Answer: SELECT substr(timestamp,1,10) AS day, COUNT(*) AS c FROM alerts WHERE alert_type = 'HAND_TOUCH' GROUP BY day ORDER BY c DESC LIMIT 1;

Example 4 (compare two types):
Question: "compare fast inspection and hand touch alert counts for today"
Answer: SELECT alert_type, COUNT(*) FROM alerts WHERE alert_type IN ('FAST_INSPECTION', 'HAND_TOUCH') AND timestamp >= '{now_s}T00:00:00' AND timestamp < '{tomorrow_s}T00:00:00' GROUP BY alert_type;

Example 5 (an explicit calendar date, not a relative phrase):
Question: "how many alerts happened on 30th July 2026"
Answer: SELECT COUNT(*) FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= '2026-07-30T00:00:00' AND timestamp < '2026-07-31T00:00:00';

Example 6 (a specific target count, not a superlative):
Question: "which days did we get 328 alerts"
Answer: SELECT substr(timestamp,1,10) AS day, COUNT(*) AS c FROM alerts WHERE alert_type != 'NORMAL_OPERATION' GROUP BY day HAVING c = 328;

Current date: {now_s}."""


ANSWER_SYSTEM_PROMPT = """You answer questions about a factory safety alert log using ONLY the exact \
SQL result rows given to you. Never invent numbers not present in the rows. You MUST state every \
single value from every row — never describe a row (e.g. a date, a type, a top result) without also \
stating its number; a "which/what had the most" answer is incomplete without the count. inspection_time \
values are ALWAYS in seconds — never call them minutes or convert them. Reply with one or two plain \
English sentences, no markdown, no code.

Example: row is day=2021-09-07, c=281
Answer: "September 7, 2021 had the most, with 281 alerts.\""""


# --- Date-range extraction / fixers (ported from the reference SQL
# implementation, which itself ported these from llm_query.py's own
# proven date-parsing logic, adapted to emit SQL literals instead of
# Mongo pipeline stages) ---

_HOUR_RE = re.compile(
    r"(?:between|from)\s+(\d{1,2})\s*(am|pm)?\s+(?:and|to)\s+(\d{1,2})\s*(am|pm)?",
    re.IGNORECASE,
)


def _to_24h(h, ampm):
    h = int(h)
    if ampm and ampm.lower() == "pm" and h != 12:
        h += 12
    if ampm and ampm.lower() == "am" and h == 12:
        h = 0
    return h


_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_NAME_RE = re.compile(r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\b", re.IGNORECASE)
_ABS_DATE_ISO_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b")
_ABS_DATE_MONTH_FIRST_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_ABS_DATE_DAY_FIRST_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(_MONTH_NAMES.keys()) + r")\.?,?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def _extract_absolute_date(question: str):
    m = _ABS_DATE_ISO_RE.search(question)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d)
        except ValueError:
            return None
    m = _ABS_DATE_MONTH_FIRST_RE.search(question)
    if m:
        try:
            return datetime(int(m.group(3)), _MONTH_NAMES[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            return None
    m = _ABS_DATE_DAY_FIRST_RE.search(question)
    if m:
        try:
            return datetime(int(m.group(3)), _MONTH_NAMES[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None
    return None


def extract_date_range(question: str, now: datetime):
    q = question.lower()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    abs_date = _extract_absolute_date(question)
    if abs_date is not None:
        day_start = abs_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
    elif "day after tomorrow" in q:
        day_start = today + timedelta(days=2)
        day_end = today + timedelta(days=3)
    elif "day before yesterday" in q:
        day_start = today - timedelta(days=2)
        day_end = today - timedelta(days=1)
    elif "yesterday" in q:
        day_start = today - timedelta(days=1)
        day_end = today
    elif "tomorrow" in q:
        day_start = today + timedelta(days=1)
        day_end = today + timedelta(days=2)
    elif "today" in q:
        day_start = today
        day_end = today + timedelta(days=1)
    else:
        day_start = day_end = None

    hour_match = _HOUR_RE.search(q)
    if day_start is not None and hour_match:
        h1 = _to_24h(hour_match.group(1), hour_match.group(2))
        h2 = _to_24h(hour_match.group(3), hour_match.group(4))
        return day_start + timedelta(hours=h1), day_start + timedelta(hours=h2)
    if day_start is not None:
        return day_start, day_end

    m = re.search(r"last\s+(\d+)\s+days?", q)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n), today + timedelta(days=1)

    if "this week" in q:
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=7)

    if "next week" in q:
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        return start, start + timedelta(days=7)

    return None


_SUPERLATIVE_RE = re.compile(
    r"\b(day|date|week|month)\b(?:\W+\w+){0,6}?\W+\b(most common|most|highest|busiest|peak|top|least|lowest|fewest)\b"
    r"|\b(most common|most|highest|busiest|peak|top|least|lowest|fewest)\b(?:\W+\w+){0,6}?\W+\b(day|date|week|month)\b",
    re.IGNORECASE,
)

_COUNT_TARGET_RE = re.compile(
    r"\b(which|what|any)\b(?:\W+\w+){0,3}?\W+\b(day|days|date|dates|week|weeks|month|months)\b"
    r"(?:\W+\w+){0,8}?\W+\b(get|got|have|had|see|saw|record(?:ed)?|reach(?:ed)?|hit)\b"
    r"(?:\W+\w+){0,6}?\W+\d+",
    re.IGNORECASE,
)


def fix_missing_superlative_limit(sql: str, question: str) -> str:
    """"Which day/date had the most/least X" needs ORDER BY + LIMIT 1 to
    actually answer "which one". Only applies to a genuine superlative
    question — a count-target question ("which days did we get 328
    alerts") is excluded here and handled instead by
    fix_misclassified_count_target_ranking below, since the two need
    opposite SQL shapes (LIMIT 1 vs. HAVING count = N)."""
    if not _SUPERLATIVE_RE.search(question) or _COUNT_TARGET_RE.search(question):
        return sql
    if re.search(r"\bORDER BY\b", sql, re.IGNORECASE):
        return sql
    if not re.search(r"\bGROUP BY\b", sql, re.IGNORECASE):
        return sql
    direction = "ASC" if re.search(r"\b(least|fewest|lowest)\b", question, re.IGNORECASE) else "DESC"
    m = re.search(r"COUNT\(\*\)\s+AS\s+(\w+)", sql, re.IGNORECASE)
    order_col = m.group(1) if m else "COUNT(*)"
    return sql.rstrip().rstrip(";") + f" ORDER BY {order_col} {direction} LIMIT 1;"


def fix_misclassified_count_target_ranking(sql: str, question: str) -> str:
    """Fixes the SQL equivalent of the bug found and fixed in llm_query.py
    (see _fix_misclassified_count_target_ranking there for the original
    Mongo write-up): "which days did we get 328 alerts" has a well-covered
    superlative worked example to pattern-match onto ("which day had the
    most alerts" -> GROUP BY + ORDER BY count DESC + LIMIT 1), and the
    model sometimes builds that exact shape here too, silently ignoring
    the specific number 328 in the question and returning the all-time
    busiest day instead — reproduced live against this exact model.
    If the question is a count-target search (not a genuine superlative)
    and the SQL has the busiest/least-shape tail (ORDER BY ... LIMIT 1,
    with or without a HAVING already present), rewrites it to keep every
    group matching the number actually named in the question, dropping
    the ORDER BY/LIMIT entirely — this deliberately can return more than
    one day, unlike a naive "just fix the top one" patch, since more than
    one day can share the same count."""
    if _SUPERLATIVE_RE.search(question) and not _COUNT_TARGET_RE.search(question):
        return sql
    if not _COUNT_TARGET_RE.search(question):
        return sql
    numbers = re.findall(r"\d+", question)
    if not numbers:
        return sql
    target = numbers[-1]

    if not re.search(r"\bGROUP BY\b", sql, re.IGNORECASE):
        return sql
    if re.search(r"\bHAVING\b", sql, re.IGNORECASE):
        # Already has a HAVING — just make sure it targets the right
        # number and isn't also carrying a stale ORDER BY/LIMIT tail.
        sql = re.sub(r"\s+ORDER BY\s+.*?(?=\bLIMIT\b|;|$)", " ", sql, flags=re.IGNORECASE | re.DOTALL)
        sql = re.sub(r"\s+LIMIT\s+\d+", "", sql, flags=re.IGNORECASE)
        return sql

    m = re.search(r"COUNT\(\*\)\s+AS\s+(\w+)", sql, re.IGNORECASE)
    count_alias = m.group(1) if m else None
    having_expr = f"{count_alias} = {target}" if count_alias else f"COUNT(*) = {target}"

    # Strip any ORDER BY .. LIMIT tail this shape came with, then append
    # HAVING right after the GROUP BY clause.
    sql = re.sub(r"\s+ORDER BY\s+.*?(?=\bLIMIT\b|;|$)", " ", sql, flags=re.IGNORECASE | re.DOTALL)
    sql = re.sub(r"\s+LIMIT\s+\d+", "", sql, flags=re.IGNORECASE)
    sql = sql.rstrip().rstrip(";")
    return sql + f" HAVING {having_expr};"


_AGG_FN_RE = re.compile(r"^\s*(COUNT|AVG|SUM|MAX|MIN)\s*\(", re.IGNORECASE)


def fix_missing_group_by(sql: str) -> str:
    if re.search(r"\bGROUP BY\b", sql, re.IGNORECASE):
        return sql
    m = re.search(r"SELECT\s+(.*?)\s+FROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql
    cols = [c.strip() for c in m.group(1).split(",")]
    plain_cols = [c for c in cols if not _AGG_FN_RE.match(c)]
    has_agg = any(_AGG_FN_RE.match(c) for c in cols)
    if not (has_agg and plain_cols):
        return sql
    group_by_cols = ", ".join(c.split(" AS ")[0].split(" as ")[0].strip() for c in plain_cols)
    if ";" in sql:
        sql = sql.rstrip()
        sql = sql[:-1] + f" GROUP BY {group_by_cols};"
    else:
        sql = sql.rstrip() + f" GROUP BY {group_by_cols}"
    return sql


_ORDER_BY_TARGET_RE = re.compile(r"\bORDER BY\s+(\w+)\b", re.IGNORECASE)
_UNALIASED_AGG_RE = re.compile(
    r"(COUNT\(\*\)|COUNT\([^)]*\)|SUM\([^)]*\)|AVG\([^)]*\)|MIN\([^)]*\)|MAX\([^)]*\))(?!\s+AS\s+\w+)",
    re.IGNORECASE,
)
_KNOWN_COLUMNS = {"alert_type", "timestamp", "inspection_time", "zone", "cloth_detected"}


def fix_unaliased_order_by(sql: str) -> str:
    m = _ORDER_BY_TARGET_RE.search(sql)
    if not m:
        return sql
    target = m.group(1)
    if target.upper() in ("ASC", "DESC") or target.lower() in _KNOWN_COLUMNS:
        return sql
    if re.search(rf"\bAS\s+{re.escape(target)}\b", sql, re.IGNORECASE):
        return sql
    aggs = _UNALIASED_AGG_RE.findall(sql)
    if len(aggs) != 1:
        return sql
    return sql.replace(aggs[0], f"{aggs[0]} AS {target}", 1)


def fix_missing_normal_operation_exclusion(sql: str, question: str) -> str:
    if re.search(r"normal.?operation|compliant", question, re.IGNORECASE):
        return sql
    if "NORMAL_OPERATION" in sql.upper():
        return sql
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if m:
        insert_at = m.end()
        return sql[:insert_at] + " alert_type != 'NORMAL_OPERATION' AND" + sql[insert_at:]
    m2 = re.search(r"\bGROUP BY\b", sql, re.IGNORECASE)
    if m2:
        return sql[:m2.start()] + "WHERE alert_type != 'NORMAL_OPERATION' " + sql[m2.start():]
    sql = sql.rstrip()
    if sql.endswith(";"):
        return sql[:-1] + " WHERE alert_type != 'NORMAL_OPERATION';"
    return sql + " WHERE alert_type != 'NORMAL_OPERATION'"


_OR_CHAIN_RE = re.compile(r"alert_type\s*=\s*'(\w+)'\s+OR\s+alert_type\s*=\s*'(\w+)'", re.IGNORECASE)


def fix_or_and_precedence(sql: str) -> str:
    m = _OR_CHAIN_RE.search(sql)
    if not m:
        return sql
    values = ", ".join(f"'{v}'" for v in m.groups())
    return _OR_CHAIN_RE.sub(f"alert_type IN ({values})", sql, count=1)


_GTE_RE = re.compile(r"timestamp\s*>=\s*'([^']+)'", re.IGNORECASE)
_LT_RE = re.compile(r"timestamp\s*<\s*'([^']+)'", re.IGNORECASE)


def fix_date_range(sql: str, question: str, now: datetime) -> str:
    computed = extract_date_range(question, now)
    if computed is None:
        return sql
    start, end = computed
    start_s = start.strftime("%Y-%m-%dT%H:%M:%S")
    end_s = end.strftime("%Y-%m-%dT%H:%M:%S")
    if _GTE_RE.search(sql) and _LT_RE.search(sql):
        sql = _GTE_RE.sub(f"timestamp >= '{start_s}'", sql, count=1)
        sql = _LT_RE.sub(f"timestamp < '{end_s}'", sql, count=1)
        return sql
    clause = f"timestamp >= '{start_s}' AND timestamp < '{end_s}'"
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if m:
        insert_at = m.end()
        return sql[:insert_at] + f" {clause} AND" + sql[insert_at:]
    m2 = re.search(r"\bGROUP BY\b|\bORDER BY\b|;|$", sql, re.IGNORECASE)
    return sql[:m2.start()].rstrip() + f" WHERE {clause} " + sql[m2.start():].lstrip()


_TS_CLAUSE_RE = re.compile(
    r"\s*(?:AND\s+)?timestamp\s*>=\s*'[^']+'\s*AND\s*timestamp\s*<\s*'[^']+'(?:\s*AND)?",
    re.IGNORECASE,
)


def strip_unwanted_date_filter(sql: str, question: str, now: datetime) -> str:
    if extract_date_range(question, now):
        return sql
    if not _TS_CLAUSE_RE.search(sql):
        return sql
    sql = _TS_CLAUSE_RE.sub(" ", sql, count=1)
    sql = re.sub(r"\bWHERE\s+AND\b", "WHERE", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bWHERE\s*(GROUP BY|ORDER BY|;|$)", r"\1", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


ENUMS = {"FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING", "NORMAL_OPERATION"}


def canonicalize_enums(sql: str) -> str:
    def fix(m):
        key = m.group(1).upper().replace(" ", "_")
        return f"'{key}'" if key in ENUMS else m.group(0)
    return re.sub(r"'([^']+)'", fix, sql)


def extract_sql(text: str) -> str | None:
    if "UNSUPPORTED" in text.upper() and "SELECT" not in text.upper():
        return None
    text = re.sub(r"```(?:sql)?", "", text)
    m = re.search(r"(SELECT\b.*?)(;|$)", text, re.S | re.I)
    if not m:
        return None
    sql = m.group(1).strip() + ";"
    return canonicalize_enums(sql)


# Fixers here take only `sql`; the rest need `question` too — kept as an
# explicit set (checked by identity) rather than inspecting each
# function's signature, so a future fixer's arity is a visible one-line
# decision here, not something that fails silently at call time.
_SQL_ONLY_FIXERS = {fix_or_and_precedence, fix_missing_group_by, fix_unaliased_order_by}
_SQL_AND_QUESTION_FIXERS = (
    fix_or_and_precedence,
    fix_missing_group_by,
    fix_missing_superlative_limit,
    fix_misclassified_count_target_ranking,
    fix_unaliased_order_by,
)


def _apply_fixers(sql: str, question: str, now: datetime) -> str:
    sql = fix_date_range(sql, question, now)
    sql = strip_unwanted_date_filter(sql, question, now)
    sql = fix_missing_normal_operation_exclusion(sql, question)
    for fixer in _SQL_AND_QUESTION_FIXERS:
        sql = fixer(sql) if fixer in _SQL_ONLY_FIXERS else fixer(sql, question)
    return sql


# --- Answer validation (deterministic safety net — ported from the
# reference implementation, same reasoning as llm_query.py's own) ---

def _row_numeric_values(rows: list[tuple]) -> list[float]:
    out = []
    for row in rows:
        for v in row:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append(float(v))
    return out


_NUM_TOKEN_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _answer_numbers(answer: str) -> list[float]:
    out = []
    for tok in _NUM_TOKEN_RE.findall(answer):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _value_confirmed(value: float, answer_numbers: list[float]) -> bool:
    tol = max(0.05, abs(value) * 0.01)
    return any(abs(n - value) <= tol for n in answer_numbers)


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def deterministic_answer(cols: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "No matching data found for that."
    if len(cols) <= 1 and len(rows) == 1:
        val = rows[0][0]
        if isinstance(val, int) and not isinstance(val, bool):
            return f"There were {_fmt_value(val)} alert{'' if val == 1 else 's'}."
        return f"The result is {_fmt_value(val)}."
    return "; ".join(f"{r[0]}: {r[-1]}" for r in rows) + "."


_COUNT_SENTENCE_RE = re.compile(r"\b(?:was|were)\s+([\d,]+)\s+(alert|alerts)\b", re.IGNORECASE)


def _grammar_ok(answer: str) -> bool:
    for m in _COUNT_SENTENCE_RE.finditer(answer):
        n = int(m.group(1).replace(",", ""))
        noun_is_plural = m.group(2).lower() == "alerts"
        if noun_is_plural == (n == 1):
            return False
    return True


def validate_answer(answer: str, cols: list[str], rows: list[tuple], fallback: str | None = None) -> tuple[str, bool]:
    fb = fallback if fallback is not None else deterministic_answer(cols, rows)
    if not answer.strip():
        return fb, True
    if not _grammar_ok(answer):
        return fb, True
    answer_numbers = _answer_numbers(answer)
    if len(rows) > 1:
        for row in rows:
            row_vals = _row_numeric_values([row])
            if row_vals and not any(_value_confirmed(v, answer_numbers) for v in row_vals):
                return fb, True
        return answer, False
    wanted = _row_numeric_values(rows)
    if wanted and not any(_value_confirmed(v, answer_numbers) for v in wanted):
        return fb, True
    return answer, False


def _stage(stages: list, name: str, description: str, start_time: float) -> None:
    stages.append({
        "name": name,
        "description": description,
        "duration_ms": round((time.time() - start_time) * 1000),
    })


def _regenerate_sql_after_error(question: str, failed_sql: str, error_msg: str) -> str | None:
    prompt = (
        f"This query was generated for the question \"{question}\", but SQLite rejected it:\n"
        f"{failed_sql}\n\nSQLite error: {error_msg}\n\n"
        f"Output a corrected SQLite query that fixes this error and still answers the question."
    )
    raw = _chat(_sql_system_prompt(datetime.now()), prompt, max_new_tokens=500, sample=True)
    print(f"[llm_query_sql] retry raw sql output: {raw!r}")
    return extract_sql(raw)


def _resolve_query(question: str):
    """Generates SQL, applies fixers, executes it, self-corrects once on a
    real SQLite error. Returns (sql, cols, rows, intent, stages,
    fast_answer)."""
    stages: list = []
    now = datetime.now()

    if _is_incomplete_time_question(question):
        return None, [], [], "unsupported", stages, None
    if _is_gap_detection_question(question):
        return None, [], [], "gap_unsupported", stages, None

    future_range = extract_date_range(question, now)
    if future_range is not None and future_range[0] >= now:
        future_label = future_range[0].strftime("%B %-d, %Y")
        fast_answer = FUTURE_DATE_TEMPLATE.format(label=future_label)
        return None, [], [], "future_date", stages, fast_answer

    t0 = time.time()
    raw = _chat(_sql_system_prompt(now), question, max_new_tokens=500)
    print(f"[llm_query_sql] raw sql output: {raw!r}")
    _stage(stages, "Query generation",
           "Asked the language model to translate the question into a SQLite query (includes "
           "structural fixers for date ranges, GROUP BY, filter precedence).", t0)

    sql = extract_sql(raw)
    if sql is None:
        return None, [], [], "unsupported", stages, None

    t1 = time.time()
    sql = _apply_fixers(sql, question, now)

    try:
        cur = _db().execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        intent = "data_query"
    except Exception as exc:
        print(f"[llm_query_sql] SQL execution failed: {exc}\n  sql: {sql}")
        retry_sql = _regenerate_sql_after_error(question, sql, str(exc))
        if retry_sql is not None:
            retry_sql = _apply_fixers(retry_sql, question, now)
            try:
                cur = _db().execute(retry_sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
                intent = "data_query"
                sql = retry_sql
                _stage(stages, "Self-correction retry",
                       "The first query failed against the database — fed the error back to the "
                       "model and got a corrected query that ran successfully.", t1)
            except Exception as exc2:
                print(f"[llm_query_sql] retry SQL execution also failed: {exc2}\n  sql: {retry_sql}")
                cols, rows, intent = [], [], "error"
        else:
            cols, rows, intent = [], [], "error"
    _stage(stages, "Database execution", "Ran the generated query against the alerts database.", t1)

    return sql, cols, rows, intent, stages, None


def _rows_as_dicts(cols: list[str], rows: list[tuple]) -> list[dict]:
    if not cols:
        cols = [f"col{i}" for i in range(len(rows[0]))] if rows else []
    return [dict(zip(cols, r)) for r in rows]


def _answer_prompt(question: str, cols: list[str], rows: list[tuple]) -> str:
    if len(cols) >= 2:
        lines = [", ".join(f"{c}={_fmt_value(v)}" for c, v in zip(cols, r)) for r in rows]
    else:
        lines = [_fmt_value(r[0]) for r in rows]
    rows_text = "\n".join(lines) if lines else "(no rows)"
    return f'Question: "{question}"\nSQL result rows:\n{rows_text}\nAnswer in plain English:'


def answer_question(question: str, history: list | None = None) -> dict:
    """`history` is accepted for main.py's call-signature compatibility.
    Matching llm_query.py's own recent decision not to feed conversation
    history into either generation step for this model (found to actively
    hurt output quality — see the Mongo backend's own comments on this),
    every question here is answered fully independently too."""
    try:
        return _answer_question_inner(question)
    except Exception as exc:
        print(f"[llm_query_sql] unhandled exception answering question {question!r}: {exc!r}")
        return {
            "question": question, "intent": "error", "pipeline": None,
            "explanation": None, "result": [], "row_count": 0,
            "answer": ERROR_ANSWER, "stages": [],
        }


def _answer_question_inner(question: str) -> dict:
    question = _normalize_common_typos(question)

    if _is_bare_greeting(question):
        return {
            "question": question, "intent": "greeting", "pipeline": None,
            "explanation": None, "result": [], "row_count": 0,
            "answer": GREETING_ANSWER, "stages": [],
        }

    sql, cols, rows, intent, stages, fast_answer = _resolve_query(question)

    if intent == "unsupported":
        return {
            "question": question, "intent": "unsupported", "pipeline": None,
            "explanation": None, "result": [], "row_count": 0,
            "answer": UNSUPPORTED_ANSWER, "stages": stages,
        }
    if intent == "gap_unsupported":
        return {
            "question": question, "intent": "gap_unsupported", "pipeline": None,
            "explanation": None, "result": [], "row_count": 0,
            "answer": GAP_QUESTION_ANSWER, "stages": stages,
        }
    if intent == "future_date":
        return {
            "question": question, "intent": "future_date", "pipeline": None,
            "explanation": None, "result": [], "row_count": 0,
            "answer": fast_answer, "stages": stages,
        }
    if intent == "error":
        return {
            "question": question, "intent": "error", "pipeline": sql,
            "explanation": None, "result": [], "row_count": 0,
            "answer": ERROR_ANSWER, "stages": stages,
        }
    if not rows:
        return {
            "question": question, "intent": intent, "pipeline": sql,
            "explanation": None, "result": [], "row_count": 0,
            "answer": "No matching data found for that.", "stages": stages,
        }

    t0 = time.time()
    prompt = _answer_prompt(question, cols, rows)
    raw_answer = _chat(ANSWER_SYSTEM_PROMPT, prompt, max_new_tokens=200)
    if not raw_answer.strip():
        raw_answer = _chat(ANSWER_SYSTEM_PROMPT, prompt, max_new_tokens=200, sample=True)
    print(f"[llm_query_sql] raw answer output: {raw_answer!r}")
    answer, _corrected = validate_answer(raw_answer.strip(), cols, rows)
    _stage(stages, "Answer generation",
           "Asked the language model to phrase the raw result rows as a natural-language answer, then "
           "verified every number in it against the real SQL result (deterministic safety net).", t0)

    return {
        "question": question, "intent": intent, "pipeline": sql,
        "explanation": None, "result": _rows_as_dicts(cols, rows), "row_count": len(rows),
        "answer": answer, "stages": stages,
    }


def answer_question_stream(question: str, history: list | None = None):
    question = _normalize_common_typos(question)

    if _is_bare_greeting(question):
        meta = {
            "question": question, "intent": "greeting", "pipeline": None,
            "explanation": None, "result": [], "row_count": 0, "stages": [],
        }
        yield "meta", meta
        yield "token", {"text": GREETING_ANSWER}
        yield "done", {"answer": GREETING_ANSWER, "stages": []}
        return

    sql, cols, rows, intent, stages, fast_answer = _resolve_query(question)
    result_dicts = _rows_as_dicts(cols, rows) if rows else []

    meta = {
        "question": question, "intent": intent, "pipeline": sql,
        "explanation": None, "result": result_dicts, "row_count": len(rows),
        "stages": stages,
    }
    yield "meta", meta

    if intent == "unsupported":
        yield "done", {"answer": UNSUPPORTED_ANSWER, "stages": stages}
        return
    if intent == "gap_unsupported":
        yield "done", {"answer": GAP_QUESTION_ANSWER, "stages": stages}
        return
    if intent == "future_date":
        yield "token", {"text": fast_answer}
        yield "done", {"answer": fast_answer, "stages": stages}
        return
    if intent == "error":
        print(f"[llm_query_sql] returning generic error answer to user")
        yield "done", {"answer": ERROR_ANSWER, "stages": stages}
        return
    if not rows:
        yield "done", {"answer": "No matching data found for that.", "stages": stages}
        return

    t0 = time.time()
    prompt = _answer_prompt(question, cols, rows)
    full_text = ""
    for delta in _chat_stream(ANSWER_SYSTEM_PROMPT, prompt, max_new_tokens=200):
        full_text += delta
        yield "token", {"text": delta}
    answer, corrected = validate_answer(full_text.strip(), cols, rows)
    _stage(stages, "Answer generation",
           "Asked the language model to phrase the raw result rows as a natural-language answer "
           "(streamed live), then verified every number in it against the real SQL result — a "
           "corrected replacement is sent here if the streamed text didn't check out.", t0)
    yield "done", {"answer": answer, "stages": stages}
