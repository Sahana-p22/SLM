"""Turn result rows into an English sentence — deterministically for fast
paths, and as a safety net over the model's phrasing for the LLM path.

Ported approach from the Retail deployment's retail_llm/phrase.py. Prompt
instructions don't reliably stop a 3B model from emitting a bare "7",
inventing a number, or claiming "no data" over a non-empty result — so every
model answer is checked here and replaced with a deterministic sentence when
it fails. This whole module is one of the two biggest differences from
slm-llama3b's original approach (the other being repair.py's pre-execution
diagnose()): slm-llama3b had a narrower NO_DATA_CLAIM_RE-only safety net,
while this also catches a bare/non-answer, a generic non-referencing answer,
and an invented number not present in any row.
"""
import re

_NO_DATA_RE = re.compile(r"no (matching )?data|no results|nothing (was )?found|"
                         r"couldn'?t find|there (are|were) no", re.I)

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2})")


def _fmt(key, v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        v = round(v, 2)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) >= 1000:
        return f"{v:,}"
    return str(v)


def _humankey(k):
    return k.replace("_", " ")


def preformat_for_prompt(rows: list) -> list:
    """Render timestamps as strings so the model copies them verbatim
    instead of re-deriving (and mangling) them. Only feeds the phrasing
    prompt — the API's result table keeps raw values."""
    rows = _clean_result_keys(rows)
    out = []
    for row in rows:
        nr = {}
        for k, v in row.items():
            if isinstance(v, str) and _ISO_RE.match(v):
                try:
                    from datetime import datetime
                    nr[k] = datetime.fromisoformat(v).strftime("%d %b %Y %H:%M")
                except ValueError:
                    nr[k] = v
            else:
                nr[k] = v
        out.append(nr)
    return out


_SQLEXPR_RE = re.compile(r"[()]|^\s*(count|sum|avg|min|max|round|total)\b", re.I)


def _clean_result_keys(rows: list) -> list:
    """The small model sometimes leaves an aggregate unaliased, so a row key
    is a raw SQL expression like 'COUNT(*)'. Rename such keys to a readable
    word before it reaches the phrasing step."""
    def clean(k):
        if not _SQLEXPR_RE.search(k):
            return k
        kl = k.lower()
        if "count" in kl:
            return "count"
        if "avg" in kl:
            return "average"
        if "sum" in kl or "total" in kl or "round" in kl:
            return "total"
        return "value"
    out = []
    for row in rows:
        seen, nr = set(), {}
        for k, v in row.items():
            nk = clean(k)
            while nk in seen:
                nk += "_2"
            seen.add(nk)
            nr[nk] = v
        out.append(nr)
    return out


_HOWMANY_TYPES_RE = re.compile(
    r"\bhow many\s+(?:types?|kinds?|different|distinct|unique)\s+(?:of\s+)?alerts?\b"
    r"|\b(?:list|what|which)\b.*\b(?:types?|kinds?)\s+of\s+alerts?\b", re.I)


def rows_to_sentence(rows: list, question: str = "") -> str:
    if not rows:
        return "Nothing matched that query."
    rows = _clean_result_keys(rows)

    # "how many types of alerts are there" — the answer is the ROW COUNT, and
    # a small model keeps grabbing a stray value (or the count of the most
    # common type) instead. This is the exact regression documented against
    # slm-llama3b's original approach ("distinct alert types" misanswered).
    if _HOWMANY_TYPES_RE.search(question) and any(
            "alert_type" in r or "type" in r for r in rows):
        key = "alert_type" if "alert_type" in rows[0] else next(
            k for k in rows[0] if "type" in k.lower())
        names = [str(r[key]) for r in rows[:12]]
        noun = "type" if len(rows) == 1 else "types"
        return f"{len(rows)} {noun}: {', '.join(names)}."

    if len(rows) == 1:
        row = rows[0]
        parts = [f"{_humankey(k)}: {_fmt(k, v)}" for k, v in row.items()]
        return _capitalize("; ".join(parts) + ".")

    # a "this period vs previous period" comparison (period column = the label)
    if len(rows) == 2 and any("period" in k.lower() for k in rows[0]):
        pk = next(k for k in rows[0] if "period" in k.lower())
        vk = next((k for k in rows[0] if k != pk and isinstance(rows[0][k], (int, float))
                   and not isinstance(rows[0][k], bool)), None)
        if vk:
            cur, prev = rows[0], rows[1]
            cv, pv = cur.get(vk) or 0, prev.get(vk) or 0
            delta = ((cv - pv) / pv * 100) if pv else 0
            arrow = "up" if cv >= pv else "down"
            return (f"{_fmt(vk, cv)} this period vs {_fmt(vk, pv)} the period before "
                    f"({arrow} {abs(delta):.0f}%).")

    # multi-row: "<label> (<value>), ..." for the leading rows.
    keys = list(rows[0].keys())
    label_key = next((k for k in keys if isinstance(rows[0][k], str)), keys[0])
    _num = [k for k in keys if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
    _METRIC = re.compile(r"count|total|alerts?|seconds?|avg|average|hour|day", re.I)
    value_key = next((k for k in _num if _METRIC.search(k)), None) or (_num[0] if _num else None)
    shown = rows[:5]
    if value_key:
        items = "; ".join(f"{r.get(label_key, '?')} ({_fmt(value_key, r.get(value_key))})"
                          for r in shown)
        metric = _humankey(value_key)
        if len(rows) > len(shown):
            return f"Top {len(shown)} of {len(rows)} by {metric} — {items}."
        return f"By {metric}: {items}."
    items = "; ".join(str(r.get(label_key, "?")) for r in shown)
    return f"{len(rows)} results: {items}" + ("…" if len(rows) > len(shown) else ".")


def _capitalize(s):
    return s[:1].upper() + s[1:] if s else s


def _is_bare(answer: str) -> bool:
    return len(re.findall(r"[A-Za-z]{2,}", answer)) < 2


def _looks_generic(answer: str, rows: list) -> bool:
    """A last-ditch check for a non-answer: the model returned prose that
    references none of the result at all. Deliberately lenient — it only
    fires when the answer shares *nothing* with the rows."""
    a = answer.lower()
    has_number_row = has_string_row = False
    for row in rows[:5]:
        for v in row.values():
            if v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                has_number_row = True
            elif isinstance(v, str) and len(v) > 2:
                has_string_row = True
                if v.lower() in a:
                    return False
                if any(w for w in re.findall(r"[a-z]{4,}", v.lower()) if w in a):
                    return False
    if has_number_row and re.search(r"\d", answer):
        return False
    if not has_number_row and not has_string_row:
        return False
    return True


def _invents_big_number(answer: str, rows: list) -> bool:
    """The model computing its own cross-row total (e.g. summing several
    per-day counts into a grand total in prose) — that figure appears in no
    row and is usually wrong. Only flags comma-grouped tokens (">=1000 with
    a comma"), so a year, an hour, or a bare count never trips this."""
    present = set()
    for row in rows:
        for v in row.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                present.add(round(v))
    nums = [round(v) for row in rows for v in row.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if 2 <= len(nums) <= 6:
        present.add(sum(nums))
    for tok in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", answer):
        n = round(float(tok.replace(",", "")))
        if n >= 1000 and not any(abs(n - p) <= 2 for p in present):
            return True
    return False


def _dedupe_sentences(answer: str) -> str:
    """Small models (esp. Llama-3.2-3B) pad an answer with the same fact
    restated. Drop a later sentence that introduces no new number and no new
    significant noun, and cap at 3."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    kept, seen_nums, seen_words = [], set(), set()
    for s in sents:
        nums = set(re.findall(r"\d[\d,]*", s))
        words = set(re.findall(r"[A-Za-z]{4,}", s.lower())) - {
            "this", "that", "there", "which", "with", "from", "have", "were",
            "also", "only", "available", "these", "those", "total", "alert"}
        new_nums = nums - seen_nums
        new_words = words - seen_words
        if kept and not new_nums and len(new_words) < 2:
            continue
        kept.append(s)
        seen_nums |= nums
        seen_words |= words
        if len(kept) >= 3:
            break
    return " ".join(kept)


def finalize(rows: list, answer: str, question: str = "") -> str:
    answer = _dedupe_sentences((answer or "").strip())
    if not rows:
        return "Nothing matched that query."
    rows = _clean_result_keys(rows)
    if _is_bare(answer):
        return rows_to_sentence(rows, question)
    if _NO_DATA_RE.search(answer):          # rows exist -> this is a hallucination
        return rows_to_sentence(rows, question)
    if _looks_generic(answer, rows):
        return rows_to_sentence(rows, question)
    if _invents_big_number(answer, rows):
        return rows_to_sentence(rows, question)
    return answer
