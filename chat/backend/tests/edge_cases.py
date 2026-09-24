"""1.12 Tricky Question Test: zero-result date ranges, far-future dates,
500+ character input, non-English input - checked for crashes, hangs, or
nonsensical answers, not for correctness per se."""
import json
import time

import requests

from bench_config import CHAT_URL

API = CHAT_URL

CASES = []

# zero-result date ranges, many years/phrasings
for year in (2005, 2008, 2010, 2012, 2015, 2018, 2019, 2020):
    CASES.append((f"zero-result date {year}", f"how many alerts happened on January 1st {year}"))
    CASES.append((f"zero-result date {year} (breakdown)", f"break down alerts by type on January 1st {year}"))
    CASES.append((f"zero-result date {year} (avg)", f"what was the average inspection time on January 1st {year}"))

# far-future dates, many phrasings/years
for year in (2500, 3000, 3500, 5000, 9999):
    CASES.append((f"far-future {year}", f"how many alerts happened in the year {year}"))
    CASES.append((f"far-future {year} (breakdown)", f"break down alerts by type in {year}"))
for phrase in ("next century", "in 100 years", "next millennium", "a thousand years from now"):
    CASES.append((f"far-future relative: {phrase}", f"how many alerts will there be {phrase}"))

# oversized input, several lengths and fillers
for n, filler in [(50, "really"), (90, "truly"), (150, "seriously"), (300, "honestly")]:
    CASES.append((f"{n}x filler input", "how many alerts " + f"{filler} " * n + "happened today"))

# non-English input, several languages
NON_ENGLISH = [
    ("Tamil", "இன்று எத்தனை எச்சரிக்கைகள் நடந்தன?"),
    ("Hindi", "आज कितने अलर्ट हुए?"),
    ("Spanish", "¿Cuántas alertas hubo hoy?"),
    ("French", "Combien d'alertes y a-t-il eu aujourd'hui?"),
    ("German", "Wie viele Warnungen gab es heute?"),
    ("Japanese", "今日は何件のアラートがありましたか？"),
    ("Arabic", "كم عدد التنبيهات التي حدثت اليوم؟"),
    ("Russian", "Сколько оповещений было сегодня?"),
    ("Chinese", "今天有多少警报？"),
    ("Korean", "오늘 알림이 몇 개 있었나요?"),
]
for lang, text in NON_ENGLISH:
    CASES.append((f"non-English ({lang})", text))

# malformed/adversarial/whitespace/punctuation variants
CASES += [
    ("empty string", ""),
    ("only whitespace", "   "),
    ("only tabs/newlines", "\t\n\t\n"),
    ("only punctuation", "???!!!..."),
    ("only punctuation 2", "..---..__**"),
    ("SQL-injection-style", "how many alerts'; DROP TABLE alerts; --"),
    ("SQL-injection-style 2", "today' OR '1'='1"),
    ("mongo-injection-style", 'how many alerts {"$where": "1==1"}'),
    ("nested quotes", 'how many alerts happened "today" or \'yesterday\''),
    ("unbalanced quotes", 'how many alerts happened "today'),
    ("unbalanced brackets", "how many alerts today [[[("),
    ("null-byte-ish", "how many alerts today\x00"),
    ("control characters", "how many alerts\x07\x08 today"),
    ("extremely large number", "how many alerts happened in the last 999999999 days"),
    ("extremely large number 2", "how many alerts happened in the last 10000000000000 minutes"),
    ("negative number", "how many alerts happened in the last -5 days"),
    ("negative number 2", "how many alerts happened in the last -100 hours"),
    ("zero number", "how many alerts happened in the last 0 days"),
    ("fractional number", "how many alerts happened in the last 2.5 days"),
    ("unicode emoji", "how many alerts today? 🏭🔔📊"),
    ("unicode emoji 2", "😀😃😄😁 alerts today?"),
    ("repeated question mark", "how many alerts today??????"),
    ("repeated exclamation", "how many alerts today!!!!!!"),
    ("all caps", "HOW MANY ALERTS HAPPENED TODAY"),
    ("all caps 2", "BREAK DOWN ALERTS BY TYPE THIS MONTH"),
    ("mixed case chaos", "hOw MaNy AlErTs ToDaY"),
    ("mixed case chaos 2", "wHaT iS tHe AvErAgE iNsPeCtIoN tImE"),
    ("very short", "alerts?"),
    ("very short 2", "today?"),
    ("very short 3", "how many?"),
    ("single word", "alerts"),
    ("single letter", "a"),
    ("just a number", "7"),
    ("just an operator", "=="),
    ("html-injection-style", "<script>alert('x')</script> how many alerts today"),
    ("markdown-injection-style", "how many alerts today ```python\nimport os\n```"),
    ("very long single word", "a" * 400),
    ("repeated word stutter", "how many many many alerts alerts today today"),
    ("backwards question", "today happened alerts many how"),
    ("question with typo", "how mnay alrets hapened todya"),
    ("question with typo 2", "waht is teh averge inspction time"),
    ("leading/trailing whitespace", "   how many alerts today   "),
    ("tab-separated", "how\tmany\talerts\ttoday"),
    ("newline-separated", "how many alerts\ntoday"),
    ("date impossible: Feb 30", "how many alerts happened on February 30th this year"),
    ("date impossible: month 13", "how many alerts happened in month 13 of this year"),
    ("date impossible: day 32", "how many alerts happened on the 32nd of this month"),
    ("time impossible: hour 25", "how many alerts happened at hour 25 today"),
    ("time impossible: negative hour", "how many alerts happened between -2am and 4am"),
    ("conflicting time range", "how many alerts happened between 5pm and 2pm today"),
    ("self-referential", "how many alerts happened when this question was asked"),
    ("recursive request", "how many times have I asked how many alerts happened today"),
    ("meta question", "what question should I be asking about alerts"),
    ("ambiguous pronoun", "how many of those happened today"),
    ("comparison with no baseline", "was today busier"),
    ("incomplete sentence", "how many alerts happened when"),
    ("incomplete sentence 2", "the number of alerts that"),
    ("just a preposition", "on"),
    ("just a conjunction", "and"),
]

results = []
for label, q in CASES:
    t0 = time.perf_counter()
    crashed = False
    status = None
    answer = None
    try:
        r = requests.post(API, json={"question": q, "history": []}, timeout=60)
        status = r.status_code
        dt = time.perf_counter() - t0
        if status == 200:
            answer = r.json().get("answer")
    except Exception as e:
        crashed = True
        dt = time.perf_counter() - t0
        answer = f"EXCEPTION: {e}"

    hung = dt > 55
    results.append({"label": label, "question": q[:80], "status": status,
                     "crashed": crashed, "hung": hung, "latency_s": round(dt, 2),
                     "answer": (answer or "")[:150]})
    flag = "CRASH" if crashed else ("HUNG" if hung else ("OK" if status == 200 else f"HTTP{status}"))
    print(f"  [{flag:6s}] {label:25s} ({dt:.1f}s) -> {(answer or '')[:90]!r}")

n_crash = sum(1 for r in results if r["crashed"])
n_hung = sum(1 for r in results if r["hung"])
n_http_error = sum(1 for r in results if r["status"] and r["status"] != 200)
print(f"\n{len(results)} cases: {n_crash} crashed, {n_hung} hung (>55s), {n_http_error} non-200 HTTP")

json.dump(results, open("/tmp/edge_cases_results.json", "w"), indent=2)
