"""Unit sanity checks for FIX GG - common single-word typos of the
day/type/period vocabulary this file's regexes key off are normalized
before any date/type extraction runs. Found live: the same typo'd
question ("missing clnaing yeaterday") produced a different wrong answer
on different runs, since typos skipped every deterministic extractor and
left the model to guess the whole shape unassisted."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX GG - common typos of the key vocabulary get corrected")
check("'yeaterday' -> 'yesterday'",
      L._normalize_common_typos("missing clnaing yeaterday") == "missing cleaning yesterday")
check("'toady' -> 'today'",
      L._normalize_common_typos("how many alerts toady") == "how many alerts today")
check("'breakdwon' -> 'breakdown'",
      L._normalize_common_typos("give me a breakdwon by type") == "give me a breakdown by type")
check("'inspecton' -> 'inspection'",
      L._normalize_common_typos("fast inspecton alerts") == "fast inspection alerts")
check("FIX KK - 'tday' (4 letters) -> 'today', previously skipped by the 5-letter minimum",
      L._normalize_common_typos("how many alerts tday") == "how many alerts today")

print("\nFIX GG negatives - correctly-spelled text and unrelated words pass through untouched")
check("already-correct sentence unchanged",
      L._normalize_common_typos("how many hand touch alerts yesterday")
      == "how many hand touch alerts yesterday")
check("an unrelated, uncommon word is NOT force-matched to the vocabulary",
      L._normalize_common_typos("what's the busiest zone recently") == "what's the busiest zone recently")
check("very short words (<4 letters) are never touched, even if superficially close",
      L._normalize_common_typos("how many day") == "how many day")
check("a real, correctly-spelled non-vocabulary word (station name) is untouched",
      L._normalize_common_typos("alerts at FQC Station 1") == "alerts at FQC Station 1")
check("FIX KK negative - common 4-letter words are NOT false-matched at the lowered threshold",
      L._normalize_common_typos("how many alerts with type data") == "how many alerts with type data")

print("\nFIX GG.2 - general dictionary fallback catches typos outside the domain list")
check("'happend' -> 'happened'",
      "happened" in L._normalize_common_typos("how many alerts happend recently"))
check("'occured' -> 'occurred'",
      "occurred" in L._normalize_common_typos("alerts that occured recently"))
check("'seperate' -> 'separate'",
      "separate" in L._normalize_common_typos("give me a seperate count"))
check("'recieved' -> 'received'",
      "received" in L._normalize_common_typos("alerts recieved yesterday"))

print("\nFIX GG.2 negatives - proper nouns and codes are never touched by the general pass")
check("'FQC' (all-caps, a real product name) untouched",
      "FQC" in L._normalize_common_typos("alerts at FQC Station 1"))
check("'MongoDB' (mixed case, a real proper noun) untouched",
      "MongoDB" in L._normalize_common_typos("does this use MongoDB for storage"))
check("a real, already-correct uncommon word untouched",
      L._normalize_common_typos("what's the busiest zone recently")
      == "what's the busiest zone recently")
check("the domain pass still wins for words in both lists (no double-processing surprises)",
      L._normalize_common_typos("missing clnaing yeaterday") == "missing cleaning yesterday")

print("\nFIX GG - wired through to a real extractor function (no model call needed)")
from datetime import datetime, timedelta, timezone
NOW = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
TODAY0 = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
normalized = L._normalize_common_typos("how many alerts yeaterday")
check("typo'd-then-corrected 'yeaterday' resolves via the real date extractor",
      L._extract_relative_date_range(normalized, NOW) == (TODAY0 - timedelta(days=1), TODAY0),
      f"normalized={normalized!r}")

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
