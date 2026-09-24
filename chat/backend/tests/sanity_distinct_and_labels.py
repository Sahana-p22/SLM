"""Unit sanity checks for FIX Q - the "Id FQC Station 1" label bug.
(The distinct/unique-count fast path this file used to also cover was
removed along with every other fast path - see FIX HH in llm_query.py.)"""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX Q.1 - _describe_label recognizes a zone value under any key")
check("key='_id', value=zone -> 'in FQC Station 1'",
      L._describe_label("_id", "FQC Station 1") == "in FQC Station 1")
check("key='zone' (already worked) still works",
      L._describe_label("zone", "FQC Station 1") == "in FQC Station 1")
check("unrelated string value untouched",
      L._describe_label("_id", "Loading Dock") == "id Loading Dock")

print("\nFIX Q.1 - the observed bug, end to end through _row_to_clause")
row = {"_id": "FQC Station 1", "count": 307755}
clause = L._row_to_clause(row)
check("no literal 'Id' leak", not clause.lower().startswith("id "), clause)
check("reads naturally", clause == "307,755 alerts in FQC Station 1", clause)

print("\nFIX Q.2 - bare single-field group rows (the 'distinct values' shape)")
check("bare alert-type row renders as the type name, not 'id X'",
      L._row_to_clause({"_id": "FAST_INSPECTION"}) == "Fast Inspection")
check("bare zone row renders as the zone name, not 'id X'",
      L._row_to_clause({"_id": "FQC Station 1"}) == "FQC Station 1")
check("bare unrecognized value still falls back to key+value (no crash)",
      L._row_to_clause({"_id": "Something Else"}) == "id Something Else", L._row_to_clause({"_id": "Something Else"}))

sentence = L._rows_to_sentence([{"_id": "FAST_INSPECTION"}, {"_id": "HAND_TOUCH"},
                                 {"_id": "MISSING_CLEANING"}])
check("a list of distinct types reads as a natural sentence",
      sentence == "Fast Inspection, Hand Touch, and Missing Cleaning.", sentence)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
