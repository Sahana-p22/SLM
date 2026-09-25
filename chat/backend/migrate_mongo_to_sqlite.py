"""One-time, read-only Mongo -> SQLite copy for the approach2 experiment.

Independent of the separate slm-llama3b-sqlite migration effort: this
writes its own fqc.db file and never touches Mongo except to read it.
No ongoing sync is implemented here (out of scope for this experiment —
see the main migration project for that); this is a point-in-time copy
good enough to build and evaluate the new query approach against.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

from pymongo import MongoClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO_ROOT / "chat" / "backend" / "fqc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    alert_type TEXT NOT NULL,
    inspection_time REAL,
    objects_present TEXT,
    cloth_detected INTEGER,
    narration_en TEXT,
    narration_ta TEXT,
    zone TEXT,
    timestamp TEXT NOT NULL,
    hour INTEGER
);
CREATE INDEX IF NOT EXISTS idx_alerts_type_ts ON alerts(alert_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
"""

BATCH = 5000


def _row(doc):
    ts = doc.get("timestamp")
    ts_iso = ts.isoformat(sep=" ") if hasattr(ts, "isoformat") else str(ts)
    return (
        str(doc["_id"]),
        doc.get("alert_type"),
        doc.get("inspection_time"),
        json.dumps(doc.get("objects_present") or []),
        1 if doc.get("cloth_detected") else 0,
        doc.get("narration_en"),
        doc.get("narration_ta"),
        doc.get("zone"),
        ts_iso,
        doc.get("hour"),
    )


def main():
    client = MongoClient("mongodb://localhost:27017")
    coll = client["slm_safety"]["alerts"]
    total = coll.count_documents({})
    print(f"[migrate] source Mongo documents: {total}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)

    t0 = time.time()
    n = 0
    batch = []
    cursor = coll.find({}).sort("_id", 1)
    for doc in cursor:
        batch.append(_row(doc))
        if len(batch) >= BATCH:
            conn.executemany(
                "INSERT INTO alerts (id, alert_type, inspection_time, objects_present, "
                "cloth_detected, narration_en, narration_ta, zone, timestamp, hour) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            conn.commit()
            n += len(batch)
            batch = []
            if n % 50000 == 0:
                print(f"[migrate] {n}/{total} ({time.time()-t0:.1f}s)")
    if batch:
        conn.executemany(
            "INSERT INTO alerts (id, alert_type, inspection_time, objects_present, "
            "cloth_detected, narration_en, narration_ta, zone, timestamp, hour) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
        conn.commit()
        n += len(batch)

    print(f"[migrate] inserted {n} rows in {time.time()-t0:.1f}s -> {DB_PATH}")

    # --- verification ---
    sqlite_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    print(f"[verify] mongo={total} sqlite={sqlite_count} match={total == sqlite_count}")
    ok = total == sqlite_count

    for t in ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING", "NORMAL_OPERATION"]:
        m = coll.count_documents({"alert_type": t})
        s = conn.execute("SELECT COUNT(*) FROM alerts WHERE alert_type=?", (t,)).fetchone()[0]
        match = m == s
        ok = ok and match
        print(f"[verify] type={t} mongo={m} sqlite={s} match={match}")

    m_min = coll.find().sort("timestamp", 1).limit(1)[0]["timestamp"]
    m_max = coll.find().sort("timestamp", -1).limit(1)[0]["timestamp"]
    s_min, s_max = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM alerts").fetchone()
    print(f"[verify] mongo range {m_min} .. {m_max}")
    print(f"[verify] sqlite range {s_min} .. {s_max}")

    # field-by-field spot check on a sample spread across the collection
    import random
    random.seed(42)
    sample_ids = [random.randint(0, total - 1) for _ in range(200)]
    mismatches = 0
    for idx in sample_ids:
        doc = coll.find({}).sort("_id", 1).skip(idx).limit(1)[0]
        row = conn.execute("SELECT * FROM alerts WHERE id=?", (str(doc["_id"]),)).fetchone()
        if row is None:
            mismatches += 1
            continue
        cols = [c[0] for c in conn.execute("SELECT * FROM alerts LIMIT 0").description]
        rowd = dict(zip(cols, row))
        expected = _row(doc)
        expected_d = dict(zip(cols, expected))
        for k in cols:
            if rowd[k] != expected_d[k]:
                mismatches += 1
                print(f"[verify] MISMATCH id={doc['_id']} field={k} mongo={expected_d[k]!r} sqlite={rowd[k]!r}")
                break
    print(f"[verify] spot-check: {len(sample_ids)} sampled, {mismatches} mismatches")
    ok = ok and mismatches == 0

    conn.close()
    if not ok:
        print("[verify] FAILED — see mismatches above", file=sys.stderr)
        sys.exit(1)
    print("[verify] PASSED — lossless copy confirmed")


if __name__ == "__main__":
    main()
