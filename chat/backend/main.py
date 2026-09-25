# backend/main.py
#
# SQLite-backed variant of this app: every read path (chat, dashboard,
# stats, recent-alerts) now goes through the local alerts.sqlite3 mirror
# via chat.backend.db_sqlite, instead of querying MongoDB directly.
# MongoDB itself is untouched and stays the write/ingestion path — see
# chat/backend/mongo_sqlite_sync.py, which keeps this SQLite mirror
# current with whatever's written there.

import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.backend.db_sqlite import get_connection
from chat.backend.llm_query_sql import ERROR_ANSWER, answer_question, answer_question_stream

app = FastAPI(title="Factory Safety Alert Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    question: str
    answer: str
    pipeline: str | None = None


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        history = [h.model_dump() for h in req.history]
        return answer_question(req.question, history)
    except Exception as exc:
        print(f"[main] /chat unhandled exception: {exc!r}")
        raise HTTPException(status_code=500, detail=ERROR_ANSWER)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    history = [h.model_dump() for h in req.history]

    def event_source():
        try:
            for event_type, payload in answer_question_stream(req.question, history):
                yield f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"
        except Exception as exc:
            print(f"[main] /chat/stream unhandled exception: {exc!r}")
            yield f"event: error\ndata: {json.dumps({'error': ERROR_ANSWER})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile):
    from chat.backend.speech import MAX_AUDIO_BYTES, transcribe

    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large (max {MAX_AUDIO_BYTES // (1024 * 1024)}MB).",
        )

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        text = transcribe(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    duration_ms = round((time.time() - t0) * 1000)

    if not text:
        print(f"[speech] transcription_ms={duration_ms} text=<no speech detected>")
        raise HTTPException(status_code=422, detail="Couldn't make out any speech in that recording.")

    print(f"[speech] transcription_ms={duration_ms} text={text!r}")
    return {"text": text}


@app.get("/stats/summary")
def stats_summary():
    """Deterministic dashboard stats — no LLM involved. Ported from the
    MongoDB aggregation version to plain SQL against the SQLite mirror;
    same NORMAL_OPERATION exclusion, same three numbers, same by-type
    breakdown, sorted the same way (highest count first)."""
    conn = get_connection()
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    today_start_s = today_start.strftime("%Y-%m-%dT%H:%M:%S")
    week_start_s = week_start.strftime("%Y-%m-%dT%H:%M:%S")

    total = conn.execute(
        "SELECT COUNT(*) AS c FROM alerts WHERE alert_type != 'NORMAL_OPERATION'"
    ).fetchone()["c"]
    today = conn.execute(
        "SELECT COUNT(*) AS c FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ?",
        (today_start_s,),
    ).fetchone()["c"]
    last_7_days = conn.execute(
        "SELECT COUNT(*) AS c FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ?",
        (week_start_s,),
    ).fetchone()["c"]
    by_type = conn.execute(
        "SELECT alert_type, COUNT(*) AS count FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
        "GROUP BY alert_type ORDER BY count DESC"
    ).fetchall()

    return {
        "total": total,
        "today": today,
        "last_7_days": last_7_days,
        "by_type": [{"alert_type": r["alert_type"], "count": r["count"]} for r in by_type],
    }


ZONES = ["FQC Station 1"]
ALERT_TYPES = ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"]


@app.get("/dashboard/fqc")
def dashboard_fqc(from_iso: str | None = None, to_iso: str | None = None, zone: str | None = None):
    """Everything the FQC Monitoring dashboard needs, in one deterministic
    call — no LLM anywhere in this path. Defaults to the last 24 hours.
    Ported from the MongoDB aggregation version to plain SQL; same
    fields, same shape, same defaults, same zone filter behavior."""
    conn = get_connection()

    now = datetime.now(timezone.utc)
    time_from = datetime.fromisoformat(from_iso) if from_iso else now - timedelta(hours=24)
    time_to = datetime.fromisoformat(to_iso) if to_iso else now
    if time_from.tzinfo is None:
        time_from = time_from.replace(tzinfo=timezone.utc)
    if time_to.tzinfo is None:
        time_to = time_to.replace(tzinfo=timezone.utc)
    # Stored timestamps are naive (no tzinfo) — compare as naive strings,
    # same convention migrate_mongo_to_sqlite.py used (isoformat() as-is).
    from_s = time_from.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
    to_s = time_to.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")

    where = "alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ?"
    params: list = [from_s, to_s]
    if zone and zone in ZONES:
        where += " AND zone = ?"
        params.append(zone)

    total_alerts = conn.execute(f"SELECT COUNT(*) AS c FROM alerts WHERE {where}", params).fetchone()["c"]

    by_type_rows = conn.execute(
        f"SELECT alert_type, COUNT(*) AS count FROM alerts WHERE {where} GROUP BY alert_type", params
    ).fetchall()
    by_type = {r["alert_type"]: r["count"] for r in by_type_rows}
    for t in ALERT_TYPES:
        by_type.setdefault(t, 0)

    avg_row = conn.execute(
        f"SELECT AVG(inspection_time) AS avg FROM alerts WHERE {where}", params
    ).fetchone()
    avg_inspection_time = round(avg_row["avg"], 2) if avg_row["avg"] is not None else None

    zone_rows = conn.execute(
        f"""
        SELECT zone,
               COUNT(*) AS total,
               SUM(CASE WHEN alert_type = 'FAST_INSPECTION' THEN 1 ELSE 0 END) AS fast_inspection,
               SUM(CASE WHEN alert_type = 'HAND_TOUCH' THEN 1 ELSE 0 END) AS hand_touch,
               SUM(CASE WHEN alert_type = 'MISSING_CLEANING' THEN 1 ELSE 0 END) AS missing_cleaning,
               AVG(inspection_time) AS avg_inspection_time,
               MAX(timestamp) AS last_alert_at
        FROM alerts WHERE {where} GROUP BY zone ORDER BY total DESC
        """,
        params,
    ).fetchall()
    by_zone = [
        {
            "zone": r["zone"],
            "total": r["total"],
            "fast_inspection": r["fast_inspection"],
            "hand_touch": r["hand_touch"],
            "missing_cleaning": r["missing_cleaning"],
            "avg_inspection_time": round(r["avg_inspection_time"], 2) if r["avg_inspection_time"] is not None else None,
            "last_alert_at": r["last_alert_at"],
        }
        for r in zone_rows
    ]

    # _id/objects_present/narration_ta stripped from the stream, matching
    # the Mongo version's own projection ({"_id": 0, "objects_present": 0,
    # "narration_ta": 0}).
    stream_rows = conn.execute(
        f"""
        SELECT alert_type, timestamp, inspection_time, zone, cloth_detected, narration_en, hour
        FROM alerts WHERE {where} ORDER BY timestamp DESC LIMIT 30
        """,
        params,
    ).fetchall()
    alert_stream = [dict(r) for r in stream_rows]
    for a in alert_stream:
        a["cloth_detected"] = bool(a["cloth_detected"])

    return {
        "time_window": {"from": time_from.isoformat(), "to": time_to.isoformat()},
        "zones_monitored": len(ZONES),
        "total_alerts": total_alerts,
        "by_type": by_type,
        "avg_inspection_time": avg_inspection_time,
        "by_zone": by_zone,
        "alert_stream": alert_stream,
    }


@app.get("/alerts/recent")
def alerts_recent(limit: int = 20):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT alert_type, timestamp, inspection_time, zone, cloth_detected,
               objects_present, narration_en, narration_ta, hour
        FROM alerts ORDER BY timestamp DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    data = [dict(r) for r in rows]
    for d in data:
        d["cloth_detected"] = bool(d["cloth_detected"])
    return data


@app.get("/health")
def health():
    return {"status": "ok"}
