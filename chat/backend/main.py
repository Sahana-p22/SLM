# backend/main.py — approach2: SQLite + Retail-style query/answer pipeline.
# All reads (chat, dashboard, stats, recent alerts) now go through SQLite.

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.backend.db import run_readonly
from chat.backend.pipeline import ERROR_ANSWER, answer_question, answer_question_stream

app = FastAPI(title="Factory Safety Alert Chatbot API (approach2 — SQLite)")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    question: str
    answer: str
    sql: str | None = None


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
    """Deterministic dashboard stats — no LLM involved. Now backed by SQLite."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    today_iso = today_start.strftime("%Y-%m-%d %H:%M:%S")
    week_iso = week_start.strftime("%Y-%m-%d %H:%M:%S")

    total = run_readonly(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type != 'NORMAL_OPERATION'")[0]["n"]
    today = run_readonly(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ?",
        (today_iso,))[0]["n"]
    last_7_days = run_readonly(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type != 'NORMAL_OPERATION' AND timestamp >= ?",
        (week_iso,))[0]["n"]
    by_type = run_readonly(
        "SELECT alert_type, COUNT(*) AS count FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
        "GROUP BY alert_type ORDER BY count DESC")

    return {"total": total, "today": today, "last_7_days": last_7_days, "by_type": by_type}


ZONES = ["FQC Station 1"]
ALERT_TYPES = ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"]


@app.get("/dashboard/fqc")
def dashboard_fqc(from_iso: str | None = None, to_iso: str | None = None, zone: str | None = None):
    now = datetime.utcnow()
    time_from = datetime.fromisoformat(from_iso) if from_iso else now - timedelta(hours=24)
    time_to = datetime.fromisoformat(to_iso) if to_iso else now
    f_iso, t_iso = time_from.strftime("%Y-%m-%d %H:%M:%S"), time_to.strftime("%Y-%m-%d %H:%M:%S")

    where = "alert_type != 'NORMAL_OPERATION' AND timestamp >= ? AND timestamp < ?"
    params = [f_iso, t_iso]
    if zone and zone in ZONES:
        where += " AND zone = ?"
        params.append(zone)

    total_alerts = run_readonly(f"SELECT COUNT(*) AS n FROM alerts WHERE {where}", params)[0]["n"]

    by_type_rows = run_readonly(
        f"SELECT alert_type, COUNT(*) AS count FROM alerts WHERE {where} GROUP BY alert_type", params)
    by_type = {r["alert_type"]: r["count"] for r in by_type_rows}
    for t in ALERT_TYPES:
        by_type.setdefault(t, 0)

    avg_rows = run_readonly(
        f"SELECT AVG(inspection_time) AS avg FROM alerts WHERE {where}", params)
    avg_inspection_time = round(avg_rows[0]["avg"], 2) if avg_rows and avg_rows[0]["avg"] is not None else None

    zone_rows = run_readonly(
        f"""SELECT zone,
                   COUNT(*) AS total,
                   SUM(CASE WHEN alert_type = 'FAST_INSPECTION' THEN 1 ELSE 0 END) AS fast_inspection,
                   SUM(CASE WHEN alert_type = 'HAND_TOUCH' THEN 1 ELSE 0 END) AS hand_touch,
                   SUM(CASE WHEN alert_type = 'MISSING_CLEANING' THEN 1 ELSE 0 END) AS missing_cleaning,
                   AVG(inspection_time) AS avg_inspection_time,
                   MAX(timestamp) AS last_alert_at
            FROM alerts WHERE {where} GROUP BY zone ORDER BY total DESC""", params)
    by_zone = [
        {**r, "avg_inspection_time": round(r["avg_inspection_time"], 2) if r["avg_inspection_time"] is not None else None}
        for r in zone_rows
    ]

    stream_rows = run_readonly(
        f"SELECT alert_type, inspection_time, cloth_detected, narration_en, zone, timestamp, hour "
        f"FROM alerts WHERE {where} ORDER BY timestamp DESC LIMIT 30", params)

    return {
        "time_window": {"from": f_iso, "to": t_iso},
        "zones_monitored": len(ZONES),
        "total_alerts": total_alerts,
        "by_type": by_type,
        "avg_inspection_time": avg_inspection_time,
        "by_zone": by_zone,
        "alert_stream": stream_rows,
    }


@app.get("/alerts/recent")
def alerts_recent(limit: int = 20):
    return run_readonly(
        "SELECT alert_type, inspection_time, objects_present, cloth_detected, narration_en, "
        "narration_ta, zone, timestamp, hour FROM alerts ORDER BY timestamp DESC LIMIT ?",
        (limit,))


@app.get("/health")
def health():
    return {"status": "ok"}
