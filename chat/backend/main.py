# backend/main.py

import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.backend.db import get_alerts_collection
from chat.backend.llm_query import answer_question, answer_question_stream

app = FastAPI(title="Factory Safety Alert Chatbot API")

app.add_middleware(
    CORSMiddleware,
    # The frontend dev server's port isn't fixed - vite bumps to the next
    # free port (5174, 5175, ...) whenever 5173 is already taken, which a
    # single hardcoded origin here silently broke: the page loaded fine
    # (it's static), but every fetch() from it - the dashboard included -
    # was blocked by the browser's own CORS check before ever reaching
    # this server, with no error visible server-side at all. A regex
    # covering any localhost/127.0.0.1 port fixes this the same way for
    # any dev port, not just whichever one happened to be free today.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    question: str
    answer: str
    pipeline: list | None = None


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
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Same underlying pipeline as /chat, but pushed to the client as
    Server-Sent Events: a "meta" event as soon as the DB query has run
    (question/pipeline/result), then "token" events as the answer is
    generated word-by-word, then a "done" event with the final answer
    text. Lets the frontend render a typing effect instead of a blank
    bubble until the whole sentence is ready.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    history = [h.model_dump() for h in req.history]

    def event_source():
        try:
            for event_type, payload in answer_question_stream(req.question, history):
                yield f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile):
    """Transcribes recorded voice input locally via faster-whisper. Returns
    plain text — the frontend then sends that text through the exact same
    /chat flow as a typed question, no separate code path.

    Imported lazily (not at module load time) so that a broken/blocked
    voice dependency (faster-whisper's `av` package, observed hitting a
    Windows Application Control policy block on this machine) only
    breaks voice input itself, not the entire backend — chat is the
    primary path and must keep working even if voice can't load.
    """
    from chat.backend.speech import transcribe

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
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
    """Deterministic dashboard stats — no LLM involved.

    NORMAL_OPERATION is a compliant, no-issue event, not an alert, so it's
    excluded from every count here (matches the exclusion applied in
    llm_query.py's default query filter).
    """
    collection = get_alerts_collection()
    not_normal = {"alert_type": {"$ne": "NORMAL_OPERATION"}}

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    total = collection.count_documents(not_normal)
    today = collection.count_documents({**not_normal, "timestamp": {"$gte": today_start}})
    last_7_days = collection.count_documents({**not_normal, "timestamp": {"$gte": week_start}})

    by_type = list(collection.aggregate([
        {"$match": not_normal},
        {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]))

    return {
        "total": total,
        "today": today,
        "last_7_days": last_7_days,
        "by_type": [{"alert_type": r["_id"], "count": r["count"]} for r in by_type],
    }


ZONES = ["FQC Station 1"]
ALERT_TYPES = ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"]


@app.get("/dashboard/fqc")
def dashboard_fqc(from_iso: str | None = None, to_iso: str | None = None, zone: str | None = None):
    """Everything the FQC Monitoring dashboard needs, in one deterministic
    call — no LLM anywhere in this path. Defaults to the last 24 hours.
    """
    collection = get_alerts_collection()

    now = datetime.now(timezone.utc)
    time_from = datetime.fromisoformat(from_iso) if from_iso else now - timedelta(hours=24)
    time_to = datetime.fromisoformat(to_iso) if to_iso else now
    if time_from.tzinfo is None:
        time_from = time_from.replace(tzinfo=timezone.utc)
    if time_to.tzinfo is None:
        time_to = time_to.replace(tzinfo=timezone.utc)

    base_filter = {
        "alert_type": {"$ne": "NORMAL_OPERATION"},
        "timestamp": {"$gte": time_from, "$lt": time_to},
    }
    if zone and zone in ZONES:
        base_filter["zone"] = zone

    total_alerts = collection.count_documents(base_filter)

    by_type_rows = list(collection.aggregate([
        {"$match": base_filter},
        {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
    ]))
    by_type = {row["_id"]: row["count"] for row in by_type_rows}
    for t in ALERT_TYPES:
        by_type.setdefault(t, 0)

    avg_rows = list(collection.aggregate([
        {"$match": base_filter},
        {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}},
    ]))
    avg_inspection_time = round(avg_rows[0]["avg"], 2) if avg_rows else None

    zone_rows = list(collection.aggregate([
        {"$match": base_filter},
        {"$group": {
            "_id": "$zone",
            "total": {"$sum": 1},
            "fast_inspection": {"$sum": {"$cond": [{"$eq": ["$alert_type", "FAST_INSPECTION"]}, 1, 0]}},
            "hand_touch": {"$sum": {"$cond": [{"$eq": ["$alert_type", "HAND_TOUCH"]}, 1, 0]}},
            "missing_cleaning": {"$sum": {"$cond": [{"$eq": ["$alert_type", "MISSING_CLEANING"]}, 1, 0]}},
            "avg_inspection_time": {"$avg": "$inspection_time"},
            "last_alert_at": {"$max": "$timestamp"},
        }},
        {"$sort": {"total": -1}},
    ]))
    by_zone = [
        {
            "zone": r["_id"],
            "total": r["total"],
            "fast_inspection": r["fast_inspection"],
            "hand_touch": r["hand_touch"],
            "missing_cleaning": r["missing_cleaning"],
            "avg_inspection_time": round(r["avg_inspection_time"], 2) if r["avg_inspection_time"] is not None else None,
            "last_alert_at": r["last_alert_at"].isoformat() if r["last_alert_at"] else None,
        }
        for r in zone_rows
    ]

    stream_cursor = (
        collection.find(base_filter, {"_id": 0, "objects_present": 0, "narration_ta": 0})
        .sort("timestamp", -1)
        .limit(30)
    )
    alert_stream = list(stream_cursor)
    for a in alert_stream:
        a["timestamp"] = a["timestamp"].isoformat()

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
    collection = get_alerts_collection()
    cursor = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    data = list(cursor)
    for d in data:
        d["timestamp"] = d["timestamp"].isoformat()
    return data


@app.get("/health")
def health():
    return {"status": "ok"}
