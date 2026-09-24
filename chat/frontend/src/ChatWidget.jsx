import { useEffect, useRef, useState } from "react";
import { API_BASE } from "./api";
import "./ChatWidget.css";

const ALERT_TYPES = ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING", "NORMAL_OPERATION"];

function formatCell(key, value) {
  if (value === null || value === undefined) return "—";
  if (key === "timestamp" && typeof value === "string" && !isNaN(Date.parse(value))) {
    return new Date(value).toLocaleString();
  }
  if (typeof value === "string" && ALERT_TYPES.includes(value)) {
    return <span className={`type-pill type-${value}`}>{value.replace("_", " ")}</span>;
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value : Math.round(value * 100) / 100;
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// Result rows are now open-ended — whatever shape the pipeline produced —
// rather than a fixed "list" or "group_count" type, so the table has to
// introspect the shape rather than branch on a known type tag.
//
// A single row is just "the answer" (a count, an average, one top result)
// — the sentence above already says it, so a one-row table adds nothing
// but visual noise. Tables only earn their place once there's an actual
// multi-row breakdown/list to scan.
function GenericTable({ rows }) {
  if (!rows || rows.length <= 1) return null;

  const columns = Array.from(
    rows.reduce((set, r) => {
      Object.keys(r).forEach((k) => set.add(k));
      return set;
    }, new Set())
  );

  return (
    <div className="alert-table-wrap">
      <table className="alert-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c === "_id" ? "key" : c.replace(/_/g, " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} className={c === "narration_en" || c === "narration_ta" ? "narration-cell" : ""}>
                  {formatCell(c, r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// A $facet "report" query returns a single document whose values are each
// an array (one per named sub-pipeline) — render each as its own labeled
// mini-table instead of one confusing flat table.
function isFacetShape(rows) {
  return (
    rows.length === 1 &&
    Object.keys(rows[0]).length > 0 &&
    Object.values(rows[0]).every((v) => Array.isArray(v))
  );
}

function FacetReport({ row }) {
  const multiRowSections = Object.entries(row).filter(([, subRows]) => subRows.length > 1);
  if (multiRowSections.length === 0) return null;

  return (
    <div className="facet-report">
      {multiRowSections.map(([name, subRows]) => (
        <div className="facet-section" key={name}>
          <div className="facet-title">{name.replace(/_/g, " ")}</div>
          <GenericTable rows={subRows} />
        </div>
      ))}
    </div>
  );
}

function ChatMessage({ role, content, meta }) {
  const rows = meta?.result;

  return (
    <div className={`msg msg-${role}`}>
      <div className="msg-bubble">
        <div className="msg-text">{content}</div>
        {rows && rows.length > 0 && (
          isFacetShape(rows) ? <FacetReport row={rows[0]} /> : <GenericTable rows={rows} />
        )}
      </div>
    </div>
  );
}

const SUGGESTIONS = [
  "How many alerts happened today?",
  "Which day had the most hand touch alerts?",
  "Give me a quarterly report",
  "Average inspection time for fast inspection alerts",
];

export default function ChatWidget({ onAnswered }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi, how can I help you?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [voiceMode, setVoiceMode] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const voiceModeRef = useRef(false);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const vadIntervalRef = useRef(null);

  // Sticky-scroll: auto-follow new content (including token-by-token
  // streaming updates, which fire this effect on every chunk) ONLY while
  // the user is already at/near the bottom. Without this, every streamed
  // token yanked the view back down, making it impossible to scroll up
  // and read earlier messages while a response was still generating.
  const stickToBottomRef = useRef(true);

  function handleScroll(e) {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    stickToBottomRef.current = scrollHeight - scrollTop - clientHeight < 60;
  }

  useEffect(() => {
    if (open && stickToBottomRef.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages, loading, open]);

  // Keep the text input focused so typing never needs a click first. The
  // input is `disabled` while a response is loading or voice mode is
  // active, which drops focus to the document body - re-focusing here,
  // once it becomes typeable again (or right when the widget opens),
  // means the cursor is always ready without the user clicking it.
  useEffect(() => {
    if (open && !loading && !voiceMode) {
      inputRef.current?.focus();
    }
  }, [open, loading, voiceMode]);

  // Release the mic/audio context if the widget unmounts while voice mode
  // is still active, so it never keeps recording in the background.
  useEffect(() => {
    return () => {
      clearInterval(vadIntervalRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioContextRef.current?.close().catch(() => {});
    };
  }, []);

  function buildHistory() {
    const pairs = [];
    for (let i = 0; i < messages.length - 1; i++) {
      if (messages[i].role === "user" && messages[i + 1].role === "assistant") {
        pairs.push({
          question: messages[i].content,
          answer: messages[i + 1].content,
          pipeline: messages[i + 1].meta?.pipeline ?? null,
        });
      }
    }
    return pairs.slice(-4);
  }

  // Mutates the trailing assistant placeholder message in place — used by
  // the streaming handler below so tokens can be appended one at a time
  // instead of only swapping in a message once the full answer is ready.
  function patchLastMessage(patch) {
    setMessages((m) => {
      const copy = [...m];
      const last = copy[copy.length - 1];
      copy[copy.length - 1] = typeof patch === "function" ? patch(last) : { ...last, ...patch };
      return copy;
    });
  }

  async function sendQuestion(question) {
    const history = buildHistory();
    stickToBottomRef.current = true;
    setMessages((m) => [...m, { role: "user", content: question }]);
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      setMessages((m) => [...m, { role: "assistant", content: "", meta: null }]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let sepIndex;
        while ((sepIndex = buf.indexOf("\n\n")) !== -1) {
          const rawEvent = buf.slice(0, sepIndex);
          buf = buf.slice(sepIndex + 2);

          const eventMatch = rawEvent.match(/^event: (.+)$/m);
          const dataMatch = rawEvent.match(/^data: (.*)$/m);
          if (!eventMatch || !dataMatch) continue;

          const eventType = eventMatch[1];
          const payload = JSON.parse(dataMatch[1]);

          if (eventType === "meta") {
            patchLastMessage({
              meta: {
                pipeline: payload.pipeline,
                explanation: payload.explanation,
                result: payload.result,
                intent: payload.intent,
                stages: payload.stages,
              },
            });
          } else if (eventType === "token") {
            patchLastMessage((last) => ({ ...last, content: last.content + payload.text }));
          } else if (eventType === "done") {
            // Authoritative final text — may differ from the streamed
            // tokens if the backend's safety net had to override a
            // bare/incomplete answer, so this always wins. `stages` here
            // supersedes the "meta" one — it includes the answer-
            // generation stage, which wasn't known yet when "meta" fired.
            patchLastMessage((last) => ({
              ...last,
              content: payload.answer,
              meta: { ...last.meta, stages: payload.stages ?? last.meta?.stages },
            }));
          } else if (eventType === "error") {
            throw new Error(payload.error);
          }
        }
      }

      onAnswered?.();
    } catch (e) {
      setError(e.message);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Something went wrong: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    sendQuestion(q);
    // Clicking "Send" (vs. pressing Enter) moves focus to the button -
    // pull it back to the input right away so the next question can be
    // typed immediately, without waiting for the response or clicking in.
    inputRef.current?.focus();
  }

  // ---- continuous hands-free voice mode ----
  // One mic click enters a listen -> auto-detect silence -> transcribe ->
  // send -> wait for answer -> listen again loop, like ChatGPT/Claude voice
  // mode. No per-question stop click needed; click the mic again anytime
  // to exit the loop.

  const SPEAKING_RMS_THRESHOLD = 12; // tune against real mic input if too sensitive/insensitive
  // 1400ms was cutting people off mid-sentence — a normal pause to take a
  // breath or think of the next word is often longer than that, so
  // "finished speaking" was being detected before the sentence actually
  // ended. 2500ms gives natural pauses enough room while still feeling
  // responsive once you're actually done.
  const SILENCE_MS = 1800; // how long silence must persist to consider the utterance finished
  const MIN_RECORDING_MS = 500; // ignore silence detection before this, avoids instant cutoff
  const MAX_RECORDING_MS = 20000; // safety cap in case VAD never detects silence

  async function enterVoiceMode() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;
      voiceModeRef.current = true;
      setVoiceMode(true);
      startTurn();
    } catch {
      setError("Couldn't access the microphone — check browser permissions.");
    }
  }

  function exitVoiceMode() {
    voiceModeRef.current = false;
    setVoiceMode(false);
    setListening(false);
    clearInterval(vadIntervalRef.current);

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.onstop = null; // don't trigger transcribe/next-turn on manual exit
      mediaRecorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioContextRef.current?.close().catch(() => {});
    streamRef.current = null;
    audioContextRef.current = null;
    analyserRef.current = null;
  }

  function startTurn() {
    if (!streamRef.current || !analyserRef.current) return;

    const recorder = new MediaRecorder(streamRef.current);
    audioChunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      clearInterval(vadIntervalRef.current);
      setListening(false);
      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      await transcribeAndSend(blob);
      if (voiceModeRef.current) startTurn();
    };

    mediaRecorderRef.current = recorder;
    recorder.start();
    setListening(true);

    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.fftSize);
    const startTime = Date.now();
    let hasSpoken = false;
    let silenceStart = null;

    vadIntervalRef.current = setInterval(() => {
      analyser.getByteTimeDomainData(dataArray);
      let sumSquares = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const dev = dataArray[i] - 128;
        sumSquares += dev * dev;
      }
      const rms = Math.sqrt(sumSquares / dataArray.length);
      const elapsed = Date.now() - startTime;

      if (rms > SPEAKING_RMS_THRESHOLD) {
        hasSpoken = true;
        silenceStart = null;
      } else if (hasSpoken) {
        if (silenceStart === null) silenceStart = Date.now();
        if (Date.now() - silenceStart >= SILENCE_MS && elapsed >= MIN_RECORDING_MS) {
          if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
        }
      }

      if (elapsed >= MAX_RECORDING_MS && mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
    }, 100);
  }

  async function transcribeAndSend(blob) {
    setTranscribing(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("audio", blob, "voice-query.webm");

      const res = await fetch(`${API_BASE}/speech-to-text`, {
        method: "POST",
        body: formData,
      });

      if (res.status === 422) {
        // Nothing intelligible in that clip (e.g. brief noise triggered
        // the VAD threshold) — in continuous mode this shouldn't interrupt
        // the loop with an error banner, just listen again.
        setTranscribing(false);
        return;
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Transcription failed (${res.status})`);
      }

      const { text } = await res.json();
      setTranscribing(false);
      if (text.trim()) await sendQuestion(text.trim());
    } catch (e) {
      setTranscribing(false);
      setError(e.message);
    }
  }

  return (
    <>
      {open && <div className="chat-widget-overlay" onClick={() => setOpen(false)} />}
      {open && (
        <div className="chat-widget-panel">
          <div className="chat-widget-header">
            <span>
              Deepinsight Assistant
              {voiceMode && (
                <span className="voice-mode-pill">
                  {listening ? "🎙️ listening" : transcribing ? "⏳ transcribing" : loading ? "💭 thinking" : "🎙️ voice mode"}
                </span>
              )}
            </span>
            <button
              className="chat-widget-close"
              onClick={() => {
                if (voiceMode) exitVoiceMode();
                setOpen(false);
              }}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          <div className="chat-scroll" ref={scrollRef} onScroll={handleScroll}>
            {messages.map((m, i) =>
              m.role === "assistant" && m.content === "" && i === messages.length - 1 ? null : (
                <ChatMessage key={i} role={m.role} content={m.content} meta={m.meta} />
              )
            )}
            {loading && messages[messages.length - 1]?.content === "" && (
              <div className="msg msg-assistant">
                <div className="msg-bubble msg-typing">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            )}
          </div>

          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="suggestion-chip"
                disabled={loading}
                onClick={() => sendQuestion(s)}
              >
                {s}
              </button>
            ))}
          </div>

          <form className="chat-input-row" onSubmit={handleSubmit}>
            <button
              type="button"
              className={`mic-btn ${voiceMode ? "voice-mode" : ""} ${listening ? "listening" : ""}`}
              onClick={voiceMode ? exitVoiceMode : enterVoiceMode}
              title={voiceMode ? "Exit voice mode" : "Start hands-free voice conversation"}
              aria-label={voiceMode ? "Exit voice mode" : "Start hands-free voice conversation"}
            >
              {voiceMode ? "⏹" : "🎤"}
            </button>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                listening
                  ? "Listening... (speak, then pause)"
                  : transcribing
                  ? "Transcribing..."
                  : voiceMode
                  ? "Voice mode active..."
                  : "Ask about alerts..."
              }
              disabled={loading || voiceMode}
              autoFocus
            />
            <button type="submit" disabled={loading || voiceMode || !input.trim()}>
              Send
            </button>
          </form>
          {transcribing && <div className="transcribing-banner">Transcribing your question...</div>}
          {error && <div className="error-banner">{error}</div>}
        </div>
      )}

      <button
        className="chat-widget-fab"
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Close assistant" : "Open assistant"}
        title="Deepinsight Assistant"
      >
        {open ? "×" : "💬"}
      </button>
    </>
  );
}
