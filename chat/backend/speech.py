# chat/backend/speech.py
#
# Local speech-to-text via faster-whisper (CTranslate2 port of OpenAI's
# Whisper, MIT license) — runs entirely on this machine, no third-party
# API call, so there's no ambiguity about commercial usability the way
# there is with browser-builtin speech recognition (which routes audio
# through the browser vendor's servers under undocumented terms).
#
# Runs on GPU with the medium model: the query-parsing LLM only commits
# a few GB of VRAM (a GGUF-quantized 3B model), leaving plenty of headroom
# on a 12GB card for accuracy-focused transcription. Voice input and the
# LLM never run inference at the same instant in a single request's
# lifecycle (transcription always finishes before the chat pipeline
# starts), so there's no contention to guard against the way `_llm_lock`
# guards concurrent LLM calls.

from faster_whisper import WhisperModel

# A voice clip for a single spoken question is a few hundred KB at most;
# 25MB comfortably covers minutes of uncompressed audio while still
# rejecting an unbounded upload before it's ever buffered into memory or
# handed to the transcriber (there was previously no limit at all here).
MAX_AUDIO_BYTES = 25 * 1024 * 1024

_model = None


def _load_model():
    global _model
    if _model is not None:
        return
    print("[speech] Loading Whisper (medium, CUDA, float16)...")
    try:
        _model = WhisperModel("medium", device="cuda", compute_type="float16")
    except Exception as exc:
        print(f"[speech] GPU load failed ({exc}), falling back to CPU/base.")
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    print("[speech] Whisper ready.")


def transcribe(audio_path: str) -> str:
    _load_model()
    segments, _info = _model.transcribe(audio_path, language="en", vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()
