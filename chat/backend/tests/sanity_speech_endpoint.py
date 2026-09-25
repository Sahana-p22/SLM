"""Unit sanity checks for the speech-to-text path (chat/backend/speech.py
and the /speech-to-text endpoint in chat/backend/main.py). No prior test
coverage existed for this path at all before this file.

Covers two things found during a synthetic-audio evaluation of the live
endpoint (18 espeak-ng clips + silence/degraded/noisy edge cases, all
transcribed correctly or gracefully rejected, ~150-210ms each on the
GPU/medium model - see the eval report for the full battery):

1. The CUDA -> CPU/base fallback in `_load_model()` actually fires and
   loads a working model when GPU init raises, instead of leaving
   `_model` as None or raising past the caller. This path had never
   actually been exercised on this machine (GPU load has always
   succeeded here), so it's covered with a fake `WhisperModel` rather
   than by forcing a real CUDA failure.
2. `MAX_AUDIO_BYTES` exists and is a sane, generous-but-finite size -
   before this fix, `/speech-to-text` read an uploaded file of any size
   fully into memory with no cap at all.

Does not load a real Whisper model (both real model sizes take real
VRAM/time to load) - this is deliberately a fast, GPU-free unit test,
run in the same `fast` tier as the other sanity_*.py files.
"""
import sys
import types

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("MAX_AUDIO_BYTES is defined and sane")
from chat.backend import speech as S

check("MAX_AUDIO_BYTES exists", hasattr(S, "MAX_AUDIO_BYTES"))
check("MAX_AUDIO_BYTES is a positive, finite size (between 1MB and 200MB)",
      1 * 1024 * 1024 <= S.MAX_AUDIO_BYTES <= 200 * 1024 * 1024,
      str(getattr(S, "MAX_AUDIO_BYTES", None)))

print("\n_load_model() falls back to CPU/base when CUDA init raises")


class _FakeModel:
    def __init__(self, size, device, compute_type):
        self.size, self.device, self.compute_type = size, device, compute_type
        if device == "cuda":
            raise RuntimeError("no CUDA device (simulated)")


fake_module = types.SimpleNamespace(WhisperModel=_FakeModel)
real_module = sys.modules.get("faster_whisper")
sys.modules["faster_whisper"] = fake_module

# speech.py imports WhisperModel at module load time, so re-import fresh
# against the faked module rather than relying on the already-imported one.
import importlib

S_fresh = importlib.reload(S)
S_fresh._model = None
S_fresh._load_model()

check("falls back to the base/CPU model after the fake CUDA failure",
      getattr(S_fresh._model, "size", None) == "base" and getattr(S_fresh._model, "device", None) == "cpu",
      f"size={getattr(S_fresh._model, 'size', None)} device={getattr(S_fresh._model, 'device', None)}")
check("fallback model does not itself raise",
      S_fresh._model is not None)

# Restore the real module and reload speech.py against it so nothing else
# in this test process is left pointed at the fake.
if real_module is not None:
    sys.modules["faster_whisper"] = real_module
else:
    del sys.modules["faster_whisper"]
importlib.reload(S)
S._model = None

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
