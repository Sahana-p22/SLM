# chat/backend/config.py
#
# Swapped from the original Qwen2.5-3B-Instruct to Llama-3.2-3B-Instruct
# (this is the slm-llama3b copy of slm-main, model swapped, context window
# increased to 4096 — see llm_query.py's n_ctx). Weights are shared with
# slm-main via the models/ symlink at the repo root (not duplicated).

import os
from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Anchored to the repo root (this file's own location), not the process's
# current working directory. A bare relative path here silently resolved
# against whatever CWD happened to start the process - fine for the live
# backend (uvicorn is always launched with cwd=REPO_ROOT), but a script
# run from a different directory (found live: a test script run from
# chat/backend/tests/) would look for "chat/backend/tests/models/..."
# instead, hit .exists() == False, and MODEL_PATH would have silently
# fallen back to the HF hub name - or, for GGUF_MODEL_PATH, llama.cpp
# would raise "Model path does not exist" outright.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_MODEL_DIR = _REPO_ROOT / "models" / "Llama-3.2-3B-Instruct"

MODEL_PATH = (
    str(_LOCAL_MODEL_DIR)
    if (_LOCAL_MODEL_DIR / "model.safetensors.index.json").exists()
    else MODEL_NAME
)

# GGUF (llama.cpp) build, used for actual chat inference. Q8_0 chosen per
# the earlier RTX-vs-Metis comparison (fqc-gpu/COMPARISON.md): near-F16
# accuracy (12/12 vs F16's 11/12) at roughly 2/3 the VRAM and faster decode.
# Overridable via FQC_GGUF_MODEL_PATH (e.g. to A/B another quant against
# the full validated test suite) without editing this file - default is
# unchanged.
GGUF_MODEL_PATH = os.environ.get(
    "FQC_GGUF_MODEL_PATH",
    str(_REPO_ROOT / "models" / "Llama-3.2-3B-Instruct-GGUF" / "Llama-3.2-3B-Instruct-Q8_0.gguf"),
)
