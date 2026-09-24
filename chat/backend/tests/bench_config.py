"""Shared configuration for the FQC chatbot benchmark suite.

Every hardware-specific path lives here and ONLY here, read from
environment variables with defaults matching the deployment this suite
was built against - so moving to different hardware (a new GPU box, a
different venv layout, a different model quantization) means editing
environment variables or this one file, never hunting through a dozen
scripts for a hardcoded path.

Usage in any benchmark script:
    from bench_config import API_BASE, CHAT_URL, restart_backend, ...
"""
import os
import subprocess
import time

import requests

# ---------------------------------------------------------------------
# Where the deployed backend lives
# ---------------------------------------------------------------------
BACKEND_HOST = os.environ.get("FQC_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = os.environ.get("FQC_BACKEND_PORT", "8002")
API_BASE = os.environ.get("FQC_API_BASE", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
CHAT_URL = f"{API_BASE}/chat"
CHAT_STREAM_URL = f"{API_BASE}/chat/stream"
HEALTH_URL = f"{API_BASE}/health"

# ---------------------------------------------------------------------
# Where the code and Python environment live
# ---------------------------------------------------------------------
REPO_ROOT = os.environ.get("FQC_REPO_ROOT", "/home/wgtech/slm-llama3b")
VENV_DIR = os.environ.get("FQC_VENV_DIR", "/home/wgtech/slm-main/.venv")
VENV_PYTHON = os.environ.get("FQC_VENV_PYTHON", os.path.join(VENV_DIR, "bin", "python"))

# ---------------------------------------------------------------------
# GPU/CUDA library paths (llama-cpp-python's pip-installed CUDA wheels
# need these on LD_LIBRARY_PATH; leave FQC_CUDA_LIB_DIRS empty - or set
# it to the literal string "none" - on a CPU-only box, or one where
# CUDA is already on the system LD_LIBRARY_PATH via /etc/ld.so.conf).
# ---------------------------------------------------------------------
_default_cuda_dirs = ":".join([
    # Both cu13 and the plain (cu12-era) cuda_runtime dir are included -
    # a shared venv can have its actually-required libcudart SONAME
    # change out from under this app when another, unrelated project
    # sharing the same venv reinstalls/upgrades llama-cpp-python or
    # its CUDA wheels (found live: another app on this box switched
    # the installed build from wanting libcudart.so.13 to .so.12).
    # Both directories can safely coexist on LD_LIBRARY_PATH since the
    # dynamic linker resolves by exact SONAME, not by directory order.
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cu13/lib",
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cublas/lib",
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn/lib",
])
CUDA_LIB_DIRS = os.environ.get("FQC_CUDA_LIB_DIRS", _default_cuda_dirs)

# ---------------------------------------------------------------------
# The raw model file, for scripts that load it directly in-process
# (token_timing_bench.py) rather than going through the HTTP API.
# ---------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "FQC_MODEL_PATH",
    f"{REPO_ROOT}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf",
)

# ---------------------------------------------------------------------
# MongoDB - db.py already reads MONGO_URI/DB_NAME itself in most
# versions of this codebase; these are here for scripts (the scale-test
# generator) that talk to Mongo directly without going through db.py.
# ---------------------------------------------------------------------
MONGO_URI = os.environ.get("FQC_MONGO_URI", "mongodb://localhost:27017")
SCALE_TEST_DB = os.environ.get("FQC_SCALE_TEST_DB", "slm_safety_scale")


def env_with_cuda():
    """A copy of the current environment with LD_LIBRARY_PATH extended
    for CUDA, for subprocess.Popen calls that spawn a fresh Python
    process needing GPU access (a backend restart, the token-timing
    microbenchmark)."""
    env = dict(os.environ)
    if CUDA_LIB_DIRS and CUDA_LIB_DIRS.lower() != "none":
        env["LD_LIBRARY_PATH"] = CUDA_LIB_DIRS + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = REPO_ROOT + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def wait_for_health(timeout_s=120, poll_every_s=1):
    """Polls HEALTH_URL until the backend responds 200 or timeout_s
    elapses. Returns True if it came up, False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(HEALTH_URL, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(poll_every_s)
    return False


def restart_backend(db_name=None, collection_name=None, extra_wait_s=0):
    """Stops any running instance of the backend and starts a fresh one,
    optionally pointed at a different Mongo database/collection (used by
    db_size_vs_speed.py to test against different data volumes). Returns
    True once the backend responds healthy, False on timeout.

    Restores db.py to its original content immediately after the new
    process has read it (whether or not this call overrode the DB name),
    so a crash or an early return here never leaves a live deployment
    quietly pointed at scratch data.
    """
    subprocess.run(["pkill", "-f", "chat.backend.main:app"], capture_output=True)
    time.sleep(2)

    db_py_path = os.path.join(REPO_ROOT, "chat", "backend", "db.py")
    original = None
    if db_name or collection_name:
        original = open(db_py_path).read()
        patched = original
        if db_name:
            patched = patched.replace('DB_NAME = "slm_safety"', f'DB_NAME = "{db_name}"')
        if collection_name:
            patched = patched.replace('ALERTS_COLLECTION = "alerts"', f'ALERTS_COLLECTION = "{collection_name}"')
        open(db_py_path, "w").write(patched)

    try:
        subprocess.Popen(
            [VENV_PYTHON, "-m", "uvicorn", "chat.backend.main:app",
             "--host", BACKEND_HOST, "--port", BACKEND_PORT],
            cwd=REPO_ROOT, env=env_with_cuda(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        ok = wait_for_health()
        if extra_wait_s:
            time.sleep(extra_wait_s)
        return ok
    finally:
        if original is not None:
            open(db_py_path, "w").write(original)
