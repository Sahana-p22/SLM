#!/bin/bash
# Starts MongoDB, the FastAPI backend, and the Vite frontend for the
# slm-llama3b app (slm-main, model swapped Qwen2.5-3B -> Llama-3.2-3B,
# context window 1024 -> 4096). Reuses slm-main's venv, node_modules, and
# MongoDB instance (shared, not duplicated) via symlinks / direct paths.
# Runs on ports 8002 (backend) / 5174 (frontend) so it can coexist with
# slm-main (8001 / 5173) if both are ever up at once.
#
# NOTE: do not run this while the GPU is in use by someone else — this
# loads a model onto it. Check with nvidia-smi first.

set -u

REPO=/home/wgtech/slm-llama3b
VENV=/home/wgtech/slm-main/.venv
MONGOBIN=/home/wgtech/mongodb-portable/mongodb-linux-x86_64-ubuntu2404-8.0.11/bin
NODEBIN=/home/wgtech/node-portable/bin
CU13_LIB=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib
CU12_WHISPER_LIB=$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
# llama.cpp's CUDA build links libcudart.so.12 specifically, which isn't
# on the cu13 pip package's own lib dir (that one ships libcudart.so.13) -
# without this, the backend process starts and answers /health fine, but
# the model fails to load on the first real /chat request ("Failed to
# load shared library ... libcudart.so.12: cannot open shared object
# file"), since llama.cpp only loads on first use, not at startup.
CUDA_RUNTIME_LIB=$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

port_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&- 3<&-
}

if port_open 27017; then
  echo "[start-all] MongoDB already running on 27017"
else
  echo "[start-all] Starting MongoDB..."
  setsid nohup "$MONGOBIN/mongod" --dbpath /home/wgtech/mongodb-data --port 27017 --bind_ip 127.0.0.1     > /home/wgtech/mongodb-data/mongod.log 2>&1 < /dev/null &
  disown
fi

if port_open 8002; then
  echo "[start-all] Backend already running on 8002"
else
  echo "[start-all] Starting backend (Llama-3.2-3B, n_ctx=4096)..."
  (
    cd "$REPO"
    source "$VENV/bin/activate"
    export LD_LIBRARY_PATH="$CU13_LIB:$CU12_WHISPER_LIB:$CUDA_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
    setsid nohup uvicorn chat.backend.main:app --host 127.0.0.1 --port 8002       > "$REPO/backend.log" 2>&1 < /dev/null &
    disown
  )
fi

if port_open 5174; then
  echo "[start-all] Frontend already running on 5174"
else
  echo "[start-all] Starting frontend..."
  (
    cd "$REPO/chat/frontend"
    export PATH="$NODEBIN:$PATH"
    setsid nohup npm run dev -- --host 127.0.0.1 --port 5174       > "$REPO/frontend.log" 2>&1 < /dev/null &
    disown
  )
fi

echo "[start-all] Waiting for services to come up..."
for i in $(seq 1 30); do
  if port_open 27017 && port_open 8002 && port_open 5174; then
    echo "[start-all] All up. Dashboard: http://127.0.0.1:5174"
    exit 0
  fi
  sleep 1
done
echo "[start-all] Timed out waiting for one or more services — check backend.log / frontend.log / mongodb-data/mongod.log"
exit 1
