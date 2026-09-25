#!/bin/bash
set -x
REPO=/home/wgtech/slm-llama3b-sqlite
VENV=/home/wgtech/slm-main/.venv
CU13_LIB=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib
CU12_WHISPER_LIB=$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
CUDA_RUNTIME_LIB=$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

cd "$REPO"
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$CU13_LIB:$CU12_WHISPER_LIB:$CUDA_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"

# This script previously only ever launched a new uvicorn process and never
# stopped the old one. If the old one was still holding port 8005 (the
# common case - it usually is, that's the whole reason to restart), the new
# process failed to bind and exited immediately, but nothing here checked
# for that: the script printed a fake success ("launched pid ...") while
# the OLD process, with the OLD code, kept right on serving every request.
# Found live: a code fix landed, this script reported success, and the
# backend kept answering with the pre-fix behavior for a further ~9 minutes
# until something else happened to free the port.
old_pid=$(ss -tlnp 2>/dev/null | awk '/:8005 /{print $0}' | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$old_pid" ]; then
    kill "$old_pid" 2>/dev/null
    for _ in $(seq 1 20); do
        ss -tln 2>/dev/null | grep -q ':8005 ' || break
        sleep 0.5
    done
    ss -tln 2>/dev/null | grep -q ':8005 ' && kill -9 "$old_pid" 2>/dev/null
fi

setsid nohup uvicorn chat.backend.main:app --host 127.0.0.1 --port 8005 > "$REPO/backend.log" 2>&1 < /dev/null &
new_pid=$!
disown

for _ in $(seq 1 30); do
    sleep 0.5
    curl -sf -o /dev/null http://127.0.0.1:8005/health && { echo "backend up, pid $new_pid"; exit 0; }
done
echo "FAILED: backend did not come up on port 8005 after restart (see $REPO/backend.log)" >&2
exit 1
