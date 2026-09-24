#!/bin/bash
# Restart the llama3b backend. Kept as a script file rather than an inline ssh
# command because a pkill pattern typed inline matches the ssh command itself.
REPO=/home/wgtech/slm-llama3b
VENV=/home/wgtech/slm-main/.venv
SP=$VENV/lib/python3.12/site-packages/nvidia
pkill -f 'chat.backend.main:app' 2>/dev/null
sleep 3
cd "$REPO" || exit 1
export PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$SP/cu13/lib:$SP/cublas/lib:$SP/cudnn/lib:$LD_LIBRARY_PATH
nohup "$VENV/bin/python" -m uvicorn chat.backend.main:app \
    --host 127.0.0.1 --port 8002 > /tmp/backend8002.log 2>&1 &
for i in $(seq 1 60); do
    sleep 2
    if curl -sf -o /dev/null http://127.0.0.1:8002/health; then
        echo "backend up after ${i}0s"; exit 0
    fi
done
echo "backend did not come up"; tail -20 /tmp/backend8002.log; exit 1
