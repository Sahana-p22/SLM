#!/bin/bash
set -x
REPO=/home/wgtech/slm-llama3b
VENV=/home/wgtech/slm-main/.venv
CU13_LIB=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib
CU12_WHISPER_LIB=$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
CUDA_RUNTIME_LIB=$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

cd "$REPO"
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$CU13_LIB:$CU12_WHISPER_LIB:$CUDA_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
export FQC_GGUF_MODEL_PATH="$REPO/models/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
setsid nohup uvicorn chat.backend.main:app --host 127.0.0.1 --port 8003 > "$REPO/backend_8b.log" 2>&1 < /dev/null &
disown
echo "launched pid $!"
