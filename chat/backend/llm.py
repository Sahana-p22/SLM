# backend/llm.py — thin llama.cpp wrapper, extracted from the original
# llm_query.py's _load_model/_chat/_chat_stream (unchanged mechanics: same
# GGUF path, same n_gpu_layers/n_ctx, same greedy-by-default /
# sampled-on-retry generation split, same single-instance lock). Kept as its
# own module (matching the Retail deployment's llm.py) so pipeline.py doesn't
# need to know about llama.cpp directly.

import os
import threading

from chat.backend.config import GGUF_MODEL_PATH

_llm = None
_llm_lock = threading.Lock()


def _load_model():
    global _llm
    if _llm is not None:
        return
    with _llm_lock:
        if _llm is not None:
            return
        print("[llm] Loading Llama-3.2-3B-Instruct (GGUF, Q8_0) via llama.cpp...")
        if hasattr(os, "add_dll_directory"):
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            os.add_dll_directory(torch_lib)
        # n_gpu_layers=0 (CPU-only): the RTX 5070 here is concurrently shared
        # with two other live deployments (slm-llama3b on 8002, the separate
        # slm-llama3b-sqlite migration on 8005), which together already hold
        # essentially all 12GB of VRAM. This experiment must not disturb
        # either of those, so it runs on CPU instead of contending for GPU
        # memory. Slower, but functionally identical output — acceptable for
        # a hallucination/accuracy comparison, where correctness is what's
        # being measured, not raw latency.
        from llama_cpp import Llama
        _llm = Llama(model_path=GGUF_MODEL_PATH, n_gpu_layers=0, n_ctx=4096, verbose=False)
        print("[llm] Model ready.")


def available() -> bool:
    try:
        _load_model()
        return _llm is not None
    except Exception as e:
        print(f"[llm] load failed: {e}")
        return False


def complete(system: str, user: str, is_retry: bool = False, max_tokens: int = 500) -> str:
    _load_model()
    gen_kwargs = {"temperature": 0.4, "top_p": 0.9, "top_k": 50} if is_retry else {"temperature": 0.0}
    with _llm_lock:
        out = _llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, repeat_penalty=1.1, **gen_kwargs)
    return (out["choices"][0]["message"]["content"] or "").strip()


def complete_stream(system: str, user: str, max_tokens: int = 220):
    _load_model()
    with _llm_lock:
        stream = _llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, repeat_penalty=1.1, temperature=0.0, stream=True)
        for chunk in stream:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                yield content
