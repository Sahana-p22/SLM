# hf_training/configs.py

from pathlib import Path


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# huggingface_hub's downloader is unreliable in this environment (keeps
# restarting instead of resuming), so the weights were fetched manually via
# curl into this local directory. Fall back to the hub ID if it's absent.
_LOCAL_MODEL_DIR = Path("models/Qwen2.5-3B-Instruct")

MODEL_PATH = (
    str(_LOCAL_MODEL_DIR)
    if (_LOCAL_MODEL_DIR / "model.safetensors.index.json").exists()
    else MODEL_NAME
)


# =========================================================
# PATHS
# =========================================================

DATASET_OUTPUT_PATH = Path(
    "model/outputs/high_quality_dataset.jsonl"
)

LORA_OUTPUT_DIR = Path(
    "model/outputs/qwen2.5_3b_lora"
)

LORA_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)