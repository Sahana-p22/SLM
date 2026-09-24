from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import (
    PeftModel
)

import torch

from model.hf_training.configs import (
    MODEL_PATH,
    LORA_OUTPUT_DIR
)


# =========================================================
# PATHS
# =========================================================

CHECKPOINT_PATH = str(
    LORA_OUTPUT_DIR / "checkpoint-1375"
)

MERGED_MODEL_PATH = (
    "model/outputs/industrial_slm"
)


# =========================================================
# LOAD TOKENIZER
# =========================================================

print(
    "\n[INFO] Loading Tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


# =========================================================
# LOAD BASE MODEL
# =========================================================

print(
    f"\n[INFO] Loading Base Model From {MODEL_PATH}..."
)

base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_PATH,

    torch_dtype=torch.float16,

    # Full fp16 3B weights (~6GB) don't fit in 4GB VRAM alongside the
    # adapter, so merge on CPU instead of the GPU used for training.
    device_map="cpu",

    trust_remote_code=True
)


# =========================================================
# LOAD LORA
# =========================================================

print(
    "\n[INFO] Loading LoRA Adapter..."
)

model = PeftModel.from_pretrained(

    base_model,

    CHECKPOINT_PATH
)


# =========================================================
# MERGE
# =========================================================

print(
    "\n[INFO] Merging Adapter Into Base Model..."
)

merged_model = model.merge_and_unload()


# =========================================================
# SAVE
# =========================================================

print(
    "\n[INFO] Saving Merged Model..."
)

merged_model.save_pretrained(
    MERGED_MODEL_PATH
)

tokenizer.save_pretrained(
    MERGED_MODEL_PATH
)

print(
    "\n[SUCCESS] Model Saved To:"
)

print(
    MERGED_MODEL_PATH
)