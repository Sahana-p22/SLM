# hf_training/test_merged_model.py

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import torch


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = (
    "model/outputs/industrial_slm"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda"


# =========================================================
# QUANTIZATION
# =========================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True
)


# =========================================================
# LOAD TOKENIZER
# =========================================================

print(
    "\n[INFO] Loading Tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# LOAD MODEL
# =========================================================

print(
    "\n[INFO] Loading Merged Model..."
)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_PATH,

    quantization_config=bnb_config,

    device_map="auto"
)

model.eval()

print(
    "\n[INFO] Model Ready"
)


# =========================================================
# TEST EVENT
# =========================================================

context = {

    "alert_type": "HAND_TOUCH",

    "inspection_time": 8.5,

    "objects_present": [

        "back_panel",

        "person"
    ],

    "cloth_detected": False
}


# =========================================================
# PROMPT
# =========================================================

prompt = f"""<|im_start|>system
You are a factory floor safety assistant. Generate short industrial alerts.<|im_end|>
<|im_start|>user
Alert Type:
{context['alert_type']}

Inspection Time:
{context['inspection_time']} seconds

Objects Present:
{', '.join(context['objects_present'])}

Cloth Detected:
{context['cloth_detected']}<|im_end|>
<|im_start|>assistant
"""


# =========================================================
# TOKENIZE
# =========================================================

inputs = tokenizer(

    prompt,

    return_tensors="pt"
).to(DEVICE)


# =========================================================
# GENERATE
# =========================================================

print(
    "\n[INFO] Generating Alert...\n"
)

with torch.no_grad():

    outputs = model.generate(

        **inputs,

        max_new_tokens=100,

        do_sample=True,

        temperature=0.7,

        top_p=0.9,

        top_k=40,

        repetition_penalty=1.2,

        eos_token_id=tokenizer.eos_token_id,

        pad_token_id=tokenizer.eos_token_id
    )


# =========================================================
# DECODE
# =========================================================

text = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)

print(
    "\n=============================="
)

print(
    "\n[RAW MODEL OUTPUT]\n"
)

print(text)

print(
    "\n==============================\n"
)