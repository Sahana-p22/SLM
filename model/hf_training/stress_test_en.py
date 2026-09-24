# hf_training/stress_test_en.py

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import torch


MODEL_PATH = "model/outputs/industrial_slm"

DEVICE = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("\n[INFO] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("\n[INFO] Loading Merged Model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()
print("\n[INFO] Model Ready\n")


def build_prompt(context):
    return f"""<|im_start|>system
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


def generate_english_only(context, temperature=0.8):
    prompt = build_prompt(context)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full.split("assistant", 1)[-1].strip()

    # pull just the English line(s), before "TAMIL:"
    if "TAMIL:" in reply:
        english = reply.split("TAMIL:")[0]
    else:
        english = reply
    english = english.replace("ENGLISH:", "").strip()
    return english


# =========================================================
# ROUND 1 — known types, wide inspection_time sweep, N samples
# =========================================================

print("\n" + "#" * 70)
print("# ROUND 1: known alert types, varied inspection_time, 5x each")
print("#" * 70)

known_types = ["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"]
time_values = [0.0, 1.2, 8.5, 22.7, 999.9]

for alert_type in known_types:
    print(f"\n=== {alert_type} ===")
    for t in time_values:
        ctx = {
            "alert_type": alert_type,
            "inspection_time": t,
            "objects_present": ["conveyor_belt", "person"],
            "cloth_detected": False
        }
        result = generate_english_only(ctx)
        print(f"  t={t:>6} -> {result}")


# =========================================================
# ROUND 2 — novel / unusual scenarios
# =========================================================

print("\n" + "#" * 70)
print("# ROUND 2: novel / edge scenarios")
print("#" * 70)

novel_cases = [
    {"alert_type": "HAND_TOUCH", "inspection_time": -5.0,
     "objects_present": ["person"], "cloth_detected": False,
     "note": "negative time"},

    {"alert_type": "fast_inspection", "inspection_time": 3.0,
     "objects_present": ["person"], "cloth_detected": False,
     "note": "lowercase alert_type"},

    {"alert_type": "SPILL_DETECTED", "inspection_time": 4.0,
     "objects_present": ["liquid", "person"], "cloth_detected": False,
     "note": "unseen alert type"},

    {"alert_type": "NO_HELMET", "inspection_time": 6.0,
     "objects_present": ["person"], "cloth_detected": False,
     "note": "unseen alert type"},

    {"alert_type": "HAND_TOUCH", "inspection_time": 8.5,
     "objects_present": ["person", "person", "person"], "cloth_detected": True,
     "note": "multiple people + cloth true (untrained field)"},

    {"alert_type": "MISSING_CLEANING", "inspection_time": 100000,
     "objects_present": ["workstation"], "cloth_detected": False,
     "note": "extreme large time"},
]

for ctx in novel_cases:
    note = ctx.pop("note")
    print(f"\n--- {note} ---")
    print(f"  input: {ctx}")
    for i in range(2):
        result = generate_english_only(ctx)
        print(f"  [{i+1}] {result}")

print("\n" + "#" * 70)
print("# DONE")
print("#" * 70)
