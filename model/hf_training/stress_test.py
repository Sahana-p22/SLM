# hf_training/stress_test.py

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


def generate(context, temperature=0.7):
    prompt = build_prompt(context)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("assistant", 1)[-1].strip()


# =========================================================
# TEST CASES
# =========================================================

test_cases = [

    # In-distribution: one of each known alert type
    {"alert_type": "FAST_INSPECTION", "inspection_time": 2.1,
     "objects_present": ["conveyor_belt", "person"], "cloth_detected": False},

    {"alert_type": "HAND_TOUCH", "inspection_time": 8.5,
     "objects_present": ["back_panel", "person"], "cloth_detected": False},

    {"alert_type": "MISSING_CLEANING", "inspection_time": 15.0,
     "objects_present": ["workstation", "person"], "cloth_detected": True},

    # Edge cases / out-of-distribution
    {"alert_type": "HAND_TOUCH", "inspection_time": 0.3,
     "objects_present": ["person"], "cloth_detected": False},

    {"alert_type": "FAST_INSPECTION", "inspection_time": 45.0,
     "objects_present": ["conveyor_belt", "back_panel", "person", "tool_cart"],
     "cloth_detected": True},

    {"alert_type": "MISSING_CLEANING", "inspection_time": 8.5,
     "objects_present": [], "cloth_detected": False},

    # Unknown alert type (not in training data at all)
    {"alert_type": "FIRE_HAZARD", "inspection_time": 5.0,
     "objects_present": ["gas_canister", "person"], "cloth_detected": False},
]

for i, ctx in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"[TEST {i}] {ctx}")
    print(f"{'='*60}")
    for j in range(2):
        result = generate(ctx)
        print(f"\n--- sample {j+1} ---")
        print(result)

print(f"\n{'='*60}")
print("[DONE]")
