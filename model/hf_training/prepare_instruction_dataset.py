import json

from pathlib import Path
from tqdm import tqdm

from model.config import (

    CONTEXT_JSON_DIR,

    NARRATIONS_DIR
)

from model.hf_training.configs import (
    DATASET_OUTPUT_PATH
)


# =========================================================
# LOAD FILES
# =========================================================

context_files = list(
    CONTEXT_JSON_DIR.glob("*.json")
)

print(
    f"[INFO] Found "
    f"{len(context_files)} samples"
)


# =========================================================
# BUILD DATASET
# =========================================================

with open(
    DATASET_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as outfile:

    for context_path in tqdm(context_files):

        sample_id = context_path.stem


        # -------------------------------------------------
        # LOAD CONTEXT
        # -------------------------------------------------

        with open(
            context_path,
            "r"
        ) as f:

            context = json.load(f)


        # -------------------------------------------------
        # LOAD NARRATION
        # -------------------------------------------------

        narration_path = (

            NARRATIONS_DIR /
            f"{sample_id}.txt"
        )

        with open(
            narration_path,
            "r",
            encoding="utf-8"
        ) as f:

            narration = f.read().strip()


        # -------------------------------------------------
        # BUILD PROMPT
        # -------------------------------------------------

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
{narration}<|im_end|>"""


        # -------------------------------------------------
        # SAVE JSONL
        # -------------------------------------------------

        sample = {

            "text": prompt
        }

        outfile.write(

            json.dumps(sample)

            + "\n"
        )


print(
    f"\n[INFO] Dataset Saved To:\n"
    f"{DATASET_OUTPUT_PATH}"
)