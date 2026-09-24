import json
import random

from pathlib import Path

from model.hf_training.high_quality_templates import (

    FAST_INSPECTION_ALERTS,

    HAND_TOUCH_ALERTS
)

from model.config import (

    CONTEXT_JSON_DIR,

    NARRATIONS_DIR
)


# =========================================================
# ALERT TYPES
# =========================================================

ALERT_TYPES = [

    "FAST_INSPECTION",

    "HAND_TOUCH"
]


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================

NUM_SAMPLES = 5000


for i in range(NUM_SAMPLES):

    alert_type = random.choice(
        ALERT_TYPES
    )


    # -----------------------------------------------------
    # FAST INSPECTION
    # -----------------------------------------------------

    if alert_type == "FAST_INSPECTION":

        context = {

            "alert_type": "FAST_INSPECTION",

            "inspection_time": round(
                random.uniform(3.0, 5.5),
                1
            ),

            "objects_present": [

                "back_panel",

                "gloved_hand"
            ],

            "cloth_detected": False
        }

        narration = random.choice(
            FAST_INSPECTION_ALERTS
        )


    # -----------------------------------------------------
    # HAND TOUCH
    # -----------------------------------------------------

    else:

        context = {

            "alert_type": "HAND_TOUCH",

            "inspection_time": round(
                random.uniform(7.0, 12.0),
                1
            ),

            "objects_present": [

                "back_panel",

                "person"
            ],

            "cloth_detected": False
        }

        narration = random.choice(
            HAND_TOUCH_ALERTS
        )


    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    context_path = (

        CONTEXT_JSON_DIR /
        f"{i}.json"
    )

    with open(
        context_path,
        "w"
    ) as f:

        json.dump(
            context,
            f,
            indent=2
        )


    # -----------------------------------------------------
    # SAVE NARRATION
    # -----------------------------------------------------

    narration_path = (

        NARRATIONS_DIR /
        f"{i}.txt"
    )

    with open(
        narration_path,
        "w"
    ) as f:

        f.write(
            narration
        )


print(
    f"\n[INFO] Generated {NUM_SAMPLES} Samples\n"
)