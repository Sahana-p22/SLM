# dataset_builder/generate_bulk_dataset.py

import random
import uuid

from model.dataset_builder.narration_generator import (
    generate_narration
)

from model.dataset_builder.dataset_writer import (
    save_json
)

from model.config import (
    CONTEXT_JSON_DIR,
    NARRATIONS_DIR
)


# =========================================================
# CONFIG
# =========================================================

NUM_SAMPLES = 3000


# =========================================================
# REAL ALERT TYPES
# =========================================================

ALERT_TYPES = [

    "FAST_INSPECTION",

    "HAND_TOUCH",

    "MISSING_CLEANING"
]


# =========================================================
# REAL DETECTION CLASSES
# =========================================================

OBJECT_CLASSES = [

    "back_panel",

    "gloved_hand",

    "approved_tray",

    "microfiber_cloth",

    "person"
]


# =========================================================
# RANDOM OBJECT GENERATOR
# =========================================================

def generate_objects(alert_type):

    objects = [

        "back_panel",

        "approved_tray",

        "person"
    ]


    # -----------------------------------------------------
    # HAND TOUCH
    # -----------------------------------------------------

    if alert_type == "HAND_TOUCH":

        objects.append(
            "gloved_hand"
        )

        # simulate missing cloth
        if random.random() > 0.7:

            objects.append(
                "microfiber_cloth"
            )


    # -----------------------------------------------------
    # FAST INSPECTION
    # -----------------------------------------------------

    elif alert_type == "FAST_INSPECTION":

        if random.random() > 0.5:

            objects.append(
                "microfiber_cloth"
            )

        objects.append(
            "gloved_hand"
        )


    # -----------------------------------------------------
    # MISSING CLEANING
    # -----------------------------------------------------

    elif alert_type == "MISSING_CLEANING":

        objects.append(
            "gloved_hand"
        )

        # cloth deliberately absent to represent missing cleaning step


    return list(set(objects))


# =========================================================
# CONTEXT GENERATOR
# =========================================================

def generate_random_context():

    alert_type = random.choice(
        ALERT_TYPES
    )


    # -----------------------------------------------------
    # FAST INSPECTION LOGIC
    # -----------------------------------------------------

    if alert_type == "FAST_INSPECTION":

        inspection_time = round(
            random.uniform(2, 10),
            1
        )

        cloth_detected = random.choice(
            [True, False]
        )


    # -----------------------------------------------------
    # HAND TOUCH LOGIC
    # -----------------------------------------------------

    elif alert_type == "HAND_TOUCH":

        inspection_time = round(
            random.uniform(15, 45),
            1
        )

        cloth_detected = False


    # -----------------------------------------------------
    # MISSING CLEANING LOGIC
    # -----------------------------------------------------

    else:

        inspection_time = round(
            random.uniform(15, 45),
            1
        )

        cloth_detected = False


    context = {

        "alert_type": alert_type,

        "inspection_time":
        inspection_time,

        "required_time": 30,

        "cloth_detected":
        cloth_detected,

        "objects_present":
        generate_objects(alert_type),

        "workflow_stage":
        "inspection"
    }

    return context


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        f"[INFO] Generating "
        f"{NUM_SAMPLES} realistic samples..."
    )

    for i in range(NUM_SAMPLES):

        # -------------------------------------------------
        # GENERATE CONTEXT
        # -------------------------------------------------

        context = generate_random_context()


        # -------------------------------------------------
        # GENERATE NARRATION
        # -------------------------------------------------

        narration = generate_narration(
            context
        )


        # -------------------------------------------------
        # SAMPLE ID
        # -------------------------------------------------

        sample_id = str(
            uuid.uuid4()
        )


        # -------------------------------------------------
        # SAVE CONTEXT
        # -------------------------------------------------

        context_path = (
            CONTEXT_JSON_DIR /
            f"{sample_id}.json"
        )

        save_json(
            context_path,
            context
        )


        # -------------------------------------------------
        # SAVE NARRATION
        # -------------------------------------------------

        narration_path = (
            NARRATIONS_DIR /
            f"{sample_id}.txt"
        )

        narration_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            narration_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "ENGLISH:\n"
            )

            f.write(
                narration["english"]
            )

            f.write(
                "\n\n"
            )

            f.write(
                "TAMIL:\n"
            )

            f.write(
                narration["tamil"]
            )

            f.write("\n")


        # -------------------------------------------------
        # LOG
        # -------------------------------------------------

        if (i + 1) % 100 == 0:

            print(
                f"[INFO] Generated "
                f"{i+1} samples"
            )


    print(
        "[INFO] Realistic dataset generation complete"
    )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()