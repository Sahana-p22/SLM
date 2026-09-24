# dataset_builder/build_dataset.py

import json
from tqdm import tqdm

from model.config import (

    DETECTION_JSON_DIR,

    CONTEXT_JSON_DIR,

    NARRATIONS_DIR
)

from model.dataset_builder.context_builder import (
    build_context
)

from model.dataset_builder.narration_generator import (
    generate_narration
)

from model.dataset_builder.dataset_writer import (
    save_json
)


# =========================================================
# LOAD ALL DETECTION JSON FILES
# =========================================================

json_files = list(
    DETECTION_JSON_DIR.glob("*.json")
)

print(
    f"[INFO] Found "
    f"{len(json_files)} detection files"
)


# =========================================================
# PROCESS EACH DETECTION FILE
# =========================================================

for json_path in tqdm(json_files):

    # -----------------------------------------------------
    # LOAD DETECTION JSON
    # -----------------------------------------------------

    with open(
        json_path,
        "r"
    ) as f:

        detection_data = json.load(f)


    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = build_context(
        detection_data
    )


    # -----------------------------------------------------
    # GENERATE BILINGUAL NARRATION
    # -----------------------------------------------------

    narration = generate_narration(
        context
    )


    sample_id = json_path.stem


    # -----------------------------------------------------
    # SAVE CONTEXT JSON
    # -----------------------------------------------------

    context_path = (
        CONTEXT_JSON_DIR /
        f"{sample_id}.json"
    )

    save_json(
        context_path,
        context
    )


    # -----------------------------------------------------
    # SAVE NARRATION TEXT
    # -----------------------------------------------------

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


    print(
        f"[INFO] Processed: {sample_id}"
    )


# =========================================================
# COMPLETE
# =========================================================

print(
    "[INFO] Dataset generation complete"
)