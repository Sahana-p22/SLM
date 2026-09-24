# dataset_builder/dataset_writer.py

import json
from pathlib import Path

def save_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )