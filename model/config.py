# config.py

from pathlib import Path

BASE_DIR = Path("model/data")

ALERT_FRAMES_DIR = BASE_DIR / "alert_frames"
DETECTION_JSON_DIR = BASE_DIR / "detection_json"

CONTEXT_JSON_DIR = BASE_DIR / "context_json"
NARRATIONS_DIR = BASE_DIR / "narrations"

TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

ALL_DIRS = [

    ALERT_FRAMES_DIR,
    DETECTION_JSON_DIR,

    CONTEXT_JSON_DIR,
    NARRATIONS_DIR,

    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]