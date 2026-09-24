# dataset_builder/synthetic_generator.py

import random

def generate_synthetic_context():

    alert_types = [

        "FAST_INSPECTION",
        "HAND_TOUCH",
        "MISSING_CLEANING"
    ]

    alert_type = random.choice(
        alert_types
    )

    context = {

        "alert_type": alert_type,

        "inspection_time":
        round(random.uniform(2, 10), 2),

        "required_time": 30,

        "cloth_detected":
        random.choice([True, False]),

        "objects_present": [

            "back_panel",
            "gloved_hand",
            "approved_tray"
        ],

        "workflow_stage":
        "inspection"
    }

    return context