# dataset_builder/narration_generator.py

import random


# =========================================================
# FAST INSPECTION ALERTS
# =========================================================

FAST_INSPECTION_ALERTS = [

    {
        "english":
        "Inspection workflow appears rushed with duration of {time} seconds. Please continue inspection carefully.",

        "tamil":
        "இன்ஸ்பெக்ஷன் கொஞ்சம் அவசரமா நடக்குது. இன்னும் கொஞ்சம் கவனமா செக் பண்ணுங்க."
    },

    {
        "english":
        "Panel inspection completed too quickly in {time} seconds. Please inspect properly before approval.",

        "tamil":
        "பேனல் ரொம்ப சீக்கிரமா செக் பண்ணப்பட்டுருக்கு. அப்புரூவ் பண்ணதுக்கு முன்னாடி நன்றாக செக் பண்ணுங்க."
    },

    {
        "english":
        "Inspection timing appears below SOP standards at {time} seconds. Please follow standard inspection timing.",

        "tamil":
        "இன்ஸ்பெக்ஷன் டைம் குறைவா இருக்கு. சரியான நேரம் எடுத்துக்கிட்டு செக் பண்ணுங்க."
    },

    {
        "english":
        "Inspection process finished too fast in {time} seconds. Please re-check the panel.",

        "tamil":
        "இன்ஸ்பெக்ஷன் ரொம்ப சீக்கிரமா முடிஞ்சுருக்கு. பேனலை மறுபடியும் செக் பண்ணுங்க."
    }
]


# =========================================================
# HAND TOUCH ALERTS
# =========================================================

HAND_TOUCH_ALERTS = [

    {
        "english":
        "Direct hand contact detected. Please use microfiber protection.",

        "tamil":
        "கையால நேரடியாக டச் பண்ணாதீங்க. மைக்ரோஃபைபர் பயன்படுத்துங்க."
    },

    {
        "english":
        "Unsafe panel handling detected. Please follow handling SOP.",

        "tamil":
        "பேனல் சரியாக ஹேண்டில் பண்ணல. SOP படி ஹேண்டில் பண்ணுங்க."
    },

    {
        "english":
        "Panel touched without protection. Please use cleaning cloth.",

        "tamil":
        "பாதுகாப்பில்லாம பேனல் டச் பண்ணப்பட்டுருக்கு. க்ளீனிங் கிளாத்து பயன்படுத்துங்க."
    }
]


# =========================================================
# MISSING CLEANING ALERTS
# =========================================================

MISSING_CLEANING_ALERTS = [

    {
        "english":
        "Cleaning step appears incomplete. Please clean the panel before inspection.",

        "tamil":
        "க்ளீனிங் சரியாக செய்யல. இன்ஸ்பெக்ஷனுக்கு முன்னாடி பேனலை சுத்தம் பண்ணுங்க."
    },

    {
        "english":
        "Panel cleaning was not detected. Please complete cleaning process.",

        "tamil":
        "பேனல் க்ளீனிங் கண்டுபிடிக்கல. முதல்ல க்ளீனிங் முடிச்சுட்டு தொடருங்க."
    },

    {
        "english":
        "Cleaning confirmation missing before inspection. Please verify panel cleanliness.",

        "tamil":
        "இன்ஸ்பெக்ஷனுக்கு முன்னாடி க்ளீனிங் சரியா செய்யுங்க."
    }
]


# =========================================================
# NORMAL OPERATION
# =========================================================

NORMAL_OPERATION_ALERTS = [

    {
        "english":
        "No safety violations detected. Operation proceeding normally.",

        "tamil":
        "பாதுகாப்பு பிரச்சனை எதுவும் இல்லை. வேலை சரியாக நடக்குது."
    },

    {
        "english":
        "Panel handling within SOP. All checks passed.",

        "tamil":
        "பேனல் ஹேண்டிலிங் சரியா இருக்கு. எல்லா செக்கும் பாஸ் ஆச்சு."
    }
]


# =========================================================
# MAIN GENERATOR
# =========================================================

def generate_narration(context):

    alert_type = context["alert_type"]


    # -----------------------------------------------------
    # FAST INSPECTION
    # -----------------------------------------------------

    if alert_type == "FAST_INSPECTION":

        selected_alert = random.choice(
            FAST_INSPECTION_ALERTS
        )

        inspection_time = context.get(
            "inspection_time",
            "unknown"
        )

        english_alert = selected_alert[
            "english"
        ].format(
            time=inspection_time
        )

        tamil_alert = selected_alert[
            "tamil"
        ]

        return {

            "english": english_alert,

            "tamil": tamil_alert
        }


    # -----------------------------------------------------
    # HAND TOUCH
    # -----------------------------------------------------

    elif alert_type == "HAND_TOUCH":

        selected_alert = random.choice(
            HAND_TOUCH_ALERTS
        )

        return {

            "english":
            selected_alert["english"],

            "tamil":
            selected_alert["tamil"]
        }


    # -----------------------------------------------------
    # MISSING CLEANING
    # -----------------------------------------------------

    elif alert_type == "MISSING_CLEANING":

        selected_alert = random.choice(
            MISSING_CLEANING_ALERTS
        )

        return {

            "english":
            selected_alert["english"],

            "tamil":
            selected_alert["tamil"]
        }


    # -----------------------------------------------------
    # NORMAL OPERATION
    # -----------------------------------------------------

    elif alert_type == "NORMAL_OPERATION":

        selected_alert = random.choice(
            NORMAL_OPERATION_ALERTS
        )

        return {

            "english":
            selected_alert["english"],

            "tamil":
            selected_alert["tamil"]
        }


    # -----------------------------------------------------
    # DEFAULT (truly unknown alert type)
    # -----------------------------------------------------

    return {

        "english":
        "Operational issue detected. Please verify workstation activity.",

        "tamil":
        "சிஸ்டத்தில் பிரச்சனை கண்டுபிடிக்கப்பட்டது. வேலைப்பகுதியை சரிபார்க்கவும்."
    }