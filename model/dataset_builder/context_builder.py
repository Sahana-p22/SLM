# dataset_builder/context_builder.py

def build_context(detection_data):

    objects_present = []

    for obj in detection_data["objects"]:

        label = obj["class"]

        if label not in objects_present:
            objects_present.append(label)

    context = {

        "alert_type":
        detection_data.get(
            "alert_type",
            "UNKNOWN"
        ),

        "inspection_time":
        detection_data.get(
            "inspection_time",
            None
        ),

        "required_time":
        detection_data.get(
            "required_time",
            30
        ),

        "cloth_detected":
        detection_data.get(
            "cloth_detected",
            False
        ),

        "objects_present":
        objects_present,

        "workflow_stage":
        detection_data.get(
            "workflow_stage",
            "inspection"
        )
    }

    return context