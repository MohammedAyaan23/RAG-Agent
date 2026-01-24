

def get_initial_state() -> dict:
    return {
        "status": "start",
        "turn_count": 0
    }

def next_state(
    current_dict: dict,
    *,
    intent: str | None = None,
    is_followup: bool = False,
    comparison: bool = False,
    unresolved: bool = False,
) -> dict:
    # Default state if dictionary is empty
    current_value = current_dict.get("status", "start")

    # Transition logic
    new_value = current_value
    if current_value == "start":
        new_value = "single_intent"
    
    if comparison or intent == "comparison":
        new_value = "multi_intent"

    if unresolved:
        new_value = "exploratory"

    # Return a fresh dictionary
    return {
        "status": new_value,
        "last_intent": intent,
        "is_followup": is_followup
    }