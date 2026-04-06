"""
Rule-based keyword classifier: natural-language question → structured filters.
"""


def classify_question(question: str) -> dict:
    """
    Map a user question to policy_category and optional activity_type using substring rules.
    Falls back to policy_category \"billing\" and activity_type None when no rule applies.
    """
    q = (question or "").lower()

    # Step 1: derive policy_category (first matching rule wins, top to bottom).
    policy_category = "billing"
    if "marketing" in q:
        policy_category = "marketing"
    elif "bill" in q or "charge" in q:
        policy_category = "billing"
    elif "payday" in q or "payroll" in q:
        policy_category = "payroll"
    elif "overtime" in q:
        policy_category = "overtime"
    elif "travel" in q or "expense" in q:
        policy_category = "travel"

    # Step 2: derive activity_type (optional).
    activity_type = None
    if "internal marketing" in q:
        activity_type = "internal marketing"
    elif "campaign" in q:
        activity_type = "client campaign"
    elif "weekend" in q:
        activity_type = "weekend overtime"

    return {
        "policy_category": policy_category,
        "activity_type": activity_type,
    }
