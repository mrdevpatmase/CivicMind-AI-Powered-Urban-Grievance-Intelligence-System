# -------- BASE SEVERITY POLICY --------

BASE_SEVERITY = {
    "water": 1,
    "sanitation": 2,
    "road": 3,
    "streetlight": 4,
    "waste": 5
}


# -------- SIMPLE SEVERITY FUNCTION --------

def calculate_severity(category: str) -> int:

    return BASE_SEVERITY.get(category, 5)
