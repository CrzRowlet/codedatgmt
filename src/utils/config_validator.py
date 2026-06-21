from pathlib import Path

REQUIRED_PATHS = [
    "models",
    "data",
    "logs"
]

def validate_project_structure(root="."):
    missing = []
    for item in REQUIRED_PATHS:
        if not Path(root, item).exists():
            missing.append(item)
    return missing
