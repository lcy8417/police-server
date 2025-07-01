import json

def safe_json_loads(value):
    if value is None:
        return []
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
