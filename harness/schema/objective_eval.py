"""Evaluate an objective predicate (schema/scene_schema.py's objective
language) against an event log (schema/event_log_schema.md). Order-blind by
design -- see event_log_schema.md for why."""


def evaluate(node: dict, event_log: list[dict]) -> bool:
    if "event" in node:
        return any(e["event"] == node["event"] and e["id"] == node["id"] for e in event_log)
    if "and" in node:
        return all(evaluate(child, event_log) for child in node["and"])
    if "or" in node:
        return any(evaluate(child, event_log) for child in node["or"])
    if "not" in node:
        return not evaluate(node["not"], event_log)
    raise ValueError(f"malformed objective node: {node}")
