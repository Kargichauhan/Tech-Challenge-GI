"""
Hybrid scene schema: fixed structure + extensible primitive library.

A scene is a plain JSON-serializable dict. Claude (or the deterministic
generator) fills this schema; nothing downstream ever executes arbitrary
code from the LLM. All coordinates are in world units, origin top-left,
y increases downward (matches pygame rendering directly, avoids a
physics<->render flip).
"""

from jsonschema import Draft202012Validator, FormatChecker

PRIMITIVE_TYPES = ["platform", "ramp", "door", "key", "hazard", "pickup"]
# Reserved for later once the core 6 work end to end (per plan, not built yet):
FUTURE_PRIMITIVE_TYPES = ["moving_platform", "timed_pickup"]

# Per-primitive-type field schemas.

_platform = {
    "properties": {
        "width": {"type": "number", "exclusiveMinimum": 0},
        "height": {"type": "number", "exclusiveMinimum": 0},
    },
    "required": ["width", "height"],
}

_ramp = {
    "properties": {
        "width": {"type": "number", "exclusiveMinimum": 0},
        "height": {"type": "number", "exclusiveMinimum": 0},
        "direction": {"enum": ["left", "right"]},  # which side is the low side
    },
    "required": ["width", "height", "direction"],
}

_door = {
    "properties": {
        "width": {"type": "number", "exclusiveMinimum": 0},
        "height": {"type": "number", "exclusiveMinimum": 0},
        "locked": {"type": "boolean"},
        "key_id": {"type": ["string", "null"]},  # required if locked=true
    },
    "required": ["width", "height", "locked", "key_id"],
}

_key = {
    "properties": {
        "radius": {"type": "number", "exclusiveMinimum": 0},
        "key_id": {"type": "string"},  # matches a door's key_id
    },
    "required": ["radius", "key_id"],
}

_hazard = {
    "properties": {
        "width": {"type": "number", "exclusiveMinimum": 0},
        "height": {"type": "number", "exclusiveMinimum": 0},
        "lethal": {"type": "boolean"},
    },
    "required": ["width", "height", "lethal"],
}

_pickup = {
    "properties": {
        "radius": {"type": "number", "exclusiveMinimum": 0},
        "pickup_id": {"type": "string"},
        "value": {"type": "number"},
    },
    "required": ["radius", "pickup_id", "value"],
}

_TYPE_FIELD_SCHEMAS = {
    "platform": _platform,
    "ramp": _ramp,
    "door": _door,
    "key": _key,
    "hazard": _hazard,
    "pickup": _pickup,
}

# Object schema, dispatches on "type".

_object_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "type": {"enum": PRIMITIVE_TYPES},
        "x": {"type": "number"},
        "y": {"type": "number"},
    },
    "required": ["id", "type", "x", "y"],
    "allOf": [
        {
            "if": {"properties": {"type": {"const": t}}},
            "then": schema,
        }
        for t, schema in _TYPE_FIELD_SCHEMAS.items()
    ],
}

# Objective predicate schema (recursive; see docs/event_log_schema.md).

_objective_schema = {
    "type": "object",
    "oneOf": [
        {
            "properties": {
                "event": {"enum": ["pickup", "zone_enter", "zone_exit", "door_open", "collision"]},
                "id": {"type": "string"},
            },
            "required": ["event", "id"],
            "additionalProperties": False,
        },
        {
            "properties": {"and": {"type": "array", "items": {"$ref": "#/$defs/objective"}, "minItems": 1}},
            "required": ["and"],
            "additionalProperties": False,
        },
        {
            "properties": {"or": {"type": "array", "items": {"$ref": "#/$defs/objective"}, "minItems": 1}},
            "required": ["or"],
            "additionalProperties": False,
        },
        {
            "properties": {"not": {"$ref": "#/$defs/objective"}},
            "required": ["not"],
            "additionalProperties": False,
        },
    ],
}

SCENE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {"objective": _objective_schema},
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "seed": {"type": ["integer", "null"]},
        "world": {
            "type": "object",
            "properties": {
                "width": {"type": "number", "exclusiveMinimum": 0},
                "height": {"type": "number", "exclusiveMinimum": 0},
                "gravity": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["width", "height", "gravity"],
        },
        "player_start": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
        "goal_zone": {
            "type": "object",
            "properties": {
                "id": {"const": "goal_zone"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number", "exclusiveMinimum": 0},
                "height": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["id", "x", "y", "width", "height"],
        },
        "objects": {"type": "array", "items": _object_schema},
        "objective": {"$ref": "#/$defs/objective"},
        "metadata": {
            "type": "object",
            "properties": {
                "prompt": {"type": ["string", "null"]},
                "generator": {"enum": ["freeform_llm", "deterministic_template"]},
                "primitive_combo": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["generator"],
        },
    },
    "required": ["id", "world", "player_start", "goal_zone", "objects", "objective", "metadata"],
}

_validator = Draft202012Validator(SCENE_SCHEMA, format_checker=FormatChecker())


def validate_scene_structure(scene: dict) -> list[str]:
    """Structural validation only (types/required fields). Returns a list of
    human-readable error strings; empty list means structurally valid.
    Does NOT check reachability, overlap, or id-reference integrity; see
    generation/validator.py for the semantic + physical validation pass.
    """
    return [f"{'.'.join(str(p) for p in e.path)}: {e.message}" for e in _validator.iter_errors(scene)]
