"""
Text command becomes a scene, via the live Claude API. This is the "upgrade
layer" (plan item 9) on top of the deterministic generator: same schema,
same validator, just a different source of scene specs.

Uses regular (non-strict) tool use, not `output_config.format` structured
outputs, because the objective predicate is a recursive schema (event/and/
or/not, self-referential via $ref), and structured outputs explicitly do
not support recursive schemas. Regular tool-call input_schema has no such
restriction. Since regular tool use doesn't guarantee schema conformance
the way strict structured outputs would, every response still goes through
the same validate_and_repair pipeline the deterministic generator uses,
with a bounded retry loop that feeds validation errors back to Claude as a
tool_result error.
"""

from __future__ import annotations

import json

import anthropic

from harness.generation.validator import validate_and_repair
from harness.schema.scene_schema import SCENE_SCHEMA

MODEL = "claude-opus-4-8"
MAX_ATTEMPTS = 4

# Tool input_schema mirrors the scene schema exactly. This is the actual
# contract Claude fills, checked afterward by the same validator the
# deterministic generator uses (schema/scene_schema.py + generation/validator.py).
SCENE_TOOL = {
    "name": "emit_scene",
    "description": (
        "Emit a complete 2D platformer scene specification conforming exactly to the "
        "provided JSON schema (the hybrid scene schema: fixed structure, extensible "
        "primitive library). This is the only way to respond -- always call this tool."
    ),
    "input_schema": {k: v for k, v in SCENE_SCHEMA.items() if k != "$schema"},
}

SYSTEM_PROMPT = """You generate 2D platformer scene specifications for a physics-based \
game harness, from a free-text description of the level the user wants.

World convention: origin top-left, x increases right, y increases DOWN (matches \
screen/render coordinates directly -- do not use a y-up math convention). A typical \
world is about 1280x720; gravity is [0, 900].

Primitive library (the only object `type`s allowed): platform, ramp, door, key, \
hazard, pickup. Every scene also has a `player_start` and a `goal_zone`, which are \
NOT objects in the `objects` array -- they are separate top-level fields.

Rules that make a scene valid (a validator checks all of these -- get them right \
the first time where you can):
- Every object needs a unique `id` string.
- `door.locked` (bool) and `door.key_id` (string) are required. If locked, some \
  `key` object's own `key_id` must equal the door's `key_id` -- that key unlocks it.
- `key.key_id` is the door-matching field; `key.id` is the object's own identity, \
  referenced by objectives (these are different strings -- e.g. id="key_1", key_id="k1").
- `hazard.lethal` (bool) is required.
- `pickup.pickup_id` and `pickup.value` are required in addition to `id`.
- Every primitive except door/key/pickup needs `width`/`height`; key/pickup use `radius` \
  instead; ramp additionally needs `direction` ("left" or "right", the low side).
- Build a walkable/jumpable route from `player_start` to `goal_zone` using platforms -- \
  a scene with no ground under the player, or gaps wider than a player can jump \
  (roughly 150px horizontal, 150-200px vertical rise), will be rejected as unsolvable.
- The `objective` is a boolean predicate over events: {"event": "pickup"|"zone_enter"| \
  "zone_exit"|"door_open"|"collision", "id": "<object id or 'goal_zone'>"}, combined \
  with {"and": [...]}, {"or": [...]}, {"not": {...}}. Every id referenced must exist. \
  A typical objective is {"and": [{"event":"pickup","id":"key_1"}, \
  {"event":"zone_enter","id":"goal_zone"}]} for a key-and-door level, or just \
  {"event":"zone_enter","id":"goal_zone"} for a plain platforming level.
- `metadata.generator` must be "freeform_llm".

Call the emit_scene tool with a complete scene. If a previous attempt was rejected, \
the rejection reasons will be given to you -- fix exactly those issues."""


def generate_scene_via_claude(prompt: str, client: anthropic.Anthropic | None = None, seed_id: str = "claude_scene"):
    """Returns (scene_or_none, report). report always has 'accepted' plus whatever
    validate_and_repair reports on the last attempt, plus 'attempts'."""
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": f"Generate a scene for: {prompt}"}]

    report = {"accepted": False}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[SCENE_TOOL],
            tool_choice={"type": "tool", "name": "emit_scene"},
            messages=messages,
        )
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            report = {"accepted": False, "attempts": attempt, "error": "no tool_use block in response"}
            break

        scene = dict(tool_use.input)
        if isinstance(scene.get("objective"), str):
            # Observed live: Claude sometimes returns the one recursive part
            # of the schema (objective, via $ref) as a JSON-encoded string
            # instead of a nested object, a known tool-call quirk with
            # deeply-nested/recursive schemas. It did this consistently
            # across retries in testing, so don't rely on re-prompting to
            # fix it; parse it defensively instead. Leave malformed JSON
            # as-is so the validator still reports it clearly.
            try:
                scene["objective"] = json.loads(scene["objective"])
            except json.JSONDecodeError:
                pass
        scene.setdefault("id", f"{seed_id}_{attempt}")
        scene.setdefault("seed", None)
        scene.setdefault("metadata", {})
        scene["metadata"]["generator"] = "freeform_llm"
        scene["metadata"]["prompt"] = prompt

        validated, validation_report = validate_and_repair(scene)
        validation_report["attempts"] = attempt
        if validated is not None:
            return validated, validation_report

        report = validation_report
        if attempt == MAX_ATTEMPTS:
            break

        errors = (
            validation_report.get("structural_errors")
            or validation_report.get("referential_errors")
            or validation_report.get("overlap_errors")
            or validation_report.get("reachability_errors")
            or ["unknown validation failure"]
        )
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": f"Scene rejected by validator: {errors}. Fix exactly these issues and call emit_scene again.",
                "is_error": True,
            }],
        })

    return None, report
