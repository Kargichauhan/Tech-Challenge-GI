"""
Claude-driven navigation, the other half of the "upgrade layer" (plan item
9). Claude picks a discrete controller action (move_left/move_right/jump/noop)
from a text description of the current game state, mirroring the vision
policy's per-frame action space (move forward/back/left/right + look) but
with a text observation standing in for the rendered frame, per the
challenge's own framing that 2D + text-level reasoning doesn't need the
vision policy.

Each decision point is a fresh, single-turn call with no growing
conversation. This keeps cost and latency bounded regardless of how long an
episode runs, and mirrors a policy that reacts to the current frame rather
than replaying full history every step.

IMPORTANT: a "jump" decision is held for the whole TICKS_PER_DECISION window
(~0.25s), and the underlying engine action model treats move/jump as mutually
exclusive per tick (matches physics_engine.step: choosing "jump" sets
horizontal velocity to 0 for that tick). A naive "hold jump for the whole
window" would rise straight up with zero horizontal movement for the entire
quarter-second, physically incapable of clearing any gap or hazard regardless
of prompting. The deterministic solvers (solvers/common.py) sidestep this by
re-deciding the action every single tick (jump for exactly one tick, then
move for the rest); the navigator can't do that per-tick replanning without a
tool call per tick, which would be far too slow and costly. Instead,
_execute_decision composes a single "jump" choice into the same shape: one
tick of vertical launch, then the rest of the window continuing whatever
horizontal direction was most recently chosen, so Claude only has to decide
when to jump, not micromanage the two-tick sequencing itself.
"""

from __future__ import annotations

import json

import anthropic

MODEL = "claude-opus-4-8"
ACTIONS = ("move_left", "move_right", "jump", "noop")
TICKS_PER_DECISION = 15  # ~0.25s of sim time per Claude decision (60 ticks/s)
HAZARD_WARNING_PX = 100  # observation flags a hazard within this horizontal range as "jump now"

ACTION_TOOL = {
    "name": "choose_action",
    "description": "Choose the controller action to hold for the next quarter-second of gameplay.",
    "input_schema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(ACTIONS)}},
        "required": ["action"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """You are playing a 2D platformer. Each turn you see the current game \
state as text and must choose ONE controller action to hold for the next quarter-second: \
move_left, move_right, jump, or noop. World convention: origin top-left, x increases \
right, y increases DOWN -- "up" means smaller y. Gravity pulls the player down \
continuously.

How jump works here: jump only takes effect if you are currently on solid ground, and a \
single jump decision automatically continues moving in whichever horizontal direction \
you were most recently traveling (defaulting to rightward if you haven't moved yet) for \
the rest of that quarter-second, so it clears gaps and hazards on its own -- you do not \
need a separate move action on the same turn. What you control is *timing*: jump too \
early and you'll land back on the ground before reaching the gap; jump too late and \
you'll already be standing in the hazard. The observation tells you the horizontal \
distance to the nearest hazard in your direction of travel -- as a rule of thumb, jump \
as soon as that distance drops under roughly 100px, don't wait until you're right next \
to it.

Locked doors block movement until their matching key is picked up (walk into a key to \
collect it automatically). Always call choose_action -- never respond with plain text."""


def _describe_observation(scene: dict, engine, last_direction: str) -> str:
    px, py = engine.player_body.position
    gz = scene["goal_zone"]
    lines = [
        f"tick={engine.tick_count} t={engine.t:.2f}s grounded={engine.grounded_count > 0}",
        f"player position: ({px:.0f}, {py:.0f}), currently heading: {last_direction}",
        f"goal_zone: x=[{gz['x']:.0f}, {gz['x'] + gz['width']:.0f}] y=[{gz['y']:.0f}, {gz['y'] + gz['height']:.0f}]",
        f"held_keys: {sorted(engine.held_keys)}",
        f"already collected: {sorted(engine.collected_ids)}",
        "objects:",
    ]
    for obj in scene["objects"]:
        if obj["id"] in engine.collected_ids:
            continue
        detail = f"  {obj['type']} '{obj['id']}' at ({obj['x']:.0f},{obj['y']:.0f})"
        if obj["type"] == "door":
            detail += f" locked={obj['locked']} key_id={obj['key_id']}"
        elif obj["type"] == "hazard":
            dx = obj["x"] - px
            detail += f" lethal={obj['lethal']} horizontal_distance={dx:+.0f}px"
            in_travel_direction = (last_direction == "move_right" and dx > 0) or (last_direction == "move_left" and dx < 0)
            if obj["lethal"] and in_travel_direction and abs(dx) < HAZARD_WARNING_PX:
                detail += "  <-- JUMP NOW, this is in your path and close"
        lines.append(detail)
    recent = engine.event_log[-5:]
    if recent:
        lines.append("recent events: " + "; ".join(f"{e['event']}({e['id']})@t={e['t']}" for e in recent))
    return "\n".join(lines)


def _execute_decision(engine, action: str, last_direction: str, max_ticks: int = TICKS_PER_DECISION) -> str:
    """Runs one Claude decision for up to `max_ticks` engine ticks. For
    "jump", composes it as one tick of vertical launch followed by the
    remaining ticks continuing `last_direction`; see module docstring for
    why a naive "hold jump the whole window" can never clear anything.
    Returns the (possibly updated) last_direction for the next decision.
    """
    if action in ("move_left", "move_right"):
        for _ in range(max_ticks):
            if engine.step(action) or engine.result is not None:
                break
        return action

    if action == "jump":
        if engine.step("jump") or engine.result is not None:
            return last_direction
        for _ in range(max_ticks - 1):
            if engine.step(last_direction) or engine.result is not None:
                break
        return last_direction

    # noop
    for _ in range(max_ticks):
        if engine.step("noop") or engine.result is not None:
            break
    return last_direction


class ClaudeNavigator:
    def __init__(self, scene: dict, client: anthropic.Anthropic | None = None):
        self.scene = scene
        self.client = client or anthropic.Anthropic()
        self.decisions_log: list[dict] = []  # for the recorded-run transcript

    def solve(self, engine, total_max_ticks: int = 60 * 25):
        objective_desc = json.dumps(self.scene["objective"])
        system = f"{SYSTEM_PROMPT}\n\nObjective (must be satisfied to win): {objective_desc}"

        last_direction = "move_right"
        while engine.result is None and engine.tick_count < total_max_ticks:
            obs = _describe_observation(self.scene, engine, last_direction)
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                tools=[ACTION_TOOL],
                tool_choice={"type": "tool", "name": "choose_action"},
                messages=[{"role": "user", "content": obs}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            action = tool_use.input["action"] if tool_use else "noop"
            self.decisions_log.append({"tick": engine.tick_count, "observation": obs, "action": action})

            last_direction = _execute_decision(engine, action, last_direction, min(TICKS_PER_DECISION, total_max_ticks - engine.tick_count))
        return engine.result, engine.event_log, engine.boundary_clips, engine.tick_count
