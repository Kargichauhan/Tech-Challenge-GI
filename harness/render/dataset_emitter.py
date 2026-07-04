"""
Per-tick (frame, action, reward) dataset export for a solved playthrough:
the actual bridge artifact toward the challenge's vision-policy action space
(move forward/back/left/right + mouse look), built on top of the corridor-
perspective raycaster (raycaster.py).

This world has one true movement axis (left/right) plus jump: there is no
lateral strafe axis and no camera that turns independently of movement. The
mapping below is a labeled, approximate bridge, not a claim of action-space
equivalence:
  - move_forward / move_backward: our move_left/move_right, resolved
    relative to the heading before this tick. Continuing the same
    direction is "forward," reversing it is "backward" (and simultaneously
    triggers the mouse turn below, matching a character that always faces
    its most recent direction of travel).
  - strafe_left / strafe_right: always 0.0, since there is no lateral axis
    to strafe into. Present-but-zero rather than omitted, so a downstream
    consumer expecting the full 6-value vocabulary doesn't have to guess
    whether the field is missing or genuinely always zero.
  - mouse_delta_x: 0 except on the tick a heading flip happens, where it
    carries a nominal +-180 degree turn, our one real "look elsewhere" event.
  - mouse_delta_y: a small positive value while airborne from a jump (a
    "glancing up mid-arc" heuristic), 0 otherwise. Not derived from any
    real camera pitch, since this world doesn't have one.

Reward is entirely event-log-derived (code-verified, not learned/guessed):
small per-tick cost, a pickup bonus per pickup event, +-1 terminal on
success/death.
"""

from __future__ import annotations

import json
import os

import numpy as np

from harness.engine.physics_engine import Engine
from harness.render.raycaster import infer_heading, render_first_person_frame
from harness.render.scene_renderer import surface_to_array

TICK_STRIDE = 4
TURN_DEGREES = 180.0
LOOK_UP_ON_JUMP = 8.0
STEP_REWARD = -0.01
PICKUP_REWARD = 0.1
SUCCESS_REWARD = 1.0
DEATH_REWARD = -1.0


def export_dataset(scene: dict, trace_runner, output_dir: str, total_max_ticks: int = 60 * 25) -> dict:
    """`trace_runner(engine, total_max_ticks)` supplies the playthrough,
    same contract as gif_builder.build_gif. Writes:
      <output_dir>/frames.npz     stacked first-person RGB frames, one per captured tick
      <output_dir>/records.jsonl  one JSON object per captured tick (frame index, our_action,
                                   mapped_action in their vocabulary, reward, cumulative_reward)
    Returns a small summary dict (n_frames, total_reward, result).
    """
    os.makedirs(output_dir, exist_ok=True)
    engine = Engine(scene)
    frames = []
    records = []
    state = {"heading": "move_right", "cumulative_reward": 0.0, "last_event_count": 0}

    orig_step = engine.step

    def wrapped_step(action):
        prev_heading = state["heading"]
        r = orig_step(action)
        state["heading"] = infer_heading(action, prev_heading)
        turned = state["heading"] != prev_heading

        reward = STEP_REWARD
        new_events = engine.event_log[state["last_event_count"]:]
        state["last_event_count"] = len(engine.event_log)
        for e in new_events:
            if e["event"] == "pickup":
                reward += PICKUP_REWARD
            elif e["event"] == "death":
                reward = DEATH_REWARD
        if engine.result == "success":
            reward = SUCCESS_REWARD
        state["cumulative_reward"] += reward

        # Always capture a stride-aligned tick for background/negative
        # examples, plus any tick with a non-default reward (pickup, death,
        # success) regardless of stride. These are single-tick events, and
        # missing them at a 1-in-TICK_STRIDE sampling rate was a real bug:
        # verified live, it made an exported multi-scene dataset >99.7% a
        # single repeated step-penalty value, with success/pickup frames
        # essentially never landing in the sample at all.
        if engine.tick_count % TICK_STRIDE == 0 or reward != STEP_REWARD:
            fp_surf = render_first_person_frame(scene, engine, heading=state["heading"])
            frames.append(surface_to_array(fp_surf))
            mapped = {
                "move_forward": 1.0 if action == prev_heading else 0.0,
                "move_backward": 1.0 if action in ("move_left", "move_right") and action != prev_heading else 0.0,
                "strafe_left": 0.0,
                "strafe_right": 0.0,
                "mouse_delta_x": (TURN_DEGREES if state["heading"] == "move_right" else -TURN_DEGREES) if turned else 0.0,
                "mouse_delta_y": LOOK_UP_ON_JUMP if action == "jump" else 0.0,
            }
            records.append({
                "tick": engine.tick_count, "frame_index": len(frames) - 1, "our_action": action,
                "mapped_action": mapped, "reward": reward, "cumulative_reward": round(state["cumulative_reward"], 4),
            })
        return r

    engine.step = wrapped_step
    trace_result, *_ = trace_runner(engine, total_max_ticks)

    if frames:
        np.savez_compressed(os.path.join(output_dir, "frames.npz"), frames=np.stack(frames))
    with open(os.path.join(output_dir, "records.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return {
        "n_frames": len(frames), "n_records": len(records),
        "total_reward": round(state["cumulative_reward"], 4), "result": trace_result,
    }
