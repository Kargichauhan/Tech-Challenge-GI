"""
Builds one self-contained GIF per environment: generated scene -> genuine
solver's trace playing through it (with live event captions) -> a
verifier-robustness result screen. Meant to be readable in well under 30s,
per the submission's own "clarity" eval criterion.

Uses the genuine solver's run for the main trace (per plan); if the
adversarial probe found an exploit, the result screen calls it out by
exploit_class + explanation rather than requiring anyone to read logs.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
from PIL import Image

from harness.engine.physics_engine import Engine
from harness.render.raycaster import infer_heading, render_first_person_frame
from harness.render.scene_renderer import COLORS, _draw_text, render_scene_frame, surface_to_array
from harness.solvers.genuine import GenuineSolver
from harness.verifier.probe import run_verifier_probe

FPS = 20
TICK_STRIDE = 4  # capture 1 frame per N engine ticks (60 ticks/s sim -> ~15 rendered fps of trace)
INTRO_SECONDS = 1.5
OUTRO_SECONDS = 3.5
CAPTION_HOLD_SECONDS = 1.0

_EVENT_CAPTIONS = {
    "pickup": lambda e: f"picked up {e['id']}",
    "door_open": lambda e: f"door {e['id']} opened",
    "zone_enter": lambda e: "entered goal zone",
    "zone_exit": lambda e: "left goal zone",
    "death": lambda e: f"died ({e['meta'].get('cause', 'hazard')})",
}


def _result_headline(label: str, result: str | None) -> tuple[str, tuple[int, int, int]]:
    word = {"success": "SUCCESS", "death": "DIED", "timeout": "TIMED OUT", None: "TIMED OUT"}[result]
    color = {"success": COLORS["clean"], "death": COLORS["exploit"], "timeout": COLORS["text_dim"], None: COLORS["text_dim"]}[result]
    return f"{label}: {word}", color


def render_result_frame(scene: dict, probe: dict, trace_label: str | None = None, trace_result: str | None = None):
    """`trace_label`/`trace_result` describe whoever's run is actually shown
    as the GIF's visual trace (e.g. "CLAUDE'S NAVIGATION", "died"). This can
    differ from `probe['genuine_result']`, which is always the deterministic
    baseline solver run separately for comparison. Conflating the two would
    mislead a viewer into thinking the displayed trace achieved whatever the
    baseline solver did. When trace_label is None (the deterministic-demo
    case), the trace is the genuine solver, so there's nothing to
    disambiguate; keep the single "GENUINE SOLVE" headline.
    """
    import pygame
    w, h = scene["world"]["width"], scene["world"]["height"]
    surf = pygame.Surface((w, h))
    surf.fill(COLORS["caption_bg"])

    y = 60
    _draw_text(surf, "VERIFIER-ROBUSTNESS RESULT", (w // 2, y), size=30, color=COLORS["text"], center=True)
    y += 60

    if trace_label is None:
        headline, color = _result_headline("GENUINE SOLVE", probe["genuine_result"])
        _draw_text(surf, headline, (w // 2, y), size=26, color=color, center=True)
        y += 50
    else:
        headline, color = _result_headline(trace_label, trace_result)
        _draw_text(surf, headline, (w // 2, y), size=26, color=color, center=True)
        y += 40
        baseline_headline, baseline_color = _result_headline("(for comparison) deterministic baseline solver", probe["genuine_result"])
        _draw_text(surf, baseline_headline, (w // 2, y), size=15, color=baseline_color, center=True)
        y += 34

    if not probe["honest_baseline_confirmed"]:
        _draw_text(surf, "(no confirmed honest solve for this scene, see below)", (w // 2, y),
                    size=16, color=COLORS["text_dim"], center=True)
        y += 40

    y += 20
    if probe["exploit_class"] != "none":
        _draw_text(surf, f"ADVERSARIAL PROBE: EXPLOIT FOUND ({probe['exploit_class']})",
                    (w // 2, y), size=24, color=COLORS["exploit"], center=True)
        y += 40
        if not probe["honest_baseline_confirmed"]:
            _draw_text(surf, "caveat: genuine solver never confirmed this scene solvable honestly,",
                        (w // 2, y), size=15, color=COLORS["text_dim"], center=True)
            y += 22
            _draw_text(surf, "so this is a weaker claim than \"shortcut past a proven-solvable level\"",
                        (w // 2, y), size=15, color=COLORS["text_dim"], center=True)
            y += 30
        explanation = probe["explanation"] or ""
        for line in _wrap(explanation, 70):
            _draw_text(surf, line, (w // 2, y), size=16, color=COLORS["text"], center=True)
            y += 24
    else:
        msg = "ADVERSARIAL PROBE: NO EXPLOIT FOUND" if probe["adversarial_success"] else "ADVERSARIAL PROBE: ALSO FAILED"
        _draw_text(surf, msg, (w // 2, y), size=24, color=COLORS["clean"], center=True)
        y += 30
        _draw_text(surf, "adversarial solver could not satisfy the objective out of order,", (w // 2, y),
                    size=15, color=COLORS["text_dim"], center=True)
        y += 22
        _draw_text(surf, "via an unintended path, or by clipping through solid geometry",
                    (w // 2, y), size=15, color=COLORS["text_dim"], center=True)

    return surf


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 > width:
            lines.append(cur)
            cur = w_
        else:
            cur = f"{cur} {w_}".strip()
    if cur:
        lines.append(cur)
    return lines


def _hstack(top_down_arr, first_person_arr):
    """Side by side, matched heights (both frames are the same world h already)."""
    return np.concatenate([top_down_arr, first_person_arr], axis=1)


def build_gif(scene: dict, output_path: str, total_max_ticks: int = 60 * 25, trace_runner=None,
              trace_label: str | None = None, dual_render: bool = False) -> dict:
    """Runs the full probe (genuine + adversarial) for the result screen,
    then a second fresh run purely to capture rendered frames (kept separate
    from the probe's own run so instrumenting one never risks perturbing the
    other's tick-for-tick determinism). Returns the probe dict (useful for
    batch reporting alongside the GIF) with an added "trace_result" key:
    whatever `trace_runner` itself returned as its result, distinct from
    `probe["genuine_result"]` (the baseline solver run separately).

    `trace_runner(engine, total_max_ticks)` supplies the visual trace: the
    genuine solver by default, or e.g. a ClaudeNavigator's `.solve` for the
    Claude-API upgrade layer (harness/claude_agent/navigator.py), so the
    displayed playthrough matches whichever agent actually generated it.
    Pass `trace_label` (e.g. "CLAUDE'S NAVIGATION") whenever trace_runner is
    NOT the genuine solver, so the result screen doesn't conflate the two.

    `dual_render=True` places the corridor-perspective first-person view
    (render/raycaster.py) alongside the existing top-down view, tracking a
    facing direction the same way navigator.py's last_direction does. False
    by default; the deterministic demo (run_demo.py) and the Claude demo
    are both unaffected, only run_dataset_export.py opts in.
    """
    probe = run_verifier_probe(scene, total_max_ticks)

    engine = Engine(scene)
    trace_runner = trace_runner or (lambda eng, ticks: GenuineSolver(scene).solve(eng, ticks))

    prompt = scene["metadata"].get("prompt") or scene["id"]
    intro_surf = render_scene_frame(scene, None, caption=f"generated scene: {prompt}")
    intro_arr = surface_to_array(intro_surf)
    if dual_render:
        blank_fp = surface_to_array(render_first_person_frame(scene, Engine(scene), caption="(first-person view)"))
        intro_arr = _hstack(intro_arr, blank_fp)

    state = {"tick": 0, "caption": None, "caption_ttl": 0, "last_event_count": 0, "heading": "move_right"}
    caption_hold_ticks = int(CAPTION_HOLD_SECONDS * 60)  # in engine ticks, not rendered frames

    # A long/dual-render trace (e.g. run_maze_demo.py) can be thousands of frames.
    # imageio's GIF writer holds every appended frame's encoder state in memory for
    # the life of the call (measured: ~3.8GB peak on this scene, vs ~80MB for the
    # rendering loop alone with the writer removed), so frames are produced by a
    # background thread and pulled one at a time into PIL's own save_all(), which
    # only keeps the current frame plus a small lookback for palette/dispose state.
    # Measured fix: same scene, same frame count, ~170MB peak instead of ~3.8GB.
    frame_queue: queue.Queue = queue.Queue(maxsize=8)
    _DONE = object()
    result_box: dict = {}

    orig_step = engine.step

    def wrapped_step(action):
        r = orig_step(action)
        state["tick"] += 1
        state["heading"] = infer_heading(action, state["heading"])

        new_events = engine.event_log[state["last_event_count"]:]
        state["last_event_count"] = len(engine.event_log)
        for e in new_events:
            fmt = _EVENT_CAPTIONS.get(e["event"])
            if fmt:
                state["caption"] = fmt(e)
                state["caption_ttl"] = caption_hold_ticks
        if state["caption_ttl"] > 0:
            state["caption_ttl"] -= 1
        else:
            state["caption"] = None

        if state["tick"] % TICK_STRIDE == 0:
            top_down = surface_to_array(render_scene_frame(scene, engine, caption=state["caption"]))
            if dual_render:
                first_person = surface_to_array(render_first_person_frame(scene, engine, heading=state["heading"]))
                frame_queue.put(_hstack(top_down, first_person))
            else:
                frame_queue.put(top_down)
        return r

    engine.step = wrapped_step

    def run_trace():
        try:
            result_box["trace_result"] = trace_runner(engine, total_max_ticks)[0]
        finally:
            frame_queue.put(_DONE)

    trace_thread = threading.Thread(target=run_trace, daemon=True)
    trace_thread.start()

    def frames_after_first():
        for _ in range(int(FPS * INTRO_SECONDS) - 1):
            yield Image.fromarray(intro_arr)
        while True:
            item = frame_queue.get()
            if item is _DONE:
                break
            yield Image.fromarray(item)
        trace_thread.join()

        trace_result = result_box["trace_result"]
        probe["trace_result"] = trace_result
        outro_surf = render_result_frame(scene, probe, trace_label=trace_label, trace_result=trace_result)
        outro_arr = surface_to_array(outro_surf)
        if dual_render:
            outro_arr = _hstack(outro_arr, np.zeros_like(blank_fp))
        for _ in range(int(FPS * OUTRO_SECONDS)):
            yield Image.fromarray(outro_arr)

    first_frame = Image.fromarray(intro_arr)
    first_frame.save(
        output_path, save_all=True, append_images=frames_after_first(),
        duration=int(1000 / FPS), loop=0, optimize=False,
    )

    return probe
