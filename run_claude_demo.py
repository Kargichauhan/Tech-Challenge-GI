#!/usr/bin/env python3
"""
The Claude-API upgrade layer (plan item 9): a free-text prompt becomes a
scene via the live Claude API, then Claude itself navigates the result
(per-decision-point tool calls, not the deterministic solver). Requires
ANTHROPIC_API_KEY -- this is explicitly NOT the guaranteed demo path; see
run_demo.py for the zero-dependency deterministic path the submission does
not depend on live API access for.

Run:  source .venv/bin/activate && python3 run_claude_demo.py "a level with a
      locked door, its key, and a hazard to jump over"
Output: runs/claude_demo/<scene_id>.gif, runs/claude_demo/<scene_id>.json
        (full event log + per-decision transcript, for inspection)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "claude_demo")

DEFAULT_PROMPT = "a level with a ramp leading up to a locked door, its key on the ground, and a hazard to jump over on the way"


def main():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "No ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) set in this environment.\n"
            "This script calls the live Claude API for both scene generation and navigation.\n"
            "Set your key first:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "The zero-dependency deterministic demo (no API key needed) is run_demo.py."
        )
        sys.exit(1)

    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import anthropic

    from harness.claude_agent.navigator import ClaudeNavigator
    from harness.claude_agent.scene_generator import generate_scene_via_claude
    from harness.render.gif_builder import build_gif

    client = anthropic.Anthropic()

    print(f"Prompt: {prompt!r}")
    print("Generating scene via Claude (tool-forced JSON, validated, bounded retry on rejection)...")
    t0 = time.time()
    scene, report = generate_scene_via_claude(prompt, client=client)
    if scene is None:
        print(f"Scene generation failed after validator retries: {report}")
        sys.exit(1)
    print(f"Scene '{scene['id']}' accepted after {report.get('attempts', 1)} attempt(s) ({time.time() - t0:.1f}s).")
    print(f"Primitive combo used: {scene['metadata'].get('primitive_combo', sorted({o['type'] for o in scene['objects']}))}")

    gif_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.gif")
    print("Running Claude-driven navigation for the visual trace, plus the standard genuine+adversarial verifier probe...")
    t0 = time.time()

    navigator_holder = {}

    def trace_runner(engine, total_max_ticks):
        navigator = ClaudeNavigator(scene, client=client)
        navigator_holder["nav"] = navigator
        return navigator.solve(engine, total_max_ticks)

    probe = build_gif(scene, gif_path, trace_runner=trace_runner, trace_label="CLAUDE'S NAVIGATION")
    elapsed = time.time() - t0

    log_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.json")
    with open(log_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "scene": scene,
            "probe": {k: v for k, v in probe.items() if not k.endswith("_log")},
            "genuine_event_log": probe["genuine_event_log"],
            "adversarial_event_log": probe["adversarial_event_log"],
            "claude_navigation_transcript": navigator_holder["nav"].decisions_log,
        }, f, indent=2, default=str)

    n_decisions = len(navigator_holder["nav"].decisions_log)
    print(f"\nDone in {elapsed:.1f}s ({n_decisions} Claude decision(s) made).")
    print(f"CLAUDE'S OWN NAVIGATION RESULT: {probe['trace_result']}")
    print(f"(for comparison) deterministic baseline solver: {probe['genuine_result']} "
          f"(honest_baseline_confirmed={probe['honest_baseline_confirmed']})")
    print(f"adversarial probe exploit_class: {probe['exploit_class']}")
    print(f"GIF: {gif_path}")
    print(f"Full log + transcript: {log_path}")


if __name__ == "__main__":
    main()
