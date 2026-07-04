#!/usr/bin/env python3
"""
The guaranteed, zero-setup demo path: deterministic scene generation, no API
key, no network access. Generates a handful of environments across the
primitive-combination space, validates them, runs the genuine solver and the
adversarial verifier-probe on each, renders a GIF per environment (generated
scene -> genuine trace -> verifier-robustness result), and prints a summary.

Run:  source .venv/bin/activate && python3 run_demo.py
Output: runs/demo/<scene_id>.gif  and  runs/demo/summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.generation.deterministic import generate_valid_scene
from harness.render.gif_builder import build_gif

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "demo")

# A fixed, hand-picked spread across the primitive library: deterministic
# (same seed -> same scene every time), no network/API dependency.
DEMO_SCENES = [
    (16, ["ramp", "door", "key", "hazard", "pickup"], "a full obstacle course with a ramp, a locked door, a hazard, and a pickup"),
    (301, ["door", "key"], "a level gated by a locked door and its key"),
    (2, ["hazard"], "a simple level with a hazard to jump over"),
    (3, ["ramp", "hazard", "pickup"], "a ramp, a hazard, and a pickup on the way to the goal"),
    (4, ["key", "door", "hazard"], "a key-and-door puzzle with a hazard along the route"),
    (5, [], "a bare platforming level with no extra primitives"),
    # The challenge brief's own example phrasing ("picked up the can from the
    # table"), built from the existing pickup+platform primitives, no new
    # schema needed. `pickup` stands in for "the can," `platform` for "the
    # table" it's resting on.
    (7, ["pickup"], "a table with a can on it, pick up the can"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    print(f"Building {len(DEMO_SCENES)} demo environments (deterministic, no API key, no network)...\n")

    for seed, combo, prompt in DEMO_SCENES:
        t0 = time.time()
        scene, report = generate_valid_scene(seed, primitive_combo=combo, prompt=prompt)
        if scene is None:
            print(f"[seed={seed} combo={combo}] validator rejected all seed attempts: {report}")
            continue

        gif_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.gif")
        probe = build_gif(scene, gif_path)
        elapsed = time.time() - t0

        row = {
            "scene_id": scene["id"],
            "primitive_combo": combo,
            "prompt": prompt,
            "seed_attempts": report.get("seed_attempts", 1),
            "genuine_result": probe["genuine_result"],
            "honest_baseline_confirmed": probe["honest_baseline_confirmed"],
            "adversarial_success": probe["adversarial_success"],
            "exploit_class": probe["exploit_class"],
            "explanation": probe["explanation"],
            "gif": os.path.relpath(gif_path, os.path.dirname(os.path.abspath(__file__))),
            "build_seconds": round(elapsed, 2),
        }
        summary.append(row)

        status = "OK" if row["honest_baseline_confirmed"] else "no honest baseline"
        exploit = f"EXPLOIT: {row['exploit_class']}" if row["exploit_class"] != "none" else "no exploit found"
        print(f"[{scene['id']:<10}] genuine={row['genuine_result']!s:<8} ({status})  adversarial={exploit}  -> {gif_path}  ({elapsed:.1f}s)")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    n_success = sum(1 for r in summary if r["genuine_result"] == "success")
    n_exploited = sum(1 for r in summary if r["exploit_class"] != "none")
    n_first_try = sum(1 for r in summary if r["seed_attempts"] == 1)
    print(f"\n{len(summary)} environments generated. Genuine solve rate: {n_success}/{len(summary)}. "
          f"Exploits found by adversarial probe: {n_exploited}/{len(summary)}.")
    print(f"Valid-yield rate (accepted on the first seed attempt, no repair/retry needed): "
          f"{n_first_try}/{len(summary)}.")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
