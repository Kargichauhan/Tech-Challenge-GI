#!/usr/bin/env python3
"""
The "maze with lava traps and a locked door" pattern: a winding, zigzagging
platform path (harness/generation/maze.py) rather than the simple generator's
flat line of platforms -- same primitive library (platform/hazard/key/door),
same validate_and_repair pipeline, no new schema. Genuine-solver reliability
on this pattern is honestly lower than the simple combos (~60% across random
seeds, similar in spirit to the existing 5-primitive combo's ~50% -- more
consecutive precision jumps means more ways for the honest solver's hop
alignment to come up short), so this demo uses a seed confirmed to solve
cleanly rather than claiming every seed does.

Run:  source .venv/bin/activate && python3 run_maze_demo.py [seed]
Output: runs/maze/<scene_id>.gif, runs/maze/<scene_id>.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "maze")


def main():
    from harness.generation.maze import generate_valid_maze_scene
    from harness.render.gif_builder import build_gif

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    scene, report = generate_valid_maze_scene(seed, prompt="a maze with lava traps and a locked door")
    print(f"Generated '{scene['id']}' ({report['seed_attempts']} seed attempt(s)).")

    gif_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.gif")
    probe = build_gif(scene, gif_path, dual_render=True)

    log_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.json")
    with open(log_path, "w") as f:
        json.dump({"scene": scene, "probe": {k: v for k, v in probe.items() if not k.endswith("_log")}}, f, indent=2, default=str)

    print(f"genuine_result={probe['genuine_result']} (honest_baseline_confirmed={probe['honest_baseline_confirmed']})")
    print(f"adversarial exploit_class={probe['exploit_class']}")
    print(f"GIF: {gif_path}")
    print(f"Scene + probe: {log_path}")


if __name__ == "__main__":
    main()
