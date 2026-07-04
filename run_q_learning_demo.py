#!/usr/bin/env python3
"""
The "WORK" stage: a tabular Q-learner (harness/solvers/q_learner.py) trains
from scratch against a scene via trial and error, no hand-built strategy, no
API key, no network. Unlike GenuineSolver/AdversarialSolver -- both fixed,
hand-built strategies that never improve with practice -- this is the one
agent in the repo that actually gets better with training; the learning
curve below is real, not illustrative.

Run:  source .venv/bin/activate && python3 run_q_learning_demo.py [seed]
Output: runs/q_learning/<scene_id>_learning_curve.png
        runs/q_learning/<scene_id>.gif  (trained policy's real playthrough)
        runs/q_learning/<scene_id>.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "q_learning")


def main():
    from harness.generation.deterministic import generate_valid_scene
    from harness.render.gif_builder import build_gif
    from harness.solvers.q_learner import QLearnerAgent
    from harness.solvers.q_learner_chart import render_learning_curve

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    combo = ["ramp", "door", "key", "hazard", "pickup"]

    scene, _report = generate_valid_scene(seed, primitive_combo=combo)
    print(f"Training a tabular Q-learner from scratch on scene '{scene['id']}'...")

    t0 = time.time()
    agent = QLearnerAgent(scene)
    returns = agent.train(episodes=3000)
    train_seconds = time.time() - t0

    chart_path = os.path.join(OUTPUT_DIR, f"{scene['id']}_learning_curve.png")
    render_learning_curve(returns, scene["id"], chart_path)

    early_avg = sum(returns[:50]) / min(50, len(returns))
    late_avg = sum(returns[-50:]) / min(50, len(returns))
    print(f"Trained {len(returns)} episodes in {train_seconds:.2f}s. "
          f"Average return: first 50 episodes = {early_avg:.2f}, last 50 = {late_avg:.2f}.")
    print(f"Learning curve: {chart_path}")

    gif_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.gif")
    print("Running the trained policy for real (pymunk engine), plus the standard verifier probe...")
    probe = build_gif(scene, gif_path, trace_runner=lambda eng, ticks: agent.solve(eng, ticks),
                       trace_label="Q-LEARNER (TRAINED POLICY)")

    log_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.json")
    with open(log_path, "w") as f:
        json.dump({
            "seed": seed, "scene": scene, "episode_returns": returns, "train_seconds": train_seconds,
            "probe": {k: v for k, v in probe.items() if not k.endswith("_log")},
        }, f, indent=2, default=str)

    print(f"Q-LEARNER RESULT: {probe['trace_result']}")
    print(f"(for comparison) deterministic baseline solver: {probe['genuine_result']} "
          f"(honest_baseline_confirmed={probe['honest_baseline_confirmed']})")
    print(f"GIF: {gif_path}")
    print(f"Full log: {log_path}")


if __name__ == "__main__":
    main()
