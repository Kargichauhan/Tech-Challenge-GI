#!/usr/bin/env python3
"""
The vision-policy bridge demo: renders a solved playthrough two ways side by
side (the existing top-down view and the new corridor-perspective first-
person view, render/raycaster.py), and exports a (frame, action, reward)
dataset in the challenge's action vocabulary (render/dataset_emitter.py;
see its docstring for the honest mapping caveat). No API key, no network.

Run:  source .venv/bin/activate && python3 run_dataset_export.py [seed]
Output: runs/dataset_export/<scene_id>_dual.gif
        runs/dataset_export/<scene_id>/frames.npz
        runs/dataset_export/<scene_id>/records.jsonl
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "dataset_export")


def main():
    from harness.generation.deterministic import generate_valid_scene
    from harness.render.dataset_emitter import export_dataset
    from harness.render.gif_builder import build_gif
    from harness.render.train_reward_model import train_linear_reward_model
    from harness.solvers.genuine import GenuineSolver

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    combo = ["ramp", "door", "key", "hazard", "pickup"]

    scene, _report = generate_valid_scene(seed, primitive_combo=combo)
    print(f"Rendering dual (top-down + first-person) GIF for scene '{scene['id']}'...")

    gif_path = os.path.join(OUTPUT_DIR, f"{scene['id']}_dual.gif")
    probe = build_gif(scene, gif_path, dual_render=True)
    print(f"genuine_result={probe['genuine_result']} (honest_baseline_confirmed={probe['honest_baseline_confirmed']})")
    print(f"Dual-render GIF: {gif_path}")

    print("Exporting (frame, action, reward) dataset from the same genuine playthrough...")
    dataset_dir = os.path.join(OUTPUT_DIR, scene["id"])
    summary = export_dataset(scene, lambda eng, ticks: GenuineSolver(scene).solve(eng, ticks), dataset_dir)
    print(f"Exported {summary['n_frames']} frames, {summary['n_records']} action records, "
          f"total_reward={summary['total_reward']}, result={summary['result']}")
    print(f"Dataset: {dataset_dir}/frames.npz, {dataset_dir}/records.jsonl")

    print("\nFitting the optional reward-model stretch piece (linear regression on "
          "downsampled pixels, not a CNN; see harness/render/train_reward_model.py)...")
    if summary["n_records"] < 10:
        print("Too few frames from this single playthrough to fit/evaluate a held-out split, skipping.")
    else:
        model_report = train_linear_reward_model(dataset_dir)
        print(f"  {model_report['model']}")
        print(f"  n_train={model_report['n_train']} n_test={model_report['n_test']} "
              f"test_mse={model_report['test_mse']} baseline_mse={model_report['baseline_mse']} r2={model_report['r2']}")


if __name__ == "__main__":
    main()
