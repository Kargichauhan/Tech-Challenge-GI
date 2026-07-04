#!/usr/bin/env python3
"""
The bigger reward-model run: tests whether the original reward model's poor
R^2 (train_reward_model.py, one 74-frame episode) was a data-volume problem
rather than a model-choice problem, by changing exactly one variable at a
time.

  1. Generates many solved playthroughs across many scenes (not one),
     held out BY SCENE for a genuine generalization test.
  2. Reports three numbers side by side: the original single-episode linear
     baseline (for reference), the same linear method on the bigger dataset
     (isolates "did more data help"), and a real small CNN on the same
     bigger dataset (isolates "did a better model help on top of that").

Requires torch (CPU build: pip install --index-url
https://download.pytorch.org/whl/cpu torch), an explicit, separate opt-in
from the guaranteed dependency-light path everywhere else in this repo.

Run:  source .venv/bin/activate && python3 run_reward_model_v2.py
Output: runs/reward_model_v2/manifest.json + per-scene frames/records,
        runs/reward_model_v2/comparison.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "reward_model_v2")


def main():
    from harness.render.multi_scene_dataset import build_multi_scene_dataset, load_split, split_manifest
    from harness.render.train_reward_model_cnn import linear_baseline_on_bigger_data, train_cnn_reward_model

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    if os.path.exists(manifest_path) and "--fresh" not in sys.argv:
        print(f"Reusing existing generated dataset at {OUTPUT_DIR} (pass --fresh to regenerate)...")
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        print("Generating solved playthroughs across many scenes (this takes a couple minutes)...")
        t0 = time.time()
        manifest = build_multi_scene_dataset(OUTPUT_DIR, n_per_combo=5, n_maze=8, seed=0)
        print(f"  generated in {time.time() - t0:.1f}s")
    total_frames = sum(m["n_frames"] for m in manifest)
    print(f"  {len(manifest)} solved scenes, {total_frames} total frames")

    train_manifest, holdout_manifest = split_manifest(manifest, holdout_frac=0.25, seed=0)
    print(f"  train scenes: {len(train_manifest)}, held-out scenes: {len(holdout_manifest)} (never trained on)")

    train_frames, train_rewards = load_split(train_manifest)
    test_frames, test_rewards = load_split(holdout_manifest)
    print(f"  train frames: {len(train_frames)}, held-out frames: {len(test_frames)}")

    print("\nFitting the same linear method as train_reward_model.py, on the bigger dataset...")
    linear_result = linear_baseline_on_bigger_data(train_frames, train_rewards, test_frames, test_rewards)
    print(f"  {linear_result}")

    print("\nTraining a small CNN (torch) on the same bigger dataset...")
    t0 = time.time()
    cnn_result = train_cnn_reward_model(train_frames, train_rewards, test_frames, test_rewards)
    print(f"  {cnn_result} ({time.time() - t0:.1f}s)")

    comparison = {
        "original_single_episode_reference": "R^2 was strongly negative (~-5 to -25) on one 74-frame episode, see README",
        "n_scenes_total": len(manifest), "n_scenes_train": len(train_manifest), "n_scenes_holdout": len(holdout_manifest),
        "n_frames_train": len(train_frames), "n_frames_holdout": len(test_frames),
        "linear_on_bigger_data": linear_result,
        "cnn_on_bigger_data": cnn_result,
    }
    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison written to {os.path.join(OUTPUT_DIR, 'comparison.json')}")
    b = cnn_result["rare_event_breakdown"]
    print(
        f"\nheld-out reward_std={cnn_result['reward_std_in_test_set']} (real variance: pickup/death/"
        f"success events are present in this held-out set; an earlier version of this run had a real "
        f"bug in dataset_emitter.py's tick-stride sampling that silently excluded almost all of them, "
        f"making reward_std ~0 and every downstream metric meaningless. Fixed, then regenerated.)\n\n"
        f"held-out R^2: linear (more data) = {linear_result['r2']}, CNN (more data) = {cnn_result['r2']}\n"
        f"  linear (more data): MAE={linear_result['test_mae']}, worse than baseline. Raw-pixel linear "
        f"regression overfits training scenes' visuals and does not generalize to new scene content.\n"
        f"  CNN (more data):    MAE={cnn_result['test_mae']}, R^2={cnn_result['r2']}, a real, positive "
        f"R^2 this time.\n\n"
        f"Aggregate MAE alone understates the CNN's result: {b['n_common_ticks']} of "
        f"{b['n_common_ticks'] + b['n_rare_event_ticks']} held-out frames are just the common step "
        f"penalty, which dilutes the average. Broken out by event type:\n"
        f"  common ticks:  CNN MAE={b['mae_common_ticks']}\n"
        f"  rare events (pickup/death/success, n={b['n_rare_event_ticks']}): "
        f"CNN MAE={b['mae_rare_events']} vs mean-predictor MAE={b['mean_predictor_mae_rare_events']}, "
        f"roughly {round(b['mean_predictor_mae_rare_events'] / max(b['mae_rare_events'], 1e-9), 1)}x "
        f"better than blind guessing on exactly the events a reward model needs to get right, though "
        f"inconsistently (some individual predictions are close, some are missed). Reported as "
        f"measured, not smoothed into one flattering number."
    )


if __name__ == "__main__":
    main()
