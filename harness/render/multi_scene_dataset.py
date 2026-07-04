"""
Aggregates (frame, action, reward) data across MANY solved scenes, not just
one -- built specifically to test whether the original reward model's poor
R^2 (train_reward_model.py, disclosed in the README) was a data-volume
problem rather than a model-choice problem. Scenes are drawn from the same
generators the rest of the repo already trusts (deterministic.py's named
combos, maze.py's zigzag layout) -- no new generation logic, no relaxed
validation.

Held out **by scene**, not by frame: a model that never saw a given scene
during training is a genuine generalization test, not just an interpolation
check on frames from episodes it already learned from.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np

from harness.generation.deterministic import generate_valid_scene
from harness.generation.maze import generate_valid_maze_scene
from harness.generation.policy_adaptation import COMBOS
from harness.render.dataset_emitter import export_dataset
from harness.solvers.genuine import GenuineSolver

MIN_FRAMES_PER_SCENE = 10


def _candidate_scenes(rng: random.Random, n_per_combo: int, n_maze: int) -> list[dict]:
    scenes = []
    for name, combo in COMBOS.items():
        for _ in range(n_per_combo):
            seed = rng.randint(0, 10_000_000)
            scene, _report = generate_valid_scene(seed, primitive_combo=combo, prompt=f"multi-scene: {name}")
            if scene is not None:
                scenes.append(scene)
    for _ in range(n_maze):
        seed = rng.randint(0, 10_000_000)
        scene, _report = generate_valid_maze_scene(seed, prompt="multi-scene maze")
        if scene is not None:
            scenes.append(scene)
    return scenes


def build_multi_scene_dataset(output_dir: str, n_per_combo: int = 5, n_maze: int = 8,
                               total_max_ticks: int = 1800, seed: int = 0) -> list[dict]:
    """Generates candidate scenes, keeps only the ones GenuineSolver actually
    solves (same honesty bar as everywhere else in this repo -- no scene
    that isn't confirmed solved contributes training data), and exports each
    via the existing dataset_emitter. Returns a manifest list of per-scene
    summaries; frames/records are written to <output_dir>/<scene_id>/."""
    rng = random.Random(seed)
    scenes = _candidate_scenes(rng, n_per_combo, n_maze)
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for scene in scenes:
        scene_dir = os.path.join(output_dir, scene["id"])
        summary = export_dataset(scene, lambda eng, ticks, s=scene: GenuineSolver(s).solve(eng, ticks),
                                  scene_dir, total_max_ticks)
        if summary["result"] == "success" and summary["n_frames"] >= MIN_FRAMES_PER_SCENE:
            manifest.append({"scene_id": scene["id"], "dir": scene_dir, **summary})

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def load_scene_frames(scene_dir: str):
    """Returns (frames: list of (H,W,3) uint8 arrays, rewards: (N,) float
    array). Frames stay a plain list, not a stacked array -- different
    scenes render at different world widths (deterministic.py/maze.py size
    the world to fit the generated content), so frames only share a common
    shape *within* one scene, never across scenes until each is individually
    downsampled to a fixed size."""
    frames = np.load(os.path.join(scene_dir, "frames.npz"))["frames"]
    with open(os.path.join(scene_dir, "records.jsonl")) as f:
        records = [json.loads(line) for line in f]
    rewards = np.array([r["reward"] for r in records], dtype=np.float32)
    return list(frames), rewards


def split_manifest(manifest: list[dict], holdout_frac: float = 0.25, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Splits by SCENE, not by frame -- see module docstring."""
    rng = random.Random(seed)
    shuffled = list(manifest)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_frac))
    return shuffled[n_holdout:], shuffled[:n_holdout]


def load_split(manifest_entries: list[dict]):
    """Returns (frames: flat list of (H,W,3) arrays, rewards: (N,) array)
    across a list of manifest entries -- see load_scene_frames for why
    frames stay a plain list rather than a stacked array."""
    all_frames, all_rewards = [], []
    for entry in manifest_entries:
        frames, rewards = load_scene_frames(entry["dir"])
        all_frames.extend(frames)
        all_rewards.append(rewards)
    return all_frames, np.concatenate(all_rewards, axis=0)
