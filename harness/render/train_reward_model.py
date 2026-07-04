"""
A learned reward model from pixels, explicitly the first thing to cut
under time pressure, per the plan. Neither torch nor scikit-learn is an
existing dependency in this environment (checked before committing to an
approach, not assumed; see WRITEUP's implemented-vs-envisioned table), and
installing a deep-learning framework purely for one optional stretch
component isn't worth the dependency weight here. This is closed-form linear
regression (numpy's lstsq, no new dependency at all) over heavily
downsampled first-person frames (render/raycaster.py) predicting the
dataset_emitter's event-log-derived reward: a genuine, honestly-scoped
stand-in for "reward model learned from pixels," not a CNN, and labeled as
exactly that everywhere it's reported.

Known, stated limitation: a single playthrough's dataset (order-100 frames)
has far fewer examples than the downsampled feature count, so this is
underdetermined. numpy's lstsq returns the minimum-norm solution, but
held-out R^2 should be read as "does this generalize at all," not expected
to be strong. That's a real property of training on one episode, not
something the model choice can fix.
"""

from __future__ import annotations

import json
import os

import numpy as np

DOWNSAMPLE = (24, 32)  # (rows, cols)


def _downsample(frame: np.ndarray, size=DOWNSAMPLE) -> np.ndarray:
    h, w, _ = frame.shape
    rows = np.linspace(0, h, size[0] + 1).astype(int)
    cols = np.linspace(0, w, size[1] + 1).astype(int)
    out = np.zeros((size[0], size[1], 3))
    for i in range(size[0]):
        for j in range(size[1]):
            block = frame[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            out[i, j] = block.reshape(-1, 3).mean(axis=0) if block.size else 0.0
    return out


def load_dataset(dataset_dir: str):
    frames = np.load(os.path.join(dataset_dir, "frames.npz"))["frames"]
    with open(os.path.join(dataset_dir, "records.jsonl")) as f:
        records = [json.loads(line) for line in f]
    return frames, records


def train_linear_reward_model(dataset_dir: str, test_frac: float = 0.2, seed: int = 0) -> dict:
    frames, records = load_dataset(dataset_dir)
    X = np.stack([_downsample(frames[r["frame_index"]]).flatten() for r in records])
    y = np.array([r["reward"] for r in records])

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    X_bias = np.concatenate([X, np.ones((len(X), 1))], axis=1)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_bias))
    n_test = max(1, int(len(idx) * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    weights, *_ = np.linalg.lstsq(X_bias[train_idx], y[train_idx], rcond=None)
    pred_test = X_bias[test_idx] @ weights
    mse = float(np.mean((y[test_idx] - pred_test) ** 2))
    baseline_mse = float(np.mean((y[test_idx] - y[train_idx].mean()) ** 2))
    r2 = (1 - mse / baseline_mse) if baseline_mse > 0 else 0.0

    return {
        "model": "linear regression (numpy lstsq) on downsampled pixels, not a CNN, see module docstring",
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "test_mse": round(mse, 5), "baseline_mse": round(baseline_mse, 5), "r2": round(r2, 4),
    }
