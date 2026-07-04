"""
The bigger reward-model run: same question as train_reward_model.py (can a
model predict the event-log-derived reward from first-person pixels alone?),
but with two changes tested independently so the result is honestly
attributable:

  1. More data -- many playthroughs across many scenes (multi_scene_dataset.py)
     instead of one 74-frame episode, held out **by scene** for a genuine
     generalization test.
  2. A real CNN (torch) instead of linear regression, trained on the same
     bigger dataset, so "did more data help" and "did a better model help"
     are reported as two separate comparisons rather than conflated into one
     number.

This module is intentionally separate from train_reward_model.py, which
stays torch-free and remains the dependency-light default -- importing this
module requires torch, an explicit opt-in (see run_reward_model_v2.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from harness.render.train_reward_model import DOWNSAMPLE, _downsample

CNN_INPUT_SIZE = (48, 64)  # (rows, cols) -- larger than the linear model's downsample; a CNN can use more pixels
STEP_REWARD_REF = -0.01  # dataset_emitter.STEP_REWARD -- duplicated to avoid importing the physics engine for one constant


def _downsample_batch(frames: np.ndarray, size) -> np.ndarray:
    return np.stack([_downsample(f, size=size) for f in frames])


def linear_baseline_on_bigger_data(train_frames, train_rewards, test_frames, test_rewards) -> dict:
    """The exact train_reward_model.py method (numpy lstsq on downsampled
    pixels), applied to the multi-scene, held-out-by-scene split -- isolates
    "did more data help the SAME model" from "did a different model help."
    """
    X_train = _downsample_batch(train_frames, DOWNSAMPLE).reshape(len(train_frames), -1)
    X_test = _downsample_batch(test_frames, DOWNSAMPLE).reshape(len(test_frames), -1)

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_train_b = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test_b = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    weights, *_ = np.linalg.lstsq(X_train_b, train_rewards, rcond=None)
    pred = X_test_b @ weights
    mse = float(np.mean((test_rewards - pred) ** 2))
    mae = float(np.mean(np.abs(test_rewards - pred)))
    baseline_mse = float(np.mean((test_rewards - train_rewards.mean()) ** 2))
    r2 = (1 - mse / baseline_mse) if baseline_mse > 0 else 0.0
    return {
        "model": "linear regression (numpy lstsq), same method as train_reward_model.py, more data",
        "n_train": len(train_rewards), "n_test": len(test_rewards),
        "test_mse": round(mse, 5), "test_mae": round(mae, 5),
        "baseline_mse": round(baseline_mse, 8), "r2": round(r2, 4),
        "reward_std_in_test_set": round(float(test_rewards.std()), 5),
    }


class TinyRewardCNN(nn.Module):
    """Small enough for a few thousand training frames, not a few hundred
    thousand -- three conv layers, no attempt at anything deeper, since more
    capacity than the data supports would just overfit."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 4, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(-1)


def train_cnn_reward_model(train_frames, train_rewards, test_frames, test_rewards,
                            epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    X_train = _downsample_batch(train_frames, CNN_INPUT_SIZE)
    X_test = _downsample_batch(test_frames, CNN_INPUT_SIZE)
    mean, std = X_train.mean(axis=(0, 1, 2)), X_train.std(axis=(0, 1, 2)) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_x = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    train_y = torch.FloatTensor(train_rewards)
    test_x = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    test_y = torch.FloatTensor(test_rewards)

    model = TinyRewardCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(test_x).numpy()
    mse = float(np.mean((test_rewards - pred) ** 2))
    mae = float(np.mean(np.abs(test_rewards - pred)))
    baseline_mse = float(np.mean((test_rewards - train_rewards.mean()) ** 2))
    r2 = (1 - mse / baseline_mse) if baseline_mse > 0 else 0.0

    # Aggregate MAE is dominated by the step-penalty ticks (~98% of frames),
    # which drowns out whatever the model does on the rare, actually-
    # informative events (pickup/death/success) -- the ones a real reward
    # model needs to get right. Broken out separately so that story is
    # visible rather than averaged away.
    common_mask = np.abs(test_rewards - STEP_REWARD_REF) < 1e-3
    rare_mask = ~common_mask
    baseline_mean = train_rewards.mean()
    breakdown = {
        "n_common_ticks": int(common_mask.sum()), "n_rare_event_ticks": int(rare_mask.sum()),
        "mae_common_ticks": round(float(np.mean(np.abs(test_rewards[common_mask] - pred[common_mask]))), 5) if common_mask.any() else None,
        "mae_rare_events": round(float(np.mean(np.abs(test_rewards[rare_mask] - pred[rare_mask]))), 5) if rare_mask.any() else None,
        "mean_predictor_mae_rare_events": round(float(np.mean(np.abs(test_rewards[rare_mask] - baseline_mean))), 5) if rare_mask.any() else None,
    }

    return {
        "model": f"TinyRewardCNN (torch, {sum(p.numel() for p in model.parameters())} params)",
        "n_train": len(train_rewards), "n_test": len(test_rewards),
        "test_mse": round(mse, 5), "test_mae": round(mae, 5),
        "baseline_mse": round(baseline_mse, 8), "r2": round(r2, 4),
        "reward_std_in_test_set": round(float(test_rewards.std()), 5),
        "rare_event_breakdown": breakdown,
    }
