"""
Phase 6: Behavioral cloning policy network.

A 4-layer MLP that maps an augmented observation (27-dim) to a
a 2-dim continuous action [acceleration, steering], both in [−1, 1].

Input (27-dim): 25 flattened 5×5 obs features + d_min + step
Architecture: 27 → 256 → 256 → 128 → 2
Output activation: tanh — enforces the [−1, 1] action range without clamping.

Usage:
    from policy_network import PolicyNetwork
    model = PolicyNetwork()          # default obs_dim=27, action_dim=2
    action = model(obs_tensor)       # (B, 27) → (B, 2)
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = 27, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # output in (-1, 1) — matches Box action space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, obs: "np.ndarray") -> "np.ndarray":
        """
        Convenience method for rollout: accepts a (5,5), (25,) or already
        normalized flat numpy array and returns a (2,) float32 numpy action.
        Normalization (mean/std) should be applied by the caller before passing
        obs here when stats are available.
        """
        import numpy as np
        x = torch.tensor(obs.reshape(-1), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.net(x).squeeze(0).numpy().astype(np.float32)
