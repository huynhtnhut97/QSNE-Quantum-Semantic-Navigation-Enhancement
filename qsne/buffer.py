"""Rollout buffer with Generalized Advantage Estimation.

Stores observations, actions, rewards, value estimates, and log-probabilities
for a fixed number of environment steps. After a rollout is collected, the
buffer computes returns and advantages with GAE(lambda) and exposes mini-batch
iterators for the PPO update loop. Matches the standard Schulman et al.
implementation up to notation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Fixed-capacity buffer for on-policy PPO updates."""

    capacity: int
    obs_dim: int                     # PQC input dim (6)
    embed_dim: int                   # LLM embedding dim (256)
    scan_dim: int                    # 720
    action_dim: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95

    obs: np.ndarray = field(init=False)
    embeds: np.ndarray = field(init=False)
    clean_scans: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)
    ptr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.embeds = np.zeros((self.capacity, self.embed_dim), dtype=np.float32)
        self.clean_scans = np.zeros(
            (self.capacity, self.scan_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        embed: np.ndarray,
        clean_scan: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer."""
        idx = self.ptr
        if idx >= self.capacity:
            raise RuntimeError("RolloutBuffer full; call reset or finalize.")
        self.obs[idx] = obs
        self.embeds[idx] = embed
        self.clean_scans[idx] = clean_scan
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = float(done)
        self.ptr += 1

    def finalize(self, last_value: float) -> None:
        """Compute GAE advantages and returns over the stored rollout."""
        adv = 0.0
        for t in reversed(range(self.ptr)):
            next_value = (
                last_value if t == self.ptr - 1 else self.values[t + 1]
            )
            next_nonterminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_nonterminal
                - self.values[t]
            )
            adv = delta + self.gamma * self.gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]
        # Normalize advantages for stable PPO updates.
        adv_slice = self.advantages[: self.ptr]
        self.advantages[: self.ptr] = (
            adv_slice - adv_slice.mean()
        ) / (adv_slice.std() + 1e-8)

    def reset(self) -> None:
        """Discard all stored data."""
        self.ptr = 0

    def iter_minibatches(self, batch_size: int):
        """Yield (obs, embed, clean_scan, action, log_prob, adv, ret) batches."""
        idx = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            chunk = idx[start : start + batch_size]
            yield (
                torch.from_numpy(self.obs[chunk]),
                torch.from_numpy(self.embeds[chunk]),
                torch.from_numpy(self.clean_scans[chunk]),
                torch.from_numpy(self.actions[chunk]),
                torch.from_numpy(self.log_probs[chunk]),
                torch.from_numpy(self.advantages[chunk]),
                torch.from_numpy(self.returns[chunk]),
            )
