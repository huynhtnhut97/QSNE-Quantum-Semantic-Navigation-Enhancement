"""PPO update loop for the QSNE dual-head policy.

Implements the clipped PPO objective of Section 2.3 of the paper,
augmented with the scan-reconstruction auxiliary loss from the policy/value
+ scan-decoder architecture.

Loss = -L_PPO(theta) + c_v * L_V(theta) - c_e * H(pi_theta)
       + c_scan * L_scan(theta)

with hyperparameters taken from the consolidated table:
    epsilon (clip)          = 0.2
    gamma                   = 0.99
    GAE lambda              = 0.95
    entropy coefficient     = 1e-2
    value coefficient       = 0.5  (standard PPO default; not in paper table)
    scan reconstruction c   = 0.1  (light auxiliary weight; ablation toggle)
    learning rate           = 3e-4
    batch size              = 64
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import RolloutBuffer
from .policy import QSNEPolicy


@dataclass
class PPOConfig:
    """PPO hyperparameters (Section 2.3 and the consolidated table)."""

    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 10
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 1e-2
    scan_recon_coef: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


class PPOTrainer:
    """Run the PPO update over a populated RolloutBuffer.

    The trainer holds the optimizer and the running statistics. The caller
    populates a RolloutBuffer through environment interaction, then calls
    `update(buffer)` to apply the PPO loss for `num_epochs` epochs.
    """

    def __init__(
        self, policy: QSNEPolicy, cfg: PPOConfig | None = None
    ) -> None:
        self.policy = policy
        self.cfg = cfg or PPOConfig()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.learning_rate
        )

    # -------------------------------------------------------------------------
    # PPO update
    # -------------------------------------------------------------------------
    def update(self, buffer: RolloutBuffer) -> dict:
        """Run num_epochs of mini-batch PPO updates on the buffer."""
        logs = {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_value": 0.0,
            "loss_entropy": 0.0,
            "loss_scan": 0.0,
            "kl": 0.0,
            "clip_frac": 0.0,
            "n_batches": 0,
        }

        for _ in range(self.cfg.num_epochs):
            for batch in buffer.iter_minibatches(self.cfg.batch_size):
                obs, embed, clean_scan, action, old_log_prob, adv, ret = batch

                outs = self.policy.evaluate_actions(
                    obs, embed, action, hidden_init=None
                )
                new_log_prob = outs["log_prob"]
                entropy = outs["entropy"].mean()
                value = outs["value"]
                scan_hat = outs["scan_hat"]

                # PPO clipped policy loss.
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Schulman et al. clipped form omitted for clarity).
                value_loss = F.mse_loss(value, ret)

                # Scan reconstruction auxiliary loss (masked MSE).
                # NaN entries in clean_scan are treated as ignored.
                mask = (~torch.isnan(clean_scan)).float()
                target = torch.nan_to_num(clean_scan, nan=0.0)
                scan_loss = ((scan_hat - target) ** 2 * mask).sum() / (
                    mask.sum() + 1e-8
                )

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                    + self.cfg.scan_recon_coef * scan_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = (old_log_prob - new_log_prob).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )

                logs["loss_total"] += float(loss.item())
                logs["loss_policy"] += float(policy_loss.item())
                logs["loss_value"] += float(value_loss.item())
                logs["loss_entropy"] += float(entropy.item())
                logs["loss_scan"] += float(scan_loss.item())
                logs["kl"] += kl
                logs["clip_frac"] += clip_frac
                logs["n_batches"] += 1

        # Average across all updates so the caller can log cleanly.
        n = max(logs["n_batches"], 1)
        for k in ("loss_total", "loss_policy", "loss_value", "loss_entropy",
                  "loss_scan", "kl", "clip_frac"):
            logs[k] /= n
        return logs
