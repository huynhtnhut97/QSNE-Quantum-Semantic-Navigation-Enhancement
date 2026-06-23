"""High-level policy that wires PQC features and LLM embeddings to the LSTM.

This module exposes a single class, QSNEPolicy, that owns the PQC, the
network with the two heads, and a small amount of state needed for an
online interaction loop. It does not own the LLM module itself; the
embedding e_t is passed in from outside so that the (potentially slow,
asynchronous) LLM call lives in a separate process or thread, consistent
with Section 2.4 of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from .networks import (
    ACTION_DIM,
    LLM_EMBED_DIM,
    PQC_FEATURE_DIM,
    QSNENetwork,
    clip_action,
)
from .pqc import PQCFeatureExtractor


class QSNEPolicy(nn.Module):
    """Full QSNE policy: PQC -> concat with LLM embedding -> LSTM -> heads."""

    def __init__(
        self,
        pqc_num_qubits: int = PQC_FEATURE_DIM,
        pqc_num_layers: int = 4,
        llm_embed_dim: int = LLM_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.pqc = PQCFeatureExtractor(
            num_qubits=pqc_num_qubits, num_layers=pqc_num_layers
        )
        self.net = QSNENetwork()
        self.llm_embed_dim = llm_embed_dim
        # Pre-allocate a zero LLM embedding for the cold-start / fallback path.
        self.register_buffer(
            "zero_embedding",
            torch.zeros(llm_embed_dim, dtype=torch.float32),
            persistent=False,
        )

    # -------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------
    def _features(
        self, u: torch.Tensor, e: torch.Tensor | None
    ) -> torch.Tensor:
        """Concatenate quantum features f_t with the LLM embedding e_t."""
        f = self.pqc(u)                                # (B, 6)
        if e is None:
            e = self.zero_embedding.unsqueeze(0).expand(f.shape[0], -1)
        if e.dim() == 1:
            e = e.unsqueeze(0).expand(f.shape[0], -1)
        return torch.cat([f, e], dim=-1)               # (B, 262)

    def forward(
        self,
        u: torch.Tensor,
        e: torch.Tensor | None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """Run one step through PQC + LSTM + heads."""
        i_t = self._features(u, e)
        return self.net(i_t, hidden)

    # -------------------------------------------------------------------------
    # Action sampling
    # -------------------------------------------------------------------------
    @staticmethod
    def _dist_from_outputs(outputs: dict) -> Independent:
        std = outputs["log_std"].exp()
        return Independent(Normal(outputs["mu"], std), 1)

    def act(
        self,
        u: torch.Tensor,
        e: torch.Tensor | None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> dict:
        """Sample an action and return everything needed by the rollout buffer.

        Returns
        -------
        dict with keys
            action      : clipped action (B, 2)
            raw_action  : pre-clip Gaussian sample (B, 2)
            log_prob    : log pi(a | h) of the raw action (B,)
            value       : V(h) (B,)
            scan_hat    : reconstructed scan (B, 720)
            hidden      : (h_n, c_n)
        """
        outputs = self.forward(u, e, hidden)
        dist = self._dist_from_outputs(outputs)
        if deterministic:
            raw_a = outputs["mu"]
        else:
            raw_a = dist.sample()
        log_prob = dist.log_prob(raw_a)
        a = clip_action(raw_a)
        return {
            "action": a,
            "raw_action": raw_a,
            "log_prob": log_prob,
            "value": outputs["value"].squeeze(-1),
            "scan_hat": outputs["scan_hat"],
            "hidden": outputs["hidden"],
        }

    def evaluate_actions(
        self,
        u_seq: torch.Tensor,
        e_seq: torch.Tensor,
        actions: torch.Tensor,
        hidden_init: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """Recompute log-prob, entropy, and value for a stored batch.

        Used inside the PPO update loop. The batch axis is treated as a set
        of independent length-1 steps, which is sufficient for the
        truncated-trajectory PPO update described in the paper.
        """
        outputs = self.forward(u_seq, e_seq, hidden_init)
        dist = self._dist_from_outputs(outputs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": outputs["value"].squeeze(-1),
            "scan_hat": outputs["scan_hat"],
        }
