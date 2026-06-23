"""LSTM backbone with the two heads of QSNE.

Implements Section 2 architecture: a single 128-unit LSTM whose hidden state
h_t feeds two parallel heads:

    Head 1 (Policy/Value)
        - Two 64-unit fully-connected layers.
        - Outputs the Gaussian mean (mu_v, mu_omega), the log-std for each
          action component, and the scalar value V(h_t).
        - Action space (clipped): v in [0, 2.0] m/s, omega in [-1.0, 1.0]
          rad/s. The Husky platform limits are enforced at sampling time.

    Head 2 (Scan Decoder)
        - Two-layer fully-connected decoder.
        - Outputs s_hat_t in R^720, the corrected LiDAR scan published on
          /scan_corrected for Gmapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Architectural constants (Section 2 and the consolidated hyperparameter table).
PQC_FEATURE_DIM: int = 6
LLM_EMBED_DIM: int = 256
LSTM_INPUT_DIM: int = PQC_FEATURE_DIM + LLM_EMBED_DIM  # 262
LSTM_HIDDEN_DIM: int = 128
FC_WIDTH: int = 64
ACTION_DIM: int = 2
SCAN_DIM: int = 720

# Action clipping limits (Husky UGV velocity envelope).
V_MAX: float = 2.0
OMEGA_MAX: float = 1.0


class QSNENetwork(nn.Module):
    """LSTM + two-head network that produces (action, value, corrected scan).

    The network takes the concatenated input i_t = [f_t, e_t] of size 262
    and returns:
        - mu_t      : Gaussian mean for the action distribution (batch, 2)
        - log_std   : Gaussian log-std for the action distribution (batch, 2)
        - value     : scalar value estimate (batch, 1)
        - scan_hat  : reconstructed LiDAR scan (batch, 720)
    """

    def __init__(
        self,
        input_dim: int = LSTM_INPUT_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        fc_width: int = FC_WIDTH,
        action_dim: int = ACTION_DIM,
        scan_dim: int = SCAN_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared LSTM backbone.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Policy head: 2 x FC(64) -> action mean.
        self.policy_trunk = nn.Sequential(
            nn.Linear(hidden_dim, fc_width),
            nn.Tanh(),
            nn.Linear(fc_width, fc_width),
            nn.Tanh(),
        )
        self.action_mean_head = nn.Linear(fc_width, action_dim)
        # State-independent log std parameter (standard PPO trick).
        self.action_log_std = nn.Parameter(
            torch.tensor([-2.3, -3.0], dtype=torch.float32)
        )

        # Value head shares its trunk style but has its own parameters.
        self.value_trunk = nn.Sequential(
            nn.Linear(hidden_dim, fc_width),
            nn.Tanh(),
            nn.Linear(fc_width, fc_width),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(fc_width, 1)

        # Scan-reconstruction head: two-layer decoder mapping h_t -> 720-D.
        self.scan_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, scan_dim),
        )

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero (h_0, c_0) suitable for a fresh episode."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self,
        i_t: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """Run one (or a batch of) step through the network.

        Parameters
        ----------
        i_t : torch.Tensor
            Either shape (batch, input_dim) for a single step, or
            (batch, seq, input_dim) for a sequence.
        hidden : (h_0, c_0), optional
            Previous LSTM state. If None, defaults to zeros.

        Returns
        -------
        dict with keys
            mu        : (batch, action_dim)
            log_std   : (batch, action_dim)
            value     : (batch, 1)
            scan_hat  : (batch, scan_dim)
            hidden    : (h_n, c_n)
        """
        if i_t.dim() == 2:
            i_t = i_t.unsqueeze(1)  # add a length-1 sequence axis
        batch = i_t.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, device=i_t.device)

        lstm_out, hidden_out = self.lstm(i_t, hidden)
        # Take the final time step's hidden state for the heads.
        h_last = lstm_out[:, -1, :]

        # Policy head.
        p_feat = self.policy_trunk(h_last)
        mu = self.action_mean_head(p_feat)
        log_std = self.action_log_std.unsqueeze(0).expand_as(mu)

        # Value head.
        v_feat = self.value_trunk(h_last)
        value = self.value_head(v_feat)

        # Scan decoder head.
        scan_hat = self.scan_decoder(h_last)

        return {
            "mu": mu,
            "log_std": log_std,
            "value": value,
            "scan_hat": scan_hat,
            "hidden": hidden_out,
        }


def clip_action(action: torch.Tensor) -> torch.Tensor:
    """Clip a (batch, 2) action tensor to the Husky velocity envelope."""
    v = torch.clamp(action[..., 0:1], 0.0, V_MAX)
    w = torch.clamp(action[..., 1:2], -OMEGA_MAX, OMEGA_MAX)
    return torch.cat([v, w], dim=-1)
