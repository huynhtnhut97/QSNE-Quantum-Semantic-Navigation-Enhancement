"""Reward function for the QSNE navigation task.

Implements the reward defined in Section 2 of the paper:

    r_t = +10                          if ||p_t - p_g|| < 0.5  (goal reached)
        = -10                          if collision is registered
        = -0.01 - 0.5 * 1[d_t >= d_{t-1}]    otherwise

where d_t = ||p_t - p_g|| is the distance to the goal. The dense per-step
term penalizes long episodes and rewards progress toward the goal; the
terminal rewards enforce success and safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Reward function parameters (Section 2 of the paper)."""

    goal_radius: float = 0.5     # meters; success threshold
    goal_reward: float = 10.0
    collision_penalty: float = -10.0
    step_cost: float = -0.01
    no_progress_penalty: float = -0.5
    safety_distance: float = 0.35  # meters; collision threshold against LiDAR


@dataclass
class StepOutcome:
    """Container for the per-step terminal status returned by an environment."""

    reached_goal: bool
    collided: bool


def compute_reward(
    prev_distance: float,
    curr_distance: float,
    outcome: StepOutcome,
    cfg: RewardConfig | None = None,
) -> float:
    """Compute r_t given the previous and current goal distances.

    Parameters
    ----------
    prev_distance : float
        d_{t-1}, the distance to the goal at the previous step.
    curr_distance : float
        d_t, the distance to the goal at the current step.
    outcome : StepOutcome
        Goal/collision flags reported by the environment.
    cfg : RewardConfig, optional
        Reward parameters. Defaults to the paper values.
    """
    cfg = cfg or RewardConfig()
    if outcome.reached_goal:
        return float(cfg.goal_reward)
    if outcome.collided:
        return float(cfg.collision_penalty)
    no_progress = float(curr_distance >= prev_distance)
    return float(cfg.step_cost + cfg.no_progress_penalty * no_progress)


def check_collision(
    scan: np.ndarray, cfg: RewardConfig | None = None
) -> bool:
    """Return True when any surviving ray is closer than the safety distance.

    NaN rays are ignored. A collision is registered when at least one valid
    range falls below `cfg.safety_distance`.
    """
    cfg = cfg or RewardConfig()
    valid = scan[~np.isnan(scan)]
    if valid.size == 0:
        return False
    return bool(np.any(valid < cfg.safety_distance))


def check_goal_reached(
    pos: np.ndarray, goal: np.ndarray, cfg: RewardConfig | None = None
) -> bool:
    """Return True when the robot is within `cfg.goal_radius` of the goal."""
    cfg = cfg or RewardConfig()
    return bool(np.linalg.norm(pos - goal) < cfg.goal_radius)
