"""Lightweight 2D simulator for training QSNE without Gazebo.

The simulator carries enough fidelity to exercise the full pipeline:
    - 720-ray LiDAR ray-cast against a polygonal occupancy map.
    - Differential-drive kinematics matching the Husky velocity envelope.
    - Per-ray dropout + Gaussian noise degradation through `degradation.py`.
    - Goal-conditioned episodes that terminate on success, collision, or
      the horizon T_max = 1500 of Section 2.

The intent is to let users compare QSNE against alternative implementations
quickly on a laptop. For high-fidelity simulation, swap this module out
for the Gazebo-backed ROS bridge in `ros_bridge.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aggregation import (
    NUM_RAYS,
    compute_sector_summary,
    encode_for_pqc,
    llm_should_query,
)
from .degradation import DegradationConfig, apply_degradation
from .reward import (
    RewardConfig,
    StepOutcome,
    check_collision,
    check_goal_reached,
    compute_reward,
)

# Husky velocity envelope (Section 2).
V_MAX: float = 2.0
OMEGA_MAX: float = 1.0
DT: float = 0.1                  # 10 Hz control loop -> 100 ms per step
T_MAX: int = 1500                # episode horizon (Section 2 termination)
LIDAR_MAX_RANGE: float = 30.0    # meters; VLP-16 effective range


# -----------------------------------------------------------------------------
# Map primitives
# -----------------------------------------------------------------------------
@dataclass
class RectObstacle:
    """Axis-aligned rectangular obstacle defined by min/max corners."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class WorldMap:
    """Rectangular world bounded by walls plus a list of inner obstacles.

    The boundary is itself treated as an obstacle for the LiDAR ray-cast,
    which prevents the robot from running off the world.
    """

    x_min: float = -10.0
    y_min: float = -10.0
    x_max: float = 10.0
    y_max: float = 10.0
    obstacles: list[RectObstacle] = field(default_factory=list)

    def all_segments(self) -> np.ndarray:
        """Return every wall segment in the world as (N, 4) coordinates."""
        segs = [
            # outer walls
            (self.x_min, self.y_min, self.x_max, self.y_min),
            (self.x_max, self.y_min, self.x_max, self.y_max),
            (self.x_max, self.y_max, self.x_min, self.y_max),
            (self.x_min, self.y_max, self.x_min, self.y_min),
        ]
        for o in self.obstacles:
            segs.extend([
                (o.x_min, o.y_min, o.x_max, o.y_min),
                (o.x_max, o.y_min, o.x_max, o.y_max),
                (o.x_max, o.y_max, o.x_min, o.y_max),
                (o.x_min, o.y_max, o.x_min, o.y_min),
            ])
        return np.asarray(segs, dtype=np.float32)

    def contains(self, x: float, y: float) -> bool:
        """Test whether (x, y) is inside the world and outside obstacles."""
        if not (self.x_min < x < self.x_max and self.y_min < y < self.y_max):
            return False
        for o in self.obstacles:
            if o.x_min < x < o.x_max and o.y_min < y < o.y_max:
                return False
        return True


def default_office_world() -> WorldMap:
    """Build a small indoor-office-style world with a few obstacles."""
    return WorldMap(
        x_min=-6.0, y_min=-6.0, x_max=6.0, y_max=6.0,
        obstacles=[
            RectObstacle(-2.0, -1.0, -1.0, 3.0),
            RectObstacle(1.5, -3.0, 3.5, -1.0),
            RectObstacle(2.0, 2.0, 4.0, 4.0),
        ],
    )


def default_outdoor_world() -> WorldMap:
    """Build a wider, sparser outdoor-style world."""
    return WorldMap(
        x_min=-12.0, y_min=-12.0, x_max=12.0, y_max=12.0,
        obstacles=[
            RectObstacle(-3.0, -6.0, -1.0, -4.0),
            RectObstacle(4.0, 3.0, 5.0, 7.0),
            RectObstacle(-6.0, 5.0, -4.0, 7.0),
        ],
    )


# -----------------------------------------------------------------------------
# 720-ray LiDAR
# -----------------------------------------------------------------------------
def _ray_segment_intersect(
    ox: float, oy: float, dx: float, dy: float,
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Return the parametric distance t at which the ray hits the segment.

    A return value of +inf means no intersection within the forward ray.
    """
    sx, sy = x2 - x1, y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return float("inf")
    t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return float("inf")


def cast_lidar(
    pos: np.ndarray, yaw: float, world: WorldMap,
    num_rays: int = NUM_RAYS, max_range: float = LIDAR_MAX_RANGE,
) -> np.ndarray:
    """Cast a 720-ray scan from the robot pose against the world segments.

    The scan covers 360 degrees with uniform angular spacing. Distances
    beyond `max_range` are clipped.
    """
    angles = yaw + np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    segs = world.all_segments()
    scan = np.full(num_rays, max_range, dtype=np.float32)
    ox, oy = float(pos[0]), float(pos[1])

    for i, a in enumerate(angles):
        dx, dy = float(np.cos(a)), float(np.sin(a))
        best = max_range
        for seg in segs:
            t = _ray_segment_intersect(ox, oy, dx, dy, *seg)
            if t < best:
                best = t
        scan[i] = best
    return scan


# -----------------------------------------------------------------------------
# Differential-drive kinematics
# -----------------------------------------------------------------------------
def step_kinematics(
    pos: np.ndarray, yaw: float, action: np.ndarray, dt: float = DT,
) -> tuple[np.ndarray, float]:
    """Forward unicycle kinematics for one control step.

    Parameters
    ----------
    pos : np.ndarray, shape (2,)
    yaw : float
    action : np.ndarray, shape (2,)
        [v, omega] in m/s and rad/s. The Husky envelope is enforced upstream.
    dt : float
        Integration step.
    """
    v, w = float(action[0]), float(action[1])
    new_yaw = yaw + w * dt
    new_pos = pos + np.array(
        [v * np.cos(yaw) * dt, v * np.sin(yaw) * dt], dtype=np.float32
    )
    # Wrap yaw to [-pi, pi].
    new_yaw = float(np.arctan2(np.sin(new_yaw), np.cos(new_yaw)))
    return new_pos.astype(np.float32), new_yaw


# -----------------------------------------------------------------------------
# Gym-style environment
# -----------------------------------------------------------------------------
@dataclass
class QSNEEnvConfig:
    """Environment hyperparameters."""

    world_factory: callable = field(default=default_office_world)
    horizon: int = T_MAX
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int | None = None


class QSNEEnv:
    """Minimal Gym-style environment for QSNE training/evaluation.

    The observation returned by `reset` and `step` is a dict that contains
    everything the policy and the LLM module need:
        u           : 6-D PQC input
        sector_sum  : SectorSummary instance
        scan        : 720-D degraded scan (with NaN for dropped rays)
        clean_scan  : 720-D clean scan (used as the scan-decoder target)
        odom        : 5-D odometry [x, y, yaw, v, omega]
        goal        : 2-D goal coordinates
        step        : current step index (for the LLM trigger)
    """

    metadata = {"render.modes": []}

    def __init__(self, cfg: QSNEEnvConfig | None = None) -> None:
        self.cfg = cfg or QSNEEnvConfig()
        self.world = self.cfg.world_factory()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._reset_internal_state()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def _reset_internal_state(self) -> None:
        self.pos = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_idx = 0
        self.prev_distance = 0.0

    def _sample_free_point(self) -> np.ndarray:
        for _ in range(1000):
            x = float(self.rng.uniform(self.world.x_min + 0.5, self.world.x_max - 0.5))
            y = float(self.rng.uniform(self.world.y_min + 0.5, self.world.y_max - 0.5))
            if self.world.contains(x, y):
                return np.array([x, y], dtype=np.float32)
        raise RuntimeError("Could not sample a free point in the world.")

    def reset(self) -> dict:
        """Start a new episode with random free start and goal points."""
        self._reset_internal_state()
        self.pos = self._sample_free_point()
        self.goal = self._sample_free_point()
        # Force a reasonable initial heading toward the goal.
        delta = self.goal - self.pos
        self.yaw = float(np.arctan2(delta[1], delta[0]))
        self.prev_distance = float(np.linalg.norm(delta))
        return self._build_observation()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Apply the action, return (obs, reward, done, info)."""
        # Clip action.
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action[0] = float(np.clip(action[0], 0.0, V_MAX))
        action[1] = float(np.clip(action[1], -OMEGA_MAX, OMEGA_MAX))
        self.v, self.w = float(action[0]), float(action[1])

        # Advance the kinematics.
        new_pos, new_yaw = step_kinematics(self.pos, self.yaw, action)
        if self.world.contains(float(new_pos[0]), float(new_pos[1])):
            self.pos, self.yaw = new_pos, new_yaw
        # If the candidate position is in collision, keep the previous one.

        self.step_idx += 1

        # Build observation, then check termination.
        obs = self._build_observation()
        outcome = StepOutcome(
            reached_goal=check_goal_reached(self.pos, self.goal, self.cfg.reward),
            collided=check_collision(obs["scan"], self.cfg.reward),
        )
        curr_distance = float(np.linalg.norm(self.goal - self.pos))
        reward = compute_reward(
            prev_distance=self.prev_distance,
            curr_distance=curr_distance,
            outcome=outcome,
            cfg=self.cfg.reward,
        )
        self.prev_distance = curr_distance

        done = (
            outcome.reached_goal
            or outcome.collided
            or self.step_idx >= self.cfg.horizon
        )
        info = {
            "reached_goal": outcome.reached_goal,
            "collided": outcome.collided,
            "distance": curr_distance,
            "step": self.step_idx,
        }
        return obs, reward, done, info

    # -------------------------------------------------------------------------
    # Observation helpers
    # -------------------------------------------------------------------------
    def _build_observation(self) -> dict:
        clean_scan = cast_lidar(self.pos, self.yaw, self.world)
        scan = apply_degradation(clean_scan, self.cfg.degradation, self.rng)
        sector_sum = compute_sector_summary(scan)
        u = encode_for_pqc(scan, self.pos, self.yaw, self.goal)
        odom = np.array(
            [self.pos[0], self.pos[1], self.yaw, self.v, self.w],
            dtype=np.float32,
        )
        return {
            "u": u,
            "sector_sum": sector_sum,
            "scan": scan,
            "clean_scan": clean_scan,
            "odom": odom,
            "goal": self.goal.copy(),
            "step": self.step_idx,
            "should_query_llm": llm_should_query(self.step_idx, sector_sum),
        }
