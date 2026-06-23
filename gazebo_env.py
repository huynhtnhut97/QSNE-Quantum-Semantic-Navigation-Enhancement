"""gym-gazebo style env for Gazebo-based QSNE training.

Mirrors the QSNEEnv interface so scripts/train.py and scripts/evaluate.py
work unchanged. The env consumes the clean LiDAR scan and odometry
published by Gazebo, applies the per-ray dropout + noise degradation of
Section 2 internally, and exposes the same observation dict.

Control pattern (synchronous, gym-gazebo convention):
    1. unpause /gazebo/unpause_physics
    2. publish /cmd_vel
    3. sleep one control period (0.1 s at 10 Hz)
    4. pause /gazebo/pause_physics
    5. read the latest /scan and /odom messages, build the observation

Prerequisites:
    - Gazebo is already running (see launch/qsne_training.launch).
    - The Husky model is spawned and its model_name matches cfg.model_name.
    - rospy and gazebo_msgs are importable.
"""

from __future__ import annotations

import time
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

# Lazy ROS import so this file is safe to read on non-ROS systems.
try:
    import rospy
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import (
        GetModelState,
        SetModelState,
        SetModelStateRequest,
    )
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
    _ROS_OK = True
except Exception:
    _ROS_OK = False


# Husky velocity envelope (Section 2 and consolidated hyperparameter table).
V_MAX: float = 2.0
OMEGA_MAX: float = 1.0
DT: float = 0.1                  # 10 Hz control loop
T_MAX: int = 1500                # episode horizon


@dataclass
class GazeboEnvConfig:
    """Parameters for the Gazebo-backed env.

    Attributes
    ----------
    model_name : str
        Gazebo model name of the Husky (Clearpath default: 'husky').
    scan_topic, odom_topic, cmd_topic : str
        Standard ROS topic names. Override if your Gazebo plugins publish
        on namespaced versions (e.g., '/husky/scan').
    base_frame, world_frame : str
        TF frames used for set_model_state and pose interpretation.
    x_min, y_min, x_max, y_max : float
        Bounding box of the free area. Start and goal points are
        rejection-sampled inside this box; points that fall inside any
        rectangle in `obstacles` are rejected.
    obstacles : list of (x0, y0, x1, y1)
        Axis-aligned obstacle bounding boxes used only for rejection
        sampling. The Gazebo physics engine still owns collision and
        contact detection at run time.
    use_bumper : bool
        If True, register collisions via the bumper topic. Otherwise fall
        back to the LiDAR-proximity rule used by QSNEEnv.
    """

    model_name: str = "husky"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    cmd_topic: str = "/cmd_vel"
    base_frame: str = "base_link"
    world_frame: str = "world"
    x_min: float = -10.0
    y_min: float = -10.0
    x_max: float = 10.0
    y_max: float = 10.0
    obstacles: list = field(default_factory=list)
    horizon: int = T_MAX
    control_period_s: float = DT
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int | None = None
    use_bumper: bool = False
    bumper_topic: str = "/bumper_states"
    service_wait_timeout_s: float = 30.0


# Preset world bounds for the four Clearpath Gazebo worlds. The user is
# expected to refine `obstacles` once they confirm the actual layout.
WORLD_PRESETS: dict[str, dict] = {
    "office":       {"x_min": -6.0, "y_min": -6.0, "x_max": 6.0, "y_max": 6.0},
    "construction": {"x_min": -8.0, "y_min": -8.0, "x_max": 8.0, "y_max": 8.0},
    "agriculture":  {"x_min": -15.0, "y_min": -15.0, "x_max": 15.0, "y_max": 15.0},
    "inspection":   {"x_min": -12.0, "y_min": -12.0, "x_max": 12.0, "y_max": 12.0},
}


class GazeboQSNEEnv:
    """Gazebo-backed env that mimics the QSNEEnv interface."""

    metadata = {"render.modes": []}

    def __init__(self, cfg: GazeboEnvConfig | None = None) -> None:
        if not _ROS_OK:
            raise RuntimeError(
                "rospy or gazebo_msgs not importable; install ROS Noetic "
                "and source your workspace before constructing this env."
            )
        self.cfg = cfg or GazeboEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # Buffered sensor state.
        self._scan: np.ndarray | None = None
        self._pose: tuple[float, float, float] | None = None
        self._vel: tuple[float, float] = (0.0, 0.0)
        self._bumper_collision: bool = False

        # Episode state.
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_idx = 0
        self.prev_distance = 0.0

        # ROS init (no-op if already initialized by an outer process).
        try:
            rospy.init_node(
                "qsne_gazebo_env", anonymous=True, disable_signals=True
            )
        except rospy.exceptions.ROSException:
            pass

        rospy.Subscriber(self.cfg.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber(self.cfg.odom_topic, Odometry, self._odom_cb, queue_size=10)
        if self.cfg.use_bumper:
            from gazebo_msgs.msg import ContactsState
            rospy.Subscriber(
                self.cfg.bumper_topic, ContactsState, self._bumper_cb,
                queue_size=1,
            )
        self.cmd_pub = rospy.Publisher(self.cfg.cmd_topic, Twist, queue_size=1)

        # Wait for Gazebo services, then build the proxies.
        for svc in (
            "/gazebo/reset_world", "/gazebo/pause_physics",
            "/gazebo/unpause_physics", "/gazebo/set_model_state",
            "/gazebo/get_model_state",
        ):
            rospy.wait_for_service(svc, timeout=self.cfg.service_wait_timeout_s)
        self._reset_world_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self._pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self._unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self._set_state_srv = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self._get_state_srv = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------
    def _scan_cb(self, msg: "LaserScan") -> None:
        arr = np.array(msg.ranges, dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0.0] = np.nan
        if arr.shape[0] != NUM_RAYS:
            xp = np.linspace(0.0, 1.0, arr.shape[0])
            xq = np.linspace(0.0, 1.0, NUM_RAYS)
            arr = np.interp(
                xq, xp, np.nan_to_num(arr, nan=0.0)
            ).astype(np.float32)
            arr[arr <= 0.0] = np.nan
        self._scan = arr

    def _odom_cb(self, msg: "Odometry") -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._pose = (float(p.x), float(p.y), float(yaw))
        self._vel = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )

    def _bumper_cb(self, msg) -> None:
        if len(msg.states) > 0:
            self._bumper_collision = True

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_free(self, x: float, y: float) -> bool:
        if not (self.cfg.x_min < x < self.cfg.x_max and
                self.cfg.y_min < y < self.cfg.y_max):
            return False
        for (ox0, oy0, ox1, oy1) in self.cfg.obstacles:
            if ox0 < x < ox1 and oy0 < y < oy1:
                return False
        return True

    def _sample_free_point(self) -> np.ndarray:
        for _ in range(1000):
            x = float(self.rng.uniform(
                self.cfg.x_min + 0.5, self.cfg.x_max - 0.5
            ))
            y = float(self.rng.uniform(
                self.cfg.y_min + 0.5, self.cfg.y_max - 0.5
            ))
            if self._is_free(x, y):
                return np.array([x, y], dtype=np.float32)
        raise RuntimeError("Could not sample a free point in the world.")

    def _teleport_robot(self, pos: np.ndarray, yaw: float) -> None:
        req = SetModelStateRequest()
        req.model_state = ModelState()
        req.model_state.model_name = self.cfg.model_name
        req.model_state.pose.position.x = float(pos[0])
        req.model_state.pose.position.y = float(pos[1])
        req.model_state.pose.position.z = 0.1
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))
        req.model_state.pose.orientation.x = qx
        req.model_state.pose.orientation.y = qy
        req.model_state.pose.orientation.z = qz
        req.model_state.pose.orientation.w = qw
        req.model_state.reference_frame = self.cfg.world_frame
        self._set_state_srv(req)

    def _wait_for_first_messages(self, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._scan is not None and self._pose is not None:
                return
            rospy.sleep(0.02)
        raise RuntimeError(
            "Timed out waiting for /scan and /odom; check Gazebo, the Husky "
            "model, and the topic names in GazeboEnvConfig."
        )

    def _build_observation(self) -> dict:
        clean_scan = (
            self._scan.copy() if self._scan is not None
            else np.full(NUM_RAYS, 30.0, dtype=np.float32)
        )
        scan = apply_degradation(clean_scan, self.cfg.degradation, self.rng)
        sector_sum = compute_sector_summary(scan)
        x, y, yaw = self._pose if self._pose is not None else (0.0, 0.0, 0.0)
        pos = np.array([x, y], dtype=np.float32)
        u = encode_for_pqc(scan, pos, yaw, self.goal)
        v_lin, w_ang = self._vel
        odom = np.array([x, y, yaw, v_lin, w_ang], dtype=np.float32)
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

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def reset(self) -> dict:
        """Reset Gazebo, sample a start/goal, return the first observation."""
        try:
            self._pause_srv()
        except rospy.ServiceException:
            pass
        self._reset_world_srv()

        start = self._sample_free_point()
        self.goal = self._sample_free_point()
        delta = self.goal - start
        start_yaw = float(np.arctan2(delta[1], delta[0]))
        self._teleport_robot(start, start_yaw)

        self.step_idx = 0
        self.prev_distance = float(np.linalg.norm(delta))
        self._bumper_collision = False
        self._scan = None
        self._pose = None

        self._unpause_srv()
        self._wait_for_first_messages()
        return self._build_observation()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Publish the action, advance one control period, return outcome."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        v = float(np.clip(action[0], 0.0, V_MAX))
        w = float(np.clip(action[1], -OMEGA_MAX, OMEGA_MAX))

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w

        # Synchronous step.
        try:
            self._unpause_srv()
        except rospy.ServiceException:
            pass
        self.cmd_pub.publish(twist)
        rospy.sleep(self.cfg.control_period_s)
        try:
            self._pause_srv()
        except rospy.ServiceException:
            pass

        self.step_idx += 1

        obs = self._build_observation()
        collided = (
            self._bumper_collision if self.cfg.use_bumper
            else check_collision(obs["scan"], self.cfg.reward)
        )
        self._bumper_collision = False  # consume the latch
        reached_goal = check_goal_reached(
            obs["odom"][:2], self.goal, self.cfg.reward
        )
        curr_distance = float(np.linalg.norm(self.goal - obs["odom"][:2]))
        outcome = StepOutcome(reached_goal=reached_goal, collided=collided)
        reward = compute_reward(
            prev_distance=self.prev_distance,
            curr_distance=curr_distance,
            outcome=outcome,
            cfg=self.cfg.reward,
        )
        self.prev_distance = curr_distance

        done = (
            reached_goal or collided or self.step_idx >= self.cfg.horizon
        )
        info = {
            "reached_goal": reached_goal,
            "collided": collided,
            "distance": curr_distance,
            "step": self.step_idx,
        }
        return obs, reward, done, info

    def close(self) -> None:
        """Publish a zero twist and release service handles."""
        twist = Twist()
        self.cmd_pub.publish(twist)
        try:
            self._unpause_srv()
        except rospy.ServiceException:
            pass
