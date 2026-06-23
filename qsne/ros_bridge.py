"""ROS bridge for QSNE deployment on the Husky UGV.

This module wires the QSNE policy into the ROS Noetic navigation stack:

    Subscribes to:
        /noisy-scan       sensor_msgs/LaserScan   (degraded LiDAR)
        /odom             nav_msgs/Odometry       (wheel + IMU odometry)
        /move_base_simple/goal   geometry_msgs/PoseStamped  (user-supplied goal)

    Publishes:
        /cmd_vel          geometry_msgs/Twist     (consumed by move_base
                                                   local-planner override)
        /scan_corrected   sensor_msgs/LaserScan   (consumed by Gmapping)

The 10 Hz control loop runs on the on-board CPU. The LLM call lives in a
background thread (see `llm_module.py`) so that the ~620 ms GPT-4o round
trip never blocks the control loop. The trigger logic of Section 2.4
(every 10 steps or sigma^2 > tau) decides when a new LLM call is issued.

The module is intentionally optional: `import rospy` is attempted lazily.
On systems without ROS installed, importing this file still succeeds and
exposes a `ros_available()` helper that returns False, which lets the
training pipeline run unaffected.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .aggregation import LLM_QUERY_INTERVAL, NUM_RAYS, compute_sector_summary, encode_for_pqc, llm_should_query
from .llm_module import AsyncLLMModule, build_prompt
from .policy import QSNEPolicy

# Lazy ROS import so that this file is safe to import on non-ROS systems.
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from tf.transformations import euler_from_quaternion
    _ROS_AVAILABLE = True
except Exception:
    _ROS_AVAILABLE = False


def ros_available() -> bool:
    """Return True when rospy and its sibling packages are importable."""
    return _ROS_AVAILABLE


# -----------------------------------------------------------------------------
# Bridge class (only meaningful when ROS is available)
# -----------------------------------------------------------------------------
class QSNEROSBridge:
    """Wire a trained QSNE policy into the ROS navigation stack.

    Parameters
    ----------
    policy : QSNEPolicy
        Trained QSNE network with PQC + LSTM + heads.
    llm : AsyncLLMModule
        Background LLM caller. The cached embedding is consumed on every
        control step without blocking.
    control_rate_hz : float
        Loop rate for the publisher. Defaults to 10 Hz (Section 2.4).
    """

    def __init__(
        self,
        policy: QSNEPolicy,
        llm: AsyncLLMModule | None = None,
        control_rate_hz: float = 10.0,
        device: str = "cpu",
    ) -> None:
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rospy is not importable; install ROS Noetic and source the "
                "workspace before constructing QSNEROSBridge."
            )
        self.policy = policy.to(device).eval()
        self.llm = llm or AsyncLLMModule()
        self.device = device
        self.rate_hz = control_rate_hz

        # State.
        self.latest_scan: np.ndarray | None = None
        self.latest_pose: tuple[float, float, float] | None = None
        self.latest_vel: tuple[float, float] = (0.0, 0.0)
        self.goal: np.ndarray | None = None
        self.step_idx: int = 0
        self.hidden = self.policy.net.init_hidden(batch_size=1, device=device)

        rospy.init_node("qsne_bridge", anonymous=False)
        rospy.Subscriber("/noisy-scan", LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self._goal_cb, queue_size=1
        )
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.scan_pub = rospy.Publisher(
            "/scan_corrected", LaserScan, queue_size=1
        )

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------
    def _scan_cb(self, msg: "LaserScan") -> None:
        arr = np.array(msg.ranges, dtype=np.float32)
        # Treat infs and zeros as dropped rays for consistency with the
        # paper's NaN convention.
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0.0] = np.nan
        # Resample to NUM_RAYS if the device reports a different count.
        if arr.shape[0] != NUM_RAYS:
            xp = np.linspace(0.0, 1.0, arr.shape[0])
            xq = np.linspace(0.0, 1.0, NUM_RAYS)
            arr = np.interp(xq, xp, np.nan_to_num(arr, nan=0.0))
            arr[arr <= 0.0] = np.nan
        self.latest_scan = arr
        self._cached_scan_meta = (
            msg.angle_min, msg.angle_max, msg.angle_increment,
            msg.range_min, msg.range_max, msg.header.frame_id,
        )

    def _odom_cb(self, msg: "Odometry") -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.latest_pose = (float(p.x), float(p.y), float(yaw))
        self.latest_vel = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )

    def _goal_cb(self, msg: "PoseStamped") -> None:
        self.goal = np.array(
            [msg.pose.position.x, msg.pose.position.y], dtype=np.float32
        )
        rospy.loginfo(f"[QSNE] new goal received: {self.goal.tolist()}")
        # Reset LSTM hidden state on a new goal so memory of past episodes
        # does not contaminate the new task.
        self.hidden = self.policy.net.init_hidden(batch_size=1, device=self.device)
        self.step_idx = 0

    # -------------------------------------------------------------------------
    # Control loop
    # -------------------------------------------------------------------------
    def spin(self) -> None:
        """Run the 10 Hz control loop until ROS shuts down."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self._step_once()
            rate.sleep()

    def _step_once(self) -> None:
        if self.latest_scan is None or self.latest_pose is None or self.goal is None:
            return
        x, y, yaw = self.latest_pose
        pos = np.array([x, y], dtype=np.float32)
        v, w = self.latest_vel
        odom = np.array([x, y, yaw, v, w], dtype=np.float32)

        # Trigger an LLM call if appropriate.
        sector_sum = compute_sector_summary(self.latest_scan)
        if llm_should_query(self.step_idx, sector_sum):
            self.llm.trigger(build_prompt(sector_sum, odom))

        # Forward pass.
        u = encode_for_pqc(self.latest_scan, pos, yaw, self.goal)
        u_t = torch.from_numpy(u).unsqueeze(0).to(self.device)
        e_t = self.llm.get_embedding().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.policy.act(u_t, e_t, self.hidden, deterministic=True)
        action = out["action"].squeeze(0).cpu().numpy()
        self.hidden = out["hidden"]

        # Publish /cmd_vel.
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        # Publish /scan_corrected.
        scan_hat = out["scan_hat"].squeeze(0).cpu().numpy()
        self._publish_corrected_scan(scan_hat)

        self.step_idx += 1

    def _publish_corrected_scan(self, scan_hat: np.ndarray) -> None:
        if not hasattr(self, "_cached_scan_meta"):
            return
        a_min, a_max, a_inc, r_min, r_max, frame_id = self._cached_scan_meta
        msg = LaserScan()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.angle_min = a_min
        msg.angle_max = a_max
        msg.angle_increment = a_inc
        msg.range_min = r_min
        msg.range_max = r_max
        # Clip the reconstructed scan to physical sensor limits.
        clipped = np.clip(scan_hat, r_min, r_max).tolist()
        msg.ranges = clipped
        self.scan_pub.publish(msg)
