#!/usr/bin/env python
"""Entry point for the QSNE ROS bridge node.

Loaded by `qsne.launch`. Reads the checkpoint path from the private rospy
parameter server and spins the control loop. Run this directly with
roslaunch; do not invoke it as a regular Python script.
"""

from __future__ import annotations

import rospy
import torch

from qsne.llm_module import AsyncLLMModule
from qsne.policy import QSNEPolicy
from qsne.ros_bridge import QSNEROSBridge, ros_available


def main() -> None:
    if not ros_available():
        raise RuntimeError(
            "rospy not importable; this script must be launched under ROS."
        )

    checkpoint_path = rospy.get_param("~checkpoint")
    control_rate_hz = rospy.get_param("~control_rate_hz", 10.0)
    use_llm = rospy.get_param("~use_llm", True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy = QSNEPolicy()
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    llm = AsyncLLMModule() if use_llm else None
    bridge = QSNEROSBridge(
        policy=policy, llm=llm, control_rate_hz=control_rate_hz
    )
    rospy.loginfo("[QSNE] bridge node ready; waiting for /move_base_simple/goal")
    bridge.spin()


if __name__ == "__main__":
    main()
