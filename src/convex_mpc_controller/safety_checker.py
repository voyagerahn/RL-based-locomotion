"""The Safety Check class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from email.mime import base

import numpy as np
from typing import Any

TORQUE_LIMIT = 13


class SafetyChecker():
    def __init__(
        self,
        robot: Any,
        state_estimator: Any,
    ):
        self._robot = robot
        self._state_estimator = state_estimator

    def CheckSafeOrientation(self):
        base_orientation = self._robot.GetBaseOrientaion()
        if np.abs(base_orientation(0)) >= TORQUE_LIMIT:
          print("Safety Checker : Orientation Fail :{}", base_orientation)
          
    def CheckSafeTorque(self, joint_id, torque):
        if np.abs(torque) >= TORQUE_LIMIT:
          print("Safety Checker : Torque is lower than Torque Limit Joint:{}, Torque:{}", joint_id, torque)
          torque = TORQUE_LIMIT
