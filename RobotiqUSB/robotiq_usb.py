"""Client implementation for the Robotiq gripper using MinimalModbus communication protocol.

This module provides a client interface to control a Robotiq gripper via USB interface.
"""

from __future__ import annotations

import minimalmodbus
import numpy as np
import numpy.typing as npt

from xrocs.common.data_type import Joints
from xrocs.entity.hand.hand_base import HandDriver
from xrocs.entity.hand.RobotiqUSB.robotiq_driver import RobotiqGripper
from typing import Optional

class RobotiqGripperUSB(HandDriver):
    """Robotiq gripper client implementation.

    This class provides a client interface to control a Robotiq gripper through their USB interface.
    """

    def __init__(self, hand_port: str) -> None:
        """Initialize the Robotiq gripper client.

        Args:
            hand_port: The port number of the gripper server.
        """
        self.gripper = None
        self.robot_port = hand_port
        self._last_val = 0

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom for the gripper.

        Returns:
            The number of degrees of freedom (always 1 for Robotiq grippers).
        """
        return 1

    def connect(self) -> bool:
        """Connect to the Robotiq gripper via USB.

        Returns:
            True if connection was successful. Note that this function always returns True.
        """
        self.gripper = RobotiqGripper(self.robot_port)
        self.gripper.resetActivate()
        return True

    def open(self) -> bool:
        """Open the gripper by setting the joint position to 0.

        Returns:
            True if the command was sent successfully, False otherwise.
        """
        return self.sync_target_joint(0)

    def close(self) -> bool:
        """Close the gripper by setting the joint position to 1.

        Returns:
            True if the command was sent successfully, False otherwise.
        """
        return self.sync_target_joint(1)

    def get_current_joint(self) -> Optional[Joints]:
        """Get the current joint position of the gripper.

        Returns:
            Joints: Current position of the gripper normalized between 0 (fully open) and 1 (fully closed).
        """
        try:
            self._last_val = gripper_pos = self.gripper.getPosition()
        except minimalmodbus.InvalidResponseError:
            print("minimalmodbus.InvalidResponseError retry")
            gripper_pos = self._last_val

        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        position = gripper_pos / 255
        return Joints(np.array([position]), num_of_dofs=self.num_dofs())

    def set_target_joint(self, target_joint: Joints) -> bool:
        """Set the target position for the gripper.

        Args:
            target_joint: The target joint position as a Joints object.

        Returns:
            True if the command was sent successfully, False otherwise.
        """
        return self.sync_target_joint(target_joint.get_radian_ndarray())

    def sync_target_joint(self, target_joint: npt.NDArray[np.float64], speed: int = 255, force: int = 255) -> bool:
        """Synchronously set the target joint position of the gripper.

        Args:
            target_joint: The target joint position as a numpy array.
                The value must be a value between 0 and 1 (0 represents fully open, 1 represents fully closed).
            speed: The speed of the gripper.
            force: The force of the gripper.

        Returns:
            True if the command was sent successfully. Note that this function always returns True.

        Raises:
            AssertionError: If the target joint value is not between 0 and 1.
        """
        target_joint = float(target_joint)
        target_joint = round(target_joint, 1)
        assert 0.0 <= target_joint <= 1.0, "Gripper control parameter must be between 0 and 1"

        gripper_pos = target_joint * 255
        try:
            self.gripper.goTo(int(gripper_pos), speed, force)
        except minimalmodbus.InvalidResponseError:
            print("minimalmodbus.InvalidResponseError retry")
