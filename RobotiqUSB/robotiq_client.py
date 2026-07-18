"""Client implementation for the Robotiq gripper using ZeroRPC communication protocol.

This module provides a client interface to control a Robotiq gripper over a network
connection using the ZeroRPC protocol for remote procedure calls.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import zerorpc

from xrocs.common.data_type import Joints
from xrocs.entity.hand.hand_base import HandDriver
from typing import Optional

class RobotiqGripperClient(HandDriver):
    """Robotiq gripper client implementation.

    This class provides a client interface to remotely control a Robotiq gripper using ZeroRPC.
    """

    def __init__(self, hand_ip: str, hand_port: str, timeout: float = 5.0) -> None:
        """Initialize the Robotiq gripper client.

        Args:
            hand_ip: The IP address of the gripper server.
            hand_port: The port number of the gripper server.
            timeout: The timeout for the ZeroRPC client.
        """
        self.driver = zerorpc.Client()
        self._hand_ip = hand_ip
        self._hand_port = hand_port

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom for the gripper.

        Returns:
            The number of degrees of freedom (always 1 for Robotiq grippers).
        """
        return 1

    def connect(self) -> bool:
        """Connect to the Robotiq gripper server.

        Returns:
            True if connection was successful. Note that this function always returns True.
        """
        self.driver.connect(f"tcp://{self._hand_ip}:{self._hand_port}")
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
            The current joint position as a Joints object or None if not available.
        """
        position = self.driver.get_gripper()
        return Joints(np.array([position]), num_of_dofs=self.num_dofs())

    def set_target_joint(self, target_joint: Joints) -> bool:
        """Set the target joint position of the gripper.

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

        self.driver.set_gripper(target_joint, speed, force)
        return True
