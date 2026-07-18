"""Client implementation for the Robotiq 2F-85 gripper using ZeroMQ communication protocol.

This module provides a client interface to control a Robotiq 2F-85 gripper over a network
connection using the ZeroMQ protocol for remote procedure calls.

Changelog:
    - 2025-09-16 (Chris Ren): Refactor the code with config.
    - 2025-10-31 (Chris Ren): Optimized with SUB mode for real-time state updates.
"""

from __future__ import annotations

import numpy as np
import zmq
import time
import threading
from typing import List, Optional, Dict, Union
import json

from xrocs.common.data_type import Joints
from xrocs.entity.hand.hand_base import HandDriver
from xrocs.utils.logger.logger_loader import logger


class RobotiqGripperClient(HandDriver):
    """Robotiq gripper client implementation."""

    def __init__(self, hand_config: Dict) -> None:
        """Initialize the Robotiq gripper client.

        Args:
            hand_config: The hand configuration. For example:
            {
                "enable": True,  # Reserved parameter for robot station initialization.
                "type": "Robotiq2f85Zmq",  # Reserved parameter for robot station initialization.
                "hand_ip": "127.0.0.1",  # The IP address of the gripper server.
                "hand_set_port": "4241",  # The port number of the gripper server for setting.
                "hand_get_port": "4242"  # The port number of the gripper server for getting (state query).
            }. The hand_set_port and hand_get_port must be different.
        """
        self.cfg_dict = hand_config
        self._hand_ip = self.cfg_dict.get("hand_ip", "127.0.0.1")
        self._hand_set_port = self.cfg_dict.get("hand_set_port")
        self._hand_get_port = self.cfg_dict.get("hand_get_port")

        self.context = zmq.Context()

        # Send commands to the gripper server (no response required).
        self.set_socket = self.context.socket(zmq.PUSH)
        self.set_socket.connect(f"tcp://{self._hand_ip}:{self._hand_set_port}")

        # Subscribe to gripper state updates (SUB mode).
        self.get_socket = self.context.socket(zmq.SUB)
        self.get_socket.connect(f"tcp://{self._hand_ip}:{self._hand_get_port}")
        self.get_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self.get_socket.setsockopt(zmq.RCVHWM, 1)  # Only keep the latest message

        # Cache for the latest gripper state.
        self._latest_position = None
        self._state_lock = threading.Lock()
        self._last_update_time = 0

        # Background thread to continuously receive state updates.
        self._running = True
        self._state_thread = threading.Thread(
            target=self._update_state_loop, 
            daemon=True,
            name="gripper_state_subscriber"
        )
        self._state_thread.start()

    def _update_state_loop(self) -> None:
        """Background thread that continuously receives the latest gripper state."""
        while self._running:
            try:
                # Receive state message from server (multipart message with JSON data)
                msg = self.get_socket.recv_multipart(flags=zmq.NOBLOCK)

                # Parse the JSON message
                if msg:
                    json_data = msg[0].decode('utf-8')
                    state_dict = json.loads(json_data)
                    
                    with self._state_lock:
                        self._latest_position = state_dict.get("position")
            except zmq.Again:
                # No message available, sleep briefly
                time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error receiving gripper state: {e}")
                time.sleep(0.01)

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
        return True

    def sync_target_joint(self, target_joint: Union[int, float, List], speed: int = 255, force: int = 255) -> bool:
        """Synchronously set the target joint position of the gripper.

        Note that the target is sent to the gripper server using ZeroMQ PUSH socket.

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
        if isinstance(target_joint, List):
            target_joint = target_joint[0]

        target_joint = float(target_joint)
        target_joint = round(target_joint, 1)
        assert 0.0 <= target_joint <= 1.0, "Gripper control parameter must be between 0 and 1"

        self.set_socket.send_json({"target_joint": target_joint, "speed": speed, "force": force})
        return True

    def open(self, timeout: float = 0., pub_interval: float = 0.01) -> bool:
        """Open the gripper by setting the joint position to 0.

        Args:
            timeout: The timeout in seconds.
            pub_interval: The publish interval in seconds.

        Returns:
            True if the command was sent successfully, False otherwise.
        """
        return self.set_target_joint(target_joint=0, timeout=timeout, pub_interval=pub_interval)

    def close(self, timeout: float = 0., pub_interval: float = 0.01) -> bool:
        """Close the gripper by setting the joint position to 1.

        Args:
            timeout: The timeout in seconds.
            pub_interval: The publish interval in seconds.

        Returns:
            True if the command was sent successfully, False otherwise.
        """
        return self.set_target_joint(target_joint=1, timeout=timeout, pub_interval=pub_interval)

    def set_target_joint(self, target_joint: Union[Joints, int, float], timeout: float = 0., pub_interval: float = 0.01) -> bool:
        """Set the target joint position of the gripper.

        Args:
            target_joint: The target joint position as a Joints object.
            timeout: The timeout in seconds.
            pub_interval: The publish interval in seconds.

        Returns:
            True if the command was sent successfully, False otherwise.
        """
        if isinstance(target_joint, (int, float)):
            target = target_joint
        else:
            target = target_joint.get_radian_ndarray()[0]

        if timeout > 0:
            start_time = time.monotonic()
            while time.monotonic() - start_time < timeout:
                self.sync_target_joint(target)
                time.sleep(pub_interval)
        else:
            self.sync_target_joint(target)  # Publish the command once.

        return True

    def get_current_joint(self) -> Optional[Joints]:
        """Get the current joint position of the gripper.

        Returns:
            The current joint position as a Joints or None if not available.
        """
        try:
            with self._state_lock:
                if self._latest_position is not None:
                    position = self._latest_position
                    return Joints(np.array([position]), num_of_dofs=1)
                else:
                    logger.warning("No gripper state received yet.")
                    return None
        except Exception as e:
            logger.error(f"Get current joint error: {e}.")
            return None
