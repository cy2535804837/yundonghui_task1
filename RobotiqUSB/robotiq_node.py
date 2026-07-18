"""ZeroRPC server implementation for Robotiq grippers.

This module provides a network-accessible interface to control Robotiq grippers via ZeroRPC protocol.
All received control commands will first be buffered in the deque, and then the server will retrieve the data
in another thread and send it to the hardware; at the same time, the server will request the gripper state
periodically for the remote client to read.
"""

import argparse
from collections import deque
import threading
import time

import minimalmodbus
import zerorpc

from xrocs.entity.hand.RobotiqUSB.robotiq_driver import RobotiqGripper


class RobotiqGripperUSBServer(zerorpc.Server):
    """Robotiq gripper server implementation.

    This class provides a server interface to remotely control a Robotiq gripper using ZeroRPC.
    """

    def __init__(self, robot_port: str, slave_id: str, state_read_freq: int) -> None:
        super().__init__()
        self.gripper = RobotiqGripper(robot_port, int(slave_id))
        self.gripper.resetActivate()

        self.device_lock = threading.Lock()

        self.state_read_freq = state_read_freq
        self._last_val = 0
        self.gripper_state = 0  # The latest gripper state
        self.update_gripper_state_thread = threading.Thread(
            target=self._update_gripper_state, daemon=True, name="update_gripper_state_thread"
        )
        self.update_gripper_state_thread.start()

        self.set_gripper_cmd_buffer = deque(maxlen=10)
        self.execute_gripper_cmd_thread = threading.Thread(
            target=self._execute_gripper_cmd, daemon=True, name="execute_gripper_cmd_thread"
        )
        self.execute_gripper_cmd_thread.start()

    def _update_gripper_state(self) -> None:
        """Update the gripper state in a separate thread at a fixed frequency."""
        while True:
            time.sleep(1.0 / self.state_read_freq)

            with self.device_lock:
                try:
                    self._last_val = gripper_pos = self.gripper.getPosition()
                except minimalmodbus.InvalidResponseError as error:
                    print(f"minimalmodbus.InvalidResponseError retry {error}")
                    gripper_pos = self._last_val

                assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
                self.gripper_state = gripper_pos / 255

    def get_gripper(self) -> float:
        """Get the current normalized position of the gripper.

        Returns:
            Current position of the gripper normalized between between 0 (open) and 1 (closed).
        """
        return self.gripper_state

    def _execute_gripper_cmd(self) -> None:
        """Execute the gripper commands in a separate thread."""
        while True:
            if len(self.set_gripper_cmd_buffer) > 0:
                cmd = self.set_gripper_cmd_buffer.popleft()

                with self.device_lock:
                    try:
                        self.gripper.goTo(cmd[0], cmd[1], cmd[2])
                    except minimalmodbus.InvalidResponseError as error:
                        print(f"minimalmodbus.InvalidResponseError retry {error}")
            else:
                time.sleep(0.005)

    def set_gripper(self, target_joint: float, speed: int = 255, force: int = 255) -> None:
        """Set the gripper position with specified speed and force.

        Args:
            target_joint: Target position normalized between 0 (open) and 1 (closed).
            speed: Speed of movement, from 0 to 255, with 255 being maximum speed.
            force: Force to apply, from 0 to 255, with 255 being maximum force.
        """
        target_joint = float(target_joint)
        assert 0.0 <= target_joint <= 1.0, "Gripper control parameter must be between 0 and 1"
        gripper_pos = target_joint * 255
        self.set_gripper_cmd_buffer.append((int(gripper_pos), speed, force))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--server_ip",
        type=str,
        default="127.0.0.1",
        help="The ip of the server, e.g., 127.0.0.1",
    )
    arg_parser.add_argument(
        "--hand_ip",
        type=str,
        default="/dev/ttyUSB0",
        help="The ip of the gripper, e.g., /dev/ttyUSB0",
    )
    arg_parser.add_argument(
        "--hand_port",
        type=str,
        default="4242",
        help="The port of the gripper, e.g., 4242",
    )
    arg_parser.add_argument(
        "--hand_slave",
        type=str,
        default="1",
        help="The slave num. of the gripper, e.g., 1",
    )
    arg_parser.add_argument(
        "--state_read_freq",
        type=int,
        default=30,
        help="The frequency of reading (updating) the gripper state, e.g., 30",
    )
    args = arg_parser.parse_args()
    server = zerorpc.Server(RobotiqGripperUSBServer(args.hand_ip, args.hand_slave, args.state_read_freq))
    server.bind(f"tcp://{args.server_ip}:{args.hand_port}")  # Bind to a specific server and hand port
    print(f"ZeroRPC robotiq server running on {args.server_ip}:{args.hand_port}...")
    server.run()
