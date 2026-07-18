"""ZeroMQ server implementation for Robotiq grippers.

This module provides a network-accessible interface to control Robotiq grippers via ZeroMQ protocol.
All received control commands will first be buffered in the deque, and then the server will retrieve the data
in another thread and send it to the hardware; at the same time, the server will request the gripper state
periodically for the remote client to read.

Changelog:
    - 2025-08-20 (Chris Ren): Initial version.
    - 2025-10-31 (Chris Ren): Optimized with PUB mode for real-time state updates.
"""

from __future__ import annotations

import argparse
from collections import deque
import threading
import time
import tqdm
import json

import minimalmodbus
import zmq

from robotiq_driver import RobotiqGripper


class RobotiqGripperUSBServer:
    """Robotiq gripper server implementation.

    This class provides a server interface to remotely control a Robotiq gripper using ZeroMQ.
    """

    def __init__(self, server_ip: str, server_set_port: str, server_get_port: str, hand_dev_ip: str, hand_slave_id: str, cmd_set_freq: int, state_read_freq: int) -> None:
        self.gripper = RobotiqGripper(hand_dev_ip, int(hand_slave_id))
        self.gripper.resetActivate()

        self.context = zmq.Context()
        # Create set socket for receiving commands (PULL).
        self.set_socket = self.context.socket(zmq.PULL)
        self.set_socket.bind(f"tcp://{server_ip}:{server_set_port}")

        # Create get socket for publishing gripper state (PUB).
        self.get_socket = self.context.socket(zmq.PUB)
        self.get_socket.bind(f"tcp://{server_ip}:{server_get_port}")
        self.get_socket.setsockopt(zmq.SNDHWM, 1)

        self.device_lock = threading.Lock()
        self.cmd_set_freq = cmd_set_freq
        self.state_read_freq = state_read_freq
        self._last_val = 0
        self.gripper_state = 0  # The latest gripper state.

        # State update thread.
        self.update_gripper_state_thread = threading.Thread(
            target=self._update_and_publish_state, daemon=True, name="update_gripper_state_thread")
        self.update_gripper_state_thread.start()

        # Command execution thread.
        self.set_gripper_cmd_buffer = deque(maxlen=10)
        self.execute_gripper_cmd_thread = threading.Thread(
            target=self._execute_gripper_cmd, daemon=True, name="execute_gripper_cmd_thread")
        self.execute_gripper_cmd_thread.start()

        for _ in tqdm.tqdm(range(1), desc="Warm Up ZMQ Server"):
            time.sleep(1)

    def _update_and_publish_state(self) -> None:
        while True:
            start_time = time.perf_counter()

            with self.device_lock:
                try:
                    self._last_val = gripper_pos = self.gripper.getPosition()
                except minimalmodbus.InvalidResponseError as error:
                    print(f"minimalmodbus.InvalidResponseError retry {error}")
                    gripper_pos = self._last_val

                assert 0.0 <= gripper_pos <= 255.0, "Gripper position must be between 0 and 255"
                self.gripper_state = gripper_pos / 255.0

            state_msg = {
                "position": self.gripper_state,
            }

            # self.get_socket.send_json(self.gripper_state)
            json_data = json.dumps(state_msg).encode('utf-8')
            self.get_socket.send_multipart([json_data])

            # Control the publish frequency.
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, 1.0 / self.state_read_freq - elapsed)
            time.sleep(sleep_time)

    def _execute_gripper_cmd(self) -> None:
        """Execute the gripper commands in a separate thread."""
        while True:
            if len(self.set_gripper_cmd_buffer) > 0:
                start_time = time.perf_counter()

                cmd = self.set_gripper_cmd_buffer.popleft()
                with self.device_lock:
                    try:
                        self.gripper.goTo(cmd[0], cmd[1], cmd[2])
                    except minimalmodbus.InvalidResponseError as error:
                        print(f"minimalmodbus.InvalidResponseError retry {error}")

                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, 1.0 / self.cmd_set_freq - elapsed)
                time.sleep(sleep_time)
            else:
                time.sleep(0.005)

    def _handle_set(self) -> None:
        """Handle the gripper set commands."""
        while True:
            try:
                command = self.set_socket.recv_json(zmq.NOBLOCK)
                target_joint = command["target_joint"]
                speed = command["speed"]
                force = command["force"]

                assert 0.0 <= target_joint <= 1.0, "Gripper control parameter must be between 0 and 1"
                gripper_pos = target_joint * 255.0
                self.set_gripper_cmd_buffer.append((int(gripper_pos), int(speed), int(force)))

            except zmq.Again:
                time.sleep(0.001)

    def start(self) -> None:
        """Start the gripper server."""
        # Handle the gripper set commands.
        set_thread = threading.Thread(target=self._handle_set)
        set_thread.daemon = True
        set_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            set_thread.join()
            self.update_gripper_state_thread.join()
            self.execute_gripper_cmd_thread.join()
            self.context.term()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--server_ip",
        type=str,
        default="127.0.0.1",
        help="The ip of the gripper server, e.g., 127.0.0.1"
    )
    arg_parser.add_argument(
        "--server_set_port",
        type=str,
        default="4242",
        help="The port of the gripper server for setting, e.g., 4242"
    )
    arg_parser.add_argument(
        "--server_get_port",
        type=str,
        default="4243",
        help="The port of the gripper server for getting, e.g., 4243"
    )
    arg_parser.add_argument(
        "--hand_dev_ip",
        type=str,
        default="/dev/ttyUSB0",
        help="The device ip of the gripper, e.g., /dev/ttyUSB0"
    )
    arg_parser.add_argument(
        "--hand_slave_id",
        type=str,
        default="1",
        help="The slave id of the gripper, e.g., 1"
    )
    arg_parser.add_argument(
        "--cmd_set_freq",
        type=int,
        default=60,
        help="The frequency of reading (updating) the gripper state, e.g., 60"
    )
    arg_parser.add_argument(
        "--state_read_freq",
        type=int,
        default=60,
        help="The frequency of reading (updating) the gripper state, e.g., 60"
    )
    args = arg_parser.parse_args()

    gripper_server = RobotiqGripperUSBServer(
        server_ip=args.server_ip,
        server_set_port=args.server_set_port,
        server_get_port=args.server_get_port,
        hand_dev_ip=args.hand_dev_ip,
        hand_slave_id=args.hand_slave_id,
        cmd_set_freq=args.cmd_set_freq,
        state_read_freq=args.state_read_freq
    )
    gripper_server.start()
