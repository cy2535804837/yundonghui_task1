"""Unified Robotiq gripper API for Modbus and ZMQ backends.

This module provides one class with a stable interface that can be used directly
from grasp pipelines (for example, sync_adaptive_grasp.py), regardless of
whether the underlying transport is:
1) USB/RS485 Modbus interface (robotiq_driver.py), or
2) ZMQ gripper server (robotiq_node_zmq.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


BackendType = Literal["modbus_rtu", "zmq"]


@dataclass
class RobotiqConfig:
    """Configuration for creating a gripper controller."""

    backend: BackendType = "zmq"
    # Modbus backend params
    serial_port: str = "auto"
    slave_id: int = 9
    # ZMQ backend params
    server_ip: str = "127.0.0.1"
    server_set_port: int = 4244
    server_get_port: int = 4245
    recv_timeout_ms: int = 200
    move_wait_timeout_s: float = 4.0
    target_tolerance_pct: float = 2.0
    # Activate on connect if not already active
    activate_on_connect: bool = False
    force_activate: bool = False


class _ZmqGripperClient:
    """Minimal client for robotiq_node_zmq.py."""

    def __init__(
        self,
        server_ip: str,
        server_set_port: int,
        server_get_port: int,
        recv_timeout_ms: int,
        move_wait_timeout_s: float,
        target_tolerance_pct: float,
    ):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._push = self._context.socket(zmq.PUSH)
        self._sub = self._context.socket(zmq.SUB)
        self._push.connect(f"tcp://{server_ip}:{server_set_port}")
        self._sub.connect(f"tcp://{server_ip}:{server_get_port}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._last_position_pct = 0
        self._move_wait_timeout_s = float(move_wait_timeout_s)
        self._target_tolerance_pct = float(target_tolerance_pct)

    @staticmethod
    def _clip_pct(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @staticmethod
    def _pct_to_joint(position_pct: float) -> float:
        return _ZmqGripperClient._clip_pct(position_pct) / 100.0

    @staticmethod
    def _pct_to_255(value_pct: float) -> int:
        return int(round(_ZmqGripperClient._clip_pct(value_pct) * 255.0 / 100.0))

    def _send_command(self, position_pct: float, speed_pct: float, force_pct: float) -> None:
        command = {
            "target_joint": self._pct_to_joint(position_pct),
            "speed": self._pct_to_255(speed_pct),
            "force": self._pct_to_255(force_pct),
        }
        self._push.send_json(command)

    def _read_one_state(self):
        import json

        try:
            parts = self._sub.recv_multipart()
            if not parts:
                return None
            data = json.loads(parts[0].decode("utf-8"))
            if "position" not in data:
                return None
            pos_pct = int(round(float(data["position"]) * 100.0))
            self._last_position_pct = max(0, min(100, pos_pct))
            return self._last_position_pct
        except self._zmq.Again:
            return None

    def move_percent(self, position_pct: float, speed_pct: float = 100, force_pct: float = 100):
        pos = int(round(self._clip_pct(position_pct)))
        self._send_command(position_pct, speed_pct, force_pct)
        return True, pos

    def warmup_subscription(self, timeout_s: float = 2.0) -> bool:
        """Wait until at least one PUB state message is received (ZMQ slow-join)."""
        import time

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self._read_one_state() is not None:
                return True
            time.sleep(0.05)
        return False

    def move_and_wait_percent(self, position_pct: float, speed_pct: float = 100, force_pct: float = 100):
        import time

        start_pos = int(self._last_position_pct)
        self._send_command(position_pct, speed_pct, force_pct)
        target = int(round(self._clip_pct(position_pct)))
        timeout = self._move_wait_timeout_s
        deadline = time.time() + timeout

        # The daemon only publishes position (no gOBJ "stopped on object" flag).
        # Closing on an object stops the fingers short of the target, so waiting
        # for position==target would burn the whole timeout every grasp. Instead
        # detect motion completion by a position PLATEAU: once the gripper has
        # started moving and then holds still for `stable_window`, it has either
        # reached the target or stalled on the object.
        stable_window = 0.35  # s of no motion => settled
        plateau_eps = 1  # percent; movement <= this counts as "not moving"
        min_travel = 5  # percent; must have moved at least this to trust a plateau
        last_pos = start_pos
        last_change_t = time.time()
        moved = False
        while time.time() < deadline:
            pos = self._read_one_state()
            now = time.time()
            if pos is None:
                continue
            if abs(pos - target) <= self._target_tolerance_pct:
                return pos, 3  # reached commanded position
            if abs(pos - last_pos) > plateau_eps:
                last_pos = pos
                last_change_t = now
                if abs(pos - start_pos) >= min_travel:
                    moved = True
            elif moved and (now - last_change_t) >= stable_window:
                # Started moving, now held still short of target -> stopped on object.
                return pos, 2

        final_pos = int(self._last_position_pct)
        if abs(final_pos - target) <= self._target_tolerance_pct:
            return final_pos, 3
        # Moved clearly toward target even if PUB feedback was sparse.
        if target >= start_pos + 15 and final_pos >= start_pos + 15:
            return final_pos, 2
        if target <= start_pos - 15 and final_pos <= start_pos - 15:
            return final_pos, 2
        return final_pos, 0  # Timed out / still moving

    def get_current_position_percent(self):
        pos = self._read_one_state()
        if pos is None:
            return self._last_position_pct
        return pos

    def stop(self):
        hold_pos = self.get_current_position_percent()
        self._send_command(hold_pos, 100, 50)
        return True

    def disconnect(self):
        self._push.close(0)
        self._sub.close(0)
        self._context.term()


class RobotiqController:
    """Unified controller exposing a stable grasping-friendly API.

    Main methods expected by grasp code:
    - move_percent(position_pct, speed_pct, force_pct)
    - move_and_wait_percent(position_pct, speed_pct, force_pct)
    - get_current_position_percent()
    - stop()
    """

    def __init__(self, config: RobotiqConfig):
        self.config = config
        self.backend: BackendType = config.backend
        self._gripper = None
        self._activation_checked = False
        self.connect()

    def connect(self) -> None:
        """Connect to the selected backend (no motion / activation here)."""
        if self.backend == "modbus_rtu":
            from robotiq_driver import RobotiqGripper as ModbusGripper

            self._gripper = ModbusGripper(
                portname=self.config.serial_port,
                slaveAddress=self.config.slave_id,
            )
            # Optional legacy path; prefer lazy ensure_activated() on first move.
            if self.config.activate_on_connect:
                self.ensure_activated()
            return

        if self.backend == "zmq":
            self._gripper = _ZmqGripperClient(
                server_ip=self.config.server_ip,
                server_set_port=self.config.server_set_port,
                server_get_port=self.config.server_get_port,
                recv_timeout_ms=self.config.recv_timeout_ms,
                move_wait_timeout_s=self.config.move_wait_timeout_s,
                target_tolerance_pct=self.config.target_tolerance_pct,
            )
            self._gripper.warmup_subscription(timeout_s=2.0)
            return

        raise ValueError(f"Unsupported backend: {self.backend}")

    def disconnect(self) -> None:
        """Disconnect the underlying backend."""
        if self._gripper is None:
            return

        if self.backend == "modbus_rtu":
            # Close serial port from minimalmodbus Instrument.
            try:
                self._gripper.serial.close()
            except Exception:
                pass
        elif self.backend == "zmq":
            self._gripper.disconnect()

        self._gripper = None

    def activate(self) -> None:
        """Activate gripper for the selected backend."""
        if self.backend == "modbus_rtu":
            self._gripper.resetActivate()
            self._activation_checked = True
        else:
            # ZMQ backend assumes server-side gripper is already activated.
            return

    def ensure_activated(self) -> None:
        """Activate only when needed; never re-activate an already-ready gripper."""
        if self.backend != "modbus_rtu":
            return
        if self._activation_checked and not self.config.force_activate:
            return
        g = self._gripper
        if self.config.force_activate:
            print("[robotiq_api] force_activate: running resetActivate()", flush=True)
            g.resetActivate()
            self._activation_checked = True
            return
        # Only skip activation when the gripper truly reports activation complete
        # (gSTA == 3). A Robotiq gripper answers READ requests even when it is NOT
        # activated, so "responds to reads" is NOT proof of readiness -- trusting
        # it leaves rACT unset and every goTo() then times out waiting for motion
        # that never starts (gOBJ stays 0).
        try:
            activated = bool(g.isActivated())
        except Exception:
            activated = False
        if activated:
            print(
                "[robotiq_api] gripper already activated (gSTA==3); skip resetActivate()",
                flush=True,
            )
            self._activation_checked = True
            return
        print(
            "[robotiq_api] gripper not activated (gSTA!=3); running resetActivate() once",
            flush=True,
        )
        g.resetActivate()
        self._activation_checked = True

    def stop(self) -> bool:
        """Immediately stop current movement.

        Returns:
            bool: True when stop command is sent.
        """
        result = self._gripper.stop()
        if isinstance(result, bool):
            return result
        return True

    def move_percent(self, position_pct: float, speed_pct: float = 100, force_pct: float = 100) -> Tuple[bool, int]:
        """Move gripper using 0-100 percent values."""
        self.ensure_activated()
        return self._gripper.move_percent(position_pct, speed_pct, force_pct)

    def move_and_wait_percent(
        self, position_pct: float, speed_pct: float = 100, force_pct: float = 100
    ) -> Tuple[int, int]:
        """Move and wait using 0-100 percent values.

        Returns:
            tuple[int, int]: (final_position_percent, motion_status_code)
        """
        self.ensure_activated()
        final_pos_pct, status = self._gripper.move_and_wait_percent(position_pct, speed_pct, force_pct)
        status_code = status.value if hasattr(status, "value") else int(status)
        return int(final_pos_pct), status_code

    def get_current_position_percent(self) -> int:
        """Read current gripper position in percentage [0, 100]."""
        return int(self._gripper.get_current_position_percent())

    def open(self, speed_pct: float = 100, force_pct: float = 50) -> Tuple[bool, int]:
        """Convenience open command (0%)."""
        return self.move_percent(0, speed_pct, force_pct)

    def close(self, speed_pct: float = 100, force_pct: float = 50) -> Tuple[bool, int]:
        """Convenience close command (100%)."""
        return self.move_percent(100, speed_pct, force_pct)


def create_gripper_controller(
    backend: BackendType = "zmq",
    *,
    serial_port: str = "auto",
    slave_id: int = 9,
    server_ip: str = "127.0.0.1",
    server_set_port: int = 4244,
    server_get_port: int = 4245,
    recv_timeout_ms: int = 200,
    move_wait_timeout_s: float = 4.0,
    target_tolerance_pct: float = 2.0,
    activate_on_connect: bool = False,
    force_activate: bool = False,
) -> RobotiqController:
    """Factory helper for quick initialization."""
    cfg = RobotiqConfig(
        backend=backend,
        serial_port=serial_port,
        slave_id=slave_id,
        server_ip=server_ip,
        server_set_port=server_set_port,
        server_get_port=server_get_port,
        recv_timeout_ms=recv_timeout_ms,
        move_wait_timeout_s=move_wait_timeout_s,
        target_tolerance_pct=target_tolerance_pct,
        activate_on_connect=activate_on_connect,
        force_activate=force_activate,
    )
    return RobotiqController(cfg)

