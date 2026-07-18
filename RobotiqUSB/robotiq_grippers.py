"""Named access to multiple Robotiq grippers on one machine.

This machine has two Robotiq grippers, each on its own FTDI USB/RS485 adapter
and both answering Modbus RTU at slave id 9. Because they share a slave id,
`--serial_port auto` is ambiguous (it just keeps whichever port it scans last),
and the kernel's /dev/ttyUSBx numbers can swap on reboot or replug.

To get a stable, unambiguous handle we address each gripper by the FTDI
adapter's serial number via /dev/serial/by-id/, and give it a friendly name.

Usage:
    from robotiq_grippers import create_gripper, list_grippers

    g = create_gripper("left", activate_on_connect=True)
    g.move_and_wait_percent(50)
    g.disconnect()

Re-mapping names: edit the GRIPPERS registry below. The left/right labels are a
convention only -- verify which physical gripper moves and swap the names if the
mapping is reversed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from robotiq_api import RobotiqController, create_gripper_controller

# Directory holding stable, serial-number-based symlinks to the tty devices.
BY_ID_DIR = "/dev/serial/by-id"


@dataclass(frozen=True)
class GripperInfo:
    """Static description of one physical gripper."""

    name: str
    # FTDI adapter serial as it appears under /dev/serial/by-id/.
    by_id: str
    slave_id: int = 9
    description: str = ""

    @property
    def port_path(self) -> str:
        """Absolute /dev/serial/by-id path for this gripper."""
        return os.path.join(BY_ID_DIR, self.by_id)


# ---------------------------------------------------------------------------
# Gripper registry. Edit names/serials here if hardware changes.
# Serial numbers come from `ls -l /dev/serial/by-id/` on this machine.
# ---------------------------------------------------------------------------
GRIPPERS: Dict[str, GripperInfo] = {
    "left": GripperInfo(
        name="left",
        by_id="usb-FTDI_FT231X_USB_UART_DU0E2IPR-if00-port0",
        slave_id=9,
        description="FTDI FT231X serial DU0E2IPR (physical LEFT gripper)",
    ),
    "right": GripperInfo(
        name="right",
        by_id="usb-FTDI_FT231X_USB_UART_D30JLK40-if00-port0",
        slave_id=9,
        description="FTDI FT231X serial D30JLK40 (physical RIGHT gripper)",
    ),
}


def list_grippers() -> List[str]:
    """Return the configured gripper names."""
    return list(GRIPPERS.keys())


def get_gripper_info(name: str) -> GripperInfo:
    """Look up a gripper's static info by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in GRIPPERS:
        available = ", ".join(GRIPPERS) or "(none)"
        raise KeyError(f"Unknown gripper '{name}'. Available: {available}")
    return GRIPPERS[key]


def resolve_port(name: str) -> str:
    """Resolve a gripper name to its serial port, validating the device exists."""
    info = get_gripper_info(name)
    if not os.path.exists(info.port_path):
        raise FileNotFoundError(
            f"Gripper '{name}' expected at {info.port_path} but that path does not "
            f"exist. Is the adapter plugged in? Check `ls -l {BY_ID_DIR}`."
        )
    return info.port_path


def create_gripper(
    name: str,
    *,
    activate_on_connect: bool = False,
    force_activate: bool = False,
    move_wait_timeout_s: float = 4.0,
    target_tolerance_pct: float = 2.0,
) -> RobotiqController:
    """Create a connected RobotiqController for the named gripper.

    Args:
        name: Friendly gripper name (see list_grippers()).
        activate_on_connect: Run the activation routine on connect (the gripper
            fully opens and closes; keep it clear of objects).
        force_activate: Force a reset+activate even if already activated.
        move_wait_timeout_s: Timeout for move_and_wait_percent().
        target_tolerance_pct: Position tolerance for "reached target".

    Returns:
        A connected RobotiqController bound to that gripper's serial port.
    """
    info = get_gripper_info(name)
    port = resolve_port(name)
    return create_gripper_controller(
        backend="modbus_rtu",
        serial_port=port,
        slave_id=info.slave_id,
        activate_on_connect=activate_on_connect,
        force_activate=force_activate,
        move_wait_timeout_s=move_wait_timeout_s,
        target_tolerance_pct=target_tolerance_pct,
    )


if __name__ == "__main__":
    # Quick discovery: print configured grippers and whether each port is present.
    print("Configured grippers:")
    for gname in list_grippers():
        ginfo = get_gripper_info(gname)
        present = "FOUND" if os.path.exists(ginfo.port_path) else "MISSING"
        print(f"  {gname:6s} [{present}] slave={ginfo.slave_id} -> {ginfo.port_path}")
        if ginfo.description:
            print(f"         {ginfo.description}")
