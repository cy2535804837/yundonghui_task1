# Robotiq Grippers — Setup & Usage

This document explains how the two Robotiq grippers on the **x86 machine**
(`192.168.41.1`) are wired, how to address each one by name, and how to test
them.

## Hardware overview

This machine has **two Robotiq grippers**, each connected through its own FTDI
USB/RS485 adapter. Both grippers answer Modbus RTU at **slave id 9**.

| Name (convention) | FTDI serial | by-id symlink | Was (volatile) |
|---|---|---|---|
| `left`  | `D30JLK40` | `usb-FTDI_FT231X_USB_UART_D30JLK40-if00-port0` | `/dev/ttyUSB1` |
| `right` | `DU0E2IPR` | `usb-FTDI_FT231X_USB_UART_DU0E2IPR-if00-port0` | `/dev/ttyUSB2` |

Other USB serial devices on the machine that are **not** grippers:
- `/dev/ttyUSB0` — QinHeng CH340 adapter (something else).
- Several Microchip **CANalyst-II** USB-CAN adapters.

> **The `left` / `right` labels are just a naming convention.** Until you have
> physically confirmed which arm each adapter belongs to, treat them as "gripper
> A / B". If a test moves the wrong physical gripper, swap the two `by_id`
> values in the `GRIPPERS` registry inside `robotiq_grippers.py`.

## Why address grippers by serial (by-id), not `/dev/ttyUSBx`

1. **`ttyUSBx` numbers are not stable** — they depend on enumeration order and
   can change across reboots or replugs.
2. **Both grippers share slave id 9**, so the driver's `--serial_port auto`
   mode is ambiguous: it scans every port and just keeps whichever gripper it
   saw *last*. You can't reliably pick a specific gripper that way.

The `/dev/serial/by-id/...` paths are derived from each FTDI adapter's unique
serial number, so they always point at the same physical gripper.

## One-time setup

### 1. Python packages

```bash
python3 -m pip install --user minimalmodbus pyserial
```

(Already installed on the x86 machine: `minimalmodbus 2.1.1`, `pyserial 3.5`.)

### 2. Serial port permissions (recommended permanent fix)

The tty device nodes are owned by `root:dialout`. Add your user to the
`dialout` group once, then log out and back in (or reboot):

```bash
sudo usermod -aG dialout $USER
```

Temporary alternative (resets on replug/reboot):

```bash
sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2
```

## Files

| File | Purpose |
|---|---|
| `robotiq_driver.py` | Low-level Modbus RTU driver (`minimalmodbus`-based). |
| `robotiq_api.py` | Unified controller API (`modbus_rtu` and `zmq` backends). |
| `robotiq_grippers.py` | **Named** access to the two grippers via by-id paths. |
| `test_both_grippers.py` | Test sequence runner for one or both grippers. |
| `test_robotiq_api.py` | Single-gripper test using the raw API. |

## Usage

### Discover configured grippers and whether they're plugged in

```bash
python3 robotiq_grippers.py
```

Example output:

```
Configured grippers:
  left   [FOUND] slave=9 -> /dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D30JLK40-if00-port0
         FTDI FT231X serial D30JLK40 (was /dev/ttyUSB1)
  right  [FOUND] slave=9 -> /dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0E2IPR-if00-port0
         FTDI FT231X serial DU0E2IPR (was /dev/ttyUSB2)
```

### Test both grippers

```bash
python3 test_both_grippers.py
```

This activates each gripper (a full open/close cycle) and then drives it through
the position sequence `0 → 50 → 100 → 0` percent.

Test a single gripper or customize the motion:

```bash
python3 test_both_grippers.py --grippers left
python3 test_both_grippers.py --positions 0 50 100 0 --speed_pct 60 --force_pct 40
```

### Use a gripper from your own code

```python
from robotiq_grippers import create_gripper

# 0% = fully open, 100% = fully closed
g = create_gripper("left", activate_on_connect=True)
try:
    pos = g.get_current_position_percent()
    g.move_and_wait_percent(50, speed_pct=60, force_pct=40)  # half closed
    g.move_percent(0)                                        # open (no wait)
finally:
    g.disconnect()
```

Key controller methods (from `robotiq_api.RobotiqController`):

- `move_percent(position_pct, speed_pct=100, force_pct=100)` → `(ok, requested)`
- `move_and_wait_percent(position_pct, speed_pct, force_pct)` → `(final_pct, status)`
- `get_current_position_percent()` → `int` in `[0, 100]`
- `open()` / `close()` — convenience for 0% / 100%
- `stop()` — halt current motion
- `disconnect()` — release the serial port

## Conventions

- **Position:** `0%` = fully **open**, `100%` = fully **closed**.
- **Speed / force:** `0–100%` (mapped internally to the gripper's `0–255`).

## Motion status codes (`gOBJ`)

| Code | Meaning |
|---|---|
| `3` | Reached the requested position (no object). |
| `2` | Stopped on an object while **closing**. |
| `1` | Stopped on an object while **opening**. |
| `0` | Still moving / timed out. |

## Activation note

Activating a gripper (`activate_on_connect=True`, or the first move) triggers a
**full open-and-close calibration cycle**. Make sure the fingers are clear of
obstacles and not holding an object when activating.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Permission denied: '/dev/ttyUSB*'` | User not in `dialout` group — see setup, or `sudo chmod 666`. |
| `No gripper detected` (auto mode) | Both grippers share slave 9; use named/by-id ports instead of `auto`. |
| `FileNotFoundError ... /dev/serial/by-id/...` | Adapter unplugged or serial changed — check `ls -l /dev/serial/by-id/`. |
| `gFLT=9` ("no communication") in a raw read | Normal before activation; clears once the gripper is activated. |
| Wrong physical gripper moves | Swap the `by_id` values for `left`/`right` in `robotiq_grippers.py`. |
