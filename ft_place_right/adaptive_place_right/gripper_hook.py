"""
adaptive_place_right/gripper_hook.py
====================================
Gripper abstraction for the **tactile-free** force-torque placement.

The original ``bottle_cup_pour_place`` / ``adaptive_place_right`` stack
coupled the gripper jaw motion to a Tac3D fingertip-tactile feedback loop
(``handover.right_gripper.RightGripperTactile`` -> ``tianyi_tactile_grasp``
-> ``PyTac3D``).  This standalone placement folder runs on a machine that
has **only wrist force-torque sensors** (no fingertip tactile), so all the
placement *decisions* are driven by the wrist FT signal and the gripper is
treated as an opaque, injectable hook.

You wire your own gripper by implementing :class:`GripperHook` (any object
with these methods works -- ``Protocol`` is structural) and passing it to
:class:`adaptive_place_right.adaptive_placer_right.RightAdaptivePlacer`.

Lifecycle as called by the placer
---------------------------------
1. ``close_to_hold()``  -- grip the object before descent (Phase A).  Only
   called when ``cfg.grasp_first=True``; skip-able if the object is already
   held.
2. ``open()``           -- fully open the jaws.  Used both for the F_empty
   weight sample (auto ``G_obj``) and as the final release before lift.
3. ``shutdown()``       -- release + drop any connection at teardown.

There is **no continuous grip-decay** in this build (the FT->cf decay PD
loop was removed because it was found to be unstable).  The object is held
closed through the descent and released in one ``open()`` once the wrist FT
confirms the table is bearing the load.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GripperHook(Protocol):
    """Structural interface the placer expects from a gripper.

    Implement these on your own Robotiq (or other) driver wrapper and pass
    an instance as ``gripper=`` to ``RightAdaptivePlacer`` / ``main``.
    All methods should be best-effort and must not raise on transient
    hardware blips (log and continue instead).
    """

    def close_to_hold(self) -> None:
        """Close the jaws onto the object and hold it (blocking until the
        grip is stable).  Called once before descent when grasp_first."""

    def open(self) -> None:
        """Fully open the jaws (release the object)."""

    def shutdown(self) -> None:
        """Release and tear down any connection (called at teardown)."""


class NoopGripper:
    """Default no-op gripper: logs every call, touches no hardware.

    Safe for dry-runs and bring-up of the FT/admittance placement motion
    before a real gripper is wired in.  Swap this out for your own
    :class:`GripperHook` implementation on the target machine.
    """

    def __init__(self, *, log_fn=None, name: str = "[GRIPPER-NOOP]") -> None:
        self._log = log_fn if log_fn is not None else (lambda m: print(f"{name} {m}"))
        self._name = name

    def close_to_hold(self) -> None:
        self._log("close_to_hold() -- (no-op stub; wire a real gripper here)")

    def open(self) -> None:
        self._log("open() -- (no-op stub; wire a real gripper here)")

    def shutdown(self) -> None:
        self._log("shutdown() -- (no-op stub; wire a real gripper here)")


__all__ = ["GripperHook", "NoopGripper"]
