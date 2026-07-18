"""
adaptive_place_right (FT-only build)
====================================
Wrist force-torque-driven placement, **right arm** variant.

The arm descends until the RIGHT wrist FT sensor (waist-frame Z delta vs
the "loaded, stationary" baseline) detects table contact, then presses
until the table bears the object's weight::

    F_support(t) = max(Fz(t) - baseline, 0)
    gamma(t)     = clip(F_support(t) / G_obj, 0, 1)

When ``gamma`` exceeds ``gamma_release_threshold`` the object is released
via an injected gripper hook (:class:`gripper_hook.GripperHook`) and the
arm lifts.

This is the **tactile-free** variant: there is no fingertip Tac3D
feedback and no grip-decay PD loop (both were removed for the new
machine, which has only wrist force-torque sensors).  The gripper is an
opaque, injectable hook -- this package never talks to a gripper driver
directly.  Provide your own ``GripperHook`` (e.g. a Robotiq-only wrapper)
or use the bundled ``NoopGripper`` for dry-runs.
"""

from .config import RightAdaptivePlaceConfig, DEFAULT_RIGHT_ADAPTIVE_PLACE  # noqa: F401
from .adaptive_placer_right import RightAdaptivePlacer  # noqa: F401
from .gripper_hook import GripperHook, NoopGripper  # noqa: F401
