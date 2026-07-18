"""Arm joint-limit awareness for the grasp executor.

The tianyi2 arms have NO soft-limit layer in the URDF, so when a controller
drives a joint into its hard firmware limit the motor faults and the joint goes
"dead" until reset. The last three joints (elbow_yaw, wrist_pitch, wrist_roll)
are the most exposed because their ranges are tight and wrist_roll is even
asymmetric left vs right.

This module provides:

* ``ARM_JOINT_NAMES`` -- the 7 arm joints in controller order.
* ``TIANYI2_ARM_LIMITS`` -- hardcoded fallback limits (rad), taken from the
  production controller config (``tianyi2_controllers.yaml`` /
  ``joint_limits.yaml`` on this machine). These intentionally differ from the
  URDF, especially the wrist.
* ``JointLimitGuard`` -- fetches the authoritative limits live from the running
  QP controller ROS params (falling back to the hardcoded table) and offers
  cheap checks against the live joint state so callers can stop motion BEFORE a
  joint reaches its hard stop.

Nothing here commands the robot; it only reads joint state / params and reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# 7 arm joints, controller order (matches xarm_sdk.config.joints_name[:7]).
ARM_JOINT_NAMES: List[str] = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "elbow_yaw",
    "wrist_pitch",
    "wrist_roll",
]

# Production operational limits in radians (lower, upper), per arm.
# Source: /home/ubuntu/XARM/install/tianyi2_bringup/.../joint_limits.yaml and
# tianyi2_controllers.yaml. wrist_roll is asymmetric L vs R by design.
TIANYI2_ARM_LIMITS: Dict[str, List[Tuple[float, float]]] = {
    "left": [
        (-2.96, 2.96),    # shoulder_pitch
        (-0.262, 2.616),  # shoulder_roll
        (-2.96, 2.96),    # shoulder_yaw
        (-2.61, 0.261),   # elbow_pitch
        (-2.96, 2.96),    # elbow_yaw
        (-0.785, 1.05),   # wrist_pitch
        (-1.65, 1.3),     # wrist_roll  (asymmetric)
    ],
    "right": [
        (-2.96, 2.96),    # shoulder_pitch
        (-2.616, 0.262),  # shoulder_roll
        (-2.96, 2.96),    # shoulder_yaw
        (-2.61, 0.261),   # elbow_pitch
        (-2.96, 2.96),    # elbow_yaw
        (-0.785, 1.05),   # wrist_pitch
        (-1.3, 1.65),     # wrist_roll  (asymmetric)
    ],
}

# Controller node names that expose joint_lower_limits / joint_upper_limits.
_QP_CTRL = {
    "left": "endpose_single_arm_qp_L_controller",
    "right": "endpose_single_arm_qp_R_controller",
}


def _log(msg: str) -> None:
    print(f"[JOINT-LIMIT] {msg}", flush=True)


def fetch_arm_limits(xarm: Any, arm: str) -> List[Tuple[float, float]]:
    """Return per-joint (lower, upper) for ``arm``.

    Prefers the live controller ROS params (authoritative for what the running
    controllers actually enforce); falls back to the hardcoded tianyi2 table on
    any failure or shape mismatch.
    """
    fallback = list(TIANYI2_ARM_LIMITS["left" if arm == "left" else "right"])
    ctrl = _QP_CTRL.get(arm)
    if ctrl is None:
        return fallback
    try:
        lo = xarm.get_node_parameter(ctrl, "joint_lower_limits", timeout_sec=2.0)
        hi = xarm.get_node_parameter(ctrl, "joint_upper_limits", timeout_sec=2.0)
    except Exception as e:  # noqa: BLE001
        _log(f"{arm}: live limit fetch failed ({e!r}); using hardcoded fallback")
        return fallback
    if (
        isinstance(lo, (list, tuple))
        and isinstance(hi, (list, tuple))
        and len(lo) >= 7
        and len(hi) >= 7
    ):
        live = [(float(lo[i]), float(hi[i])) for i in range(7)]
        _log(f"{arm}: using live controller limits from {ctrl}")
        return live
    _log(f"{arm}: live limits unavailable/!=7 dof; using hardcoded fallback")
    return fallback


class JointLimitGuard:
    """Checks live arm joint angles against per-joint limits with a margin.

    Recovery-aware abort (``allow_recovery``): a joint inside the margin only
    forces an abort if it is *moving toward* its limit (or stuck against it).
    If the commanded motion is pulling the joint *away* from the limit, the
    motion is allowed to continue. This preserves the original protection (a
    joint driving inward into the margin still aborts immediately) while letting
    a reduced-alignment retry escape a joint that is parked at the margin from a
    previously aborted phase -- otherwise the retry would re-abort on its very
    first live check (arm still parked at the limit) and never move.
    """

    def __init__(
        self,
        xarm: Any,
        *,
        margin_rad: float = 0.10,
        enabled: bool = True,
        allow_recovery: bool = True,
        recovery_grace_iters: int = 12,
        recovery_eps_rad: float = 0.002,
    ) -> None:
        self.xarm = xarm
        self.margin = max(0.0, float(margin_rad))
        self.enabled = bool(enabled)
        self.allow_recovery = bool(allow_recovery)
        self.recovery_grace_iters = max(0, int(recovery_grace_iters))
        self.recovery_eps = max(0.0, float(recovery_eps_rad))
        self._limits: Dict[str, List[Tuple[float, float]]] = {}
        # Optional per-joint margin overrides (rad). A joint listed here uses its
        # own margin instead of the global one -- e.g. the elbow-high grasp lets
        # wrist_pitch travel closer to its stop (it physically has the room) while
        # every other joint keeps the full protective margin.
        self.margin_overrides: Dict[str, float] = {}
        # Per-arm, per-joint recovery state for the recovery-aware abort logic.
        self._dist_hist: Dict[str, Dict[str, float]] = {}
        self._grace: Dict[str, Dict[str, int]] = {}
        # Set by check_live() when an ABORT is decided, for the caller to act on.
        self.last_event: Optional[Dict[str, Any]] = None

    def reset_recovery_state(self, arm: str) -> None:
        """Clear the recovery trend history/grace for ``arm``.

        Call this at the start of each streamed motion so the trend starts
        fresh: a joint that is already inside the margin at phase start (e.g. the
        arm parked at the limit after a previous abort) gets its grace window to
        let the new, less-aggressive command pull it back out.
        """
        key = "left" if arm == "left" else "right"
        self._dist_hist.pop(key, None)
        self._grace.pop(key, None)

    # ----- limits -------------------------------------------------------
    def limits_for(self, arm: str) -> List[Tuple[float, float]]:
        key = "left" if arm == "left" else "right"
        if key not in self._limits:
            self._limits[key] = fetch_arm_limits(self.xarm, key)
        return self._limits[key]

    # ----- joint readback ----------------------------------------------
    def _read_joints(self, arm: str) -> Optional[List[float]]:
        try:
            if arm == "left":
                joints = self.xarm.xarm_left_arm_joint_angles()
            else:
                joints = self.xarm.xarm_right_arm_joint_angles()
        except Exception:  # noqa: BLE001
            return None
        if not joints or len(joints) < 7 or any(j is None for j in joints[:7]):
            return None
        return [float(j) for j in joints[:7]]

    # ----- evaluation ---------------------------------------------------
    def evaluate(self, arm: str, joints: List[float]) -> Dict[str, Any]:
        """Per-joint distance to the nearest limit; flag joints within margin."""
        limits = self.limits_for(arm)
        per_joint: List[Dict[str, Any]] = []
        breached: List[Dict[str, Any]] = []
        closest: Optional[Dict[str, Any]] = None
        for i, name in enumerate(ARM_JOINT_NAMES):
            lo, hi = limits[i]
            val = float(joints[i])
            d_lo = val - lo  # >0 inside
            d_hi = hi - val  # >0 inside
            dist = min(d_lo, d_hi)
            side = "lower" if d_lo <= d_hi else "upper"
            info = {
                "joint": name,
                "value": round(val, 4),
                "lower": lo,
                "upper": hi,
                "dist_to_limit": round(dist, 4),
                "nearest_side": side,
            }
            per_joint.append(info)
            if closest is None or dist < closest["dist_to_limit"]:
                closest = info
            joint_margin = self.margin_overrides.get(name, self.margin)
            if dist <= joint_margin:
                breached.append(info)
        return {
            "arm": arm,
            "margin_rad": self.margin,
            "ok": len(breached) == 0,
            "closest": closest,
            "breached": breached,
            "per_joint": per_joint,
        }

    def _decide_abort(self, arm: str, ev: Dict[str, Any]) -> bool:
        """Recovery-aware abort decision for a streamed motion.

        Returns True when the motion should stop to protect a joint. With
        ``allow_recovery`` off this is simply "any joint within margin" (the
        original behaviour). With it on, a joint inside the margin aborts only if
        it is moving toward its limit or stuck; a joint moving away (being pulled
        back out by the new command) is allowed to continue.
        """
        if not self.allow_recovery:
            return not ev["ok"]

        key = "left" if arm == "left" else "right"
        hist = self._dist_hist.setdefault(key, {})
        grace = self._grace.setdefault(key, {})
        breached_names = {b["joint"] for b in ev["breached"]}

        abort = False
        for info in ev["per_joint"]:
            name = info["joint"]
            cur = float(info["dist_to_limit"])
            prev = hist.get(name)
            hist[name] = cur  # update trend history for every joint

            if name not in breached_names:
                grace[name] = 0  # outside margin -> reset grace
                continue

            if prev is None:
                # First read of this joint inside the margin (e.g. parked at the
                # limit at phase start): grant the grace window so the new
                # command can begin pulling it out before we judge the trend.
                grace[name] = self.recovery_grace_iters
                continue

            delta = cur - prev  # >0 means moving AWAY from the limit
            if delta < -self.recovery_eps:
                # Moving toward the limit -> stop now (original protection).
                abort = True
            elif delta > self.recovery_eps:
                # Moving away (recovering) -> allow, refresh grace.
                grace[name] = self.recovery_grace_iters
            else:
                # Stuck against the margin -> allow only while grace remains.
                remaining = grace.get(name, 0)
                if remaining > 0:
                    grace[name] = remaining - 1
                else:
                    abort = True
        return abort

    def check_live(self, arm: str) -> Optional[Dict[str, Any]]:
        """Read current joints and evaluate. Returns None if no joint data."""
        joints = self._read_joints(arm)
        if joints is None:
            return None
        ev = self.evaluate(arm, joints)
        ev["should_abort"] = self._decide_abort(arm, ev)
        if ev["should_abort"]:
            self.last_event = ev
        return ev

    def report(self, arm: str) -> Optional[Dict[str, Any]]:
        """Compact diagnostic snapshot (closest joint + any breaches)."""
        joints = self._read_joints(arm)
        if joints is None:
            return None
        ev = self.evaluate(arm, joints)
        c = ev["closest"] or {}
        _log(
            f"{arm} closest-to-limit: {c.get('joint')}={c.get('value')} "
            f"[{c.get('lower')},{c.get('upper')}] dist={c.get('dist_to_limit')}rad"
            + (
                f"  BREACHED(<= {self.margin}): "
                + ", ".join(b["joint"] for b in ev["breached"])
                if ev["breached"]
                else ""
            )
        )
        return ev
