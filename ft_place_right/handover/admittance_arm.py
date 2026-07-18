"""
handover/admittance_arm.py
==========================
Threaded, runtime-tunable admittance runner for ONE arm (left or right).

This is a class-based re-packaging of the math in:
    ../admittance_control/AdmittanceController_v4_2_fixed.py
so that the handover pipeline can:

  * run LEFT and RIGHT admittance simultaneously (each in its own thread),
    sharing a single rclpy node / executor;
  * switch gains live (e.g. "receiver" → "holding" after the object is
    transferred);
  * move the equilibrium point live (e.g. retract the right arm after
    releasing the object).

Threading model
---------------
* One `XARM_manager` (ROS node) is owned by the pipeline.
* The pipeline spins the node in a `MultiThreadedExecutor` on a daemon
  thread so subscribers/TF listeners keep running.
* Each `AdmittanceArm.start()` call launches a worker thread that runs
  the admittance loop; the loop only READS shared state (force, TF) and
  PUBLISHES commands, so no additional spin is needed inside the loop.
* Gain / target updates are lock-protected.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose, WrenchStamped

# Make the sibling admittance_control package importable so we can reuse the
# well-tested math helpers.
_ADMITTANCE_CONTROL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "admittance_control")
)
if _ADMITTANCE_CONTROL_DIR not in sys.path:
    sys.path.insert(0, _ADMITTANCE_CONTROL_DIR)

from AdmittanceController_v3 import (  # noqa: E402
    TFHelper,
    ForceFilter,
    PayloadCompensator,
    ForceOffsetCalibrator,
    transform_wrench,
    smooth_deadzone,
    quat_from_transform,
    quat_to_axis_angle,
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    omega_to_delta_quat,
)
from AdmittanceController_v4_2_fixed import (  # noqa: E402
    TranslationAdmittanceFixed,
    RotationAdmittanceFixed,
)

from .config import (
    AdmittanceGains,
    ForceProcessing,
    DEFAULT_FORCE_PROC,
    DEFAULT_LEFT_FT_CALIB,
    DEFAULT_RIGHT_FT_CALIB,
)


def _load_ft_calibration(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


class _SideForceSubscriber:
    """Subscribe to /arm_6dof_left or /arm_6dof_right on a shared node."""

    def __init__(self, node, topic: str):
        self.force = np.zeros(6)
        node.create_subscription(WrenchStamped, topic, self._cb, 10)

    def _cb(self, msg):
        self.force[0] = msg.wrench.force.x
        self.force[1] = msg.wrench.force.y
        self.force[2] = msg.wrench.force.z
        self.force[3] = msg.wrench.torque.x
        self.force[4] = msg.wrench.torque.y
        self.force[5] = msg.wrench.torque.z


class AdmittanceArm:
    """One-arm admittance controller designed to run alongside others.

    Parameters
    ----------
    side : {"left", "right"}
    xarm_manager : XARM_manager
        Shared ROS node.
    topic_pub : TopicPublisher
        Shared publisher helper (exposes publish_endposetarget_L/R).
    tf_helper : TFHelper
        Shared TF listener.
    tcp_frame : str
        e.g. "left_tcp_link".
    waist_frame : str
        e.g. "waist_yaw_link".
    force_topic : str
        e.g. "/arm_6dof_left".
    qp_controller : str
        QP controller name to deactivate on stop, e.g.
        "endpose_single_arm_qp_L_controller".
    calib_path : str | None
        FT-calibration JSON path; if None, use the default for the side.
    initial_gains : AdmittanceGains
        Gains to start with (can be changed later with `set_gains`).
    force_proc : ForceProcessing
        Dead-zones, filter alpha, calibrator sample count.
    loop_period : float
        Desired loop period (seconds).  The loop sleeps the remainder.
    name : str
        Human-readable label used in logs.
    log_periodic_telemetry : bool
        If True (default), the admittance loop logs F/disp/hz every 200
        iterations at INFO. Set False for quiet sessions (e.g. pose recorders).
    """

    def __init__(
        self,
        *,
        side: str,
        xarm_manager,
        topic_pub,
        tf_helper: TFHelper,
        tcp_frame: str,
        waist_frame: str,
        force_topic: str,
        qp_controller: str,
        calib_path: Optional[str] = None,
        initial_gains: Optional[AdmittanceGains] = None,
        force_proc: Optional[ForceProcessing] = None,
        loop_period: float = 0.004,
        name: Optional[str] = None,
        log_periodic_telemetry: bool = True,
        hold_orientation: bool = False,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        self.side = side
        self.name = name or f"[{side.upper()}]"
        # When True, the rotation admittance is bypassed and the wrist
        # orientation is held rigidly at the captured reference quaternion.
        # Useful for drag-to-teach point recording where only the fingertip
        # position matters and a wandering wrist feels like "fighting".
        self.hold_orientation = bool(hold_orientation)
        self.node = xarm_manager
        self.topic_pub = topic_pub
        self.tf_helper = tf_helper
        self.tcp_frame = tcp_frame
        self.waist_frame = waist_frame
        self.qp_controller = qp_controller
        self.loop_period = loop_period

        if calib_path is None:
            calib_path = DEFAULT_LEFT_FT_CALIB if side == "left" else DEFAULT_RIGHT_FT_CALIB
        self.calib_path = calib_path

        self.force_proc = force_proc or DEFAULT_FORCE_PROC

        gains = initial_gains or AdmittanceGains()
        self._gains_lock = threading.Lock()
        self._gains = gains

        # Admittance math objects — instantiated with initial gains and kept
        # alive across gain swaps (to preserve velocity state).
        self.adm_trans = TranslationAdmittanceFixed(
            M=gains.M, B=gains.B, K=gains.K, max_vel=gains.max_vel
        )
        self.adm_rot = RotationAdmittanceFixed(
            M_rot=gains.M_rot, B_rot=gains.B_rot, K_rot=gains.K_rot,
            max_omega=gains.max_omega, rot_lead_time=gains.rot_lead_time,
        )

        # Force pipeline
        self.force_sub = _SideForceSubscriber(xarm_manager, force_topic)
        self.force_filter = ForceFilter(alpha=self.force_proc.filter_alpha)

        calib = _load_ft_calibration(self.calib_path)
        if calib is not None:
            self.payload = PayloadCompensator(mass=calib["mass"], com=calib["com"])
            self.ft_bias = np.array(calib["F_bias"] + calib["T_bias"])
            self.R_sensor_tcp = np.array(calib["R_sensor_tcp"])
            self.use_file_calib = True
            self.node.get_logger().info(
                f"{self.name} Loaded F/T calibration from {self.calib_path}"
            )
        else:
            self.node.get_logger().warn(
                f"{self.name} No F/T calibration at {self.calib_path}; "
                "using identity fallback (expect drift)."
            )
            self.payload = PayloadCompensator(mass=0.06, com=[0, 0, 0.05])
            self.ft_bias = None
            self.R_sensor_tcp = np.eye(3)
            self.use_file_calib = False

        self.p_sensor_tcp = np.zeros(3)
        self.runtime_offset = ForceOffsetCalibrator(num_samples=self.force_proc.calib_samples)

        # Target pose (equilibrium): updated via `set_target`.
        self._target_lock = threading.Lock()
        self._target_p_ref: Optional[np.ndarray] = None
        self._target_q_ref: Optional[np.ndarray] = None

        # Thread bookkeeping
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._paused = threading.Event()
        self._paused.clear()

        # Diagnostics
        self.last_F_ctrl = np.zeros(6)
        self.last_v_cmd = np.zeros(3)
        self.last_x_current = np.zeros(3)
        self.last_q_current = np.array([0.0, 0.0, 0.0, 1.0])
        self.loop_count = 0
        self.hz_window: deque = deque(maxlen=200)
        self._log_periodic_telemetry = bool(log_periodic_telemetry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_gains(self, gains: AdmittanceGains) -> None:
        """Swap admittance gains live. Integrator velocity state is retained
        so there is no step discontinuity in the commanded velocity."""
        with self._gains_lock:
            self._gains = gains
            self.adm_trans.M = np.array(gains.M, dtype=float)
            self.adm_trans.B = np.array(gains.B, dtype=float)
            self.adm_trans.K = np.array(gains.K, dtype=float)
            self.adm_trans.max_vel = gains.max_vel

            self.adm_rot.M_rot = np.array(gains.M_rot, dtype=float)
            self.adm_rot.B_rot = np.array(gains.B_rot, dtype=float)
            self.adm_rot.K_rot = np.array(gains.K_rot, dtype=float)
            self.adm_rot.max_omega = gains.max_omega
            self.adm_rot.rot_lead_time = gains.rot_lead_time
        self.node.get_logger().info(
            f"{self.name} gains updated: "
            f"B={gains.B} K={gains.K} B_rot={gains.B_rot} K_rot={gains.K_rot}"
        )

    def set_target(self, p_ref, q_ref) -> None:
        """Update equilibrium position / orientation (waist frame).

        This RESETS the admittance velocity / acceleration state -- appropriate
        when switching modes (e.g. initial capture, re-homing).  If you want
        to slew the equilibrium smoothly while a motion is already in
        progress (e.g. adaptive descent / lift) use ``update_target`` instead,
        otherwise the integrator is zeroed every cycle and the arm never
        builds up enough velocity to actually follow ``p_ref`` down.
        """
        with self._target_lock:
            self._target_p_ref = np.asarray(p_ref, dtype=float).copy()
            self._target_q_ref = np.asarray(q_ref, dtype=float).copy()
        self.adm_trans.set_reference(self._target_p_ref)
        self.adm_rot.set_reference(self._target_q_ref)
        self.node.get_logger().info(
            f"{self.name} target set: p={self._target_p_ref.tolist()} "
            f"q={self._target_q_ref.tolist()}"
        )

    def update_target(self, p_ref, q_ref, *, log: bool = False) -> None:
        """Slew the admittance equilibrium WITHOUT resetting integrator state.

        Use this when the caller is continuously moving the reference
        (e.g. ramping z_eq down at 10 mm/s during a placement descent).
        Unlike ``set_target`` this does not call
        ``TranslationAdmittanceFixed.set_reference`` -- which zeroes
        ``v_state`` / ``a_prev`` and would prevent the admittance from
        reaching the steady-state velocity needed to follow ``p_ref``.
        """
        p = np.asarray(p_ref, dtype=float).copy()
        q = np.asarray(q_ref, dtype=float).copy()
        with self._target_lock:
            self._target_p_ref = p
            self._target_q_ref = q
        self.adm_trans.p_ref = p
        self.adm_rot.q_ref = q
        if log:
            self.node.get_logger().info(
                f"{self.name} target slewed: p={p.tolist()} q={q.tolist()}"
            )

    def capture_current_pose_as_target(self, timeout_sec: float = 0.3) -> bool:
        """Look up TF and lock the current TCP pose as the admittance reference.

        Also publishes that pose to the QP streaming controller ONCE so that
        the arm has a fresh target during the gap between QP activation and
        the first admittance loop iteration.  Without this seed the QP
        receives no commands for several hundred ms and the arm can sag
        under gravity before offset calibration finishes.
        """
        t = self.tf_helper.lookup_transform(
            self.waist_frame, self.tcp_frame, timeout_sec=timeout_sec
        )
        if t is None:
            self.node.get_logger().error(
                f"{self.name} cannot look up {self.waist_frame}->{self.tcp_frame}"
            )
            return False
        p = TFHelper.position_from_transform(t)
        q = quat_from_transform(t)
        self.set_target(p, q)
        seed_pose = Pose()
        seed_pose.position.x = float(p[0])
        seed_pose.position.y = float(p[1])
        seed_pose.position.z = float(p[2])
        seed_pose.orientation.x = float(q[0])
        seed_pose.orientation.y = float(q[1])
        seed_pose.orientation.z = float(q[2])
        seed_pose.orientation.w = float(q[3])
        try:
            self._publish_target(seed_pose)
            self.node.get_logger().info(
                f"{self.name} seeded QP target at captured pose "
                f"(z={float(p[2]):+.4f})."
            )
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warning(
                f"{self.name} could not seed QP target: {exc}"
            )
        return True

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"AdmittanceArm-{self.side}", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def _publish_target(self, target_pose: Pose) -> None:
        if self.side == "left":
            self.topic_pub.publish_endposetarget_L(target_pose, from_frame=self.waist_frame)
        else:
            self.topic_pub.publish_endposetarget_R(target_pose, from_frame=self.waist_frame)

    def _run(self) -> None:
        self.node.get_logger().info(
            f"{self.name} admittance loop starting (calibrating force offset; "
            "keep hands off the arm briefly)..."
        )
        target_pose = Pose()
        ref_set = False
        last_time = time.time()

        while not self._stop_evt.is_set():
            loop_start = time.monotonic()

            if self._paused.is_set():
                time.sleep(0.01)
                last_time = time.time()
                continue

            now = time.time()
            dt_real = now - last_time
            last_time = now
            dt = self.loop_period  # fixed-step integration for smoother behaviour
            self.hz_window.append(1.0 / max(dt_real, 1e-6))

            # --- 1. Read filtered F/T ----------------------------------
            F_raw = self.force_sub.force.copy()
            F_filtered = self.force_filter.update(F_raw)

            # --- 2. Get current TCP pose -------------------------------
            t_stamped = self.tf_helper.lookup_transform(
                self.waist_frame, self.tcp_frame, timeout_sec=0.05
            )
            if t_stamped is None:
                time.sleep(0.001)
                continue

            R_waist_sensor = (
                TFHelper.rotation_from_transform(t_stamped) @ self.R_sensor_tcp
            )
            x_current = TFHelper.position_from_transform(t_stamped)
            q_current = quat_from_transform(t_stamped)
            self.last_x_current = x_current
            self.last_q_current = q_current

            # --- 3. Payload / bias compensation ------------------------
            F_ext = self.payload.compensate(F_filtered, R_waist_sensor)
            if self.use_file_calib and self.ft_bias is not None:
                F_ext = F_ext - self.ft_bias
            F_ctrl = transform_wrench(F_ext, R_waist_sensor, self.p_sensor_tcp)

            # --- 4. Runtime offset calibration -------------------------
            if not self.runtime_offset.calibrated:
                self.runtime_offset.add_sample(F_ctrl)
                if self.runtime_offset.calibrated:
                    with self._target_lock:
                        p_ref = (
                            self._target_p_ref if self._target_p_ref is not None else x_current.copy()
                        )
                        q_ref = (
                            self._target_q_ref if self._target_q_ref is not None else q_current.copy()
                        )
                    self.adm_trans.set_reference(p_ref)
                    self.adm_rot.set_reference(q_ref)
                    ref_set = True
                    self.node.get_logger().info(
                        f"{self.name} runtime offset calibrated. "
                        f"p_ref={p_ref.tolist()}, ready."
                    )
                # IMPORTANT: publish a *hold* target to the QP during the
                # calibration window so the arm does not sag under gravity.
                # Without this the QP receives no commands for ~0.5 s, the
                # arm drifts, and the runtime offset we capture ends up
                # tied to a moving pose -- leaving a persistent residual
                # (several N) after calibration that later drives the arm
                # down.  We prefer the explicitly-captured target pose
                # (from capture_current_pose_as_target()) over the live
                # x_current/q_current so that noise in TF cannot propagate
                # into the QP target.
                with self._target_lock:
                    hold_p = (
                        self._target_p_ref if self._target_p_ref is not None else x_current
                    )
                    hold_q = (
                        self._target_q_ref if self._target_q_ref is not None else q_current
                    )
                target_pose.position.x = float(hold_p[0])
                target_pose.position.y = float(hold_p[1])
                target_pose.position.z = float(hold_p[2])
                target_pose.orientation.x = float(hold_q[0])
                target_pose.orientation.y = float(hold_q[1])
                target_pose.orientation.z = float(hold_q[2])
                target_pose.orientation.w = float(hold_q[3])
                self._publish_target(target_pose)
                time.sleep(0.001)
                continue

            if not ref_set:
                with self._target_lock:
                    p_ref = (
                        self._target_p_ref if self._target_p_ref is not None else x_current.copy()
                    )
                    q_ref = (
                        self._target_q_ref if self._target_q_ref is not None else q_current.copy()
                    )
                self.adm_trans.set_reference(p_ref)
                self.adm_rot.set_reference(q_ref)
                ref_set = True

            F_ctrl = self.runtime_offset.remove_offset(F_ctrl)
            # Store pre-deadzone force for diagnostics (so light pushes show up).
            self.last_F_ctrl = F_ctrl.copy()
            F_ctrl[:3] = smooth_deadzone(F_ctrl[:3], self.force_proc.force_deadzone)
            F_ctrl[3:] = smooth_deadzone(F_ctrl[3:], self.force_proc.torque_deadzone)

            # --- 5. Admittance integration -----------------------------
            with self._gains_lock:
                dx, v_cmd, _ = self.adm_trans.update(F_ctrl[:3], x_current, dt)
                if self.hold_orientation:
                    # Rigidly hold the captured reference orientation: no
                    # rotation admittance, so the wrist cannot wander/fight.
                    q_hold = (
                        self.adm_rot.q_ref
                        if self.adm_rot.q_ref is not None
                        else q_current
                    )
                    q_target = np.asarray(q_hold, dtype=float)
                    omega_cmd = np.zeros(3)
                else:
                    q_target, omega_cmd, _ = self.adm_rot.update(F_ctrl[3:], q_current, dt)
            self.last_v_cmd = np.asarray(v_cmd, dtype=float).copy()

            # --- 6. Publish target pose --------------------------------
            target_pose.position.x = x_current[0] + dx[0]
            target_pose.position.y = x_current[1] + dx[1]
            target_pose.position.z = x_current[2] + dx[2]
            target_pose.orientation.x = float(q_target[0])
            target_pose.orientation.y = float(q_target[1])
            target_pose.orientation.z = float(q_target[2])
            target_pose.orientation.w = float(q_target[3])
            self._publish_target(target_pose)

            # --- 7. Periodic diagnostics -------------------------------
            self.loop_count += 1
            if (
                self._log_periodic_telemetry
                and self.loop_count % 200 == 0
            ):
                disp = x_current - self.adm_trans.p_ref
                hz = np.mean(self.hz_window) if self.hz_window else 0.0
                self.node.get_logger().info(
                    f"{self.name} n={self.loop_count} "
                    f"hz={hz:.1f} "
                    f"F=[{F_ctrl[0]:+.2f},{F_ctrl[1]:+.2f},{F_ctrl[2]:+.2f}]N "
                    f"disp=[{disp[0]*100:+.1f},{disp[1]*100:+.1f},{disp[2]*100:+.1f}]cm "
                    f"v=[{v_cmd[0]:+.2f},{v_cmd[1]:+.2f},{v_cmd[2]:+.2f}]"
                )

            elapsed = time.monotonic() - loop_start
            remaining = self.loop_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self.node.get_logger().info(f"{self.name} admittance loop stopped.")
