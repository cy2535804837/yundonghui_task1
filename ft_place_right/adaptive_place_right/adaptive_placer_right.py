"""
adaptive_place_right/adaptive_placer_right.py
=============================================
Orchestrator for **wrist force-torque-driven placement** on the right arm.

This is the *tactile-free* build: all placement decisions come from the
RIGHT wrist FT sensor (``/arm_6dof_right``).  There is **no** fingertip
tactile feedback and **no** continuous grip-decay loop (the FT->contact
coefficient decay PD loop was removed as it was found unstable).  The
gripper is an injected, opaque hook (see ``gripper_hook.GripperHook``);
this module never talks to a gripper driver directly.

Expected inputs (see :class:`RightAdaptivePlacer`):
    * ``cfg``         -- :class:`RightAdaptivePlaceConfig`
    * ``xarm_manager`` -- running ``XARM_manager`` ROS node
    * ``topic_pub``   -- shared ``TopicPublisher``
    * ``tf_helper``   -- shared ``TFHelper``
    * ``gripper``     -- a :class:`gripper_hook.GripperHook` implementation
      (``close_to_hold`` / ``open`` / ``shutdown``).  Use ``NoopGripper``
      for dry-runs; wire your own Robotiq-only wrapper on the target
      machine.  May be ``None`` if the object is already held and you
      never release in-process.
    * ``right_adm``   -- :class:`handover.admittance_arm.AdmittanceArm`
      built for ``side="right"``; the orchestrator starts / stops it
      via its already-running control thread, and only swaps gains +
      updates equilibrium reference.

Phases:
    A.  Grasp the object (optional)   -- ``gripper.close_to_hold()``.
        Also samples F_empty / F_loaded to estimate ``G_obj`` (purely
        from the wrist FT; no tactile).
    B.  Descent until contact         -- fast direct-QP descent down
        to ``fast_slow_band_m`` above the safety floor, then slow
        admittance until a waist-Z force delta trips.
    C.  Load transfer                 -- each cycle:

            F_support(t) = max(Fz(t) - baseline, 0)
            gamma(t)     = clip(F_support / G_obj, 0, 1)

        Hold the press until ``gamma >= gamma_release_threshold`` (the
        table is bearing the load), then release.  No grip modulation.

    D.  Release + lift                -- ``gripper.open()`` once, then
        admittance equilibrium stepped upward by ``lift_after_release_m``.

All motion goes through the shared RIGHT ``AdmittanceArm``
(equilibrium setpoints via
:meth:`AdmittanceArm.update_target`) so the arm's reaction to
contact is inherent, not scripted.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Pose, WrenchStamped

# --- Make siblings importable when run as a script ------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (
    os.path.join(_WORKSPACE_ROOT, "admittance_control"),
    _WORKSPACE_ROOT,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AdmittanceController_v3 import TFHelper, quat_from_transform  # noqa: E402

from .config import RightAdaptivePlaceConfig  # noqa: E402


# =====================================================================
# Light wrench subscriber (local copy; we don't want to depend on
# detect_handover_place per the user's request)
# =====================================================================
class _RightWrenchSubscriber:
    def __init__(self, node, topic: str) -> None:
        self._lock = threading.Lock()
        self._f = np.zeros(3, dtype=float)
        self._t = np.zeros(3, dtype=float)
        self._got_msg = False
        node.create_subscription(WrenchStamped, topic, self._cb, 10)

    def _cb(self, msg: WrenchStamped) -> None:
        with self._lock:
            self._f[0] = msg.wrench.force.x
            self._f[1] = msg.wrench.force.y
            self._f[2] = msg.wrench.force.z
            self._t[0] = msg.wrench.torque.x
            self._t[1] = msg.wrench.torque.y
            self._t[2] = msg.wrench.torque.z
            self._got_msg = True

    @property
    def got_message(self) -> bool:
        with self._lock:
            return self._got_msg

    def get_force(self) -> np.ndarray:
        with self._lock:
            return self._f.copy()


def _load_ft_calib(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# =====================================================================
# RightAdaptivePlacer
# =====================================================================
class RightAdaptivePlacer:
    """Force-driven contact-decay placement orchestrator (right arm)."""

    def __init__(
        self,
        cfg: RightAdaptivePlaceConfig,
        *,
        xarm_manager,
        topic_pub,
        tf_helper: TFHelper,
        right_gripper,
        right_adm,
    ) -> None:
        self.cfg = cfg
        self.xarm = xarm_manager
        self.topic_pub = topic_pub
        self.tf_helper = tf_helper
        self.right_gripper = right_gripper
        self.right_adm = right_adm

        # Wrench subscriber (we own this; separate from the one the
        # admittance loop uses internally).
        self.wrench_sub = _RightWrenchSubscriber(
            self.xarm, topic=self.cfg.right_force_topic,
        )

        # FT-sensor -> TCP rotation (from calib JSON).  If missing we
        # fall back to identity (expect some Z bias).
        calib = _load_ft_calib(self.cfg.right_ft_calibration_path)
        if calib is not None and "R_sensor_tcp" in calib:
            self.R_sensor_tcp = np.asarray(
                calib["R_sensor_tcp"], dtype=float,
            )
        else:
            self.R_sensor_tcp = np.eye(3)
            self.xarm.get_logger().warn(
                f"[ADP-R] No FT calibration at "
                f"{self.cfg.right_ft_calibration_path} -- using identity "
                "R_sensor_tcp (expect Z bias)."
            )

        # EMA state for waist-Z force.
        self._fz_filt: float = 0.0
        self._fz_primed: bool = False

        # Retained only for the CSV logging columns (kept for trace
        # compatibility; there is no tactile contact-coefficient decay loop
        # in this FT-only build).
        self._target_cf_ref: float = 0.0

        # Legacy flag referenced by the descent / release code paths.
        # Always False in this FT-only build (there is no parent tactile
        # hold thread to drain).
        self._use_parent_pf_hold: bool = False

        # Runtime state that helpers look at.
        self._C_initial: Optional[float] = None
        self._G_obj: Optional[float] = None
        self._baseline_Fz: Optional[float] = None
        self._contact_z: Optional[float] = None

        # Diagnostics
        self._csv_fh = None
        self._csv_writer = None
        if cfg.log_csv_path:
            self._csv_fh = open(cfg.log_csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_fh)
            self._csv_writer.writerow([
                "t", "phase", "Fz", "F_support", "gamma",
                "target_cf", "cf", "mu", "z_eq",
            ])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            self.xarm.get_logger().info(f"[ADP-R] {msg}")

    def _read_R_waist_tcp(self) -> Optional[np.ndarray]:
        t = self.tf_helper.lookup_transform(
            self.cfg.waist_frame, self.cfg.right_tcp_frame, timeout_sec=0.0,
        )
        if t is None:
            return None
        return TFHelper.rotation_from_transform(t)

    def _read_tcp_pose(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray]]:
        t = self.tf_helper.lookup_transform(
            self.cfg.waist_frame, self.cfg.right_tcp_frame, timeout_sec=0.0,
        )
        if t is None:
            return None, None, None
        return (
            TFHelper.position_from_transform(t),
            quat_from_transform(t),
            TFHelper.rotation_from_transform(t),
        )

    def _waist_Fz(self) -> float:
        """Return EMA-filtered waist-frame Z force (sensor -> TCP -> waist)."""
        R_waist_tcp = self._read_R_waist_tcp()
        f_sensor = self.wrench_sub.get_force()
        if R_waist_tcp is None:
            return self._fz_filt
        f_waist = R_waist_tcp @ self.R_sensor_tcp @ f_sensor
        fz = float(f_waist[2])
        a = float(self.cfg.fz_filter_alpha)
        if not self._fz_primed:
            self._fz_filt = fz
            self._fz_primed = True
        else:
            self._fz_filt = a * self._fz_filt + (1.0 - a) * fz
        return self._fz_filt

    def _sample_Fz_mean(self, n_samples: int, sleep_sec: float) -> float:
        """Sample the filtered waist-Z force while the arm is stationary."""
        self._fz_primed = False  # reset filter to absorb any payload step
        samples: List[float] = []
        for _ in range(n_samples):
            fz = self._waist_Fz()
            samples.append(fz)
            time.sleep(sleep_sec)
        return float(np.mean(samples)) if samples else 0.0

    def _wait_for_wrench(self, timeout_sec: float = 5.0) -> bool:
        end = time.time() + timeout_sec
        while time.time() < end and not self.wrench_sub.got_message:
            time.sleep(0.02)
        return self.wrench_sub.got_message

    def _publish_csv(
        self, phase: str, Fz: float, F_support: float, gamma: float,
        target_cf: float, cf: float, mu: float, z_eq: float,
    ) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            f"{time.time():.4f}", phase, f"{Fz:.4f}", f"{F_support:.4f}",
            f"{gamma:.4f}", f"{target_cf:.4f}", f"{cf:.4f}",
            f"{mu:.4f}", f"{z_eq:.4f}",
        ])
        self._csv_fh.flush()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Phase A -- grasp + characterise object weight
    # ------------------------------------------------------------------
    def _grip_close_to_hold(self) -> None:
        """Close the gripper onto the object via the injected hook.

        FT-only build: this just delegates to ``gripper.close_to_hold()``
        (the original tactile ``close_to_contact`` + ``adjust_grasp`` hold
        loop has been removed).  Admittance is paused around the close so
        finger-squeeze reaction does not perturb the TCP, mirroring the
        original grasp window.
        """
        if self.right_gripper is None:
            return
        paused = False
        try:
            if self.right_adm is not None:
                self.right_adm.pause()
                paused = True
                time.sleep(0.02)
            self._log("Closing gripper to hold (hook.close_to_hold) ...")
            try:
                self.right_gripper.close_to_hold()
            except Exception as e:
                self.xarm.get_logger().warn(
                    f"[ADP-R] gripper.close_to_hold() warning: {e}"
                )
            if paused and self.right_adm is not None:
                if not self.right_adm.capture_current_pose_as_target(
                    timeout_sec=0.5
                ):
                    self.xarm.get_logger().warning(
                        "[ADP-R] TCP re-capture after grip close failed."
                    )
            time.sleep(float(self.cfg.grasp_settle_sec))
        finally:
            if paused and self.right_adm is not None:
                self.right_adm.resume()
                self._log("[ADP-R] RIGHT admittance resumed after grip close.")

    def characterise_object(self) -> None:
        """Populate ``self._G_obj`` (and a dummy ``self._C_initial``).

        Tactile-free (wrist FT only):

        * ``cfg.g_obj_estimation = "manual"``  -- ``G_obj = cfg.object_weight_N``.
          If ``grasp_first`` the gripper is closed via the hook; otherwise the
          object is assumed already held.
        * ``cfg.g_obj_estimation = "auto"`` + ``grasp_first=True``:
            1. ``gripper.open()`` -> sample F_empty.
            2. ``gripper.close_to_hold()`` -> sample F_loaded.
            3. ``G_obj = |F_empty - F_loaded|``.
        * ``cfg.g_obj_estimation = "auto"`` + ``grasp_first=False``
          (object already held):
            1. Sample F_loaded (held).
            2. ``gripper.open()`` -> sample F_empty.
            3. ``gripper.close_to_hold()`` again so Phase B still has the payload.

        ``G_obj`` is used only to normalise ``gamma`` (the fraction of the
        load the table has taken) for the Phase C release trigger.
        """
        if not self._wait_for_wrench(timeout_sec=5.0):
            raise RuntimeError(
                f"[ADP-R] No message on {self.cfg.right_force_topic} "
                "after 5 s; is the FT driver up?"
            )

        # Defensively apply hold_gains so the arm stays stiff while the
        # gripper closes.  With soft (descent) K_z, closing the gripper
        # adds -G_obj in waist-Z and the arm sags before Phase B even
        # starts.
        if self.right_adm is not None:
            self._log(
                f"Applying HOLD gains during characterisation "
                f"(K={list(self.cfg.hold_gains.K)})."
            )
            self.right_adm.set_gains(self.cfg.hold_gains)
            self.right_adm.capture_current_pose_as_target(timeout_sec=0.5)

        manual_mode = (self.cfg.g_obj_estimation == "manual")
        # No contact-coefficient in this build; kept only for trace columns.
        self._C_initial = 0.0
        self._target_cf_ref = 0.0

        # 1. Auto weight characterisation (F_empty vs F_loaded).
        F_empty: Optional[float] = None
        F_loaded_pregrasp: Optional[float] = None
        if not manual_mode:
            if self.right_gripper is None:
                raise RuntimeError(
                    "[ADP-R] auto G_obj needs a gripper hook to open/close; "
                    "pass one or use --g-obj-mode manual."
                )
            if self.cfg.grasp_first:
                self._log("Opening gripper for F_empty sampling ...")
                try:
                    self.right_gripper.open()
                except Exception as e:
                    self.xarm.get_logger().warn(
                        f"[ADP-R] gripper.open() warning: {e}"
                    )
                time.sleep(0.5)
                self._log("Sampling F_empty (gripper open) ...")
                F_empty = self._sample_Fz_mean(
                    self.cfg.weight_samples, self.cfg.weight_sleep_sec,
                )
                self._log(f"F_empty = {F_empty:+.3f} N (waist-Z)")
            else:
                self._log(
                    "[ADP-R] grasp_first=False (object already held): "
                    "F_loaded with gripper closed, then open for F_empty."
                )
                self._log("Sampling F_loaded (gripper closed on object) ...")
                F_loaded_pregrasp = self._sample_Fz_mean(
                    self.cfg.weight_samples, self.cfg.weight_sleep_sec,
                )
                self._log(
                    f"F_loaded = {F_loaded_pregrasp:+.3f} N (waist-Z)"
                )
                self._log("Opening gripper for F_empty sampling ...")
                try:
                    self.right_gripper.open()
                except Exception as e:
                    self.xarm.get_logger().warn(
                        f"[ADP-R] gripper.open() warning: {e}"
                    )
                time.sleep(0.5)
                self._log("Sampling F_empty (gripper open) ...")
                F_empty = self._sample_Fz_mean(
                    self.cfg.weight_samples, self.cfg.weight_sleep_sec,
                )
                self._log(f"F_empty = {F_empty:+.3f} N (waist-Z)")

        # 2. Close the gripper onto the object so Phase B has the payload.
        #    * grasp_first=True  -> close now (in both auto and manual).
        #    * grasp_first=False -> only re-close in auto mode (we opened
        #      to sample F_empty); in manual mode the object is already held.
        if self.cfg.grasp_first or (not manual_mode):
            self._grip_close_to_hold()

        # 3. Resolve G_obj.
        if manual_mode:
            self._G_obj = float(self.cfg.object_weight_N)
            self._log(
                f"Manual G_obj = {self._G_obj:.3f} N; skipping F_loaded "
                "sample (object weight taken from --object-weight-N)."
            )
            if self._G_obj < self.cfg.min_object_weight_N:
                self._log(
                    f"WARNING: manual G_obj = {self._G_obj:.3f} N < "
                    f"min={self.cfg.min_object_weight_N:.2f} N.  Release "
                    "will fall back to threshold-on-contact."
                )
            return

        # --- auto mode from here on ---------------------------------
        assert F_empty is not None  # sampled above in auto mode
        if self.cfg.grasp_first:
            self._log("Sampling F_loaded (gripper closed on object) ...")
            F_loaded = self._sample_Fz_mean(
                self.cfg.weight_samples, self.cfg.weight_sleep_sec,
            )
            self._log(f"F_loaded = {F_loaded:+.3f} N (waist-Z)")
        else:
            assert F_loaded_pregrasp is not None
            F_loaded = float(F_loaded_pregrasp)
            self._log(
                "[ADP-R] Using F_loaded from pre-open held sample "
                f"(F_loaded={F_loaded:+.3f} N)."
            )

        G = float(abs(F_empty - F_loaded))
        self._G_obj = G
        # A truly tiny G_obj (< ~50 mN) almost always means the gripper
        # closed on *nothing* -- no real payload.  Descending further
        # would just drive the arm into the table.  Abort early.
        negligible_G = 0.05  # 50 mN -- well below the useful floor
        if G < negligible_G:
            msg = (
                f"G_obj = {G:.3f} N is below the sensor-noise floor "
                f"({negligible_G:.2f} N).  The gripper appears to be "
                "closed on NOTHING (F_empty and F_loaded are "
                "indistinguishable).  Aborting before descent -- "
                "please place an object in the gripper and rerun, or "
                "set --g-obj-mode=manual --object-weight-N=<N> if you "
                "know the weight."
            )
            self._log(msg)
            raise RuntimeError(msg)
        if G < self.cfg.min_object_weight_N:
            self._log(
                f"WARNING: G_obj = {G:.3f} N < "
                f"min={self.cfg.min_object_weight_N:.2f} N.  Release will "
                "fall back to threshold-on-contact."
            )
        else:
            self._log(f"G_obj = {G:.3f} N (|F_empty - F_loaded|)")

    # ------------------------------------------------------------------
    # Phase B -- descend until contact
    # ------------------------------------------------------------------
    def _fast_descend_qp(
        self,
        *,
        start_x: float,
        start_y: float,
        start_z: float,
        q0: np.ndarray,
        fast_target_z: float,
        baseline_Fz: float,
    ) -> Tuple[bool, float]:
        """Fast Cartesian descent by streaming poses directly to the
        right QP controller with admittance PAUSED.

        Admittance is too soft to track > ~15 mm/s (tracking lag
        grows as B*v/K).  The raw QP stream on the same topic can
        track ~50-60 mm/s (the same setting used by
        ``tactile_place/tactile_place.py`` on this exact arm for the
        grasp approach) because there is no compliance loop between
        command and controller.

        While the QP is streaming we still read the wrist F/T
        independently (via ``_waist_Fz`` -- wrench_sub is alive) so a
        contact trip immediately stops the fast phase.

        Returns
        -------
        (contact, final_tcp_z)
            ``contact`` True iff fast phase tripped on contact; in
            that case the caller should skip the slow phase.
            ``final_tcp_z`` is the actual TCP z at the end of the
            fast phase (for logging only).
        """
        assert self.right_adm is not None
        self._log(
            f"FAST descent start z={start_z:+.4f} -> {fast_target_z:+.4f} "
            f"(span={(start_z - fast_target_z) * 1000.0:.1f} mm at "
            f"{self.cfg.fast_descent_speed_mps * 1000.0:.0f} mm/s)"
        )

        self.right_adm.pause()
        time.sleep(0.02)  # let the admittance loop notice the pause

        pose_msg = Pose()
        pose_msg.orientation.x = float(q0[0])
        pose_msg.orientation.y = float(q0[1])
        pose_msg.orientation.z = float(q0[2])
        pose_msg.orientation.w = float(q0[3])

        target_z = start_z
        contact = False
        debounce = 0
        last_t = time.time()
        loop = 0
        throttle_events = 0
        min_tcp_drop_m = float(self.cfg.descent_min_tcp_drop_m)
        max_lag_m = float(self.cfg.fast_descent_max_lag_m)

        final_tcp_z = start_z
        t_start = time.time()
        deadline = t_start + float(self.cfg.fast_descent_max_time_sec)
        try:
            while target_z > fast_target_z:
                if time.time() > deadline:
                    self.xarm.get_logger().warning(
                        "[ADP-R] fast descent hit time budget "
                        f"({self.cfg.fast_descent_max_time_sec:.1f} s) "
                        f"at z_cmd={target_z:+.4f}; handing off to slow "
                        "phase from current pose."
                    )
                    break
                now = time.time()
                dt = float(np.clip(now - last_t, 0.001, 0.05))
                last_t = now

                tcp_p_pre, _, _ = self._read_tcp_pose()
                if tcp_p_pre is not None:
                    current_lag_m = float(tcp_p_pre[2]) - target_z
                    final_tcp_z = float(tcp_p_pre[2])
                else:
                    current_lag_m = 0.0

                # --- Contact BEFORE slewing z down -----------------------------
                # Old bug: we decremented target_z and *then* read Fz, and we
                # kept slewing for every debounce sample while "maybe contact"
                # built up — the arm was commanded 5+ steps deeper *after* the
                # table had already started supporting load.  That spuriously
                # raises F_support / gamma and feels like the gripper "won't
                # let go" because the table force kept climbing.  If |ΔFz| is
                # past threshold, do NOT lower target_z this iteration; only
                # count debounce and hold the last equilibrium.
                fz = self._waist_Fz()
                d_fz = fz - baseline_Fz
                th = float(self.cfg.descent_force_threshold_N)
                if self.cfg.descent_contact_signed_support:
                    contact_force = d_fz > th
                else:
                    contact_force = abs(d_fz) > th
                if not contact_force:
                    debounce = 0

                if current_lag_m > max_lag_m:
                    throttle_events += 1
                    if throttle_events == 1 or throttle_events % 100 == 0:
                        self.xarm.get_logger().warning(
                            f"[ADP-R] fast throttle: lag "
                            f"{current_lag_m * 1000.0:+.1f}mm > "
                            f"{max_lag_m * 1000.0:.0f}mm at z="
                            f"{target_z:+.4f}; HOLDING."
                        )
                elif not contact_force:
                    target_z -= self.cfg.fast_descent_speed_mps * dt
                    target_z = max(target_z, fast_target_z)

                pose_msg.position.x = float(start_x)
                pose_msg.position.y = float(start_y)
                pose_msg.position.z = float(target_z)
                self.topic_pub.publish_endposetarget_R(
                    pose_msg, from_frame=self.cfg.waist_frame
                )

                if contact_force:
                    debounce += 1
                    if debounce >= self.cfg.descent_force_debounce:
                        tcp_p, _, _ = self._read_tcp_pose()
                        if tcp_p is not None:
                            actual_drop = start_z - float(tcp_p[2])
                            if actual_drop >= min_tcp_drop_m:
                                final_tcp_z = float(tcp_p[2])
                                self._log(
                                    f"FAST CONTACT at tcp_z={final_tcp_z:+.4f} "
                                    f"(ΔFz={d_fz:+.3f} N, tcp_drop="
                                    f"{actual_drop * 1000.0:+.1f} mm)"
                                )
                                contact = True
                                break
                            else:
                                debounce = 0  # probably start-up jerk
                        else:
                            debounce = 0

                loop += 1
                if loop % 100 == 0:
                    tcp_p, _, _ = self._read_tcp_pose()
                    tcp_z = float(tcp_p[2]) if tcp_p is not None else float("nan")
                    self._log(
                        f"  fast loop {loop:4d}: z_cmd={target_z:+.4f} "
                        f"tcp_z={tcp_z:+.4f} lag="
                        f"{(tcp_z - target_z) * 1000.0:+.1f}mm "
                        f"Fz={fz:+.3f} ΔFz={d_fz:+.3f}"
                    )
                self._publish_csv(
                    "fast_descent", fz, max(fz - baseline_Fz, 0.0), 0.0,
                    self._target_cf_ref, 0.0, 0.0, target_z,
                )
                time.sleep(self.cfg.descent_loop_period_sec)
        finally:
            # Settle: keep publishing the last target during the dwell
            # so the QP has time to decelerate, then re-sync admittance
            # to the current TCP and resume.
            dwell = float(self.cfg.fast_to_slow_dwell_sec)
            dwell_end = time.time() + dwell
            while time.time() < dwell_end:
                pose_msg.position.z = float(target_z)
                self.topic_pub.publish_endposetarget_R(
                    pose_msg, from_frame=self.cfg.waist_frame
                )
                time.sleep(self.cfg.descent_loop_period_sec)

            if not self.right_adm.capture_current_pose_as_target(
                timeout_sec=0.3
            ):
                self.xarm.get_logger().warning(
                    "[ADP-R] could not re-capture TCP pose after fast "
                    "descent; admittance will resume with stale ref."
                )
            self.right_adm.resume()
            tcp_p, _, _ = self._read_tcp_pose()
            if tcp_p is not None:
                final_tcp_z = float(tcp_p[2])

        if throttle_events > 0:
            self._log(
                f"FAST descent: throttle engaged {throttle_events}x "
                f"(QP was momentarily behind)."
            )
        return contact, final_tcp_z

    def descend_until_contact(self) -> Tuple[bool, float]:
        """Fast direct-QP descent down to the "slow band", then slow
        admittance until the waist-Z force delta trips.  Returns
        ``(contact, final_z_eq)``.
        """
        assert self.right_adm is not None
        self.right_adm.set_gains(self.cfg.descent_gains)
        if not self.right_adm.capture_current_pose_as_target(timeout_sec=0.5):
            raise RuntimeError(
                "[ADP-R] Could not capture RIGHT TCP as admittance target -- "
                "TF stale or QP controller not ready?"
            )

        # Sample the "stationary + loaded" baseline.
        time.sleep(float(self.cfg.descent_baseline_settle_sec))
        baseline = self._sample_Fz_mean(
            n_samples=max(50, self.cfg.weight_samples // 2),
            sleep_sec=self.cfg.weight_sleep_sec,
        )
        self._baseline_Fz = baseline
        if getattr(self, "_use_parent_pf_hold", False) and self.right_adm is not None:
            if self.right_adm.capture_current_pose_as_target(timeout_sec=0.5):
                self._log(
                    "[ADP-R] TCP equilibrium re-capture after Fz baseline "
                    "(chained tactile hold; recenters z_eq vs finger loads)."
                )
        trip = (
            f"(Fz-baseline) > {self.cfg.descent_force_threshold_N:.3f} N"
            if self.cfg.descent_contact_signed_support
            else f"|Fz-baseline| > {self.cfg.descent_force_threshold_N:.3f} N"
        )
        self._log(
            f"Descent baseline Fz = {baseline:+.3f} N; contact trip {trip}; "
            f"min_tcp_drop={self.cfg.descent_min_tcp_drop_m * 1000.0:.0f} mm"
        )

        with self.right_adm._target_lock:  # noqa: SLF001
            p0 = self.right_adm._target_p_ref.copy()  # noqa: SLF001
            q0 = self.right_adm._target_q_ref.copy()  # noqa: SLF001
        start_x, start_y, start_z = float(p0[0]), float(p0[1]), float(p0[2])

        # ----- PHASE B1: optional fast descent (QP direct) ------------
        if (
            self.cfg.fast_descent_enabled
            and start_z > self.cfg.descent_min_z_m + self.cfg.fast_slow_band_m
        ):
            fast_target_z = max(
                self.cfg.descent_min_z_m + self.cfg.fast_slow_band_m,
                self.cfg.descent_min_z_m + 0.01,
            )
            fast_contact, fast_end_tcp_z = self._fast_descend_qp(
                start_x=start_x, start_y=start_y, start_z=start_z,
                q0=np.asarray(q0, dtype=float),
                fast_target_z=fast_target_z,
                baseline_Fz=baseline,
            )
            if fast_contact:
                self._contact_z = fast_end_tcp_z
                return True, self._contact_z
            # Re-sync local cache of admittance reference -- capture_*
            # inside _fast_descend_qp moved p_ref / q_ref.
            with self.right_adm._target_lock:  # noqa: SLF001
                p0 = self.right_adm._target_p_ref.copy()  # noqa: SLF001
                q0 = self.right_adm._target_q_ref.copy()  # noqa: SLF001
            start_x, start_y = float(p0[0]), float(p0[1])

            # Re-sample baseline Fz in the new pose.
            time.sleep(0.2)
            new_baseline = self._sample_Fz_mean(
                n_samples=30, sleep_sec=self.cfg.weight_sleep_sec,
            )
            baseline_shift = new_baseline - baseline
            self._log(
                f"Re-baselined Fz after fast descent: "
                f"{baseline:+.3f} -> {new_baseline:+.3f} N "
                f"(shift={baseline_shift:+.3f} N)"
            )
            baseline = new_baseline
            self._baseline_Fz = baseline
        else:
            if not self.cfg.fast_descent_enabled:
                self._log("Fast descent disabled (fast_descent_enabled=False).")
            else:
                self._log(
                    f"Fast descent SKIPPED -- only "
                    f"{(start_z - self.cfg.descent_min_z_m) * 1000.0:.1f} mm "
                    "to floor, below fast_slow_band."
                )

        # ----- PHASE B2: slow descent (admittance) --------------------
        target_z = float(p0[2])
        debounce = 0
        contact = False
        last_t = time.time()
        loop = 0
        min_tcp_drop_m = float(self.cfg.descent_min_tcp_drop_m)
        max_lag_m: float = float(self.cfg.descent_max_lag_m)
        throttle_events = 0

        slow_label = (
            "SLOW descent start" if self.cfg.fast_descent_enabled
            else "Descent start"
        )
        self._log(
            f"{slow_label} p=[{start_x:+.4f},{start_y:+.4f},{target_z:+.4f}] "
            f"speed={self.cfg.descent_speed_mps:.3f} m/s "
            f"min_z={self.cfg.descent_min_z_m:.3f} "
            f"max_lag={max_lag_m * 1000.0:.0f}mm"
        )

        while target_z > self.cfg.descent_min_z_m:
            now = time.time()
            dt = float(np.clip(now - last_t, 0.001, 0.05))
            last_t = now

            tcp_p_pre, _, _ = self._read_tcp_pose()
            if tcp_p_pre is not None:
                current_lag_m = float(tcp_p_pre[2]) - target_z
            else:
                current_lag_m = 0.0

            # Same "contact before slew" fix as _fast_descend_qp: once the
            # table supports load, stop commanding deeper z_eq while debounce
            # counts — otherwise F_support and gamma are polluted before
            # Phase C.
            fz = self._waist_Fz()
            d_fz = fz - baseline
            th = float(self.cfg.descent_force_threshold_N)
            if self.cfg.descent_contact_signed_support:
                contact_force = d_fz > th
            else:
                contact_force = abs(d_fz) > th
            if not contact_force:
                debounce = 0

            if current_lag_m > max_lag_m:
                throttle_events += 1
                if throttle_events == 1 or throttle_events % 100 == 0:
                    self.xarm.get_logger().warning(
                        f"[ADP-R] descent throttle: tcp lag "
                        f"{current_lag_m * 1000.0:+.1f}mm > max "
                        f"{max_lag_m * 1000.0:.0f}mm at z_eq="
                        f"{target_z:+.4f}; HOLDING z_eq until arm "
                        "catches up."
                    )
            elif not contact_force:
                target_z -= self.cfg.descent_speed_mps * dt
                target_z = max(target_z, self.cfg.descent_min_z_m)

            # Use update_target (not set_target) so the admittance
            # integrator state is preserved across ref slews.
            self.right_adm.update_target(
                [start_x, start_y, target_z],
                [float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])],
            )

            if contact_force:
                debounce += 1
                if debounce >= self.cfg.descent_force_debounce:
                    tcp_p, _, _ = self._read_tcp_pose()
                    if tcp_p is None:
                        self.xarm.get_logger().warning(
                            "[ADP-R] contact candidate but TCP TF is "
                            "stale; ignoring."
                        )
                        debounce = 0
                    else:
                        actual_drop = start_z - float(tcp_p[2])
                        if actual_drop < min_tcp_drop_m:
                            self.xarm.get_logger().warning(
                                f"[ADP-R] Ignoring force trip: TCP has "
                                f"only descended {actual_drop * 1000.0:+.1f}"
                                f" mm (need {min_tcp_drop_m * 1000.0:.1f}"
                                f" mm). Likely wrist reaction, not contact."
                            )
                            debounce = 0
                        else:
                            self._log(
                                f"CONTACT at z_eq={target_z:+.4f} m "
                                f"(ΔFz={d_fz:+.3f} N, "
                                f"tcp_drop={actual_drop * 1000.0:+.1f} mm)"
                            )
                            contact = True
                            break

            loop += 1
            if loop % 100 == 0:
                tcp_p, _, _ = self._read_tcp_pose()
                tcp_z = float(tcp_p[2]) if tcp_p is not None else float("nan")
                lag_mm = (tcp_z - target_z) * 1000.0
                self._log(
                    f"  descent loop {loop:4d}: z_eq={target_z:+.4f} "
                    f"tcp_z={tcp_z:+.4f} lag={lag_mm:+.1f}mm "
                    f"Fz={fz:+.3f} ΔFz={d_fz:+.3f}"
                )
            self._publish_csv(
                "descent", fz, max(fz - baseline, 0.0), 0.0,
                self._target_cf_ref, 0.0, 0.0, target_z,
            )
            time.sleep(self.cfg.descent_loop_period_sec)

        if not contact:
            tcp_p_end, _, _ = self._read_tcp_pose()
            tcp_z_end = (
                float(tcp_p_end[2]) if tcp_p_end is not None else float("nan")
            )
            actual_drop_mm = (
                (start_z - tcp_z_end) * 1000.0
                if tcp_p_end is not None else float("nan")
            )
            self.xarm.get_logger().warn(
                f"[ADP-R] Descent reached safety floor z={target_z:+.4f} "
                "without contact; aborting transfer."
            )
            self.xarm.get_logger().warn(
                f"[ADP-R]   summary: tcp descended "
                f"{actual_drop_mm:+.1f} mm (start z={start_z:+.4f} -> "
                f"end tcp_z={tcp_z_end:+.4f}); "
                f"throttle engaged {throttle_events} times "
                f"(max_lag={max_lag_m * 1000.0:.0f} mm)."
            )
        self._contact_z = float(target_z)
        return contact, self._contact_z

    # ------------------------------------------------------------------
    # Phase C -- load transfer / decay release
    # ------------------------------------------------------------------
    def load_transfer(self) -> bool:
        """Press the object onto the table and wait until the wrist FT
        confirms the table bears the load.

        Each cycle::

            F_support = max(Fz - baseline, 0)
            gamma     = clip(F_support / G_obj, 0, 1)

        Returns True once ``gamma >= gamma_release_threshold`` for
        ``gamma_release_debounce`` consecutive samples (or on timeout
        fallback).  There is no grip modulation here -- the actual jaw
        opening happens once in :meth:`release_and_lift`.
        """
        assert self._G_obj is not None
        assert self._baseline_Fz is not None

        baseline = self._baseline_Fz
        G_obj = self._G_obj

        if G_obj < self.cfg.min_object_weight_N:
            self._log(
                "G_obj below the useful floor; skipping gamma transfer and "
                "releasing on contact (pre-open unload in release_and_lift)."
            )
            return True

        # Switch admittance to transfer gains (firm K_z so the arm
        # resists the table's push-back and holds the commanded
        # equilibrium Z).
        self.right_adm.set_gains(self.cfg.transfer_gains)

        # ---- Press the object onto the table ---------------------------
        # Drop the admittance target Z by ``transfer_press_depth_m``.
        # The table stops the arm from actually going that low, so the
        # residual position error becomes a steady press force of
        # ``K_z * press_depth`` Newtons on the object -- the SAME force
        # appears on the F/T sensor as F_support and drives gamma.
        # Without this step, for light objects (<1 N) F_support stays
        # stuck at 0 until the gripper finally releases, which is the
        # 8 s stall observed on the previous run.
        press_depth = float(getattr(self.cfg, "transfer_press_depth_m", 0.0))
        if press_depth > 0.0:
            with self.right_adm._target_lock:  # noqa: SLF001
                p_cur = self.right_adm._target_p_ref.copy()  # noqa: SLF001
                q_cur = self.right_adm._target_q_ref.copy()  # noqa: SLF001
            p_press = p_cur.copy()
            p_press[2] = float(p_cur[2]) - press_depth
            self.right_adm.update_target(
                [float(p_press[0]), float(p_press[1]), float(p_press[2])],
                [float(q_cur[0]), float(q_cur[1]), float(q_cur[2]),
                 float(q_cur[3])],
            )
            K_z = float(list(self.cfg.transfer_gains.K)[2])
            expected_press_N = K_z * press_depth
            self._log(
                f"Transfer press engaged: target_z "
                f"{float(p_cur[2]):+.4f} -> {float(p_press[2]):+.4f} "
                f"(depth {press_depth * 1000.0:.1f} mm, K_z={K_z:.1f}, "
                f"expected F_press~{expected_press_N:.2f} N / "
                f"gamma_start~{min(1.0, expected_press_N / max(G_obj, 1e-6)):.2f})"
            )

        self._log(
            f"Transfer gains applied (K={list(self.cfg.transfer_gains.K)}). "
            f"G_obj={G_obj:.3f} N baseline_Fz={baseline:+.3f} N"
        )

        # FT-only build: there is no grip-decay thread.  The gripper stays
        # closed (held by the hook / its own motor) through the press; we
        # only monitor gamma and release once the table bears the load.
        dt_target = 1.0 / max(1.0, float(self.cfg.transfer_loop_hz))
        t_start = time.time()
        t_end = t_start + float(self.cfg.transfer_timeout_sec)
        debounce = 0
        released = False
        last_gamma = 0.0

        try:
            while time.time() < t_end:
                fz = self._waist_Fz()
                F_support = max(fz - baseline, 0.0)
                gamma = float(np.clip(F_support / max(G_obj, 1e-6), 0.0, 1.0))
                last_gamma = gamma

                now = time.time()
                dwell = now - t_start

                in_window = (
                    dwell >= float(self.cfg.transfer_min_dwell_sec)
                    and gamma >= float(self.cfg.gamma_release_threshold)
                )
                if in_window:
                    debounce += 1
                    if debounce >= int(self.cfg.gamma_release_debounce):
                        self._log(
                            f"Transfer complete at t={dwell:.2f}s "
                            f"gamma={gamma:.3f}"
                        )
                        released = True
                        break
                else:
                    debounce = 0

                # Periodic log + CSV.
                if int(dwell * 10) % 5 == 0:
                    self._log(
                        f"  t={dwell:5.2f}s  Fz={fz:+.3f}  Fs={F_support:.3f}  "
                        f"gamma={gamma:.3f}"
                    )
                    self._publish_csv(
                        "transfer", fz, F_support, gamma,
                        0.0, 0.0, 0.0,
                        float(self._contact_z or 0.0),
                    )

                time.sleep(dt_target)

            if not released:
                self._log(
                    f"Transfer timeout after "
                    f"{self.cfg.transfer_timeout_sec:.1f}s; last gamma="
                    f"{last_gamma:.3f} (needed >= "
                    f"{self.cfg.gamma_release_threshold:.2f}).  Releasing "
                    "anyway (fallback)."
                )
        finally:
            self._target_cf_ref = 0.0

        return released

    # ------------------------------------------------------------------
    # Phase D -- release + lift
    # ------------------------------------------------------------------
    def release_and_lift(self) -> None:
        # Unload the table **before** opening: Phase B used ``descent_gains``
        # (K_z=20) and Phase C may have used ``transfer_gains`` + press.
        # Both leave a stiff vertical spring.  The wrist then shows a
        # large "support" force while the object is still pinched.  Very
        # soft K_z + re-capturing the TCP as p_ref at the *current* pose
        # (``set_target``) zeros the position error in the spring so the
        # arm can yield to the table instead of fighting it.
        if self.right_adm is not None:
            self._log(
                "Pre-open: soft pre_release_gains + TCP re-capture to "
                "reduce table contact load before the gripper opens."
            )
            self.right_adm.set_gains(self.cfg.pre_release_gains)
            if not self.right_adm.capture_current_pose_as_target(
                timeout_sec=0.5,
            ):
                self.xarm.get_logger().warning(
                    "[ADP-R] pre-open TCP re-capture failed; continuing."
                )
            time.sleep(float(self.cfg.pre_release_settle_sec))

        if self.right_gripper is not None:
            self._log("Opening gripper (full release).")
            try:
                self.right_gripper.open()
            except Exception as e:
                self.xarm.get_logger().warn(
                    f"[ADP-R] gripper.open() warning: {e}"
                )
        time.sleep(float(self.cfg.post_release_dwell_sec))

        # Stiffer Z for the lift: pre_release was for unloading only.
        if self.right_adm is not None:
            self.right_adm.set_gains(self.cfg.descent_gains)

        # Lift via admittance equilibrium step-up.
        assert self.right_adm is not None
        with self.right_adm._target_lock:  # noqa: SLF001
            p = self.right_adm._target_p_ref.copy()  # noqa: SLF001
            q = self.right_adm._target_q_ref.copy()  # noqa: SLF001
        start_x, start_y = float(p[0]), float(p[1])
        z_now = float(p[2])
        z_end = z_now + float(self.cfg.lift_after_release_m)

        self._log(f"Lifting z from {z_now:+.4f} to {z_end:+.4f}")
        last_t = time.time()
        z = z_now
        while z < z_end:
            now = time.time()
            dt = float(np.clip(now - last_t, 0.001, 0.05))
            last_t = now
            z = min(z + float(self.cfg.lift_speed_mps) * dt, z_end)
            self.right_adm.update_target(
                [start_x, start_y, z],
                [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            )
            time.sleep(0.02)
        self._log("Lift complete.")

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Full sequence: characterise object -> descend -> transfer ->
        release + lift.  Returns True if the object was released with
        contact detected."""
        try:
            self.characterise_object()
            contact, _ = self.descend_until_contact()
            if not contact:
                return False
            ok = self.load_transfer()
            self.release_and_lift()
            return ok
        finally:
            if self._csv_fh is not None:
                try:
                    self._csv_fh.close()
                except Exception:
                    pass


__all__ = ["RightAdaptivePlacer"]
