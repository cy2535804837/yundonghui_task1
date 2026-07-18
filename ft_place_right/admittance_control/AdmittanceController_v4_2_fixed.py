"""
AdmittanceController_v4_2_fixed.py
===================================
Fixes bugs in v4_2 for BOTH translation and rotation:

TRANSLATION (TranslationAdmittanceFixed):
  BUG 1 — x_state not accumulated:
      Original: self.x_state = self.v_state * dt    ← resets every loop → K*x ≈ 0 always
      Fixed:    self.x_state += self.v_state * dt
  BUG 2 — x_state drifts from real TCP:
      Fix: compute displacement from a fixed p_ref using actual TF position.

ROTATION (RotationAdmittanceFixed):
  BUG 3 — theta_state drifts from real TCP orientation:
      Fix: compute angular displacement from a fixed q_ref using actual TF quaternion.
  BUG 4 — Euler integration instead of trapezoidal:
      Fixed:    omega_cmd_raw = omega_state + dt * (alpha + alpha_prev) / 2
  BUG 5 — No set_reference / q_ref wired up:
      Fix: added set_reference(q_current) called at calibration time.
  BUG 6 — Per-step rotation increment too small for controller:
      omega*dt ≈ 0.0004 rad << OTG_R_STEP (0.005 rad), so the QP controller
      ignores the delta. Fix: use rot_lead_time > dt to project the target
      further ahead — same relative-to-current pattern as translation
      (target = dq * q_current) but with a longer projection window.

Both translation and rotation now use the same pattern:
  target = current_pose + velocity * lead_time
No accumulation, no divergence, no conflict between axes.

CUSTOM TARGET POSE:
  --target-joints '0.0,1.18,0.0,-1.3,0.0,-0.13,0.18'
      Move there first, read Cartesian pose from TF, use as p_ref/q_ref.
  --target-xyz-quat 'x,y,z,qx,qy,qz,qw'
      Directly specify the Cartesian target in waist frame.
  --start-joints '...'
      Override the default working-position joint angles.
  If no target is given, the spring reference is set to the TCP position
  after force-offset calibration (original behavior).
"""

import os
import sys
import json
import time
import argparse
from collections import deque
import numpy as np
import rclpy
from geometry_msgs.msg import Pose

from AdmittanceController_v3 import (
    XARM_manager,
    TopicPublisher,
    ActionCall,
    TFHelper,
    ForceSubscriber,
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

CALIB_PATH = os.path.join(os.path.dirname(__file__), "ft_calibration.json")


def load_ft_calibration(path=CALIB_PATH):
    """Load calibration JSON.  Returns dict or None if file missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


class TranslationAdmittanceFixed:
    """
    Second-order admittance: M*a + B*v + K*(p - p_ref) = F
    Displacement is measured from a fixed p_ref using actual TCP position from TF,
    so K stiffness correctly resists real motion regardless of inner-loop tracking.

    dx returned is the incremental step to ADD to x_current when publishing target.
    """

    def __init__(self, M, B, K, max_vel=2.0):
        self.M = np.array(M, dtype=float)
        self.B = np.array(B, dtype=float)
        self.K = np.array(K, dtype=float)
        self.max_vel = max_vel
        self.v_state = np.zeros(3)
        self.a_prev = np.zeros(3)
        self.p_ref = None  # set once at calibration time, then locked

    def set_reference(self, p_current):
        self.p_ref = np.array(p_current, dtype=float).copy()
        self.v_state = np.zeros(3)
        self.a_prev = np.zeros(3)

    def update(self, F, p_current, dt):
        """
        F        : 3-vector, external force in waist frame [N]
        p_current: 3-vector, actual TCP position in waist frame [m]
        dt       : timestep [s]
        Returns  : (dx, v_cmd, v_cmd_raw)
                   dx is the position increment to apply on top of p_current.
        """
        if self.p_ref is None:
            self.set_reference(p_current)

        displacement = p_current - self.p_ref          # actual displacement, not integrated cmd

        a = (F - self.B * self.v_state - self.K * displacement) / self.M
        # Trapezoidal velocity integration
        v_raw = self.v_state + dt * (a + self.a_prev) / 2.0
        v_cmd = np.clip(v_raw, -self.max_vel, self.max_vel)

        self.a_prev = a.copy()
        self.v_state = v_cmd.copy()

        dx = v_cmd * dt
        return dx, v_cmd, v_raw


class RotationAdmittanceFixed:
    """
    Second-order admittance for rotation: M*alpha + B*omega + K*theta = tau
    Angular displacement is measured from a fixed q_ref using actual TCP orientation
    from TF, so K stiffness correctly resists real rotation regardless of inner-loop
    tracking — mirrors the TranslationAdmittanceFixed approach.

    Target is RELATIVE to q_current (same pattern as translation: target = x + v*dt).
    A rot_lead_time parameter projects omega further ahead than a single dt step,
    producing a large enough angular offset for the controller to track.
    Translation gets away with dt because v is large; rotation needs a longer
    projection because omega is smaller relative to the OTG step size.
    """

    def __init__(self, M_rot, B_rot, K_rot, max_omega=1.0, rot_lead_time=0.1):
        self.M_rot = np.array(M_rot, dtype=float)
        self.B_rot = np.array(B_rot, dtype=float)
        self.K_rot = np.array(K_rot, dtype=float)
        self.max_omega = max_omega
        self.rot_lead_time = rot_lead_time
        self.omega_state = np.zeros(3)
        self.alpha_prev = np.zeros(3)
        self.q_ref = None

    def set_reference(self, q_current):
        self.q_ref = np.array(q_current, dtype=float).copy()
        self.omega_state = np.zeros(3)
        self.alpha_prev = np.zeros(3)

    def update(self, tau, q_current, dt):
        """
        tau       : 3-vector, external torque in waist frame [Nm]
        q_current : 4-vector quaternion [x, y, z, w], actual TCP orientation
        dt        : timestep [s]
        Returns   : (q_target, omega_cmd, omega_cmd_raw)
        """
        if self.q_ref is None:
            self.set_reference(q_current)

        q_error = quat_multiply(q_current, quat_conjugate(self.q_ref))
        axis, angle = quat_to_axis_angle(q_error)
        theta_displacement = axis * angle

        alpha = (tau - self.B_rot * self.omega_state - self.K_rot * theta_displacement) / self.M_rot
        omega_cmd_raw = self.omega_state + dt * (alpha + self.alpha_prev) / 2.0
        omega_cmd = np.clip(omega_cmd_raw, -self.max_omega, self.max_omega)

        self.alpha_prev = alpha.copy()
        self.omega_state = omega_cmd.copy()

        # Target = q_current + omega * rot_lead_time  (same pattern as translation)
        # rot_lead_time > dt so the angular offset is large enough for the controller.
        dq = omega_to_delta_quat(omega_cmd, self.rot_lead_time)
        q_target = quat_normalize(quat_multiply(dq, q_current))
        return q_target, omega_cmd, omega_cmd_raw


def parse_float_list(s):
    """Parse a comma-separated string of floats."""
    return [float(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Admittance Controller v4.2 fixed")
    parser.add_argument(
        "--target-joints", type=str, default=None,
        help="Target pose as 7 joint angles (comma-separated, radians). "
             "The spring will pull the TCP toward this configuration's Cartesian pose. "
             "Example: --target-joints '0.0,1.18,0.0,-1.3,0.0,-0.13,0.18'"
    )
    parser.add_argument(
        "--target-xyz-quat", type=str, default=None,
        help="Target pose as x,y,z,qx,qy,qz,qw in waist frame (comma-separated). "
             "Example: --target-xyz-quat '0.1,0.2,0.3,0,0,0,1'"
    )
    parser.add_argument(
        "--target-offset", type=str, default=None,
        help="Target as a position offset from the start pose in meters (comma-separated, 3 values). "
             "Orientation stays the same as the start pose. "
             "Example: --target-offset '0.05,0,0' for +5cm in X"
    )
    parser.add_argument(
        "--start-joints", type=str, default=None,
        help="Starting joint angles (comma-separated). If omitted, uses default working pose."
    )
    args, _ = parser.parse_known_args()

    rclpy.init()

    xarm_manager = XARM_manager()
    topic_pub = TopicPublisher(xarm_manager)
    action = ActionCall(xarm_manager)
    tf_helper = TFHelper(xarm_manager)

    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.hardware_arm_enable(True)
    xarm_manager.hardware_arm_mode(3)

    # ── Resolve target reference pose ─────────────────────────────────────
    target_p_ref = None
    target_q_ref = None

    if args.target_joints is not None:
        target_joints = parse_float_list(args.target_joints)
        xarm_manager.get_logger().info(
            f"Moving to target joint config to read Cartesian pose: {target_joints}"
        )
        action.jointspace_arm_L_controller(target_joints)
        input("Press Enter after arm reaches TARGET pose...")
        time.sleep(0.5)

        for _ in range(50):
            rclpy.spin_once(xarm_manager, timeout_sec=0.0)

        t_ref = tf_helper.lookup_transform("waist_yaw_link", "left_tcp_link", timeout_sec=0.0)
        if t_ref is not None:
            target_p_ref = TFHelper.position_from_transform(t_ref)
            target_q_ref = quat_from_transform(t_ref)
            xarm_manager.get_logger().info(
                f"Target reference pose captured:\n"
                f"  position = [{target_p_ref[0]:.4f}, {target_p_ref[1]:.4f}, {target_p_ref[2]:.4f}]\n"
                f"  quaternion = [{target_q_ref[0]:.4f}, {target_q_ref[1]:.4f}, {target_q_ref[2]:.4f}, {target_q_ref[3]:.4f}]"
            )
        else:
            xarm_manager.get_logger().error("TF lookup failed at target pose!")

    elif args.target_xyz_quat is not None:
        vals = parse_float_list(args.target_xyz_quat)
        target_p_ref = np.array(vals[:3])
        target_q_ref = np.array(vals[3:7])
        xarm_manager.get_logger().info(
            f"Target reference from command line:\n"
            f"  position = {target_p_ref}\n"
            f"  quaternion = {target_q_ref}"
        )

    # ── Move to starting position ─────────────────────────────────────────
    start_joints = (
        parse_float_list(args.start_joints) if args.start_joints
        else [0.0, 1.18, 0.0, -1.3, 1.4, -0.13, 0.18]
    )
    action.jointspace_arm_L_controller(start_joints)
    input("Press Enter after arm reaches working position...")

    # ── Resolve --target-offset (needs TF at start pose) ──────────────────
    if args.target_offset is not None and target_p_ref is None:
        offset = np.array(parse_float_list(args.target_offset))
        time.sleep(0.5)
        for _ in range(50):
            rclpy.spin_once(xarm_manager, timeout_sec=0.0)

        t_start = tf_helper.lookup_transform("waist_yaw_link", "left_tcp_link", timeout_sec=0.0)
        if t_start is not None:
            start_p = TFHelper.position_from_transform(t_start)
            start_q = quat_from_transform(t_start)
            target_p_ref = start_p + offset
            target_q_ref = start_q.copy()
            xarm_manager.get_logger().info(
                f"Target from offset {offset.tolist()}:\n"
                f"  start  = [{start_p[0]:.4f}, {start_p[1]:.4f}, {start_p[2]:.4f}]\n"
                f"  target = [{target_p_ref[0]:.4f}, {target_p_ref[1]:.4f}, {target_p_ref[2]:.4f}]\n"
                f"  orientation = same as start (unchanged)"
            )
        else:
            xarm_manager.get_logger().error("TF lookup failed at start pose for --target-offset!")

    xarm_manager.xarm_activate_controller(["endpose_single_arm_qp_L_controller"])

    # Increase OTG step limits so QP controller can track admittance velocity output.
    # From logs: max |dx| per loop ≈ 7.4mm (vector norm), at ~520 Hz.
    # Default 0.005 clips that. Set to 0.01 (10mm) = ~1.35× headroom over observed max.
    # If you increase max_vel or lower loop rate, raise this proportionally.
    OTG_P_STEP = 0.005   # meters per QP cycle  (was 0.005, max observed dx ≈ 0.0074)
    OTG_R_STEP = 0.005   # radians per QP cycle  (was 0.005)
    from xarm_sdk.tools import set_node_parameter
    set_node_parameter(xarm_manager, "endpose_single_arm_qp_L_controller", "otg_p_step", OTG_P_STEP)
    set_node_parameter(xarm_manager, "endpose_single_arm_qp_L_controller", "otg_r_step", OTG_R_STEP)
    xarm_manager.get_logger().info(
        f"Mode 3 + OTG steps raised: otg_p_step={OTG_P_STEP}, otg_r_step={OTG_R_STEP}"
    )

    t_stamped_init = tf_helper.lookup_transform("waist_yaw_link", "left_tcp_link", timeout_sec=0.0)
    if t_stamped_init is None:
        xarm_manager.get_logger().error("Cannot get initial pose from TF")
        rclpy.shutdown()
        return

    init_quat = quat_from_transform(t_stamped_init)
    target_pose = Pose()

    force_sub = ForceSubscriber(xarm_manager)
    force_filter = ForceFilter(alpha=0.8)

    # ── Load F/T calibration (from ft_calibration.py) ────────────────────
    ft_calib = load_ft_calibration()
    if ft_calib is not None:
        xarm_manager.get_logger().info(
            f"Loaded F/T calibration from {CALIB_PATH}\n"
            f"  mass={ft_calib['mass']:.4f} kg  "
            f"com={ft_calib['com']}  "
            f"F_bias={ft_calib['F_bias']}  "
            f"T_bias={ft_calib['T_bias']}"
        )
        payload = PayloadCompensator(
            mass=ft_calib["mass"],
            com=ft_calib["com"],
        )
        ft_bias = np.array(ft_calib["F_bias"] + ft_calib["T_bias"])
        R_sensor_tcp = np.array(ft_calib["R_sensor_tcp"])
        use_file_calib = True
    else:
        xarm_manager.get_logger().error(
            f"[LEFT] No calibration file at {CALIB_PATH}!\n"
            f"  The left arm REQUIRES calibration for correct force compensation.\n"
            f"  Run: python ft_calibration.py\n"
            f"  Using identity R_sensor_tcp as fallback (likely incorrect)."
        )
        payload = PayloadCompensator(mass=0.06, com=[0, 0, 0.05])
        ft_bias = None
        R_sensor_tcp = np.eye(3)
        use_file_calib = False

    p_sensor_tcp = np.array([0.0, 0.0, 0.0])

    # ── Tuning ──────────────────────────────────────────────────────────────
    # For K>0, critical damping requires B_crit = 2*sqrt(M*K).
    # With M=0.1, K=20: B_crit = 2*sqrt(2) ≈ 2.83.  Use B=3.0 (slightly overdamped).
    # adm_trans = TranslationAdmittanceFixed(
    #     M=[0.1, 0.1, 0.1],
    #     B=[2.8, 0.2, 0.2],
    #     K=[20.0, 0.0, 0.0],
    #     max_vel=20.0,
    # )

    # adm_trans = TranslationAdmittanceFixed(
    #     M=[0.1, 0.1, 0.1],
    #     B=[1.8, 1.8, 1.8],
    #     K=[0.0, 0.0, 0.0],
    #     max_vel=20.0,
    # )

    adm_trans = TranslationAdmittanceFixed(
        M=[0.1, 0.1, 0.1],
        B=[0.01, 0.01, 0.01],
        K=[0.0, 0.0, 0.0],
        max_vel=20.0,
    )

    # adm_rot = RotationAdmittanceFixed(
    #     M_rot=[0.01, 0.01, 0.01],
    #     B_rot=[0.3, 0.3, 0.3],
    #     K_rot=[0.0, 0.0, 0.0],
    #     max_omega=20.0,
    #     rot_lead_time=0.15,
    # )

    # adm_rot = RotationAdmittanceFixed(
    #     M_rot=[0.01, 0.01, 0.01],
    #     B_rot=[1.3, 1.3, 1.3],
    #     K_rot=[10.0, 10.0, 10.0],
    #     max_omega=20.0,
    #     rot_lead_time=0.15,
    # )

    adm_rot = RotationAdmittanceFixed(
        M_rot=[0.01, 0.01, 0.01],
        B_rot=[0.5, 0.5, 0.5],
        K_rot=[0.0, 0.0, 0.0],
        max_omega=20.0,
        rot_lead_time=0.15,
    )

    force_deadzone  = 0.5
    torque_deadzone = 0.05
    force_threshold  = 0.5
    torque_threshold = 0.15

    # Per-axis torque sign.  +1 = keep original direction, -1 = flip.
    TORQUE_SIGN = np.array([1.0, 1.0, 1.0])

    calibrator = ForceOffsetCalibrator(num_samples=200)
    hz_window = deque(maxlen=120)
    v_cmd_norm_window  = deque(maxlen=60)
    v_ach_norm_window  = deque(maxlen=60)
    omega_cmd_norm_window = deque(maxlen=60)
    omega_ach_norm_window = deque(maxlen=60)
    prev_x_current    = None
    prev_q_current    = None
    prev_tf_stamp_sec = None
    ref_set = False

    last_time  = time.time()
    loop_count = 0
    start_time = time.time()

    xarm_manager.get_logger().info(
        "Calibrating force offset — keep hands off the robot for ~2 seconds..."
    )

    while time.time() - start_time < 2000:
        rclpy.spin_once(xarm_manager, timeout_sec=0.0)

        now = time.time()
        dt  = now - last_time
        last_time = now
        # dt = np.clip(dt, 0.001, 0.004)
        dt = 0.004
        hz_window.append(1.0 / dt)

        F_raw = force_sub.force.copy()
        F_filtered = force_filter.update(F_raw)

        t_stamped = tf_helper.lookup_transform(
            "waist_yaw_link", "left_tcp_link", timeout_sec=0.0
        )
        if t_stamped is None:
            continue

        R_waist_sensor = TFHelper.rotation_from_transform(t_stamped) @ R_sensor_tcp
        x_current      = TFHelper.position_from_transform(t_stamped)
        q_current      = quat_from_transform(t_stamped)
        tf_stamp_sec   = (
            float(t_stamped.header.stamp.sec)
            + float(t_stamped.header.stamp.nanosec) * 1e-9
        )

        # Payload gravity compensation
        F_ext  = payload.compensate(F_filtered, R_waist_sensor)

        # Apply file-based bias BEFORE frame transform (bias is in sensor frame)
        if use_file_calib:
            F_ext -= ft_bias

        F_ctrl = transform_wrench(F_ext, R_waist_sensor, p_sensor_tcp)

        # Runtime offset calibrator catches residual drift not in the file
        if not calibrator.calibrated:
            calibrator.add_sample(F_ctrl)
            if calibrator.calibrated:
                p_ref = target_p_ref if target_p_ref is not None else x_current
                q_ref = target_q_ref if target_q_ref is not None else q_current
                xarm_manager.get_logger().info(
                    f"Runtime offset calibration done. Reference set to:\n"
                    f"  p_ref = [{p_ref[0]:.4f}, {p_ref[1]:.4f}, {p_ref[2]:.4f}]\n"
                    f"  q_ref = [{q_ref[0]:.4f}, {q_ref[1]:.4f}, {q_ref[2]:.4f}, {q_ref[3]:.4f}]\n"
                    f"  {'(from --target-joints/--target-xyz-quat)' if target_p_ref is not None else '(current TCP)'}\n"
                    f"You can now drag the robot."
                )
                adm_trans.set_reference(p_ref)
                adm_rot.set_reference(q_ref)
                ref_set = True
            continue

        if not ref_set:
            p_ref = target_p_ref if target_p_ref is not None else x_current
            q_ref = target_q_ref if target_q_ref is not None else q_current
            adm_trans.set_reference(p_ref)
            adm_rot.set_reference(q_ref)
            ref_set = True

        F_ctrl = calibrator.remove_offset(F_ctrl)
        F_ctrl_before_deadzone = F_ctrl.copy()
        F_ctrl[:3] = smooth_deadzone(F_ctrl[:3], force_deadzone)
        F_ctrl[3:] = smooth_deadzone(F_ctrl[3:], torque_deadzone)

        force_active   = np.max(np.abs(F_ctrl[:3])) > force_threshold
        torque_active  = np.max(np.abs(F_ctrl[3:])) > torque_threshold
        human_active   = force_active or torque_active

        dx, v_cmd, v_cmd_raw = adm_trans.update(F_ctrl[:3], x_current, dt)

        tau_input = TORQUE_SIGN * F_ctrl[3:]
        q_target, omega_cmd, omega_cmd_raw = adm_rot.update(tau_input, q_current, dt)

        target_pose.position.x = x_current[0] + dx[0]
        target_pose.position.y = x_current[1] + dx[1]
        target_pose.position.z = x_current[2] + dx[2]
        target_pose.orientation.x = q_target[0]
        target_pose.orientation.y = q_target[1]
        target_pose.orientation.z = q_target[2]
        target_pose.orientation.w = q_target[3]

        topic_pub.publish_endposetarget_L(target_pose, from_frame="waist_yaw_link")

        # ── Achieved velocity estimate ──────────────────────────────────────
        tf_dt = None
        if prev_tf_stamp_sec is not None:
            tf_dt = tf_stamp_sec - prev_tf_stamp_sec
            if tf_dt <= 1e-6 or tf_dt > 0.2:
                tf_dt = None

        dt_vel = float(tf_dt) if tf_dt is not None else float(dt)
        dt_vel = max(dt_vel, 1e-6)

        if prev_x_current is None:
            v_ach = np.zeros(3)
        else:
            v_ach = (x_current - prev_x_current) / dt_vel

        if prev_q_current is None:
            omega_ach = np.zeros(3)
        else:
            q_delta   = quat_multiply(q_current, quat_conjugate(prev_q_current))
            axis_d, angle_d = quat_to_axis_angle(q_delta)
            omega_ach = axis_d * (angle_d / dt_vel)

        prev_x_current    = x_current.copy()
        prev_q_current    = q_current.copy()
        prev_tf_stamp_sec = tf_stamp_sec
        v_cmd_norm_window.append(np.linalg.norm(v_cmd))
        v_ach_norm_window.append(np.linalg.norm(v_ach))
        omega_cmd_norm_window.append(np.linalg.norm(omega_cmd))
        omega_ach_norm_window.append(np.linalg.norm(omega_ach))

        # ── Diagnostics ───────────────────────────────────────────────────
        loop_count += 1
        if loop_count % 50 == 0:
            mode = "ACTIVE" if human_active else "IDLE"
            disp = x_current - adm_trans.p_ref
            Kx   = adm_trans.K * disp

            print(
                f"[{loop_count:5d}] {mode:6s}  "
                f"F=[{F_ctrl[0]:+.2f} {F_ctrl[1]:+.2f} {F_ctrl[2]:+.2f}]  "
                f"disp=[{disp[0]*100:+.1f} {disp[1]*100:+.1f} {disp[2]*100:+.1f}]cm  "
                f"K*x=[{Kx[0]:+.2f} {Kx[1]:+.2f} {Kx[2]:+.2f}]N  "
                f"v=[{v_cmd[0]:+.3f} {v_cmd[1]:+.3f} {v_cmd[2]:+.3f}]  "
                f"dx=[{dx[0]*1000:+.2f} {dx[1]*1000:+.2f} {dx[2]*1000:+.2f}]mm  "
                f"w=[{omega_cmd[0]:+.3f} {omega_cmd[1]:+.3f} {omega_cmd[2]:+.3f}]"
            )

        time.sleep(0.001)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
