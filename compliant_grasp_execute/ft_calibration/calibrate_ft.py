#!/usr/bin/env python3
"""
ft_calibration/calibrate_ft.py — Wrist F/T Sensor Calibration (BOTH arms)
=========================================================================
Adapted from ft_place_right/admittance_control/ft_calibration_right.py into a
single tool that calibrates EITHER arm via ``--arm {left,right}`` and writes
``ft_calibration_left.json`` / ``ft_calibration_right.json`` next to this file.

This robot is a NEW machine with a different hardware configuration, so BOTH
arms must be (re)calibrated here before the compliant grasp can be trusted.
R_sensor_tcp is AUTO-DETECTED for both arms (24 proper-rotation candidates),
which is more robust than the old hardcoded left-arm value.

The procedure moves the selected arm through several wrist orientations,
records the gravity-loaded wrench at each (operator presses Enter once settled),
and least-squares fits the payload mass, centre-of-mass, 6-axis bias and the
sensor->TCP rotation.

Usage:
  python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm right
  python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm left
  # update only the zero-drift bias of an existing calibration:
  python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm left --drift-only

Prerequisites (same as a normal run):
  * the F/T driver is publishing on /arm_6dof_left and /arm_6dof_right;
  * TF is publishing waist_yaw_link -> {left,right}_tcp_link;
  * remove every external load EXCEPT the permanently-attached end-effector/tool.
"""

import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from rclpy.duration import Duration

try:
    import tf_transformations
except ImportError:
    # Pure-numpy fallback (no tf_transformations / transforms3d dependency).
    class _TfCompat:
        @staticmethod
        def quaternion_matrix(q_xyzw):
            x, y, z, w = q_xyzw
            n = x * x + y * y + z * z + w * w
            T = np.eye(4)
            if n < 1e-12:
                return T
            s = 2.0 / n
            xx, yy, zz = x * x * s, y * y * s, z * z * s
            xy, xz, yz = x * y * s, x * z * s, y * z * s
            wx, wy, wz = w * x * s, w * y * s, w * z * s
            T[0, 0] = 1.0 - (yy + zz); T[0, 1] = xy - wz; T[0, 2] = xz + wy
            T[1, 0] = xy + wz; T[1, 1] = 1.0 - (xx + zz); T[1, 2] = yz - wx
            T[2, 0] = xz - wy; T[2, 1] = yz + wx; T[2, 2] = 1.0 - (xx + yy)
            return T

    tf_transformations = _TfCompat()

from xarm_sdk import XARM_manager, TopicPublisher, ActionCall  # noqa: F401


GRAVITY = np.array([0.0, 0.0, -9.81])
NUM_SETTLE_SAMPLES = 50
NUM_AVG_SAMPLES = 200
_HERE = os.path.dirname(os.path.abspath(__file__))


# ── Per-arm configuration ───────────────────────────────────────────────────
def arm_config(arm: str) -> dict:
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    # RIGHT calibration poses, built around the actual GRASP nominal posture
    # (gripper parallel to the ground), so the payload/bias fit is most accurate
    # exactly in the configuration the arm uses while grasping. The LEFT set is
    # the sagittal mirror (signs of joints 1,2,4,6 flipped):
    #   RIGHT nominal = [0.0, -1.18, 0.0, -1.3,  0.3, -0.13, -0.18]
    #   LEFT  nominal = [0.0,  1.18, 0.0, -1.3, -0.3, -0.13,  0.18]
    #
    # Diversity strategy: vary the joints that rotate the wrist/sensor most
    # (j5 forearm-roll = idx4, j6 wrist-pitch = idx5, j7 wrist-roll = idx6),
    # plus some elbow (idx3) and shoulder-yaw (idx0), to spread the gravity
    # vector across the sensor frame. The tool auto-skips any pose that is
    # unreachable or too similar to a previous one, so a generous list is safe.
    right_poses = [
        # 0: nominal grasp posture (gripper parallel to ground)
        [0.0, -1.18, 0.0, -1.3, 0.3, -0.13, -0.18],
        # --- wrist pitch (j6 / idx5) sweep ---
        [0.0, -1.18, 0.0, -1.3, 0.3, 0.50, -0.18],    # 1: wrist pitched up
        [0.0, -1.18, 0.0, -1.3, 0.3, -0.70, -0.18],   # 2: wrist pitched down
        # --- wrist roll (j7 / idx6) sweep ---
        [0.0, -1.18, 0.0, -1.3, 0.3, -0.13, 1.00],    # 3: wrist rolled +
        [0.0, -1.18, 0.0, -1.3, 0.3, -0.13, -1.40],   # 4: wrist rolled -
        # --- forearm roll (j5 / idx4) sweep (this is the joint set to 0.3 to
        #     make the gripper parallel; sweeping it tilts the sensor a lot) ---
        [0.0, -1.18, 0.0, -1.3, 1.10, -0.13, -0.18],  # 5: forearm rolled +
        [0.0, -1.18, 0.0, -1.3, -0.50, -0.13, -0.18], # 6: forearm rolled -
        # --- elbow (j4 / idx3) variation + roll ---
        [0.0, -1.18, 0.0, -0.95, 0.3, -0.13, -0.70],  # 7: elbow up + roll
        [0.0, -1.18, 0.0, -1.55, 0.3, -0.13, 0.60],   # 8: elbow down + roll
        # --- shoulder yaw (j1 / idx0) + pitch ---
        [-0.30, -1.18, 0.0, -1.3, 0.3, 0.30, -0.18],  # 9: shoulder yaw + pitch up
        [0.30, -1.18, 0.0, -1.3, 0.3, -0.40, -0.18],  # 10: shoulder yaw + pitch down
        # --- combined wrist pitch + roll (max gravity diversity) ---
        [0.0, -1.18, 0.0, -1.3, 0.3, 0.45, 0.85],     # 11: pitch up + roll +
        [0.0, -1.18, 0.0, -1.3, 0.3, -0.55, -0.90],   # 12: pitch down + roll -
        # --- combined forearm roll + wrist pitch / roll ---
        [0.0, -1.18, 0.0, -1.3, 0.90, 0.35, -0.18],   # 13: forearm + wrist pitch
        [0.0, -1.18, 0.0, -1.3, -0.40, -0.13, 0.90],  # 14: forearm - + wrist roll
    ]
    if arm == "right":
        poses = right_poses
    else:
        # Sagittal mirror to the LEFT arm: flip the sign of the mirrored joints
        # (indices 1,2,4,6), matching the home-pose mirror used by the pipeline.
        mirror_idx = (1, 2, 4, 6)
        poses = [
            [(-j if i in mirror_idx else j) for i, j in enumerate(p)]
            for p in right_poses
        ]
    return {
        "tcp_frame": f"{arm}_tcp_link",
        "force_topic": f"/arm_6dof_{arm}",
        "joint_method": f"jointspace_arm_{'L' if arm == 'left' else 'R'}_controller",
        "out_path": os.path.join(_HERE, f"ft_calibration_{arm}.json"),
        "poses": poses,
        "label": f"[{arm.upper()}]",
    }


# ── Helpers ─────────────────────────────────────────────────────────────────
def rotation_from_tf(t_stamped):
    q = t_stamped.transform.rotation
    return tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]


def collect_samples(node, force_sub, n_settle, n_avg, spin_rate=0.002):
    samples = []
    count = 0
    while count < n_settle + n_avg:
        rclpy.spin_once(node, timeout_sec=0.01)
        f = force_sub.force.copy()
        if np.all(f == 0):
            continue
        count += 1
        if count > n_settle:
            samples.append(f)
        time.sleep(spin_rate)
    return np.array(samples)


def solve_payload_and_bias(pose_data):
    """Least-squares payload mass, CoM and 6-axis bias from multi-pose data."""
    n_poses = len(pose_data)

    A_f = np.zeros((3 * n_poses, 4))
    b_f = np.zeros(3 * n_poses)
    for i, (R_world_sensor, F_meas, _) in enumerate(pose_data):
        g_sensor = R_world_sensor.T @ GRAVITY
        A_f[3 * i:3 * i + 3, :3] = np.eye(3)
        A_f[3 * i:3 * i + 3, 3] = g_sensor
        b_f[3 * i:3 * i + 3] = F_meas
    cond_f = np.linalg.cond(A_f)
    x_f, _, _, _ = np.linalg.lstsq(A_f, b_f, rcond=None)
    F_bias = x_f[:3]
    mass = x_f[3]

    A_t = np.zeros((3 * n_poses, 6))
    b_t = np.zeros(3 * n_poses)
    for i, (R_world_sensor, _, T_meas) in enumerate(pose_data):
        g_sensor = R_world_sensor.T @ GRAVITY
        w = mass * g_sensor
        S = np.array([[0, w[2], -w[1]],
                      [-w[2], 0, w[0]],
                      [w[1], -w[0], 0]])
        A_t[3 * i:3 * i + 3, :3] = np.eye(3)
        A_t[3 * i:3 * i + 3, 3:] = S
        b_t[3 * i:3 * i + 3] = T_meas
    cond_t = np.linalg.cond(A_t)
    x_t, _, _, _ = np.linalg.lstsq(A_t, b_t, rcond=None)
    T_bias = x_t[:3]
    com = x_t[3:]

    residuals = []
    for i, (R_world_sensor, F_meas, T_meas) in enumerate(pose_data):
        g_sensor = R_world_sensor.T @ GRAVITY
        F_pred = F_bias + mass * g_sensor
        w = mass * g_sensor
        T_pred = T_bias + np.cross(com, w)
        residuals.append({
            "pose": i,
            "force_err": float(np.linalg.norm(F_meas - F_pred)),
            "torque_err": float(np.linalg.norm(T_meas - T_pred)),
        })

    return {
        "mass": float(mass),
        "com": com.tolist(),
        "F_bias": F_bias.tolist(),
        "T_bias": T_bias.tolist(),
        "cond_force": float(cond_f),
        "cond_torque": float(cond_t),
        "residuals": residuals,
    }


def generate_rotation_candidates():
    candidates = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = signs[i]
            if abs(np.linalg.det(R) - 1.0) < 0.01:
                candidates.append(R)
    return candidates


def rotation_label(R):
    cols = []
    for i in range(3):
        for j in range(3):
            if abs(R[i, j]) > 0.5:
                sign = "+" if R[i, j] > 0 else "-"
                cols.append(f"{'XYZ'[i]}={sign}{'xyz'[j]}")
    return ", ".join(cols)


def find_best_R_sensor_tcp(raw_pose_data):
    candidates = generate_rotation_candidates()
    results = []
    for R_cand in candidates:
        transformed = [(R_tcp @ R_cand, F, T) for (R_tcp, F, T) in raw_pose_data]
        result = solve_payload_and_bias(transformed)
        avg_f_err = np.mean([r["force_err"] for r in result["residuals"]])
        max_f_err = max(r["force_err"] for r in result["residuals"])
        results.append((avg_f_err, max_f_err, R_cand, result))
    results.sort(key=lambda x: x[0])
    best_avg, best_max, best_R, best_result = results[0]
    return best_R, best_result, results


def estimate_zero_drift(node, force_sub, tf_helper, calib_result, R_sensor_tcp,
                        tcp_frame, label, n_samples=500):
    mass = calib_result["mass"]
    com = np.array(calib_result["com"])
    print(f"\n{'=' * 60}")
    print(f"{label} Zero-drift estimation - keep hands off the robot")
    print(f"Collecting {n_samples} samples...")
    print(f"{'=' * 60}")

    F_residuals, T_residuals = [], []
    count, settled = 0, 0
    while count < n_samples:
        rclpy.spin_once(node, timeout_sec=0.01)
        f = force_sub.force.copy()
        if np.all(f == 0):
            continue
        settled += 1
        if settled <= NUM_SETTLE_SAMPLES:
            continue
        t_stamped = tf_helper.lookup_transform("waist_yaw_link", tcp_frame, timeout_sec=0.05)
        if t_stamped is None:
            continue
        R_world_sensor = rotation_from_tf(t_stamped) @ R_sensor_tcp
        g_sensor = R_world_sensor.T @ GRAVITY
        F_gravity = mass * g_sensor
        T_gravity = np.cross(com, F_gravity)
        F_residuals.append(f[:3] - F_gravity)
        T_residuals.append(f[3:] - T_gravity)
        count += 1
        time.sleep(0.002)

    F_bias_new = np.mean(F_residuals, axis=0)
    T_bias_new = np.mean(T_residuals, axis=0)
    print(f"  F_bias = [{F_bias_new[0]:+.4f}, {F_bias_new[1]:+.4f}, {F_bias_new[2]:+.4f}] N")
    print(f"  T_bias = [{T_bias_new[0]:+.5f}, {T_bias_new[1]:+.5f}, {T_bias_new[2]:+.5f}] Nm")
    return F_bias_new.tolist(), T_bias_new.tolist()


def save_calibration(calib, path, label):
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n{label} Calibration saved to: {path}")


def load_calibration(path):
    with open(path, "r") as f:
        return json.load(f)


class ForceSubscriberCalib:
    def __init__(self, node, topic):
        self.force = np.zeros(6)
        node.create_subscription(WrenchStamped, topic, self._cb, 10)

    def _cb(self, msg):
        self.force[0] = msg.wrench.force.x
        self.force[1] = msg.wrench.force.y
        self.force[2] = msg.wrench.force.z
        self.force[3] = msg.wrench.torque.x
        self.force[4] = msg.wrench.torque.y
        self.force[5] = msg.wrench.torque.z


class TFHelperCalib:
    def __init__(self, node):
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

    def lookup_transform(self, target, source, timeout_sec=0.05):
        try:
            return self.buffer.lookup_transform(
                target, source, Time(), Duration(seconds=timeout_sec)
            )
        except TransformException:
            return None


def main():
    parser = argparse.ArgumentParser(description="Wrist F/T Sensor Calibration (both arms)")
    parser.add_argument("--arm", required=True, choices=["left", "right"],
                        help="Which arm to calibrate")
    parser.add_argument("--file", default=None,
                        help="Output calibration JSON path (default: ft_calibration_<arm>.json here)")
    parser.add_argument("--drift-only", action="store_true",
                        help="Only update zero-drift offset (requires existing calib)")
    parser.add_argument("--poses", default=None,
                        help="Comma-separated pose indices to use (e.g. '0,1,2,3')")
    args = parser.parse_args()

    cfg = arm_config(args.arm)
    tcp_frame = cfg["tcp_frame"]
    label = cfg["label"]
    out_path = args.file or cfg["out_path"]

    rclpy.init()
    node = XARM_manager()
    action = ActionCall(node)
    tf_helper = TFHelperCalib(node)
    force_sub = ForceSubscriberCalib(node, cfg["force_topic"])
    joint_move = getattr(action, cfg["joint_method"])

    print(f"{label} Waiting for TF and F/T sensor data on {cfg['force_topic']}...")
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(0.01)

    if args.drift_only:
        if not os.path.exists(out_path):
            print(f"ERROR: calibration file {out_path} not found. Run full calibration first.")
            rclpy.shutdown()
            return
        calib = load_calibration(out_path)
        R_sensor_tcp = np.array(calib["R_sensor_tcp"])
        print(f"{label} Loaded existing calibration from {out_path}")
        F_bias_new, T_bias_new = estimate_zero_drift(
            node, force_sub, tf_helper, calib, R_sensor_tcp, tcp_frame, label
        )
        calib["F_bias"] = F_bias_new
        calib["T_bias"] = T_bias_new
        calib["drift_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_calibration(calib, out_path, label)
        rclpy.shutdown()
        return

    all_poses = cfg["poses"]
    if args.poses:
        pose_indices = [int(x) for x in args.poses.split(",")]
    else:
        pose_indices = list(range(len(all_poses)))
    poses_to_run = [(i, all_poses[i]) for i in pose_indices if i < len(all_poses)]

    print(f"\n{'=' * 60}")
    print(f"{label} F/T Sensor Calibration - {len(poses_to_run)} poses")
    print(f"{'=' * 60}")
    print(f"This moves the {args.arm.upper()} arm through several orientations and")
    print("collects F/T readings at each to identify the payload.\n")
    print("IMPORTANT: Remove any external loads EXCEPT the permanently-attached tool.\n")

    node.xarm_deactivate_all_controller()
    node.hardware_arm_enable(True)

    pose_data = []
    collected_rotations = []

    def orientations_too_similar(R_new, threshold_deg=5.0):
        for R_prev in collected_rotations:
            R_diff = R_prev.T @ R_new
            angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            if np.degrees(angle) < threshold_deg:
                return True
        return False

    for pose_idx, (orig_idx, joints) in enumerate(poses_to_run):
        print(f"\n-- Pose {pose_idx + 1}/{len(poses_to_run)} (config #{orig_idx}) --")
        print(f"   joints = {[f'{j:.3f}' for j in joints]}")

        t_before = tf_helper.lookup_transform("waist_yaw_link", tcp_frame, timeout_sec=1.0)
        R_before = rotation_from_tf(t_before) if t_before is not None else None

        joint_move(joints)
        time.sleep(1.5)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.01)

        t_after = tf_helper.lookup_transform("waist_yaw_link", tcp_frame, timeout_sec=1.0)
        if t_after is None:
            print("   WARNING: TF lookup failed, skipping this pose")
            continue
        R_after = rotation_from_tf(t_after)

        if R_before is not None:
            R_diff = R_before.T @ R_after
            move_angle = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
            if move_angle < 2.0:
                print(f"   WARNING: arm barely moved ({move_angle:.1f}deg) - likely unreachable, SKIPPING")
                continue

        if orientations_too_similar(R_after):
            print("   WARNING: orientation too similar to a previous pose, SKIPPING")
            continue

        input(f"   Press Enter when the {args.arm} arm has settled at pose {pose_idx + 1}...")
        print(f"   Settling + collecting {NUM_AVG_SAMPLES} samples...")
        time.sleep(0.5)
        samples = collect_samples(node, force_sub, NUM_SETTLE_SAMPLES, NUM_AVG_SAMPLES)
        F_mean = np.mean(samples[:, :3], axis=0)
        T_mean = np.mean(samples[:, 3:], axis=0)

        t_stamped = tf_helper.lookup_transform("waist_yaw_link", tcp_frame, timeout_sec=1.0)
        R_waist_tcp = rotation_from_tf(t_stamped) if t_stamped is not None else R_after
        pose_data.append((R_waist_tcp, F_mean, T_mean))
        collected_rotations.append(R_waist_tcp.copy())
        print(f"   F_mean = [{F_mean[0]:+.4f}, {F_mean[1]:+.4f}, {F_mean[2]:+.4f}] N")
        print(f"   T_mean = [{T_mean[0]:+.5f}, {T_mean[1]:+.5f}, {T_mean[2]:+.5f}] Nm")

    print(f"\n   Collected {len(pose_data)} valid poses out of {len(poses_to_run)} attempted.")
    if len(pose_data) < 2:
        print("\nERROR: need at least 2 poses for calibration.")
        rclpy.shutdown()
        return

    print(f"\n{'=' * 60}")
    print(f"{label} Auto-detecting R_sensor_tcp from 24 rotation candidates...")
    print(f"{'=' * 60}")
    R_sensor_tcp, result, all_results = find_best_R_sensor_tcp(pose_data)

    print("\n  Top 5 R_sensor_tcp candidates (by avg force residual):")
    for rank, (avg_err, max_err, R_cand, res) in enumerate(all_results[:5]):
        print(f"    #{rank + 1}: avg_F_err={avg_err:.4f} N, max_F_err={max_err:.4f} N  "
              f"mass={res['mass']:.4f} kg  [{rotation_label(R_cand)}]")
    print(f"\n  *** Best R_sensor_tcp: [{rotation_label(R_sensor_tcp)}] ***")
    print(f"  mass   = {result['mass']:.4f} kg")
    print(f"  com    = [{result['com'][0]:.5f}, {result['com'][1]:.5f}, {result['com'][2]:.5f}] m")
    print(f"  Condition numbers: force={result['cond_force']:.1f}, torque={result['cond_torque']:.1f}")
    if result["cond_force"] > 50 or result["cond_torque"] > 50:
        print("  WARNING: high condition number - add poses with more wrist pitch/roll variation.")

    result["n_poses"] = len(pose_data)
    result["calibration_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["R_sensor_tcp"] = R_sensor_tcp.tolist()
    result["arm"] = args.arm
    save_calibration(result, out_path, label)

    print(f"\n{'=' * 60}")
    print(f"{label} Verification - moving back to first pose...")
    print(f"{'=' * 60}")
    first_joints = poses_to_run[0][1]
    joint_move(first_joints)
    input("Press Enter when the arm has settled...")
    samples = collect_samples(node, force_sub, NUM_SETTLE_SAMPLES, NUM_AVG_SAMPLES)
    F_raw = np.mean(samples[:, :3], axis=0)
    T_raw = np.mean(samples[:, 3:], axis=0)
    t_stamped = tf_helper.lookup_transform("waist_yaw_link", tcp_frame, timeout_sec=1.0)
    if t_stamped is not None:
        R_ws = rotation_from_tf(t_stamped) @ R_sensor_tcp
        g_sensor = R_ws.T @ GRAVITY
        F_gravity = result["mass"] * g_sensor
        T_gravity = np.cross(np.array(result["com"]), F_gravity)
        F_comp = F_raw - np.array(result["F_bias"]) - F_gravity
        T_comp = T_raw - np.array(result["T_bias"]) - T_gravity
        print(f"  Raw      : F=[{F_raw[0]:+.4f} {F_raw[1]:+.4f} {F_raw[2]:+.4f}]")
        print(f"  After cal: F=[{F_comp[0]:+.4f} {F_comp[1]:+.4f} {F_comp[2]:+.4f}]  (ideal: ~0)")

    print(f"\n{label} Done. Calibration saved to: {out_path}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
