"""
ft_calibration_right.py — Force/Torque Sensor Calibration (RIGHT arm)
=====================================================================
Right-arm version of ft_calibration.py with AUTO-DETECTION of R_sensor_tcp.

Unlike the left-arm version which uses a hardcoded R_sensor_tcp = diag(-1,1,-1),
this version collects raw data and then tries all 24 proper rotation matrices
to find the R_sensor_tcp that produces the lowest calibration residuals.

Differences from left-arm:
  - TF frame:    right_tcp_link  (was left_tcp_link)
  - Joint action: jointspace_arm_R_controller  (was _L_)
  - F/T topic:   /arm_6dof_right  (was /arm_6dof_left)
  - Calibration poses: j2 mirrored  (-1.18 instead of +1.18)
  - Output file:  ft_calibration_right.json
  - R_sensor_tcp: auto-detected (not hardcoded)

Usage:
  python ft_calibration_right.py                              # interactive calibration
  python ft_calibration_right.py --file calib_right.json      # custom output path
  python ft_calibration_right.py --drift-only                 # update bias only
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
    from transforms3d.quaternions import quat2mat

    class _TfCompat:
        @staticmethod
        def quaternion_matrix(q_xyzw):
            x, y, z, w = q_xyzw
            R = quat2mat([w, x, y, z])
            T = np.eye(4)
            T[:3, :3] = R
            return T

    tf_transformations = _TfCompat()

from xarm_sdk import XARM_manager, TopicPublisher, ActionCall


# ── Constants ─────────────────────────────────────────────────────────────
GRAVITY = np.array([0.0, 0.0, -9.81])
DEFAULT_CALIB_PATH = os.path.join(os.path.dirname(__file__), "ft_calibration_right.json")
NUM_SETTLE_SAMPLES = 50
NUM_AVG_SAMPLES    = 200

TCP_FRAME = "right_tcp_link"


# ── Helpers ───────────────────────────────────────────────────────────────
def rotation_from_tf(t_stamped):
    q = t_stamped.transform.rotation
    return tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]


def collect_samples(node, force_sub, n_settle, n_avg, spin_rate=0.002):
    """Collect n_avg wrench samples after discarding n_settle."""
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
    """
    Solve for payload mass, CoM, and 6-axis bias from multi-pose data.
    Identical math to the left-arm version.
    """
    n_poses = len(pose_data)

    # Stage 1: mass and force bias
    A_f = np.zeros((3 * n_poses, 4))
    b_f = np.zeros(3 * n_poses)

    for i, (R_world_sensor, F_meas, _) in enumerate(pose_data):
        g_sensor = R_world_sensor.T @ GRAVITY
        A_f[3*i:3*i+3, :3] = np.eye(3)
        A_f[3*i:3*i+3, 3]  = g_sensor
        b_f[3*i:3*i+3]     = F_meas

    cond_f = np.linalg.cond(A_f)
    x_f, res_f, _, _ = np.linalg.lstsq(A_f, b_f, rcond=None)
    F_bias = x_f[:3]
    mass   = x_f[3]

    # Stage 2: CoM and torque bias
    A_t = np.zeros((3 * n_poses, 6))
    b_t = np.zeros(3 * n_poses)

    for i, (R_world_sensor, _, T_meas) in enumerate(pose_data):
        g_sensor = R_world_sensor.T @ GRAVITY
        w = mass * g_sensor
        S = np.array([[ 0,    w[2], -w[1]],
                      [-w[2], 0,     w[0]],
                      [ w[1],-w[0],  0   ]])
        A_t[3*i:3*i+3, :3] = np.eye(3)
        A_t[3*i:3*i+3, 3:] = S
        b_t[3*i:3*i+3]     = T_meas

    cond_t = np.linalg.cond(A_t)
    x_t, res_t, _, _ = np.linalg.lstsq(A_t, b_t, rcond=None)
    T_bias = x_t[:3]
    com    = x_t[3:]

    # Residuals
    residuals = []
    for i, (R_world_sensor, F_meas, T_meas) in enumerate(pose_data):
        g_sensor = R_world_sensor.T @ GRAVITY
        F_pred = F_bias + mass * g_sensor
        w = mass * g_sensor
        T_pred = T_bias + np.cross(com, w)
        residuals.append({
            "pose": i,
            "force_err":  np.linalg.norm(F_meas - F_pred),
            "torque_err": np.linalg.norm(T_meas - T_pred),
        })

    return {
        "mass":    float(mass),
        "com":     com.tolist(),
        "F_bias":  F_bias.tolist(),
        "T_bias":  T_bias.tolist(),
        "cond_force":  float(cond_f),
        "cond_torque": float(cond_t),
        "residuals": residuals,
    }


def generate_rotation_candidates():
    """Generate all 24 proper rotation matrices (octahedral symmetry group)."""
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
    """Human-readable label for an axis-aligned rotation matrix."""
    cols = []
    for i in range(3):
        for j in range(3):
            if abs(R[i, j]) > 0.5:
                sign = "+" if R[i, j] > 0 else "-"
                cols.append(f"{'XYZ'[i]}={sign}{'xyz'[j]}")
    return ", ".join(cols)


def find_best_R_sensor_tcp(raw_pose_data):
    """
    Try all 24 proper rotation candidates as R_sensor_tcp.
    Returns (best_R, best_result, all_results_sorted).
    raw_pose_data: list of (R_waist_tcp, F_mean, T_mean).
    """
    candidates = generate_rotation_candidates()
    results = []

    for R_cand in candidates:
        transformed = [(R_tcp @ R_cand, F, T)
                       for (R_tcp, F, T) in raw_pose_data]
        result = solve_payload_and_bias(transformed)
        avg_f_err = np.mean([r["force_err"] for r in result["residuals"]])
        max_f_err = max(r["force_err"] for r in result["residuals"])
        results.append((avg_f_err, max_f_err, R_cand, result))

    results.sort(key=lambda x: x[0])
    best_avg, best_max, best_R, best_result = results[0]
    return best_R, best_result, results


# ── Predefined calibration poses (RIGHT arm) ─────────────────────────────
# Mirrored from left arm: j2 sign flipped (-1.18 instead of +1.18).
# j1 sign also flipped where non-zero for symmetric shoulder rotation.
# Wrist joints (j6, j7) stay the same — they are local to the wrist.
CALIBRATION_POSES = [
    # Pose 0: nominal working pose
    [0.0, -1.18, 0.0, -1.3,  -1.4, -0.13, 0.18],
    # Pose 1: wrist pitched up ~40°
    [0.0, -1.18, 0.0, -1.3,  -1.4,  0.55,  0.18],
    # Pose 2: wrist pitched down ~25°
    [0.0, -1.18, 0.0, -1.3,  -1.4, -0.40,  0.18],
    # Pose 3: wrist rolled +60°
    [0.0, -1.18, 0.0, -1.3,  -1.4, -0.13,  1.20],
    # Pose 4: wrist rolled -90°
    [0.0, -1.18, 0.0, -1.3,  -1.4, -0.13, -1.20],
    # Pose 5: shoulder rotated + slight pitch (j1 mirrored)
    [-0.3, -1.18, 0.0, -1.3,  -1.4,  0.30,  0.18],
    # Pose 6: elbow varied + opposite roll
    [0.0, -1.18, 0.0, -1.0,  -1.4, -0.13, -0.80],
    # Pose 7: combined — pitch + roll for maximum gravity diversity
    [0.0, -1.18, 0.0, -1.3,  -1.4,  0.40,  0.90],
]


# ── Zero-drift estimation ────────────────────────────────────────────────
def estimate_zero_drift(node, force_sub, tf_helper, calib_result,
                        R_sensor_tcp, n_samples=500):
    """
    With the payload model known, measure the current offset residual.
    Returns updated F_bias and T_bias.
    """
    mass = calib_result["mass"]
    com  = np.array(calib_result["com"])

    print(f"\n{'='*60}")
    print("[RIGHT] Zero-drift estimation — keep hands off the robot")
    print(f"Collecting {n_samples} samples...")
    print(f"{'='*60}")

    F_residuals = []
    T_residuals = []
    count = 0
    settled = 0

    while count < n_samples:
        rclpy.spin_once(node, timeout_sec=0.01)
        f = force_sub.force.copy()
        if np.all(f == 0):
            continue

        settled += 1
        if settled <= NUM_SETTLE_SAMPLES:
            continue

        t_stamped = tf_helper.lookup_transform(
            "waist_yaw_link", TCP_FRAME, timeout_sec=0.05
        )
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

    F_std = np.std(F_residuals, axis=0)
    T_std = np.std(T_residuals, axis=0)

    print(f"  F_bias = [{F_bias_new[0]:+.4f}, {F_bias_new[1]:+.4f}, {F_bias_new[2]:+.4f}] N")
    print(f"  T_bias = [{T_bias_new[0]:+.5f}, {T_bias_new[1]:+.5f}, {T_bias_new[2]:+.5f}] Nm")
    print(f"  F_std  = [{F_std[0]:.4f}, {F_std[1]:.4f}, {F_std[2]:.4f}] N")
    print(f"  T_std  = [{T_std[0]:.5f}, {T_std[1]:.5f}, {T_std[2]:.5f}] Nm")

    return F_bias_new.tolist(), T_bias_new.tolist()


# ── Save / Load ──────────────────────────────────────────────────────────
def save_calibration(calib, path):
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n[RIGHT] Calibration saved to: {path}")


def load_calibration(path):
    with open(path, "r") as f:
        return json.load(f)


# ── Force subscriber (right arm) ─────────────────────────────────────────
class ForceSubscriberCalib:
    def __init__(self, node, topic="/arm_6dof_right"):
        self.force = np.zeros(6)
        node.create_subscription(WrenchStamped, topic, self._cb, 10)

    def _cb(self, msg):
        self.force[0] = msg.wrench.force.x
        self.force[1] = msg.wrench.force.y
        self.force[2] = msg.wrench.force.z
        self.force[3] = msg.wrench.torque.x
        self.force[4] = msg.wrench.torque.y
        self.force[5] = msg.wrench.torque.z


# ── TF helper (standalone) ───────────────────────────────────────────────
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


# ── Main calibration routine ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="F/T Sensor Calibration — RIGHT arm"
    )
    parser.add_argument("--file", default=DEFAULT_CALIB_PATH,
                        help="Output calibration JSON path")
    parser.add_argument("--drift-only", action="store_true",
                        help="Only update zero-drift offset (requires existing calib)")
    parser.add_argument("--poses", default=None,
                        help="Comma-separated pose indices to use (e.g., '0,1,2,3')")
    args = parser.parse_args()

    rclpy.init()
    node = XARM_manager()
    action = ActionCall(node)
    tf_helper = TFHelperCalib(node)
    force_sub = ForceSubscriberCalib(node)

    print("[RIGHT] Waiting for TF and F/T sensor data...")
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(0.01)

    if args.drift_only:
        if not os.path.exists(args.file):
            print(f"ERROR: calibration file {args.file} not found. "
                  f"Run full calibration first.")
            rclpy.shutdown()
            return
        calib = load_calibration(args.file)
        R_sensor_tcp = np.array(calib["R_sensor_tcp"])
        print(f"[RIGHT] Loaded existing calibration from {args.file}")
        print(f"  mass = {calib['mass']:.4f} kg")
        print(f"  com  = {calib['com']}")
        print(f"  R_sensor_tcp = {rotation_label(R_sensor_tcp)}")

        F_bias_new, T_bias_new = estimate_zero_drift(
            node, force_sub, tf_helper, calib, R_sensor_tcp
        )
        calib["F_bias"] = F_bias_new
        calib["T_bias"] = T_bias_new
        calib["drift_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_calibration(calib, args.file)
        rclpy.shutdown()
        return

    # ── Full calibration ──────────────────────────────────────────────
    if args.poses:
        pose_indices = [int(x) for x in args.poses.split(",")]
    else:
        pose_indices = list(range(len(CALIBRATION_POSES)))

    poses_to_run = [(i, CALIBRATION_POSES[i]) for i in pose_indices
                    if i < len(CALIBRATION_POSES)]

    print(f"\n{'='*60}")
    print(f"[RIGHT] F/T Sensor Calibration — {len(poses_to_run)} poses")
    print(f"{'='*60}")
    print("This will move the RIGHT arm to several orientations and")
    print("collect F/T readings at each to identify the payload.\n")
    print("IMPORTANT: Remove any external loads EXCEPT the")
    print("end-effector/tool that is permanently attached.\n")

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
        print(f"\n── Pose {pose_idx+1}/{len(poses_to_run)} "
              f"(config #{orig_idx}) ──")
        print(f"   joints = {[f'{j:.3f}' for j in joints]}")

        t_before = tf_helper.lookup_transform(
            "waist_yaw_link", TCP_FRAME, timeout_sec=1.0
        )
        R_before = rotation_from_tf(t_before) if t_before is not None else None

        action.jointspace_arm_R_controller(joints)

        time.sleep(1.5)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.01)

        t_after = tf_helper.lookup_transform(
            "waist_yaw_link", TCP_FRAME, timeout_sec=1.0
        )
        if t_after is None:
            print("   WARNING: TF lookup failed, skipping this pose")
            continue

        R_after = rotation_from_tf(t_after)

        if R_before is not None:
            R_diff = R_before.T @ R_after
            move_angle = np.degrees(
                np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            )
            if move_angle < 2.0:
                print(f"   WARNING: arm barely moved ({move_angle:.1f}°) — "
                      f"pose likely unreachable, SKIPPING")
                continue

        if orientations_too_similar(R_after):
            print(f"   WARNING: orientation too similar to a previous pose, SKIPPING")
            continue

        input(f"   Press Enter when right arm has settled at pose {pose_idx+1}...")

        print(f"   Settling ({NUM_SETTLE_SAMPLES} samples)...")
        time.sleep(0.5)

        print(f"   Collecting {NUM_AVG_SAMPLES} samples...")
        samples = collect_samples(
            node, force_sub, NUM_SETTLE_SAMPLES, NUM_AVG_SAMPLES
        )
        F_mean = np.mean(samples[:, :3], axis=0)
        T_mean = np.mean(samples[:, 3:], axis=0)
        F_std  = np.std(samples[:, :3], axis=0)
        T_std  = np.std(samples[:, 3:], axis=0)

        t_stamped = tf_helper.lookup_transform(
            "waist_yaw_link", TCP_FRAME, timeout_sec=1.0
        )
        R_waist_tcp = rotation_from_tf(t_stamped) if t_stamped is not None else R_after

        pose_data.append((R_waist_tcp, F_mean, T_mean))
        collected_rotations.append(R_waist_tcp.copy())

        print(f"   F_mean = [{F_mean[0]:+.4f}, {F_mean[1]:+.4f}, {F_mean[2]:+.4f}] N")
        print(f"   T_mean = [{T_mean[0]:+.5f}, {T_mean[1]:+.5f}, {T_mean[2]:+.5f}] Nm")
        print(f"   F_std  = [{F_std[0]:.4f}, {F_std[1]:.4f}, {F_std[2]:.4f}] N")
        print(f"   T_std  = [{T_std[0]:.5f}, {T_std[1]:.5f}, {T_std[2]:.5f}] Nm")

    print(f"\n   Collected {len(pose_data)} valid poses out of {len(poses_to_run)} attempted.")

    if len(pose_data) < 2:
        print("\nERROR: need at least 2 poses for calibration.")
        rclpy.shutdown()
        return

    # ── Solve: auto-detect R_sensor_tcp ─────────────────────────────
    print(f"\n{'='*60}")
    print("[RIGHT] Auto-detecting R_sensor_tcp from 24 rotation candidates...")
    print(f"{'='*60}")

    R_sensor_tcp, result, all_results = find_best_R_sensor_tcp(pose_data)

    print(f"\n  Top 5 R_sensor_tcp candidates (by avg force residual):")
    for rank, (avg_err, max_err, R_cand, res) in enumerate(all_results[:5]):
        label = rotation_label(R_cand)
        print(f"    #{rank+1}: avg_F_err={avg_err:.4f} N, max_F_err={max_err:.4f} N  "
              f"mass={res['mass']:.4f} kg  [{label}]")

    print(f"\n  *** Best R_sensor_tcp: [{rotation_label(R_sensor_tcp)}] ***")
    print(f"  R_sensor_tcp =")
    for row in R_sensor_tcp:
        print(f"    [{row[0]:+.0f}  {row[1]:+.0f}  {row[2]:+.0f}]")

    print(f"\n  mass   = {result['mass']:.4f} kg")
    print(f"  com    = [{result['com'][0]:.5f}, {result['com'][1]:.5f}, {result['com'][2]:.5f}] m")
    print(f"  F_bias = [{result['F_bias'][0]:+.4f}, {result['F_bias'][1]:+.4f}, {result['F_bias'][2]:+.4f}] N")
    print(f"  T_bias = [{result['T_bias'][0]:+.5f}, {result['T_bias'][1]:+.5f}, {result['T_bias'][2]:+.5f}] Nm")

    cond_f = result["cond_force"]
    cond_t = result["cond_torque"]
    print(f"\n  Condition numbers: force={cond_f:.1f}, torque={cond_t:.1f}")
    if cond_f > 50 or cond_t > 50:
        print("  WARNING: high condition number — poses may lack orientation diversity.")
        print("  Try adding poses with more wrist pitch/roll variation.")

    print("\n  Per-pose residuals:")
    for r in result["residuals"]:
        print(f"    Pose {r['pose']}: F_err={r['force_err']:.4f} N, "
              f"T_err={r['torque_err']:.5f} Nm")

    result["n_poses"] = len(pose_data)
    result["calibration_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["R_sensor_tcp"] = R_sensor_tcp.tolist()

    save_calibration(result, args.file)

    # ── Verify: move back to first pose and check ─────────────────────
    print(f"\n{'='*60}")
    print("[RIGHT] Verification — moving back to first pose...")
    print(f"{'='*60}")
    first_joints = poses_to_run[0][1]
    action.jointspace_arm_R_controller(first_joints)
    input("Press Enter when right arm has settled...")

    samples = collect_samples(node, force_sub, NUM_SETTLE_SAMPLES, NUM_AVG_SAMPLES)
    F_raw = np.mean(samples[:, :3], axis=0)
    T_raw = np.mean(samples[:, 3:], axis=0)

    t_stamped = tf_helper.lookup_transform(
        "waist_yaw_link", TCP_FRAME, timeout_sec=1.0
    )
    if t_stamped is not None:
        R_ws = rotation_from_tf(t_stamped) @ R_sensor_tcp
        g_sensor = R_ws.T @ GRAVITY
        F_gravity = result["mass"] * g_sensor
        T_gravity = np.cross(np.array(result["com"]), F_gravity)

        F_compensated = F_raw - np.array(result["F_bias"]) - F_gravity
        T_compensated = T_raw - np.array(result["T_bias"]) - T_gravity

        print(f"  Raw      : F=[{F_raw[0]:+.4f} {F_raw[1]:+.4f} {F_raw[2]:+.4f}]  "
              f"T=[{T_raw[0]:+.5f} {T_raw[1]:+.5f} {T_raw[2]:+.5f}]")
        print(f"  After cal: F=[{F_compensated[0]:+.4f} {F_compensated[1]:+.4f} {F_compensated[2]:+.4f}]  "
              f"T=[{T_compensated[0]:+.5f} {T_compensated[1]:+.5f} {T_compensated[2]:+.5f}]")
        print(f"  (ideal: all near zero)")

    print(f"\n[RIGHT] Done. Calibration saved to: {args.file}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
