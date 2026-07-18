"""
Admittance Controller v3 — 6-DOF (Translation + Rotation)
==========================================================
Based on v2, adds rotation admittance control using the torque channels.

Translation: v    = F   / B_trans   (first-order, same as v2)
Rotation:    omega = tau / B_rot    (first-order, torque → angular velocity)

Both use the hold-position/orientation mechanism to prevent drift after release.
"""

import time
import rclpy
import numpy as np
from geometry_msgs.msg import Pose, WrenchStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from rclpy.duration import Duration

try:
    import tf_transformations
except ImportError:
    # Pure-numpy fallback so this self-contained project needs neither
    # tf_transformations nor transforms3d (numpy is already a dependency).
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

import logging
from xarm_sdk import XARM_manager, TopicPublisher, ActionCall

logger = logging.getLogger(__name__)


# ============================================================
# Quaternion utilities  (all use ROS convention: [x, y, z, w])
# ============================================================
def quat_multiply(q1, q2):
    """Hamilton product of two quaternions in [x,y,z,w] format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_conjugate(q):
    """Inverse of a unit quaternion [x,y,z,w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_to_axis_angle(q):
    """Convert [x,y,z,w] quaternion to (axis, angle). Always shortest path."""
    q = quat_normalize(q)
    if q[3] < 0:
        q = -q
    sin_half = np.linalg.norm(q[:3])
    if sin_half < 1e-10:
        return np.zeros(3), 0.0
    axis = q[:3] / sin_half
    angle = 2.0 * np.arctan2(sin_half, q[3])
    return axis, angle


def axis_angle_to_quat(axis, angle):
    """Convert axis + angle to [x,y,z,w] quaternion."""
    half = angle * 0.5
    s = np.sin(half)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(half)])


def omega_to_delta_quat(omega, dt):
    """Angular velocity (rad/s, 3D) × dt → incremental quaternion [x,y,z,w]."""
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = omega / np.linalg.norm(omega)
    return axis_angle_to_quat(axis, angle)


def quat_from_pose(pose):
    """Extract [x,y,z,w] numpy array from a Pose message."""
    o = pose.orientation
    return np.array([o.x, o.y, o.z, o.w])


def quat_from_transform(t_stamped):
    """Extract [x,y,z,w] numpy array from a TransformStamped."""
    r = t_stamped.transform.rotation
    return np.array([r.x, r.y, r.z, r.w])


# ============================================================
# Force Filter
# ============================================================
class ForceFilter:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None

    def update(self, F):
        if self.prev is None:
            self.prev = F.copy()
            return self.prev.copy()
        self.prev = self.alpha * F + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self):
        self.prev = None


# ============================================================
# Payload (gravity) compensator
# ============================================================
class PayloadCompensator:
    def __init__(self, mass, com):
        self.m = mass
        self.com = np.array(com)
        self.g = np.array([0, 0, -9.81])

    def compensate(self, F_meas, R_world_sensor):
        Fg = R_world_sensor.T @ (self.m * self.g)
        tau_g = np.cross(self.com, Fg)
        F_ext = np.zeros(6)
        F_ext[:3] = F_meas[:3] - Fg
        F_ext[3:] = F_meas[3:] - tau_g
        return F_ext


# ============================================================
# Wrench coordinate transform
# ============================================================
def transform_wrench(F, R, p):
    F_out = np.zeros(6)
    F_out[:3] = R @ F[:3]
    F_out[3:] = R @ F[3:] + np.cross(p, F_out[:3])
    return F_out


# ============================================================
# TF helper
# ============================================================
class TFHelper:
    def __init__(self, node):
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

    def lookup_transform(self, target, source, timeout_sec=0.05):
        try:
            return self.buffer.lookup_transform(
                target, source, Time(), Duration(seconds=timeout_sec)
            )
        except TransformException as ex:
            logger.warning(f"TF lookup {source}->{target} failed: {ex}")
            return None

    @staticmethod
    def rotation_from_transform(t_stamped):
        q = t_stamped.transform.rotation
        return tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

    @staticmethod
    def position_from_transform(t_stamped):
        t = t_stamped.transform.translation
        return np.array([t.x, t.y, t.z])

    @staticmethod
    def pose_from_transform(t_stamped):
        p = Pose()
        p.position.x = t_stamped.transform.translation.x
        p.position.y = t_stamped.transform.translation.y
        p.position.z = t_stamped.transform.translation.z
        p.orientation.x = t_stamped.transform.rotation.x
        p.orientation.y = t_stamped.transform.rotation.y
        p.orientation.z = t_stamped.transform.rotation.z
        p.orientation.w = t_stamped.transform.rotation.w
        return p


# ============================================================
# Auto-calibrator for 6-axis force/torque offset
# ============================================================
class ForceOffsetCalibrator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
        self.samples = []
        self.offset = np.zeros(6)
        self.calibrated = False

    def add_sample(self, F):
        if self.calibrated:
            return
        self.samples.append(F.copy())
        if len(self.samples) >= self.num_samples:
            self.offset = np.mean(self.samples, axis=0)
            self.calibrated = True
            logger.info(f"Force offset calibrated: {self.offset}")

    def remove_offset(self, F):
        return F - self.offset


# ============================================================
# Smooth deadzone (works for both force and torque)
# ============================================================
def smooth_deadzone(F, threshold):
    result = np.zeros_like(F)
    for i in range(len(F)):
        if abs(F[i]) > threshold:
            result[i] = F[i] - np.sign(F[i]) * threshold
    return result


# ============================================================
# Translation admittance (first-order, same as v2)
# ============================================================
class TranslationAdmittance:
    """
    v = F / B_trans

    force_threshold / K_hold: when no human contact detected, a spring
    gently pulls back to the hold position to prevent drift.
    """
    def __init__(self, B, K=None, max_vel=0.5, force_threshold=1.5, K_hold=5.0):
        self.B = np.array(B, dtype=float)
        self.K = np.zeros(3) if K is None else np.array(K, dtype=float)
        self.max_vel = max_vel
        self.force_threshold = force_threshold
        self.K_hold = K_hold

    def update(self, F, x_current, x_hold, dt, human_active):
        if human_active:
            x_err = x_current - x_hold
            xd = (F - self.K * x_err) / self.B
        else:
            x_err = x_current - x_hold
            xd = -self.K_hold * x_err

        xd = np.clip(xd, -self.max_vel, self.max_vel)
        return xd * dt


# ============================================================
# Rotation admittance (first-order)
# ============================================================
class RotationAdmittance:
    """
    omega = tau / B_rot

    Maps external torques directly to angular velocity.
    When no human contact, a spring returns to the hold orientation.

    All quaternions use [x, y, z, w] (ROS convention).
    """
    def __init__(self, B_rot, max_omega=1.0, K_hold_rot=3.0):
        self.B_rot = np.array(B_rot, dtype=float)
        self.max_omega = max_omega
        self.K_hold_rot = K_hold_rot

    def update(self, tau, q_current, q_hold, dt, human_active):
        """
        tau:       3D torque in waist frame [tx, ty, tz]
        q_current: current orientation [x,y,z,w]
        q_hold:    hold orientation    [x,y,z,w]
        dt:        time step
        human_active: True if human is touching the robot

        Returns: target quaternion [x,y,z,w]
        """
        if human_active:
            omega = tau / self.B_rot
        else:
            # Orientation error: rotation from hold to current, expressed in waist frame
            q_err = quat_multiply(q_current, quat_conjugate(q_hold))
            axis, angle = quat_to_axis_angle(q_err)
            omega = -self.K_hold_rot * angle * axis

        omega = np.clip(omega, -self.max_omega, self.max_omega)

        # Apply angular velocity as a world-frame (waist) rotation
        dq = omega_to_delta_quat(omega, dt)
        q_target = quat_multiply(dq, q_current)
        return quat_normalize(q_target)


# ============================================================
# Force subscriber
# ============================================================
class ForceSubscriber:
    def __init__(self, node):
        self.force = np.zeros(6)
        node.create_subscription(
            WrenchStamped, '/arm_6dof_left', self._callback, 10
        )

    def _callback(self, msg):
        self.force[0] = msg.wrench.force.x
        self.force[1] = msg.wrench.force.y
        self.force[2] = msg.wrench.force.z
        self.force[3] = msg.wrench.torque.x
        self.force[4] = msg.wrench.torque.y
        self.force[5] = msg.wrench.torque.z


# ============================================================
# Main
# ============================================================
def main():
    rclpy.init()

    xarm_manager = XARM_manager()
    topic_pub = TopicPublisher(xarm_manager)
    action = ActionCall(xarm_manager)
    tf_helper = TFHelper(xarm_manager)

    # ===== Initialize arm =====
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.hardware_arm_enable(True)
    action.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    input("Press Enter after arm reaches zero position...")
    action.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    input("Press Enter after arm reaches working position...")

    xarm_manager.xarm_activate_controller(['endpose_single_arm_qp_L_controller'])
    xarm_manager.get_logger().info("Controller activated, starting 6-DOF admittance control...")

    # ===== Get initial pose =====
    t_stamped_init = tf_helper.lookup_transform("waist_yaw_link", "left_tcp_link", timeout_sec=1.0)
    if t_stamped_init is None:
        xarm_manager.get_logger().error("Cannot get initial pose from TF")
        rclpy.shutdown()
        return

    init_position = TFHelper.position_from_transform(t_stamped_init)
    init_quat = quat_from_transform(t_stamped_init)

    target_pose = Pose()

    # ===== Control modules =====
    force_sub = ForceSubscriber(xarm_manager)
    force_filter = ForceFilter(alpha=0.6)
    payload = PayloadCompensator(mass=0.09, com=[0, 0, 0.05])

    # --- Translation admittance ---
    adm_trans = TranslationAdmittance(
        # B=[2.0, 3.0, 3.0],
        B=[0.1, 0.1, 0.1],
        K=[0.0, 0.0, 0.0],
        max_vel=1.0,
        force_threshold=1.5,
        K_hold=5.0,
    )

    # --- Rotation admittance ---
    # B_rot: Nm·s/rad — lower = easier to twist.
    #   0.3 → 1 Nm gives ~3.3 rad/s (fast, light feel)
    #   1.0 → 1 Nm gives  1.0 rad/s (moderate)
    #   3.0 → 1 Nm gives ~0.33 rad/s (stiff)
    adm_rot = RotationAdmittance(
        B_rot=[0.5, 0.5, 0.5],
        max_omega=1.5,
        K_hold_rot=3.0,
    )

    # --- Thresholds ---
    force_deadzone = 0.5       # N
    torque_deadzone = 0.05     # Nm
    force_threshold = 0.5      # N  — above this → human dragging
    torque_threshold = 0.15    # Nm — above this → human twisting

    # Sensor-to-TCP rotation (180° around y-axis)
    R_sensor_tcp = np.eye(3)
    R_sensor_tcp[0, 0] = -1
    R_sensor_tcp[2, 2] = -1
    p_sensor_tcp = np.array([0.0, 0.0, 0.0])

    calibrator = ForceOffsetCalibrator(num_samples=200)

    # Hold pose — updated while human is active, anchored on release
    hold_position = init_position.copy()
    hold_quat = init_quat.copy()

    # ===== Control loop =====
    last_time = time.time()
    loop_count = 0
    start_time = time.time()

    xarm_manager.get_logger().info(
        "Calibrating force offset — keep hands off the robot for ~2 seconds..."
    )

    while time.time() - start_time < 2000:
        rclpy.spin_once(xarm_manager, timeout_sec=0.0)

        # --- Measure actual dt ---
        now = time.time()
        dt = now - last_time
        last_time = now
        dt = np.clip(dt, 0.001, 0.1)

        # --- 1. Read & filter ---
        F_raw = force_sub.force.copy()
        F_filtered = force_filter.update(F_raw)

        # --- 2. TF lookup ---
        t_stamped = tf_helper.lookup_transform(
            "waist_yaw_link", "left_tcp_link", timeout_sec=0.05
        )
        if t_stamped is None:
            continue

        R_waist_sensor = TFHelper.rotation_from_transform(t_stamped) @ R_sensor_tcp
        x_current = TFHelper.position_from_transform(t_stamped)
        q_current = quat_from_transform(t_stamped)

        # --- 3. Gravity compensation ---
        F_ext = payload.compensate(F_filtered, R_waist_sensor)

        # --- 4. Transform to waist frame (both force and torque) ---
        F_ctrl = transform_wrench(F_ext, R_waist_sensor, p_sensor_tcp)

        # --- 5. Calibrate or remove offset ---
        if not calibrator.calibrated:
            calibrator.add_sample(F_ctrl)
            if calibrator.calibrated:
                xarm_manager.get_logger().info(
                    "Calibration done. You can now drag the robot."
                )
            continue

        F_ctrl = calibrator.remove_offset(F_ctrl)

        # --- 6. Smooth deadzone (force and torque separately) ---
        F_ctrl[:3] = smooth_deadzone(F_ctrl[:3], force_deadzone)
        F_ctrl[3:] = smooth_deadzone(F_ctrl[3:], torque_deadzone)

        # --- 7. Detect human contact (force OR torque) ---
        force_active = np.max(np.abs(F_ctrl[:3])) > force_threshold
        torque_active = np.max(np.abs(F_ctrl[3:])) > torque_threshold
        human_active = force_active or torque_active

        if human_active:
            hold_position = x_current.copy()
            hold_quat = q_current.copy()

        # --- 8. Translation admittance ---
        dx = adm_trans.update(F_ctrl[:3], x_current, hold_position, dt, human_active)

        # --- 9. Rotation admittance ---
        q_target = adm_rot.update(F_ctrl[3:], q_current, hold_quat, dt, human_active)

        # --- 10. Build target pose ---
        target_pose.position.x = x_current[0] + dx[0]
        target_pose.position.y = x_current[1] + dx[1]
        target_pose.position.z = x_current[2] + dx[2]
        target_pose.orientation.x = q_target[0]
        target_pose.orientation.y = q_target[1]
        target_pose.orientation.z = q_target[2]
        target_pose.orientation.w = q_target[3]

        # --- 11. Publish ---
        topic_pub.publish_endposetarget_L(target_pose, from_frame="waist_yaw_link")

        # --- Diagnostics ---
        loop_count += 1
        if loop_count % 500 == 0:
            mode = "DRAG" if human_active else "HOLD"
            axis_cur, angle_cur = quat_to_axis_angle(
                quat_multiply(q_current, quat_conjugate(init_quat))
            )
            print(
                f"[loop {loop_count}] dt={dt*1000:.1f}ms  "
                f"F=[{F_ctrl[0]:+.2f} {F_ctrl[1]:+.2f} {F_ctrl[2]:+.2f}]  "
                f"T=[{F_ctrl[3]:+.3f} {F_ctrl[4]:+.3f} {F_ctrl[5]:+.3f}]  "
                f"dx=[{dx[0]*1000:+.2f} {dx[1]*1000:+.2f} {dx[2]*1000:+.2f}]mm  "
                f"rot={np.degrees(angle_cur):+.1f}deg  "
                f"[{mode}]"
            )

        time.sleep(0.001)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
