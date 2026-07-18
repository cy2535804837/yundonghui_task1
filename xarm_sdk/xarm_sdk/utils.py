import numpy as np

def quaternion_to_euler(x, y, z, w):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    欧拉角单位为弧度
    欧拉角顺序: roll (x轴), pitch (y轴), yaw (z轴)
    """
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def euler_to_quaternion(roll_x, pitch_y, yaw_z):
    """
    将欧拉角转换为四元数
    四元数顺序: x, y, z, w
    """
    w = np.cos(roll_x / 2) * np.cos(pitch_y / 2) * np.cos(yaw_z / 2) + np.sin(roll_x / 2) * np.sin(pitch_y / 2) * np.sin(yaw_z / 2)
    x = np.sin(roll_x / 2) * np.cos(pitch_y / 2) * np.cos(yaw_z / 2) - np.cos(roll_x / 2) * np.sin(pitch_y / 2) * np.sin(yaw_z / 2)
    y = np.cos(roll_x / 2) * np.sin(pitch_y / 2) * np.cos(yaw_z / 2) + np.sin(roll_x / 2) * np.cos(pitch_y / 2) * np.sin(yaw_z / 2)
    z = np.cos(roll_x / 2) * np.cos(pitch_y / 2) * np.sin(yaw_z / 2) - np.sin(roll_x / 2) * np.sin(pitch_y / 2) * np.cos(yaw_z / 2)
    return x, y, z, w


