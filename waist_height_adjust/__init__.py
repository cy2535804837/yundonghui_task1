"""
Adjust the robot's waist (torso) height to a commanded value.

The tianyi2 body is a 4-DOF chain [first_leg_pitch, second_leg_pitch,
waist_pitch, waist_yaw] whose Cartesian endpose is controlled as
``[x, z, pitch, yaw]`` (SDK ``ActionCall.endpose_body_controller``, zero pose
``[0.05, 0.68, 0.0, 0.0]``). This tool drives the ``z`` component (the waist
height, in metres) to a target while keeping x / pitch / yaw at safe defaults.
"""
