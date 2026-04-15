import numpy as np

constants = {
    # Observation bounds
    'max_vel': 10.0,
    'min_z_pos': 0.0,
    'max_z_pos': 2.5,
    'max_angle': 1.0,
    'max_joint_angle': 1.5,
    'max_foot_angle': 1.5,

    # Default setpoint: upright standing, some set forward velocity 
    # state is [z_pos, angle, thigh, leg, foot, x_vel, z_vel, ang_vel, thigh_vel, leg_vel, foot_vel]

    'default_setpoint': np.array(
        [1.25, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32
    ),

    # Normalization ranges for closeness metric
    # [z_pos, angle, thigh, leg, foot, x_vel, z_vel, ang_vel, thigh_vel, leg_vel, foot_vel]
    'closeness_ranges': np.array(
        [2.5, 2.0, 3.0, 3.0, 3.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        dtype=np.float32
    ),
}
