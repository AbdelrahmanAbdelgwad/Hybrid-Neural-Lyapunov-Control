"""
Hopper environment wrapped for the Lyapunov control framework.

Wraps Gymnasium's Hopper-v4 (MuJoCo) with bounded observation space
and a setpoint definition suitable for Lyapunov stabilization.

State (11D) — same order as Hopper-v4:
    [z_pos, angle_torso, thigh_angle, leg_angle, foot_angle,
     x_vel, z_vel, ang_vel_torso, thigh_vel, leg_vel, foot_vel]

Setpoint (11D) — full state target:
    Default: [1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (upright, stationary)

Action (3D):
    Joint torques for thigh, leg, foot in [-1, 1].

Requires: pip install gymnasium[mujoco]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from sd.envs.modelable_env import ModelableEnv
from sd import fpl
from sd.envs.hopper.constant import constants


class HopperLyapunovEnv(ModelableEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

    def __init__(self, render_mode=None, **kwargs):
        # terminate_when_unhealthy=False so we collect data from diverse states
        self.base_env = gym.make(
            "Hopper-v4",
            render_mode=render_mode,
            terminate_when_unhealthy=False,
            **kwargs,
        )

        max_vel = constants["max_vel"]
        obs_low = np.array([
            constants["min_z_pos"],     # z_pos
            -constants["max_angle"],    # angle_torso
            -constants["max_joint_angle"],  # thigh
            -constants["max_joint_angle"],  # leg
            -constants["max_foot_angle"],   # foot
            -max_vel, -max_vel,         # x_vel, z_vel
            -max_vel, -max_vel,         # ang_vel_torso, thigh_vel
            -max_vel, -max_vel,         # leg_vel, foot_vel
        ], dtype=np.float32)

        obs_high = np.array([
            constants["max_z_pos"],
            constants["max_angle"],
            constants["max_joint_angle"],
            constants["max_joint_angle"],
            constants["max_foot_angle"],
            max_vel, max_vel,
            max_vel, max_vel,
            max_vel, max_vel,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = self.base_env.action_space

        # Setpoint = full state (for stabilization, target is a full state config)
        self.setpoint_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def _clip_obs(self, obs):
        return np.clip(
            obs, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        return self._clip_obs(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        return self._clip_obs(obs), info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()

    def default_setpoint(self):
        return constants["default_setpoint"].copy()

    @staticmethod
    @tf.function
    def closeness_fpl(obs1, obs2):
        abs_diff = tf.abs(obs1 - obs2)
        return 1.0 / fpl.p_mean((1.0 + abs_diff), 1.0)


if __name__ == "__main__":
    env = HopperLyapunovEnv(render_mode="human")
    obs, _ = env.reset()
    print("obs shape:", obs.shape)
    print("obs:", obs)
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("setpoint_space:", env.setpoint_space)
    print("default_setpoint:", env.default_setpoint())

    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
