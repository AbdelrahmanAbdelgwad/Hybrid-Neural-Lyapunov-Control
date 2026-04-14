"""
Modeled Hopper: uses a learned neural network dynamics model instead of MuJoCo.
Used for Lyapunov training (differentiable forward model) and testing.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from sd import utils
from sd.envs.hopper.hopper_env import HopperLyapunovEnv


class ModeledHopperEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, model_path=None, render_mode=None, **kwargs):
        self.env = HopperLyapunovEnv(render_mode=render_mode, **kwargs)
        if model_path is not None:
            model_path = Path(model_path)
            self.dynamic_model = utils.load_checkpoint(model_path / "model.keras")
        else:
            self.dynamic_model = None

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.setpoint_space = self.env.setpoint_space
        self.state = None

    def run_nn(self, obs, action):
        latent_shape = self.dynamic_model.input["latent"].shape
        if latent_shape[1] == 0:
            latent = np.array([[]], dtype=np.float32)
        else:
            latent = np.random.normal(size=(1,) + tuple(latent_shape[1:])).astype(
                np.float32
            )
        inputs = {
            "state": np.array([obs], dtype=np.float32),
            "action": np.array([action], dtype=np.float32),
            "latent": latent,
        }
        return self.dynamic_model(inputs, training=False)[0].numpy()

    def step(self, action):
        new_state = self.run_nn(self.state, action)
        self.state = np.clip(
            new_state,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)
        return self.state, 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.state = obs
        return obs, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def default_setpoint(self):
        return self.env.default_setpoint()
