"""
Test a trained Lyapunov controller on the Hopper environment.

Usage:
    python -m sd.test_hopper --actor_path models/HopperLyapunov-v0/<run>/checkpoints/<ckpt>/controller_ckpts/<epoch>/actor.keras

    # With Lyapunov value printing:
    python -m sd.test_hopper --actor_path <path>/actor.keras --lyapunov_path <path>/lyapunov.keras

    # With random actions (baseline comparison):
    python -m sd.test_hopper --random_actor
"""

import argparse
import numpy as np
import gymnasium as gym
import keras
from pathlib import Path
import sd.envs  # register environments
from sd.lyapunov import ActionLayer  # register custom Keras layer for model loading
from sd.envs.hopper.constant import constants


def run_test(actor, lyapunov=None, num_steps=1000, render=True, seed=None):
    render_mode = "human" if render else None
    env = gym.make("HopperLyapunov-v0", render_mode=render_mode)

    setpoint = constants["default_setpoint"]

    obs, _ = env.reset(seed=seed)
    print(f"Initial obs: {obs}")
    print(f"Setpoint:    {setpoint}")

    total_reward = 0.0
    for step in range(num_steps):
        if callable(actor) and not hasattr(actor, "predict"):
            action = actor(obs)
        else:
            inputs = {
                "state": np.array([obs], dtype=np.float32),
                "setpoint": np.array([setpoint], dtype=np.float32),
            }
            action = actor(inputs, training=False)[0].numpy()

        if lyapunov is not None and step % 50 == 0:
            v_inputs = {
                "state": np.array([obs], dtype=np.float32),
                "setpoint": np.array([setpoint], dtype=np.float32),
            }
            v_val = lyapunov(v_inputs, training=False)[0].numpy().item()
            error = np.linalg.norm(obs - setpoint)
            print(f"  step {step:4d} | V={v_val:.4f} | error={error:.4f}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step} (terminated={terminated})")
            obs, _ = env.reset()

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained Lyapunov controller on the Hopper"
    )
    parser.add_argument(
        "--actor_path", type=Path, required=False, default=None,
        help="Path to actor.keras",
    )
    parser.add_argument(
        "--lyapunov_path", type=Path, required=False, default=None,
        help="Path to lyapunov.keras (optional, for V value printing)",
    )
    parser.add_argument(
        "--random_actor", action="store_true",
        help="Use random actions instead of trained actor",
    )
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    if args.random_actor:
        env_tmp = gym.make("HopperLyapunov-v0")
        action_space = env_tmp.action_space
        env_tmp.close()
        actor = lambda obs: action_space.sample()
        print("Using random actor")
    else:
        if args.actor_path is None:
            print("Error: provide --actor_path or use --random_actor")
            exit(1)
        actor = keras.models.load_model(str(args.actor_path))
        print(f"Loaded actor from {args.actor_path}")
        actor.summary()

    lyapunov = None
    if args.lyapunov_path:
        lyapunov = keras.models.load_model(str(args.lyapunov_path))
        print(f"Loaded lyapunov from {args.lyapunov_path}")

    run_test(
        actor,
        lyapunov=lyapunov,
        num_steps=args.num_steps,
        render=not args.no_render,
        seed=args.seed,
    )
