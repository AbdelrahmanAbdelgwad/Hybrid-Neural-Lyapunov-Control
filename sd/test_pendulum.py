"""
Test a trained Lyapunov controller on the Pendulum environment.

Usage:
    # Auto-discover latest controller checkpoint:
    python -m sd.test_pendulum

    # Specify paths explicitly:
    python -m sd.test_pendulum --actor_path <path>/actor.keras --lyapunov_path <path>/lyapunov.keras

    # Plot Lyapunov function (phase portrait):
    python -m sd.test_pendulum --plot

    # Random actor baseline:
    python -m sd.test_pendulum --random_actor
"""

import argparse
import numpy as np
import gymnasium as gym
import keras
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sd.envs  # register environments
from sd.lyapunov import ActionLayer  # register custom Keras layer for model loading
from sd.envs.Pendulum.PendulumKerasModel import PendulumDifferenceEq  # register for model loading


# Pendulum upright setpoint: cos(0)=1, sin(0)=0, angular_vel=0
DEFAULT_SETPOINT = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def find_latest_controller(dynamics_path=None):
    """Auto-discover the latest controller checkpoint.

    Searches for controller_ckpts/<epoch>/ under the dynamics model checkpoint
    directory, returning paths to the highest-epoch actor.keras and lyapunov.keras.

    Args:
        dynamics_path: Path to model.keras for the dynamics model. If None,
                       uses utils.latest_model() to find the latest one.

    Returns:
        (actor_path, lyapunov_path, dynamics_path) or raises FileNotFoundError
    """
    from sd import utils

    if dynamics_path is None:
        dynamics_path = utils.latest_model()

    ckpt_dir = dynamics_path.parent  # e.g. .../checkpoints/checkpoint0/
    controller_dir = ckpt_dir / "controller_ckpts"

    if not controller_dir.exists():
        raise FileNotFoundError(
            f"No controller_ckpts/ found in {ckpt_dir}.\n"
            "Train a controller first with: python -m sd.lyapunov"
        )

    # Find highest-numbered epoch directory
    epoch_dirs = [d for d in controller_dir.iterdir() if d.is_dir()]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories in {controller_dir}")

    def epoch_number(d):
        try:
            return int(d.name)
        except ValueError:
            return -1

    latest_epoch = max(epoch_dirs, key=epoch_number)
    actor_path = latest_epoch / "actor.keras"
    lyapunov_path = latest_epoch / "lyapunov.keras"

    if not actor_path.exists():
        raise FileNotFoundError(f"actor.keras not found in {latest_epoch}")

    print(f"Auto-discovered controller at epoch {latest_epoch.name}")
    print(f"  actor:    {actor_path}")
    print(f"  lyapunov: {lyapunov_path}")
    return actor_path, lyapunov_path, dynamics_path


def plot_lyapunov(lyapunov, actor=None, fname="V_pendulum", interactive=False):
    """Plot the Lyapunov function as a phase portrait (theta vs thetadot).

    Creates a heatmap of V(state, setpoint) over the (theta, thetadot) plane,
    where the setpoint is the upright position.
    """
    pts = 200
    theta = np.linspace(-np.pi, np.pi, pts)
    thetadot = np.linspace(-8.0, 8.0, pts)
    TH, TD = np.meshgrid(theta, thetadot)

    # Build state array: [cos(theta), sin(theta), thetadot]
    cos_th = np.cos(TH).reshape(-1)
    sin_th = np.sin(TH).reshape(-1)
    td_flat = TD.reshape(-1)
    states = np.stack([cos_th, sin_th, td_flat], axis=-1).astype(np.float32)

    setpoints = np.tile(DEFAULT_SETPOINT, (states.shape[0], 1))

    V_vals = lyapunov({"state": states, "setpoint": setpoints}, training=False)
    V_vals = np.array(V_vals).reshape(pts, pts)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(TH, TD, V_vals, vmin=0.0, vmax=1.0, shading="auto")
    plt.colorbar(mesh, ax=ax, label="V(state, setpoint)")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title("Lyapunov Function (Pendulum)")
    ax.plot(0, 0, "w*", markersize=15, label="setpoint (upright)")
    ax.legend()

    if interactive:
        plt.show()
    else:
        plt.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")
        print(f"Saved Lyapunov plot to {fname}.png")


def run_test(actor, lyapunov=None, num_steps=1000, render=True, seed=None):
    """Run the Pendulum environment with the trained controller."""
    render_mode = "human" if render else None
    env = gym.make("Pendulum-v2", render_mode=render_mode)

    setpoint = DEFAULT_SETPOINT.copy()

    obs, _ = env.reset(seed=seed)
    print(f"Initial obs: {obs}")
    print(f"Setpoint:    {setpoint} (upright, zero velocity)")

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
            theta = np.arctan2(obs[1], obs[0])
            print(f"  step {step:4d} | V={v_val:.4f} | theta={np.degrees(theta):+.1f} deg | thetadot={obs[2]:+.3f}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step} (terminated={terminated})")
            obs, _ = env.reset()

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained Lyapunov controller on the Pendulum"
    )
    parser.add_argument(
        "--actor_path", type=Path, default=None,
        help="Path to actor.keras. If not provided, auto-discovers the latest.",
    )
    parser.add_argument(
        "--lyapunov_path", type=Path, default=None,
        help="Path to lyapunov.keras. If not provided, auto-discovers the latest.",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to dynamics model.keras (used for auto-discovery base path)",
    )
    parser.add_argument(
        "--random_actor", action="store_true",
        help="Use random actions instead of trained actor",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot the Lyapunov function as a phase portrait",
    )
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--no_test", action="store_true",
                        help="Skip the simulation (useful with --plot)")
    args = parser.parse_args()

    # --- Load actor ---
    if args.random_actor:
        env_tmp = gym.make("Pendulum-v2")
        action_space = env_tmp.action_space
        env_tmp.close()
        actor = lambda obs: action_space.sample()
        print("Using random actor")
        lyapunov = None
    else:
        if args.actor_path is not None:
            # Explicit paths
            actor = keras.models.load_model(str(args.actor_path))
            print(f"Loaded actor from {args.actor_path}")
        else:
            # Auto-discover
            actor_path, lyap_path, _ = find_latest_controller(args.model)
            actor = keras.models.load_model(str(actor_path))
            if args.lyapunov_path is None and lyap_path.exists():
                args.lyapunov_path = lyap_path

        actor.summary()

        lyapunov = None
        if args.lyapunov_path:
            lyapunov = keras.models.load_model(str(args.lyapunov_path))
            print(f"Loaded lyapunov from {args.lyapunov_path}")

    # --- Plot Lyapunov ---
    if args.plot and lyapunov is not None:
        plot_lyapunov(lyapunov, actor, interactive=not args.no_render)

    # --- Run simulation ---
    if not args.no_test:
        run_test(
            actor,
            lyapunov=lyapunov,
            num_steps=args.num_steps,
            render=not args.no_render,
            seed=args.seed,
        )
