"""
Test a trained Lyapunov controller on the Pendulum environment.

Usage:
    # Auto-discover latest controller checkpoint:
    python -m sd.test_pendulum

    # Specify paths explicitly:
    python -m sd.test_pendulum --actor_path <path>/actor.keras --lyapunov_path <path>/lyapunov.keras

    # Plot Lyapunov function (phase portrait, interactive):
    python -m sd.test_pendulum --plot

    # Random actor baseline:
    python -m sd.test_pendulum --random_actor

Interactive setpoints:
    The trained controller is parameterized by setpoint, so it can stabilize the
    pendulum to any target angle (not just upright). Two ways to drive it:

    Matplotlib Lyapunov plot (with --plot):
        Left click  -> change the target angle to the clicked theta and redraw V
        Right click -> simulate a trajectory from the clicked (theta, thetadot)
                       using actor + dynamics, drawn as red dots fading to dark
    Pygame simulation window:
        Hold left mouse button anywhere in the window to set the target angle
        relative to the pivot (center of the screen). Release to keep that target.
"""

import argparse
import numpy as np
import gymnasium as gym
import keras
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import pygame
import sd.envs  # register environments
from sd.lyapunov import ActionLayer  # register custom Keras layer for model loading
from sd.envs.Pendulum.PendulumKerasModel import PendulumDifferenceEq  # register for model loading
from sd import utils


# Pendulum upright setpoint: cos(0)=1, sin(0)=0, angular_vel=0
DEFAULT_SETPOINT = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def angle_to_setpoint(angle):
    """Convert a target angle (radians) to a Pendulum setpoint vector.

    Pendulum state is [cos(theta), sin(theta), thetadot]. The setpoint encodes
    a desired angle with zero angular velocity (a stable rest position).
    angle=0 corresponds to upright; angle=pi to hanging down.
    """
    return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)


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


def plot_lyapunov(
    lyapunov, actor=None, dynamics=None, fname="V_pendulum", interactive=False
):
    """Plot the Lyapunov function as a phase portrait (theta vs thetadot).

    Creates a heatmap of V(state, setpoint) over the (theta, thetadot) plane.
    Initially the setpoint is upright (theta=0).

    Interactive mode (matplotlib mouse events):
      Left click  -> change setpoint to the clicked theta and redraw V
      Right click -> roll out actor+dynamics from the clicked (theta, thetadot)
                     and scatter the resulting trajectory in red shades

    The right-click trajectory uses the dynamics model (not the real env), so
    we can step it differentiably from any initial state -- useful for sanity-
    checking what the controller would do without actually running the real env.
    """
    pts = 200

    # Cached grid for speed: each redraw only changes V values, not the meshgrid.
    theta = np.linspace(-np.pi, np.pi, pts)
    thetadot = np.linspace(-8.0, 8.0, pts)
    TH, TD = np.meshgrid(theta, thetadot)
    cos_th = np.cos(TH).reshape(-1)
    sin_th = np.sin(TH).reshape(-1)
    td_flat = TD.reshape(-1)
    states = np.stack([cos_th, sin_th, td_flat], axis=-1).astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 6))
    colorbar_state = {"_init": True}

    # Mutable state captured by the mouse callback.
    cur_setpoint = [DEFAULT_SETPOINT.copy()]
    scatter_artists = []

    def draw_lyapunov(setpoint):
        """Redraw the V heatmap for the given setpoint."""
        ax.clear()
        setpoints_grid = np.tile(setpoint, (states.shape[0], 1)).astype(np.float32)
        V_vals = lyapunov(
            {"state": states, "setpoint": setpoints_grid}, training=False
        )
        V_vals = np.array(V_vals).reshape(pts, pts)

        mesh = ax.pcolormesh(TH, TD, V_vals, vmin=0.0, vmax=1.0, shading="auto")
        if colorbar_state["_init"]:
            plt.colorbar(mesh, ax=ax, label="V(state, setpoint)")
            colorbar_state["_init"] = False
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
        target_angle = float(np.arctan2(setpoint[1], setpoint[0]))
        ax.set_title(
            f"Lyapunov Function (Pendulum) -- target theta = {np.degrees(target_angle):+.1f} deg\n"
            "left click: set target theta | right click: simulate trajectory"
        )
        ax.plot(target_angle, 0, "w*", markersize=15, label="setpoint")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-8.0, 8.0)
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    MAX_TRAJ_STEPS = 200
    CLOSE_THRESHOLD = 0.05

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            # Left click: change target angle to the clicked theta, redraw V.
            target_angle = float(np.clip(event.xdata, -np.pi, np.pi))
            cur_setpoint[0] = angle_to_setpoint(target_angle)
            print(f"new setpoint: target theta = {np.degrees(target_angle):+.1f} deg")
            draw_lyapunov(cur_setpoint[0])
        elif event.button == 3:
            # Right click: roll out a trajectory from the clicked phase point.
            if dynamics is None or actor is None:
                print("right-click trajectory requires --model (dynamics) and a trained actor")
                return
            for artist in scatter_artists:
                artist.remove()
            scatter_artists.clear()

            theta0 = float(event.xdata)
            thetadot0 = float(event.ydata)
            state = np.array(
                [[np.cos(theta0), np.sin(theta0), thetadot0]], dtype=np.float32
            )
            setpoint = cur_setpoint[0].astype(np.float32).reshape(1, 3)
            colors = cm.Reds(np.linspace(1.0, 0.2, MAX_TRAJ_STEPS))
            latent_shape = (1,) + tuple(dynamics.input["latent"].shape[1:])

            for step in range(MAX_TRAJ_STEPS):
                # Map back to phase coordinates for plotting.
                th = float(np.arctan2(state[0, 1], state[0, 0]))
                td = float(state[0, 2])
                if -np.pi < th < np.pi and -8.0 < td < 8.0:
                    art = ax.scatter(th, td, c=[colors[step]], s=12, zorder=3)
                    scatter_artists.append(art)
                    fig.canvas.draw_idle()
                    fig.canvas.start_event_loop(0.01)

                # Distance to setpoint (in state space): close enough -> stop early.
                dist = float(np.linalg.norm(state[0] - cur_setpoint[0]))
                if dist < CLOSE_THRESHOLD:
                    break

                action = actor(
                    {"state": state, "setpoint": setpoint}, training=False
                )
                state = dynamics(
                    {
                        "state": state,
                        "action": action,
                        "latent": np.random.normal(size=latent_shape).astype(np.float32),
                    },
                    training=False,
                ).numpy()

    fig.canvas.mpl_connect("button_press_event", on_click)
    draw_lyapunov(cur_setpoint[0])

    if interactive:
        plt.show()
    else:
        plt.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")
        print(f"Saved Lyapunov plot to {fname}.png")


def run_test(actor, lyapunov=None, num_steps=1000, render=True, seed=None):
    """Run the Pendulum environment with the trained controller.

    When rendering, holding the left mouse button anywhere in the window updates
    the target setpoint to the angle of the cursor relative to the screen center
    (the pendulum's pivot). This lets you steer the pendulum live to arbitrary
    target angles, demonstrating the setpoint-conditioned controller.
    """
    setpoint = DEFAULT_SETPOINT.copy()
    screen = None

    if render:
        # Manage our own pygame window so we can read mouse events alongside
        # the env's rendering (env writes onto the surface we provide).
        pygame.init()
        pygame.display.init()
        screen_dim = 500
        screen = pygame.display.set_mode((screen_dim, screen_dim))
        pygame.display.set_caption("Pendulum -- hold left mouse to set target angle")
        env = gym.make("Pendulum-v2", render_mode="human", screen=screen)
    else:
        env = gym.make("Pendulum-v2")

    obs, _ = env.reset(seed=seed)
    print(f"Initial obs: {obs}")
    print(f"Setpoint:    {setpoint} (upright, zero velocity)")

    total_reward = 0.0
    last_target_print = None
    for step in range(num_steps):
        # Mouse-driven setpoint: while holding left click, point the target at the cursor.
        if render and pygame.mouse.get_pressed()[0]:
            cur_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
            center = np.array(screen.get_size(), dtype=np.float32) / 2.0
            # Convention matches Pendulum's rendering: theta=0 is straight up,
            # theta=pi/2 is to the left, theta=-pi/2 is to the right.
            # We use atan2(center_x - cur_x, center_y - cur_y) so that:
            #   cursor above pivot  -> theta = 0       (upright)
            #   cursor left  of pivot -> theta = pi/2  (to the left)
            #   cursor right of pivot -> theta = -pi/2 (to the right)
            dx_inv = center[0] - cur_pos[0]
            dy_inv = center[1] - cur_pos[1]
            target_angle = float(np.arctan2(dx_inv, dy_inv))
            setpoint = angle_to_setpoint(target_angle)
            if last_target_print is None or abs(target_angle - last_target_print) > 0.05:
                print(f"  -> target theta = {np.degrees(target_angle):+.1f} deg")
                last_target_print = target_angle

        if render:
            # Drain pygame events so the OS keeps the window responsive.
            pygame.event.pump()

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
            target_angle = float(np.arctan2(setpoint[1], setpoint[0]))
            print(
                f"  step {step:4d} | V={v_val:.4f} | theta={np.degrees(theta):+.1f} deg "
                f"| thetadot={obs[2]:+.3f} | target={np.degrees(target_angle):+.1f} deg"
            )

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
        help="Path to dynamics model.keras (used for auto-discovery base path "
             "and right-click trajectory simulation in --plot)",
    )
    parser.add_argument(
        "--random_actor", action="store_true",
        help="Use random actions instead of trained actor",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot the Lyapunov function as a phase portrait (interactive)",
    )
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--no_test", action="store_true",
                        help="Skip the simulation (useful with --plot)")
    args = parser.parse_args()

    # --- Load actor ---
    dynamics_path = None
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
            dynamics_path = args.model
        else:
            # Auto-discover
            actor_path, lyap_path, dynamics_path = find_latest_controller(args.model)
            actor = keras.models.load_model(str(actor_path))
            if args.lyapunov_path is None and lyap_path.exists():
                args.lyapunov_path = lyap_path

        actor.summary()

        lyapunov = None
        if args.lyapunov_path:
            lyapunov = keras.models.load_model(str(args.lyapunov_path))
            print(f"Loaded lyapunov from {args.lyapunov_path}")

    # --- Plot Lyapunov (interactive) ---
    if args.plot and lyapunov is not None:
        # Load the dynamics model so right-click can roll out trajectories.
        dynamics = None
        try:
            if dynamics_path is None:
                dynamics_path = utils.latest_model()
            dynamics = utils.load_checkpoint(dynamics_path)
            print(f"Loaded dynamics from {dynamics_path}")
        except Exception as e:
            print(f"(could not load dynamics for trajectory simulation: {e})")

        plot_lyapunov(
            lyapunov, actor, dynamics=dynamics, interactive=not args.no_render
        )

    # --- Run simulation ---
    if not args.no_test:
        run_test(
            actor,
            lyapunov=lyapunov,
            num_steps=args.num_steps,
            render=not args.no_render,
            seed=args.seed,
        )
