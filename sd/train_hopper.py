"""
Lyapunov-based controller training for the Hopper environment.

This script learns a Lyapunov function V and a stabilizing actor (controller)
jointly, using a pre-trained dynamics model of the Hopper.

Pipeline:
    # 1. Learn dynamics model from MuJoCo rollouts
    python -m sd.dynamics_learning --env_name HopperLyapunov-v0 --epochs 200 --episode_size 500

    # 2. Train Lyapunov controller (uses latest dynamics model by default)
    python -m sd.train_hopper

    # Or specify a checkpoint explicitly:
    python -m sd.train_hopper --ckpt_path models/HopperLyapunov-v0/<run>/checkpoints/checkpoint<N>/model.keras

    # 3. Test the controller
    python -m sd.test --model models/HopperLyapunov-v0/<run>/checkpoints/checkpoint<N>/model.keras
"""

import os
import gymnasium as gym
import numpy as np
from pathlib import Path
import argparse
import tensorflow as tf
import keras
from sd.dfl import (
    p_mean,
    Constraints,
    dfl_scalar,
    build_piecewise,
    scale_gradient,
    transform,
    move_toward_zero,
)
from sd import utils
from sd.lyapunov import V_def, actor_def
from sd.envs.hopper.constant import constants
import sd.envs  # register environments


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def random_setpoint():
    """Sample a random setpoint for the Hopper.

    State layout: [z_pos, angle, thigh, leg, foot,
                   x_vel, z_vel, ang_vel, thigh_vel, leg_vel, foot_vel]

    We randomize the target forward velocity (x_vel) so the controller
    learns to stabilize at different speeds, not just standing still.
    """
    sp = constants["default_setpoint"].copy()
    # Randomize forward velocity: 0 (stand) to 3.0 (run)
    sp[5] = np.random.uniform(0.0, 3.0)
    return sp.astype(np.float32)


def generate_dataset(env):
    """Generates (state, setpoint) pairs for Lyapunov training.

    States are sampled by resetting the env and taking a random number of
    steps with a random policy, producing diverse initial conditions.
    Setpoints are randomized (especially forward velocity) so the controller
    learns to track different targets.
    """

    def gen_sample():
        while True:
            obs, _ = env.reset()
            # Take random steps to explore diverse states
            n_steps = np.random.randint(0, 50)
            for _ in range(n_steps):
                obs, _, terminated, _, _ = env.step(env.action_space.sample())
                if terminated:
                    obs, _ = env.reset()
                    break
            yield {
                "state": np.array(obs, dtype=np.float32),
                "setpoint": random_setpoint(),
            }

    return gen_sample


# ---------------------------------------------------------------------------
# Hopper-specific closeness metric
# ---------------------------------------------------------------------------

CLOSENESS_RANGES = tf.constant(constants["closeness_ranges"])


@tf.function
def hopper_closeness_dfl(states, setpoints):
    """Normalized closeness between hopper states and setpoints.

    Positions (indices 0-4) are weighted more heavily than velocities (5-10)
    via a power transform on the velocity errors.

    Args:
        states:    (11, ...) tensor
        setpoints: (11, ...) tensor, broadcast-compatible with states

    Returns:
        Scalar in [0, 1].  1 = identical, 0 = maximally far apart.
    """
    errors = tf.abs(states - setpoints)
    normalized = [
        tf.clip_by_value(errors[i] / CLOSENESS_RANGES[i], 0.0, 1.0)
        for i in range(11)
    ]
    # De-emphasise small velocity errors with a square
    weighted = normalized[:5] + [n ** 2.0 for n in normalized[5:]]
    stacked = tf.clip_by_value(tf.stack(weighted), 0.0, 1.0)
    return p_mean(1.0 - stacked, 0.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(batches, dynamics_model, actor, V, state_shape, setpoint_shape, args):
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    actor_and_before_tanh = tf.keras.Model(
        actor.input,
        {"action": actor.output, "before_tanh": actor.layers[-3].output},
    )
    V_and_before_sigmoid = tf.keras.Model(
        V.input,
        {"output": V.output, "before_sigmoid": V.layers[-2].output},
    )

    # ---- dynamics roll-out ------------------------------------------------

    @tf.function
    def run_full_model(initial_states, set_points, repeat=1):
        states = tf.TensorArray(tf.float32, size=repeat)
        vs = tf.TensorArray(tf.float32, size=repeat)
        lines = tf.TensorArray(tf.float32, size=repeat)
        actions = tf.TensorArray(tf.float32, size=repeat)
        before_tanhs = tf.TensorArray(tf.float32, size=repeat)
        current_states = initial_states
        decrease_by = 10.0 / 100.0
        batch_size = tf.shape(initial_states)[0]
        latent_shape = (batch_size,) + tuple(
            dynamics_model.input["latent"].shape[1:]
        )
        for i in range(repeat):
            outputs = actor_and_before_tanh(
                {"state": current_states, "setpoint": set_points}
            )
            current_actions = outputs["action"]
            before_tanh = outputs["before_tanh"]
            current_states = dynamics_model(
                {
                    "state": current_states,
                    "action": current_actions,
                    "latent": tf.random.normal(latent_shape),
                },
                training=True,
            )
            lines = lines.write(
                i, decrease_by * tf.cast(i, tf.dtypes.float32)
            )
            vs = vs.write(
                i, V({"state": current_states, "setpoint": set_points})
            )
            states = states.write(i, current_states)
            actions = actions.write(i, current_actions)
            before_tanhs = before_tanhs.write(i, before_tanh)
        return (
            current_states,
            tf.transpose(states.stack(), [1, 0, 2]),
            actions.stack(),
            before_tanhs.stack(),
            vs.stack(),
            lines.stack(),
        )

    # ---- batch loss (DFL) -------------------------------------------------

    @tf.function
    def batch_value(batch, percent_completion):
        maxRepetitions = int(5 + 50 * percent_completion)
        repetitions = tf.random.uniform(
            shape=[], minval=1, maxval=maxRepetitions + 1, dtype=tf.dtypes.int32
        )
        prev_states = batch["state"]
        set_points = batch["setpoint"]

        fxu, states, actions, before_tanhs, vs, lines = run_full_model(
            prev_states, set_points, repeat=repetitions
        )

        outputs = V_and_before_sigmoid(
            {"state": prev_states, "setpoint": set_points}, training=True
        )
        Vx = outputs["output"]
        Vx_before_sigmoid = outputs["before_sigmoid"]
        V_fxu = V({"state": fxu, "setpoint": set_points}, training=True)

        # Zero constraint:  V(setpoint, setpoint) == 0
        # For Hopper the setpoint IS a full state, so feed it as both inputs.
        zero = p_mean(
            (1.0 - V({"state": set_points, "setpoint": set_points}) ** 0.5),
            -1.0,
        )

        # V should decrease along the trajectory
        diff = Vx - vs

        # Transpose states for per-dimension closeness computation
        # transposed_states shape: (state_dim, batch_size, repeat)
        transposed_states = tf.transpose(states, [2, 0, 1])
        tmp_ts = tf.expand_dims(tf.transpose(set_points), axis=-1)
        target_shape = (
            tf.shape(tmp_ts)[0],
            tf.shape(tmp_ts)[1],
            tf.shape(transposed_states)[2],
        )
        transposed_setpoints = tf.broadcast_to(tmp_ts, target_shape)

        close_to_setpoints = hopper_closeness_dfl(
            transposed_states, transposed_setpoints
        )

        actor_reg = 1 - tf.tanh(tf.reduce_mean(actor.losses))
        lyapunov_reg = 1 - tf.tanh(tf.reduce_mean(V.losses))

        decrease_by = 10.0 / 100.0
        line = tf.minimum(lines, Vx)

        proof_of_performance = p_mean(
            build_piecewise(
                [
                    (-1.0, 0.0),
                    (-0.1, 1e-5),
                    (0.0, 0.01),
                    (line, 0.9),
                    (1.0, 1.0),
                ],
                diff,
                clipped=True,
            ),
            0.0,
        )

        # V should be large for states far from the setpoint
        non_setpoint_Vx = tf.where(
            hopper_closeness_dfl(
                tf.transpose(prev_states), tf.transpose(set_points)
            )
            > 0.95,
            1.0,
            Vx,
        )
        small_actions = p_mean(1.0 - tf.abs(actions), 0) ** 0.5
        large_elsewhere = p_mean(
            tf.minimum(non_setpoint_Vx * 2, 1.0), -2.0
        )

        dfl = Constraints(
            0.0,
            {
                "close_setpoints": scale_gradient(close_to_setpoints, 1e2),
                "small_actions": scale_gradient(small_actions, 1e-3),
                "lyapunov": Constraints(
                    0.0,
                    {
                        "pop": scale_gradient(proof_of_performance, 1.0),
                        "large": large_elsewhere,
                        "zero": zero,
                        "lyapunov_reg": scale_gradient(
                            tf.minimum(
                                transform(lyapunov_reg, 0.0, 1.0, 0.0, 1.1),
                                1.0,
                            ),
                            1.0,
                        ),
                    },
                ),
                "actor_reg": scale_gradient(
                    p_mean(move_toward_zero(before_tanhs), 0), 1e-4
                ),
                "V_reg": scale_gradient(
                    p_mean(move_toward_zero(Vx_before_sigmoid), 0), 1.0
                ),
            },
        )
        return dfl

    # ---- gradient step ----------------------------------------------------

    @tf.function
    def train_step(batch, epoch):
        with tf.GradientTape() as tape:
            dfl = batch_value(batch, epoch / float(args.epochs))
            scalar = dfl_scalar(dfl)
            loss = 1 - scalar
        grads = tape.gradient(
            loss, actor.trainable_weights + V.trainable_weights
        )
        optimizer.apply_gradients(
            zip(grads, actor.trainable_weights + V.trainable_weights)
        )
        return scalar, dfl

    # ---- checkpoint saving ------------------------------------------------

    def save_models(epoch):
        ckpt_dir = Path(args.ckpt_path).parent
        for name, model in [("actor", actor), ("lyapunov", V)]:
            path = ckpt_dir / "controller_ckpts" / str(epoch) / f"{name}.keras"
            os.makedirs(path.parent, exist_ok=True)
            print("saving:", path)
            model.save(str(path))

    # ---- main loop --------------------------------------------------------

    def train_and_show(batch, epoch):
        scalar, metrics = train_step(batch, epoch)
        return f"Scalar: {scalar:.2e}|||{metrics}"

    utils.train_loop(
        [batches] * args.epochs,
        train_step=train_and_show,
        every_n_seconds={"freq": args.save_freq, "callback": save_models},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Lyapunov controller for the Hopper environment"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=None,
        help="Path to a dynamics model checkpoint (model.keras). "
        "Defaults to the most recently trained model.",
    )
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument(
        "--save_freq",
        type=int,
        default=15,
        help="Save controller checkpoints every N seconds",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="Load previously saved actor/lyapunov from the checkpoint dir",
    )
    parser.add_argument(
        "--actor_hidden_sizes", nargs="+", type=int, default=[64, 64],
        help="Hidden layer sizes for the actor network (default: 64 64)",
    )
    parser.add_argument(
        "--lyapunov_hidden_sizes", nargs="+", type=int, default=[64, 64],
        help="Hidden layer sizes for the Lyapunov network (default: 64 64)",
    )
    args = parser.parse_args()

    if args.ckpt_path is None:
        args.ckpt_path = utils.latest_model()

    env_name = utils.extract_env_name(args.ckpt_path)
    print("env_name:", env_name)
    env = gym.make(env_name)

    state_shape = utils.infer_shape(env, "observation_space")
    try:
        setpoint_shape = utils.infer_shape(env, "setpoint_space")
    except AssertionError:
        setpoint_shape = state_shape

    print(f"state_shape: {state_shape}, setpoint_shape: {setpoint_shape}")

    dynamics_model = utils.load_checkpoint(args.ckpt_path)
    print("\ndynamics_model:")
    dynamics_model.summary()

    actor = (
        keras.models.load_model(args.ckpt_path.parent / "actor.keras")
        if args.load_saved
        else actor_def(
            state_shape, env.action_space,
            input_setpoint_shape=setpoint_shape,
            hidden_sizes=args.actor_hidden_sizes,
        )
    )

    lyapunov_model = (
        keras.models.load_model(args.ckpt_path.parent / "lyapunov.keras")
        if args.load_saved
        else V_def(
            state_shape,
            input_setpoint_shape=setpoint_shape,
            hidden_sizes=args.lyapunov_hidden_sizes,
        )
    )

    state_spec = tf.TensorSpec(state_shape)
    setpoint_spec = tf.TensorSpec(setpoint_shape)
    dataset_spec = {"state": state_spec, "setpoint": setpoint_spec}

    dataset = tf.data.Dataset.from_generator(
        generate_dataset(env), output_signature=dataset_spec
    )
    batched_dataset = (
        dataset.batch(args.batch_size).take(args.num_batches).cache()
    )

    train(
        batched_dataset,
        dynamics_model,
        actor,
        lyapunov_model,
        state_shape,
        setpoint_shape,
        args,
    )
