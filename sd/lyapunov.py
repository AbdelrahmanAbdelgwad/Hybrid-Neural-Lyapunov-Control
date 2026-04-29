import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import math
from typing import Tuple
from os import path
import tensorflow as tf
import keras
from keras import layers
from functools import reduce
from pathlib import Path
import argparse
from .fpl import *
from . import utils
from sd.envs.amazingball.constant import constants
from sd.envs.Pendulum.PendulumKerasModel import (
    PendulumDifferenceEq,
)  # register for model loading


def V_def(state_shape: Tuple[int, ...], input_setpoint_shape=None, hidden_sizes=None):
    """Defines the Lyapunov function V(x, setpoint) as a neural network.

    Architecture: [state; setpoint] -> Dense(64, tanh) -> Dense(64, tanh) -> Dense(1, sigmoid)
    The sigmoid bounds V to (0,1). tanh hidden activations keep internal values bounded,
    which helps training stability compared to relu.
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    input_state = keras.Input(shape=state_shape)
    input_setpoint = (
        keras.Input(shape=input_setpoint_shape)
        if input_setpoint_shape
        else keras.Input(shape=state_shape)
    )
    dense = layers.Concatenate()([input_state, input_setpoint])
    for size in hidden_sizes:
        dense = layers.Dense(
            size, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.01)
        )(dense)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.01)
    )(dense)

    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_setpoint},
        outputs=outputs,
        name="V",
    )
    print()
    print()
    model.summary()
    return model


@keras.saving.register_keras_serializable(package="MyLayers")
class ActionLayer(keras.layers.Layer):
    """Rescales sigmoid output [0,1] to the action space [low, high].

    Kept for backward compatibility with saved models that used this layer.
    New actors use tanh * action_high instead.
    """

    def __init__(self, high, low):
        super().__init__()
        if type(high) is dict:
            high = high["config"]["value"]
            low = low["config"]["value"]
        self.high = np.array(high)
        self.low = np.array(low)

    def call(self, inputs):
        return self.low + inputs * (self.high - self.low)

    def get_config(self):
        return {"high": np.array(self.high), "low": np.array(self.low)}


def actor_def(state_shape, action_space, input_setpoint_shape=None, hidden_sizes=None):
    """Defines the controller (actor) network: pi(state, setpoint) -> action.

    Architecture: [state; setpoint] -> Dense(64, tanh) -> Dense(64, tanh) -> Dense(action_dim, tanh) * action_high
    The tanh output is in [-1, 1], scaled by action_high to fill the action range.
    For Pendulum: tanh * 2.0 gives actions in [-2, 2].
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    high = tf.constant(action_space.high, dtype=tf.float32)
    input_state = keras.Input(shape=state_shape)
    input_set_point = (
        keras.Input(shape=input_setpoint_shape)
        if input_setpoint_shape
        else keras.Input(shape=state_shape)
    )
    dense = layers.Concatenate()([input_state, input_set_point])
    for size in hidden_sizes:
        dense = layers.Dense(
            size, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.01)
        )(dense)
    prescaled = layers.Dense(
        action_space.shape[0],
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
    )(dense)
    outputs = prescaled * high
    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_set_point}, outputs=outputs
    )
    model.summary()
    return model


def generate_dataset(env: gym.Env):
    """Creates a generator that yields {state, setpoint} training samples.

    For Pendulum: state=[cos(theta), sin(theta), thetadot]. The thetadot dimension
    is scaled by 7x because env.reset() only samples thetadot in [-1, 1] but the
    actual range is [-8, 8]. Without this scaling, the controller never trains on
    fast-spinning states and fails to handle them at test time.

    We sample a random target angle per sample so the controller learns to stabilize
    to any target, not just upright.
    """
    obs_shape = env.observation_space.shape
    try:
        sp_shape = env.setpoint_space.shape
    except AttributeError:
        sp_shape = obs_shape

    def gen_sample():
        while True:
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32)
            if sp_shape == (4,) and obs_shape == (8,):
                setpoint = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif obs_shape == (3,):
                # Expand thetadot range from [-1,1] to [-7,7] to cover fast spins
                obs[2] *= 7.0
                target_angle = np.random.uniform(-np.pi, np.pi)
                setpoint = np.array(
                    [np.cos(target_angle), np.sin(target_angle), 0.0],
                    dtype=np.float32,
                )
            else:
                setpoint = np.zeros(sp_shape, dtype=np.float32)
            yield {"state": obs, "setpoint": setpoint}

    return gen_sample


def save_model(model, rel_path):
    path = Path(args.ckpt_path.parent, rel_path)
    os.makedirs(path.parent, exist_ok=True)
    print(str(path))
    model.save(str(path))


pi = tf.constant(math.pi)


@tf.function
def angular_similarity(v1, v2):
    v1_angle = tf.math.atan2(v1[1], v1[0])  # atan2's range [-pi, pi]
    v2_angle = tf.math.atan2(v2[1], v2[0])
    return tf.abs(tf.abs(v1_angle - v2_angle) - pi) / pi


@tf.function
def ball_pos_distance_fpl(ball_pos1, ball_pos2):
    errors = tf.abs(ball_pos1 - ball_pos2)
    errors_x = errors[0] / (constants["max_ball_pos_x"] * 2.0)
    errors_y = errors[1] / (constants["max_ball_pos_y"] * 2.0)
    errors_vx = errors[2] / (constants["max_ball_vel"] * 2.0)
    errors_vy = errors[3] / (constants["max_ball_vel"] * 2.0)

    normalized_errors = tf.clip_by_value(
        tf.stack([errors_x, errors_y, errors_vx**4.0, errors_vy**4.0]), 0.0, 1.0
    )

    return p_mean(1.0 - normalized_errors, 0.0)


@tf.function
def generic_closeness_fpl(states, setpoints, ranges):
    """Generic closeness metric for envs where state_dim == setpoint_dim."""
    errors = tf.abs(states - setpoints)
    extra_dims = len(states.shape) - 1
    ranges_shape = tf.concat([tf.shape(ranges), tf.ones(extra_dims, dtype=tf.int32)], 0)
    ranges_bc = tf.reshape(ranges, ranges_shape)
    normalized = tf.clip_by_value(errors / ranges_bc, 0.0, 0.99)
    return p_mean(1.0 - normalized, 0.0)


def train(
    batches,
    dynamics_model,
    actor,
    V,
    state_shape,
    args,
    obs_range=None,
    action_high=None,
):
    """Jointly trains the actor (controller) and V (Lyapunov function).

    Training logic matches the "LYAPUNOV PENDULUM FULLY WORKS" approach:
      - Consecutive-step V decrease check (prev_V vs next_V at each step)
      - Squared penalty for V increases, accumulated across steps
      - Adaptive decrease target proportional to Vx^0.5
      - smooth_constraint scoring with harmonic mean aggregation (p=-1)
      - close_angles objective active in the constraint tree
      - Fixed rollout horizon (maxRepetitions=15)

    With --alternate: even epochs update V only, odd epochs update actor only.
    """
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    @tf.function
    def run_full_model(initial_states, set_points, repeat=1):
        """Simulates the closed-loop system for `repeat` timesteps.

        Returns:
            current_states: final state after all steps, shape (batch, state_dim)
            states: all intermediate states, shape (batch, repeat, state_dim)
        """
        states = tf.TensorArray(tf.float32, size=repeat)
        current_states = initial_states
        batch_size = tf.shape(initial_states)[0]
        latent_shape = (batch_size,) + tuple(dynamics_model.input["latent"].shape[1:])
        for i in range(repeat):
            current_states = dynamics_model(
                {
                    "state": current_states,
                    "action": actor(
                        {"state": current_states, "setpoint": set_points}
                    ),
                    "latent": tf.random.normal(latent_shape),
                },
                training=True,
            )
            states = states.write(i, current_states)
        return current_states, tf.transpose(states.stack(), [1, 0, 2])

    @tf.function
    def batch_value(batch):
        """Computes how well the Lyapunov conditions are satisfied for one batch.

        Returns an FPL Constraints tree that gets scalarized into a loss.
        """
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 1: ROLLOUT
        # ═══════════════════════════════════════════════════════════════════════════
        # Fixed horizon of 15 steps maximum. Each batch picks a random horizon
        # in [1, 15] for data augmentation.
        maxRepetitions = 15
        repetitions = tf.random.uniform(
            shape=[], minval=1, maxval=maxRepetitions + 1, dtype=tf.dtypes.int32
        )
        prev_states = batch["state"]
        set_points = batch["setpoint"]
        set_points_dim = tf.shape(set_points)[-1]

        fxu, states = run_full_model(
            prev_states, set_points, repeat=repetitions
        )

        # V at the initial states (before any rollout)
        Vx = V({"state": prev_states, "setpoint": set_points}, training=True)

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 2: V(setpoint) = 0
        # ═══════════════════════════════════════════════════════════════════════════
        # Geometric mean of (1 - V(setpoint)). Equals 1 when V(setpoint) = 0.
        state_dim = prev_states.shape[-1]
        sp_dim = set_points.shape[-1]
        if state_dim == sp_dim:
            zero_states = set_points
        else:
            zero_states = tf.concat(
                [prev_states[:, :set_points_dim], set_points], axis=1
            )
        zero = p_mean(
            1.0 - V({"state": zero_states, "setpoint": set_points}), 0
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 3: V decreases at EVERY consecutive step
        # ═══════════════════════════════════════════════════════════════════════════
        # For each step, compute (prev_V - next_V), normalize to [0,1] via
        # (delta + 1)/2, then penalize deviations from 1 (= perfect decrease).
        # The squared penalty means V increases are punished much harder than
        # V staying flat. Penalties accumulate across steps, then diff = 1 - sqrt(sum).
        #
        # diff close to 1.0 = V decreased nicely at every step.
        # diff close to 0.0 = V increased at some step (bad).
        diff = tf.zeros_like(Vx)
        prev_V = Vx
        for i in tf.range(repetitions):
            next_V = V(
                {"state": states[:, i, :], "setpoint": set_points}, training=True
            )
            norm_diff = (prev_V - next_V + 1.0) / 2.0
            diff += (1.0 - norm_diff) ** 2
            prev_V = next_V
        diff = 1.0 - diff**0.5

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 3b: Scoring the V decrease via smooth_constraint
        # ═══════════════════════════════════════════════════════════════════════════
        # The "line" is the adaptive target: how much diff should be, given the
        # number of steps and the initial V value.
        #   line = tanh(transform(reps, ...)) * Vx^0.5
        # This means:
        #   - More steps -> higher target (must decrease more)
        #   - Higher initial V -> higher target (farther from setpoint = expect more decrease)
        #   - The sqrt makes the target gentler near V=0 (near the setpoint)
        #
        # smooth_constraint maps diff to [0,1] via a sigmoidal curve:
        #   diff << 0 (V increased) -> score near 0
        #   diff = line (met target) -> score near 0.97
        #   diff >> line -> score near 1
        #
        # Harmonic mean (p=-1) is very sensitive to outliers: one bad batch item
        # with a low score tanks the entire proof_of_performance.
        repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
        maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
        line = (
            tf.math.tanh(
                transform(
                    repetitionsf,
                    0.05,
                    4.0 * maxRepetitionsf * Vx**0.5 + 0.05,
                    0.0,
                    1.0,
                )
            )
            * Vx**0.5
        )
        decreases_everywhere = smooth_constraint(diff, 0.0, line)
        proof_of_performance = tf.squeeze(
            p_mean(decreases_everywhere, -1.0, slack=1e-13)
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 4: Angular similarity (close_angles)
        # ═══════════════════════════════════════════════════════════════════════════
        # Measures how close trajectory states are angularly to the setpoint.
        # Uses atan2 on the first two state dims [cos θ, sin θ].
        # p_mean with p=2.0 (quadratic mean) rewards getting close on average
        # while tolerating some steps being far.
        transposed_states = tf.transpose(states, [2, 0, 1])
        if state_dim == sp_dim:
            transposed_setpoints = tf.broadcast_to(
                tf.expand_dims(tf.transpose(set_points), axis=-1),
                tf.shape(transposed_states),
            )
            as_all = angular_similarity(transposed_states, transposed_setpoints)
            close_angles = p_mean(as_all, 2.0)
        else:
            close_angles = None

        # Regularization (computed but only used if re-enabled in tree)
        actor_reg = 1 - tf.tanh(tf.reduce_mean(actor.losses))
        lyapunov_reg = 1 - tf.tanh(tf.reduce_mean(V.losses))

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 5: V(x) > 0 for x != setpoint (avg_large)
        # ═══════════════════════════════════════════════════════════════════════════
        # geometric mean of Vx * 1.5, capped at 1.0.
        # Rewards V being large (> 0.67) at non-setpoint states.
        # No explicit setpoint masking — relies on the zero condition to keep
        # V(setpoint) low while avg_large pushes V up everywhere else.
        avg_large = tf.minimum(p_mean(Vx, 0.0) * 1.5, 1.0)

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 6: FPL CONSTRAINT TREE
        # ═══════════════════════════════════════════════════════════════════════════
        # Geometric mean (p=0.0) at every level:
        #   - One badly-violated objective drags the entire score down
        #   - All objectives must be satisfied simultaneously
        #
        # close_angles: actor drives states toward the setpoint angularly
        # proof_of_performance: V decreases at every consecutive step
        # avg_large: V is positive away from setpoint
        # zero: V(setpoint) = 0
        lyapunov_constraints = {
            "proof_of_performance": proof_of_performance,
            "avg_large": avg_large,
            "zero": zero,
        }

        tree = {
            "lyapunov": Constraints(0.0, lyapunov_constraints),
        }
        if close_angles is not None:
            tree["close_angles"] = close_angles

        fpl = Constraints(0.0, tree)
        return fpl

    # @tf.function
    def set_gradient_size(gradients, size):
        return size * gradients / tf.norm(gradients)

    @tf.function
    def train_step_both(batch):
        """Update both actor and V weights in a single step (default mode)."""
        with tf.GradientTape() as tape:
            fpl = batch_value(batch)
            scalar = fpl_scalar(fpl)
            loss = 1 - scalar
        grads = tape.gradient(loss, actor.trainable_weights + V.trainable_weights)
        optimizer.apply_gradients(
            zip(grads, actor.trainable_weights + V.trainable_weights)
        )
        return scalar, fpl

    @tf.function
    def train_step_V_only(batch):
        """Update only V weights. Actor is frozen this epoch."""
        with tf.GradientTape() as tape:
            fpl = batch_value(batch)
            scalar = fpl_scalar(fpl)
            loss = 1 - scalar
        grads = tape.gradient(loss, V.trainable_weights)
        optimizer.apply_gradients(zip(grads, V.trainable_weights))
        return scalar, fpl

    @tf.function
    def train_step_actor_only(batch):
        """Update only actor weights. V is frozen this epoch."""
        with tf.GradientTape() as tape:
            fpl = batch_value(batch)
            scalar = fpl_scalar(fpl)
            loss = 1 - scalar
        grads = tape.gradient(loss, actor.trainable_weights)
        optimizer.apply_gradients(zip(grads, actor.trainable_weights))
        return scalar, fpl

    def save_models(epoch):
        save_model(actor, Path("controller_ckpts", str(epoch), "actor.keras"))
        save_model(
            lyapunov_model, Path("controller_ckpts", str(epoch), "lyapunov.keras")
        )

    def train_and_show(batch, epoch):
        if args.alternate:
            if epoch % 2 == 0:
                scalar, metrics = train_step_V_only(batch)
                label = "V"
            else:
                scalar, metrics = train_step_actor_only(batch)
                label = "actor"
        else:
            scalar, metrics = train_step_both(batch)
            label = "both"
        return f"[{label}] Scalar: {scalar:.2e}|||{metrics}"

    utils.train_loop(
        [batches] * args.epochs,
        train_step=train_and_show,
        every_n_seconds={"freq": args.save_freq, "callback": save_models},
    )


if __name__ == "__main__":
    """Entry point: loads a pre-trained dynamics model, creates fresh actor + V networks,
    then trains them jointly using only the Lyapunov conditions.

    Usage:
      python -m sd.lyapunov                          # uses latest dynamics checkpoint
      python -m sd.lyapunov --ckpt_path path/to/model.keras
      python -m sd.lyapunov --load_saved              # resume training from saved actor+V
      python -m sd.lyapunov --alternate               # alternating V/actor epochs
    """
    # tf.config.run_functions_eagerly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument(
        "--save_freq", type=int, default=15, help="save the checkpoints every n seconds"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--load_saved", action="store_true")
    parser.add_argument(
        "--alternate", action="store_true",
        help="Alternate training: even epochs update V only, odd epochs update actor only.",
    )
    args = parser.parse_args()
    if args.ckpt_path is None:
        args.ckpt_path = utils.latest_model()

    env_name = utils.extract_env_name(args.ckpt_path)
    print("env_name:", env_name)
    env = gym.make(env_name)

    action_shape = utils.infer_shape(env, "action_space")
    state_shape = utils.infer_shape(env, "observation_space")
    try:
        setpoint_shape = utils.infer_shape(env, "setpoint_space")
    except AssertionError:
        print("setpoint_space not specified in env, using observation_space instead")
        setpoint_shape = state_shape

    dynamics_model = utils.load_checkpoint(args.ckpt_path)
    print()
    print()
    print("dynamics_model:")
    dynamics_model.summary()

    actor = (
        keras.models.load_model(args.ckpt_path.parent / "actor.keras")
        if args.load_saved
        else actor_def(
            state_shape, env.action_space, input_setpoint_shape=setpoint_shape
        )
    )

    lyapunov_model = (
        keras.models.load_model(args.ckpt_path.parent / "lyapunov.keras")
        if args.load_saved
        else V_def(state_shape, input_setpoint_shape=setpoint_shape)
    )

    state_spec = tf.TensorSpec(state_shape)
    setpoint_spec = tf.TensorSpec(setpoint_shape)
    dataset_spec = {"state": state_spec, "setpoint": setpoint_spec}
    dataset = tf.data.Dataset.from_generator(
        generate_dataset(env), output_signature=dataset_spec
    )
    batched_dataset = dataset.batch(args.batch_size).take(args.num_batches).cache()

    obs_range = tf.constant(
        env.observation_space.high - env.observation_space.low, dtype=tf.float32
    )
    action_high = tf.constant(env.action_space.high, dtype=tf.float32)
    train(
        batched_dataset,
        dynamics_model,
        actor,
        lyapunov_model,
        state_shape,
        args,
        obs_range=obs_range,
        action_high=action_high,
    )
