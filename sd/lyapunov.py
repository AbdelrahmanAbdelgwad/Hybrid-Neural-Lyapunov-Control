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

    V takes a state and a setpoint and outputs a scalar in (0, 1) via sigmoid.
    The Lyapunov conditions require:
      - V(setpoint) = 0     (zero at equilibrium)
      - V(x) > 0 for x != setpoint  (positive definite)
      - V(x_t) > V(x_{t+1}) along trajectories  (decreasing => stability)

    Architecture: [state; setpoint] -> Dense(64, relu) -> Dense(64, relu) -> Dense(1) -> sigmoid
    The sigmoid bounds V to (0,1). The layer before sigmoid ("before_sigmoid") is also
    exposed during training to allow regularization on pre-activation values.
    L2 regularization on all weights prevents overfitting and keeps weights small.
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
            size, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
        )(dense)
    before_sigmoid = layers.Dense(
        1, activation=None, kernel_regularizer=keras.regularizers.l2(0.01)
    )(dense)
    outputs = layers.Activation("sigmoid")(before_sigmoid)

    # outputs = layers.Lambda(lambda x: tf.clip_by_value(x+0.5, 0.0, 1.0))(activation)
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

    The actor network ends with sigmoid -> ActionLayer, so:
      output = low + sigmoid_output * (high - low)
    This guarantees actions are always within the environment's valid range.
    For Pendulum: low=-2, high=2, so output is in [-2, 2].
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

    Architecture: [state; setpoint] -> Dense(64, relu) -> Dense(64, relu) -> Dense(action_dim, linear) -> sigmoid -> ActionLayer
    The sigmoid -> ActionLayer chain maps the output to [action_low, action_high].
    For Pendulum: outputs a single torque in [-2, 2].

    The layer before sigmoid ("regularize_me") is exposed during training so that
    before-activation values can be regularized (to keep the actor away from sigmoid
    saturation where gradients vanish).
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    low = tf.constant(action_space.low)
    high = tf.constant(action_space.high)
    input_state = keras.Input(shape=state_shape)
    input_set_point = (
        keras.Input(shape=input_setpoint_shape)
        if input_setpoint_shape
        else keras.Input(shape=state_shape)
    )
    dense = layers.Concatenate()([input_state, input_set_point])
    for size in hidden_sizes:
        dense = layers.Dense(
            size, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
        )(dense)
    dense3 = layers.Dense(
        action_space.shape[0],
        activation="linear",
        name="regularize_me",
        kernel_regularizer=keras.regularizers.l2(0.01),
    )(dense)
    sigmoided = layers.Activation("sigmoid")(dense3)
    outputs = ActionLayer(high, low)(sigmoided)
    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_set_point}, outputs=outputs
    )
    model.summary()
    return model


def generate_dataset(env: gym.Env):
    """Creates a generator that yields {state, setpoint} training samples.

    Each sample is produced by resetting the environment (which randomizes the state),
    then pairing it with a setpoint. For envs that take a setpoint as input, we
    randomize the setpoint per-sample so the trained controller learns to stabilize
    to ANY target — not just one fixed equilibrium. This is what makes a single
    network usable for different goals (e.g. swing-up to any angle).

    For Pendulum: state=[cos(theta), sin(theta), thetadot]. We sample a random
    target angle in [-pi, pi] and convert to setpoint [cos(target), sin(target), 0].
    env.reset() randomizes theta in [-pi, pi] and thetadot in [-1, 1].
    The (state, setpoint) pair is therefore uniformly distributed over the joint
    space, giving the controller broad coverage during training.

    For AmazingBall: setpoint is the desired ball position+velocity (4D); state
    is 8D. We keep the setpoint at zero (centered ball, at rest) for now.
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
                # Random target angle uniformly in [-pi, pi].
                # Setpoint is the unit-circle representation [cos, sin] with zero
                # angular velocity. The controller learns to stabilize the pendulum
                # at this target — not just upright.
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
    # max_ball_pos = tf.constant([constants["max_ball_pos_x"], constants["max_ball_pos_y"]])
    # repeated_max_ball_pos = tf.repeat(max_ball_pos, tf.shape(ball_pos1))
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
    """Generic closeness metric for envs where state_dim == setpoint_dim.

    Args:
        states, setpoints: tensors with shape (state_dim, ...) (transposed layout)
        ranges: (state_dim,) tensor of per-dimension normalization ranges
    """
    errors = tf.abs(states - setpoints)
    # Reshape ranges for broadcasting: (state_dim,) -> (state_dim, 1, ...)
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

    The core idea: we don't use reward signals or RL. Instead, we enforce the three
    Lyapunov stability conditions as a differentiable loss via FPL (Fuzzy Predicate Logic).
    Both networks are trained end-to-end with a single Adam optimizer.

    The training loop for each batch:
      1. Sample random initial states from the dataset
      2. Roll out the actor through the dynamics model for N steps (N is random)
      3. Evaluate V at the initial states and at each rollout step
      4. Compute how well the three Lyapunov conditions are satisfied (as FPL constraints)
      5. Scalarize the constraint tree into a single loss via generalized means
      6. Backprop through: V + actor + dynamics_model (frozen) to update V and actor weights

    Args:
        batches: TF Dataset of {state, setpoint} batches
        dynamics_model: frozen learned dynamics f(state, action) -> next_state
        actor: the controller network to train (weights are updated)
        V: the Lyapunov network to train (weights are updated)
        obs_range: per-dimension range of observations, for normalizing closeness metrics
        action_high: max action magnitude, for normalizing action sizes
    """
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    # Wrap actor to also expose the pre-sigmoid activations ("before_tanh" is a legacy name).
    # This allows regularizing the actor's pre-activation values.
    actor_and_before_tanh = tf.keras.Model(
        actor.input,
        {"action": actor.output, "before_tanh": actor.layers[-3].output},
    )
    # Wrap V to also expose pre-sigmoid values, used for V_reg.
    V_and_before_sigmoid = tf.keras.Model(
        V.input,
        {"output": V.output, "before_sigmoid": V.layers[-2].output},
    )

    @tf.function
    def run_full_model(initial_states, set_points, repeat=1):
        """Simulates the closed-loop system for `repeat` timesteps.

        Starting from initial_states, at each step:
          1. Actor chooses action: u = actor(state, setpoint)
          2. Dynamics predicts next state: x_{t+1} = dynamics(x_t, u_t, noise)
          3. V evaluates the Lyapunov value at the new state: V(x_{t+1})

        This is the "multi-step rollout" approach: rather than just checking dV/dt
        at a single point (like the Lie derivative in Chang et al.), we simulate N
        steps of the closed-loop system and check V decreases along the trajectory.

        The entire rollout is differentiable: gradients flow back through
          V -> dynamics -> actor -> V -> dynamics -> actor -> ...
        so both V and actor get gradient signal about long-horizon behavior.

        Returns:
            current_states: final state after all steps, shape (batch, state_dim)
            states: all intermediate states, shape (batch, repeat, state_dim)
            actions: all actions taken, shape (repeat, batch, action_dim)
            before_tanhs: pre-activation values of actor, shape (repeat, batch, action_dim)
            vs: V evaluated at each step's state, shape (repeat, batch, 1)
            lines: target decrease amounts per step, shape (repeat,)
                   lines[i] = decrease_by * i, encoding "by step i, V should have
                   decreased by this much from V(x_0)"
        """
        states = tf.TensorArray(tf.float32, size=repeat)
        vs = tf.TensorArray(tf.float32, size=repeat)
        lines = tf.TensorArray(tf.float32, size=repeat)
        actions = tf.TensorArray(tf.float32, size=repeat)
        before_tanhs = tf.TensorArray(tf.float32, size=repeat)
        current_states = initial_states
        decrease_by = 10.0 / 100.0
        batch_size = tf.shape(initial_states)[0]
        latent_shape = (batch_size,) + tuple(dynamics_model.input["latent"].shape[1:])
        for i in range(repeat):
            # Step 1: actor picks an action given current state and desired setpoint
            outputs = actor_and_before_tanh(
                {"state": current_states, "setpoint": set_points}
            )
            current_actions = outputs["action"]
            before_tanh = outputs["before_tanh"]
            # Step 2: dynamics model predicts next state (with noise for robustness)
            current_states = dynamics_model(
                {
                    "state": current_states,
                    "action": current_actions,
                    "latent": tf.random.normal(latent_shape),
                },
                training=True,
            )
            # lines[i] = 0.1 * i: the cumulative target decrease by step i.
            # At step 0: target = 0 (no decrease required yet).
            # At step 10: target = 1.0 (V should have decreased by 1.0 from initial).
            # At step 100: target = 10 (but capped to Vx later, so effectively V -> 0).
            lines = lines.write(i, decrease_by * tf.cast(i, tf.dtypes.float32))
            # Evaluate V at the new state — this is what we check for decrease
            vs = vs.write(i, V({"state": current_states, "setpoint": set_points}))
            states = states.write(i, current_states)
            actions = actions.write(i, current_actions)
            before_tanhs = before_tanhs.write(i, before_tanh)
        return (
            current_states,
            # Transpose states from (repeat, batch, state_dim) to (batch, repeat, state_dim)
            tf.transpose(states.stack(), [1, 0, 2]),
            actions.stack(),
            before_tanhs.stack(),
            vs.stack(),       # shape: (repeat, batch, 1)
            lines.stack(),    # shape: (repeat,)
        )

    @tf.function
    def batch_value(batch, percent_completion):
        """Computes how well the Lyapunov conditions are satisfied for one batch.

        This is the heart of the training. It returns an FPL Constraints tree where
        each leaf is a scalar in [0, 1] measuring satisfaction of one condition.
        The tree is then scalarized into a single loss by train_step.

        Args:
            batch: {"state": (batch_size, state_dim), "setpoint": (batch_size, sp_dim)}
                   For Pendulum: state_dim = sp_dim = 3, state = [cos θ, sin θ, θ̇]
            percent_completion: float in [0, 1], fraction of training done.
                   Used to increase rollout horizon as training progresses.
        """
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 1: ROLLOUT — simulate the closed-loop system for N steps
        # ═══════════════════════════════════════════════════════════════════════════

        # Rollout horizon grows during training: starts at 5, ends at 15.
        # Early: short horizon (fast, easy gradients). Late: longer horizon (harder task).
        maxRepetitions = int(5 + 10 * percent_completion)
        # Each batch uses a random horizon in [1, maxRepetitions].
        # This acts as data augmentation: the network must satisfy Lyapunov conditions
        # for any number of steps, not just a fixed horizon.
        repetitions = tf.random.uniform(
            shape=[], minval=1, maxval=maxRepetitions + 1, dtype=tf.dtypes.int32
        )
        prev_states = batch["state"]      # (batch_size, state_dim) — random initial states
        set_points = batch["setpoint"]    # (batch_size, sp_dim) — desired equilibrium
        set_points_dim = tf.shape(set_points)[-1]

        # Run the closed-loop: actor picks actions, dynamics predicts next states, repeat N times.
        # fxu = final state after N steps. states/vs/lines = values at each intermediate step.
        fxu, states, actions, before_tanhs, vs, lines = run_full_model(
            prev_states, set_points, repeat=repetitions
        )

        # Evaluate V at the initial states (before any rollout steps).
        # Vx shape: (batch_size, 1). This is V(x_0) — the starting Lyapunov value.
        outputs = V_and_before_sigmoid(
            {"state": prev_states, "setpoint": set_points}, training=True
        )
        Vx = outputs["output"]
        Vx_before_sigmoid = outputs["before_sigmoid"]
        # V at the final state after all rollout steps (not currently used in constraints)
        V_fxu = V({"state": fxu, "setpoint": set_points}, training=True)

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 2: LYAPUNOV CONDITION 1 — V(setpoint) = 0
        # ═══════════════════════════════════════════════════════════════════════════
        # Evaluate V at the setpoint itself. For a correct Lyapunov function, V(setpoint) = 0.
        # We compute: 1 - V(setpoint)^0.5, which equals 1 when V(setpoint) = 0 (perfect).
        # p_mean with p=-1.0 (harmonic mean) is sensitive to any batch item where V(setpoint) != 0.
        state_dim = prev_states.shape[-1]
        sp_dim = set_points.shape[-1]
        if state_dim == sp_dim:
            # For Pendulum: state and setpoint have the same dimensionality,
            # so we evaluate V directly at the setpoint states
            zero_states = set_points
        else:
            # For AmazingBall: state (8D) != setpoint (4D), so we construct
            # a full state by keeping non-setpoint dims from prev_states
            zero_states = tf.concat(
                [prev_states[:, :set_points_dim], set_points], axis=1
            )
        zero = p_mean(
            (1.0 - V({"state": zero_states, "setpoint": set_points}) ** 0.5), -1.0
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 3: LYAPUNOV CONDITION 3 — V̇ < 0 (V decreases along trajectories)
        # ═══════════════════════════════════════════════════════════════════════════
        # This is the key stability condition. We check:
        #   V(x_0) - V(x_t) > 0  for each rollout step t
        #
        # diff = Vx - vs
        #   Vx shape:  (batch_size, 1)           — V at initial states
        #   vs shape:  (repeat, batch_size, 1)   — V at each rollout step
        #   diff shape: (repeat, batch_size, 1)  — how much V decreased from initial
        #
        # diff > 0 means V decreased (good). diff < 0 means V increased (bad).
        # The commented-out alternative "diff = Vx - V_fxu" would only check the final state.
        # diff = (Vx - V_fxu)

        diff = Vx - vs

        # Reshape trajectory states for computing closeness to setpoint.
        # states shape: (batch, repeat, state_dim) -> transposed: (state_dim, batch, repeat)
        transposed_states = tf.transpose(states, [2, 0, 1])

        # Broadcast setpoints to match trajectory states shape.
        # set_points: (batch, sp_dim) -> transposed: (sp_dim, batch) -> expanded: (sp_dim, batch, 1)
        # Then broadcast to (sp_dim, batch, repeat) to align with transposed_states.
        tmp_ts = tf.expand_dims(tf.transpose(set_points), axis=-1)
        target_shape = (
            tf.shape(tmp_ts)[0],
            tf.shape(tmp_ts)[1],
            tf.shape(transposed_states)[2],
        )
        transposed_setpoints = tf.broadcast_to(tmp_ts, target_shape)
        # close_to_setpoints: scalar in [0, 1] measuring how close trajectory states are
        # to the setpoint (geometric mean over dimensions, batch, and repeat).
        # Currently commented out in the constraint tree but computed for potential use.
        if state_dim == sp_dim:
            close_to_setpoints = generic_closeness_fpl(
                transposed_states, transposed_setpoints, obs_range
            )
        else:
            close_to_setpoints = ball_pos_distance_fpl(
                transposed_states[4:8], transposed_setpoints[0:4]
            )
        # Regularization: L2 penalty on weights, mapped to [0,1] via 1-tanh.
        # = 1.0 when weights are small, = 0.0 when weights are large.
        actor_reg = 1 - tf.tanh(tf.reduce_mean(actor.losses))
        lyapunov_reg = 1 - tf.tanh(tf.reduce_mean(V.losses))

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 3b: "proof of performance" — scoring the V decrease
        # ═══════════════════════════════════════════════════════════════════════════
        # "line" sets the TARGET decrease at each step:
        #   lines[i] = decrease_by * i = 0.1 * i (from run_full_model)
        #   line = min(lines, Vx) — cap the target at the initial V value.
        #     At step 0:  line = min(0.0, Vx) = 0      → no decrease required
        #     At step 5:  line = min(0.5, Vx)           → V should drop by 0.5
        #     At step 10: line = min(1.0, Vx) ≈ Vx     → V should reach ~0
        #
        # The piecewise function maps diff (actual decrease) to a score in [0, 1]:
        #   diff < -0.1  →  score ≈ 0.0      (V increased a lot — very bad)
        #   diff = 0     →  score = 0.01      (V didn't change — poor)
        #   diff = line  →  score = 0.9       (V decreased by target amount — good)
        #   diff ≥ 1.0   →  score = 1.0       (V decreased maximally — perfect)
        #
        # The geometric mean (p=0) over all (step, batch) entries produces the
        # final "pop" score. Geometric mean means: one badly-violated step pulls
        # the whole score down (unlike arithmetic mean which would let good steps
        # compensate for bad ones).
        #
        # NOTE on shapes: lines is (repeat,) and Vx is (batch, 1).
        # tf.minimum(lines, Vx) broadcasts to (batch, repeat) via TF rules.
        # diff is (repeat, batch, 1). The piecewise function broadcasts line and diff
        # together, creating shape (repeat, batch, repeat) — a cross-product where each
        # diff entry is compared against every line target. p_mean then reduces all axes.
        repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
        maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
        decrease_by = (
            10.0 / 100.0
        )  # should arrive to the target within 100 steps, think about maximizing this parameter
        line = tf.minimum(
            lines, Vx
        )  # how much we would like taking a step to reduce V by
        proof_of_performance = p_mean(
            build_piecewise(
                [(-1.0, 0.0), (-0.1, 1e-5), (0.0, 0.01), (line, 0.9), (1.0, 1.0)],
                diff,
                clipped=True,
            ),
            0.0,
        )
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 4: LYAPUNOV CONDITION 2 — V(x) > 0 for x ≠ setpoint
        # ═══════════════════════════════════════════════════════════════════════════
        # We check V > 0 at the initial batch states (NOT at trajectory states).
        # States very close to the setpoint (closeness > 0.95) are excluded by setting
        # their V to 1.0 (automatically satisfied) — because V should be ~0 there.
        #
        # non_setpoint_Vx: for each batch state, either its actual V value (if far from
        # setpoint) or 1.0 (if near setpoint, don't penalize low V).
        if state_dim == sp_dim:
            non_setpoint_Vx = tf.where(
                generic_closeness_fpl(
                    tf.transpose(prev_states), tf.transpose(set_points), obs_range
                )
                > 0.95,
                1.0,
                Vx,
            )
        else:
            non_setpoint_Vx = tf.where(
                ball_pos_distance_fpl(
                    tf.transpose(prev_states)[4:8], tf.transpose(set_points)[0:4]
                )
                > 0.95,
                1.0,
                Vx,
            )

        # small_actions: currently commented out in the constraint tree.
        # Would reward the actor for using small torques (energy efficiency).
        if action_high is not None:
            normalized_actions = tf.abs(actions) / action_high
        else:
            normalized_actions = tf.abs(actions)
        small_actions = p_mean(tf.maximum(1.0 - normalized_actions, 0.0), 0) ** 0.5
        # positive_elsewhere: V should be > 0.5 at all non-setpoint states.
        # The "* 2" maps V=0.5 to 1.0 (just barely passing) and V=0 to 0.0 (failing).
        # p_mean with p=-2.0 (near-minimum) means the WORST batch item dominates:
        # even one state with V ≈ 0 drags the whole score down.
        positive_elsewhere = p_mean(
            tf.minimum(non_setpoint_Vx * 2, 1.0), -2.0
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 5: FPL CONSTRAINT TREE — combining all objectives
        # ═══════════════════════════════════════════════════════════════════════════
        # The Constraints tree aggregates multiple objectives using generalized means.
        # Each leaf is a scalar in [0, 1]:  0 = fully violated,  1 = fully satisfied.
        #
        # Structure:
        #   outer Constraints(p=0.0)           ← geometric mean of all top-level objectives
        #     "lyapunov" Constraints(p=0.0)    ← geometric mean of the 4 Lyapunov sub-objectives
        #       "pop"       = proof_of_performance   ← V̇ < 0 (V decreases along trajectories)
        #       "positive"  = positive_elsewhere     ← V(x) > 0 for x ≠ setpoint
        #       "zero"      = zero                   ← V(setpoint) = 0
        #       "lyapunov_reg" = weight regularization on V network
        #
        # Commented-out objectives that can be re-enabled:
        #   "close_setpoints": actor drives states TOWARD the setpoint (convergence)
        #   "small_actions": actor uses minimal torque (energy efficiency)
        #   "actor_reg": keep actor's pre-activations small (prevent saturation)
        #   "V_reg": keep V's pre-sigmoid values small
        #
        # Currently only the "lyapunov" subtree is active — testing the hypothesis
        # that Lyapunov conditions alone are sufficient for stable control.
        #
        # scale_gradient(x, s) multiplies the gradient by s without changing the value.
        # This lets you control how much gradient signal each objective contributes
        # without changing its score in the constraint tree.
        fpl = Constraints(
            0.0,
            {
                # "close_setpoints": scale_gradient(close_to_setpoints, 1e2),
                # "small_actions": scale_gradient(small_actions, 1e-3),
                "lyapunov": Constraints(
                    0.0,
                    {
                        "pop": scale_gradient(proof_of_performance, 1.0),
                        "positive": positive_elsewhere,
                        "zero": zero,
                        "lyapunov_reg": scale_gradient(
                            tf.minimum(
                                transform(lyapunov_reg, 0.0, 1.0, 0.0, 1.1), 1.0
                            ),
                            1.0,
                        ),
                    },
                ),
                # "actor_reg": scale_gradient(p_mean(move_toward_zero(before_tanhs), 0), 1e-4),
                # "V_reg": scale_gradient(p_mean(move_toward_zero(Vx_before_sigmoid), 0), 1.0),
            },
        )

        return fpl

    # @tf.function
    def set_gradient_size(gradients, size):
        return size * gradients / tf.norm(gradients)

    @tf.function
    def train_step(batch, epoch):
        """One gradient step: evaluate constraints, compute loss, update weights.

        The loss is simply: loss = 1 - scalar, where scalar = fpl_scalar(constraints).
        fpl_scalar recursively scalarizes the Constraints tree using generalized means:
          - Each Constraints node combines its children with p_mean(children, p)
          - Leaf tensors are used directly
        The result is a single scalar in [0, 1]: 1 = all conditions perfectly met, 0 = total failure.

        GradientTape records all operations inside batch_value (including the rollout through
        the dynamics model), so gradients flow through:
          loss -> V weights (to shape V correctly)
          loss -> actor weights (to choose actions that make V decrease)
        The dynamics_model weights are NOT updated (it's frozen, used in inference mode).
        """
        with tf.GradientTape() as tape:
            fpl = batch_value(batch, epoch / float(args.epochs))
            scalar = fpl_scalar(fpl)
            loss = 1 - scalar
        grads = tape.gradient(loss, actor.trainable_weights + V.trainable_weights)
        optimizer.apply_gradients(
            zip(grads, actor.trainable_weights + V.trainable_weights)
        )

        return scalar, fpl

    def save_models(epoch):
        save_model(actor, Path("controller_ckpts", str(epoch), "actor.keras"))
        save_model(
            lyapunov_model, Path("controller_ckpts", str(epoch), "lyapunov.keras")
        )

    def train_and_show(batch, epoch):
        scalar, metrics = train_step(batch, epoch)
        return f"Scalar: {scalar:.2e}|||{metrics}"

    utils.train_loop(
        [batches] * args.epochs,
        train_step=train_and_show,
        every_n_seconds={"freq": args.save_freq, "callback": save_models},
    )


if __name__ == "__main__":
    """Entry point: loads a pre-trained dynamics model, creates fresh actor + V networks,
    then trains them jointly using only the Lyapunov conditions.

    Prerequisites:
      - A trained dynamics model checkpoint (created by training the env model separately).
        The dynamics model learns f(state, action) -> next_state from environment rollouts.
        It is frozen during Lyapunov training — only actor and V are updated.

    Pipeline:
      1. Load dynamics model from checkpoint
      2. Create fresh actor (controller) and V (Lyapunov) networks (or load saved ones)
      3. Generate training data: random states paired with the stabilization setpoint
      4. Train: for each batch, roll out actor through dynamics, enforce Lyapunov conditions
      5. Periodically save actor + V checkpoints

    Usage:
      python -m sd.lyapunov                          # uses latest dynamics checkpoint
      python -m sd.lyapunov --ckpt_path path/to/model.keras
      python -m sd.lyapunov --load_saved              # resume training from saved actor+V
    """
    # tf.config.run_functions_eagerly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument(
        "--save_freq", type=int, default=15, help="save the checkpoints every n seconds"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--load_saved", action="store_true")
    args = parser.parse_args()
    if args.ckpt_path is None:
        args.ckpt_path = utils.latest_model()

    # The env name is embedded in the checkpoint path structure:
    # models/<env_name>/<run_id>/checkpoints/<ckpt_id>/model.keras
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

    # Load the frozen dynamics model — this predicts next_state = f(state, action, noise)
    dynamics_model = utils.load_checkpoint(args.ckpt_path)
    print()
    print()
    print("dynamics_model:")
    dynamics_model.summary()

    # Create or load the actor (controller) and Lyapunov networks
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

    # Build the training dataset: random states from env.reset() paired with setpoint.
    # .batch(128): each training step sees 128 states.
    # .take(200): 200 batches per epoch = 25,600 states per epoch.
    # .cache(): after first epoch, data is served from memory (env.reset() only called once).
    state_spec = tf.TensorSpec(state_shape)
    setpoint_spec = tf.TensorSpec(setpoint_shape)
    dataset_spec = {"state": state_spec, "setpoint": setpoint_spec}
    dataset = tf.data.Dataset.from_generator(
        generate_dataset(env), output_signature=dataset_spec
    )
    batched_dataset = dataset.batch(args.batch_size).take(args.num_batches).cache()

    # obs_range: per-dimension range of the observation space, used to normalize
    # closeness metrics (so each state dimension contributes equally).
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
