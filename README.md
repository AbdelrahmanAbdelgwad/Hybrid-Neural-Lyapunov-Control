# Hybrid Neural Lyapunov Control

Jointly learn a neural Lyapunov function and a stabilizing controller for continuous-control systems using Differentiable Fuzzy Logic (FPL).

## Demo

Pendulum stabilization with a parameterized Lyapunov function:

[Screencast From 2024-12-09 00-29-48.webm](https://github.com/user-attachments/assets/8b208b17-cb24-4e68-9e59-0dd2b0e4c992)

[Screencast from 2023-08-22 02-17-25.webm](https://github.com/user-attachments/assets/0e3e52f1-282d-404e-88e3-ad8e1a6c2716)

## Overview

The framework learns controllers with formal stability guarantees via three stages:

1. **Dynamics Learning** -- Learn a neural surrogate of the environment's transition function from random rollouts.
2. **Lyapunov Controller Training** -- Co-train a Lyapunov certificate V and a control policy using the learned dynamics and a FPL loss.
3. **Evaluation** -- Deploy the trained controller in the real (MuJoCo) environment.

Both V and the actor are parameterized by a **setpoint**, so a single trained controller can stabilize to different targets (e.g., stand vs. run).

See [`docs/formulation.tex`](docs/formulation.tex) for the full mathematical formulation.

## Supported Environments

| Environment | ID | State dim | Action dim |
|---|---|---|---|
| Pendulum | `Pendulum-v2` | 3 | 1 |
| Acrobot (continuous) | `Acrobot_continuous-v0` | 6 | 1 |
| Bipedal Walker | `CustomBipedalWalker-v0` | 24 | 4 |
| Drone Attitude | `AttitudeEnv-v0` | 12 | 4 |
| Amazing Ball | `SetpointedAmazingBallEnv-v0` | 8 | 2 |
| Hopper | `HopperLyapunov-v0` | 11 | 3 |

## Installation

### Ubuntu

```bash
apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake \
    zlib1g zlib1g-dev swig

pip install -r requirements.txt
pip install -e .
```

### Nix

```bash
nix develop
```

## Usage

### Step 1: Learn Dynamics

Train a neural dynamics model from random rollouts:

```bash
# Pendulum (default)
python -m sd.dynamics_learning

# Hopper
python -m sd.dynamics_learning --env_name HopperLyapunov-v0 --epochs 200 --episode_size 500

# Custom settings
python -m sd.dynamics_learning \
    --env_name HopperLyapunov-v0 \
    --epochs 200 \
    --episode_size 500 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --save_freq 15
```

Checkpoints are saved to `models/<env_name>/<run_id>/checkpoints/`.

**Key metric:** `closeness` -- measures prediction accuracy (0 = bad, 1 = perfect). Aim for > 0.95.

### Step 2: Train Lyapunov Controller

Co-train the Lyapunov function and actor using the learned dynamics:

```bash
# Pendulum (auto-discovers latest dynamics checkpoint)
python -m sd.lyapunov

# Pendulum with explicit dynamics checkpoint
python -m sd.lyapunov --ckpt_path models/Pendulum-v2/<run>/checkpoints/checkpoint<N>/model.keras

# Resume training from a previously saved controller
python -m sd.lyapunov --load_saved --ckpt_path <dynamics_ckpt>

# Custom training parameters
python -m sd.lyapunov --epochs 200 --batch_size 256 --lr 5e-4

# Alternating training: even epochs update V only, odd epochs update actor only
python -m sd.lyapunov --alternate

# Hopper (uses sd/train_hopper.py)
python -m sd.train_hopper

# Hopper with explicit checkpoint
python -m sd.train_hopper --ckpt_path models/HopperLyapunov-v0/<run>/checkpoints/checkpoint<N>/model.keras
```

If no `--ckpt_path` is given, the most recently saved dynamics model is used automatically.

Controller checkpoints are saved to `<dynamics_ckpt_dir>/controller_ckpts/<epoch>/`.

**Key metrics in training output:**

```
Scalar: 8.00e-01|||0.0<close_setpoints:9.3e-01 small_actions:5.3e-01 lyapunov:0.0<pop:7.0e-01 large:9.5e-01 zero:4.5e-01 lyapunov_reg:1.0e+00> actor_reg:1.0e+00 V_reg:1.0e+00>
```

- `Scalar` -- Overall satisfaction (0-1). Higher is better.
- `close_setpoints` -- How close trajectories get to the setpoint.
- `pop` (proof of performance) -- Lyapunov decrease along trajectories.
- `large` -- V is nonzero away from the setpoint.
- `zero` -- V equals zero at the setpoint.
- `small_actions` -- Control effort penalty.

### Step 3: Test the Controller

#### Pendulum

```bash
# Auto-discover latest trained controller and run
python -m sd.test_pendulum

# Plot the Lyapunov function as a phase portrait (theta vs thetadot)
python -m sd.test_pendulum --plot

# Plot only (skip simulation)
python -m sd.test_pendulum --plot --no_test

# Explicit paths
python -m sd.test_pendulum \
    --actor_path models/Pendulum-v2/<run>/checkpoints/<ckpt>/controller_ckpts/<epoch>/actor.keras \
    --lyapunov_path models/Pendulum-v2/<run>/checkpoints/<ckpt>/controller_ckpts/<epoch>/lyapunov.keras

# Random actor baseline
python -m sd.test_pendulum --random_actor

# Longer simulation, no rendering
python -m sd.test_pendulum --num_steps 2000 --no_render
```

**Interactive setpoints.** Both the controller and the Lyapunov function are
parameterized by setpoint, so a single trained model can stabilize the pendulum
to any target angle (not just upright). Two ways to drive it:

- **Pygame simulation window** -- hold the left mouse button anywhere in the
  window to set the target angle. The cursor's angle relative to the pivot
  (screen center) becomes the new setpoint, and the controller drives the
  pendulum there in real time.
- **Matplotlib Lyapunov plot** (with `--plot`):
  - Left click on the phase portrait -> change the target `theta` and redraw V
  - Right click on a phase point -> simulate a trajectory from that
    `(theta, thetadot)` using actor + dynamics, scattered as red dots fading dark

Training already exposes the model to random setpoints (a fresh random target
angle is sampled per training sample), so the controller generalizes across
target angles by construction.

#### Hopper

```bash
# Stand (default setpoint)
python -m sd.test_hopper \
    --actor_path models/HopperLyapunov-v0/<run>/checkpoints/<ckpt>/controller_ckpts/<epoch>/actor.keras

# Run at target velocity
python -m sd.test_hopper \
    --actor_path <path>/actor.keras \
    --target_vel 2.0

# With Lyapunov value printing
python -m sd.test_hopper \
    --actor_path <path>/actor.keras \
    --lyapunov_path <path>/lyapunov.keras

# Random baseline
python -m sd.test_hopper --random_actor
```

#### AmazingBall

```bash
python -m sd.test --random_actor
```

### RL Baseline (SAC)

```bash
python -m sd.rl.sac
```

## Project Structure

```
sd/
  fpl.py                  # Differentiable Fuzzy Logic (p-mean, Constraints, piecewise)
  lyapunov.py             # V and actor network definitions, Pendulum/AmazingBall training
  train_hopper.py         # Hopper-specific Lyapunov training with randomized setpoints
  test_pendulum.py        # Pendulum evaluation with auto-discovery and Lyapunov plotting
  test_hopper.py          # Hopper evaluation script
  test.py                 # AmazingBall evaluation script
  dynamics_learning.py    # Neural dynamics model training (GAN or direct)
  utils.py                # Checkpointing, train loop, helpers
  envs/
    __init__.py            # Environment registration
    hopper/                # Hopper environment + constants
    Pendulum/              # Pendulum environment
    amazingball/           # AmazingBall environment
    drone/                 # Drone attitude environment
    bipedal_walker/        # Bipedal walker environment
    Acrobot_continuous/    # Continuous acrobot environment
  rl/                     # RL baselines (SAC)
docs/
  formulation.tex         # Full mathematical formulation (LaTeX)
```

## Method

The core idea: express Lyapunov stability conditions as a differentiable constraint tree using **generalized means** (p-means), then optimize the controller and Lyapunov function jointly via gradient descent.

- **p-mean with p < 0** acts like AND (all constraints must hold)
- **p-mean with p > 0** acts like OR (at least one must hold)
- **Gradient scaling** controls relative learning rates per constraint branch
- **Piecewise transforms** shape the loss landscape (e.g., heavily penalize Lyapunov increases)

The Lyapunov function and actor both take the setpoint as input, enabling a single model to stabilize to any target state.
