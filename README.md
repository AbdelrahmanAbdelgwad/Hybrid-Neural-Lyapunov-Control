# Hybrid Neural Lyapunov Control

Jointly learn a neural Lyapunov function and a stabilizing controller for continuous-control systems using Differentiable Fuzzy Logic (DFL).

## Demo

Pendulum stabilization with a parameterized Lyapunov function:

[Screencast From 2024-12-09 00-29-48.webm](https://github.com/user-attachments/assets/8b208b17-cb24-4e68-9e59-0dd2b0e4c992)

[Screencast from 2023-08-22 02-17-25.webm](https://github.com/user-attachments/assets/0e3e52f1-282d-404e-88e3-ad8e1a6c2716)

## Overview

The framework learns controllers with formal stability guarantees via three stages:

1. **Dynamics Learning** -- Learn a neural surrogate of the environment's transition function from random rollouts.
2. **Lyapunov Controller Training** -- Co-train a Lyapunov certificate V and a control policy using the learned dynamics and a DFL loss.
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
# Pendulum / AmazingBall (uses sd/lyapunov.py)
python -m sd.lyapunov

# Hopper (uses sd/train_hopper.py)
python -m sd.train_hopper

# Specify a dynamics checkpoint explicitly
python -m sd.train_hopper --ckpt_path models/HopperLyapunov-v0/<run>/checkpoints/checkpoint<N>/model.keras

# Resume from saved controller
python -m sd.train_hopper --load_saved --ckpt_path <dynamics_ckpt>
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

```bash
# Hopper -- stand (default setpoint)
python -m sd.test_hopper \
    --actor_path models/HopperLyapunov-v0/<run>/checkpoints/<ckpt>/controller_ckpts/<epoch>/actor.keras

# Hopper -- run at target velocity
python -m sd.test_hopper \
    --actor_path <path>/actor.keras \
    --target_vel 2.0

# With Lyapunov value printing
python -m sd.test_hopper \
    --actor_path <path>/actor.keras \
    --lyapunov_path <path>/lyapunov.keras \
    --target_vel 2.0

# Random baseline
python -m sd.test_hopper --random_actor

# Pendulum / AmazingBall
python -m sd.test --random_actor
```

### RL Baseline (SAC)

```bash
python -m sd.rl.sac
```

## Project Structure

```
sd/
  dfl.py                  # Differentiable Fuzzy Logic (p-mean, Constraints, piecewise)
  lyapunov.py             # V and actor network definitions, Pendulum/AmazingBall training
  train_hopper.py         # Hopper-specific Lyapunov training with randomized setpoints
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
