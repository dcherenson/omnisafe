# RL Final Velocity Scenario

## Overview

`RL_final_velocity.py` implements a new three-stage scenario for the velocity-constrained locomotion tasks:

- `Ant`
- `HalfCheetah`
- `Humanoid`

The intended control structure is:

1. Train a fast nominal PPO policy on the original velocity task.
2. Train a PPO recovery policy that learns to slow the robot to a stable equilibrium.
3. Freeze both policies and train a PPO switching policy that decides which frozen policy should control the robot at each step.

The new code is self-contained in:

- `examples/benchmarks/RL_final_velocity.py`

No existing repo files are modified.

## What Is Implemented

### 1. Stage runner

The script can execute any subset of:

- `nominal`
- `recovery`
- `switch`

It still runs them in dependency order:

- `nominal -> recovery -> switch`

If you run only `recovery` or `switch`, the script auto-discovers previously saved checkpoints from the output directory.

### 2. Supported robots

The script currently supports:

- `ant`
- `halfcheetah`
- `humanoid`

Each robot has:

- its base Safety-Gymnasium velocity env id
- a custom recovery env id
- a custom switch env id
- a stop-speed heuristic
- a recovery horizon
- a switch horizon

### 3. Custom recovery environment

Custom env ids:

- `VelocityRecoveryAnt-v0`
- `VelocityRecoveryHalfCheetah-v0`
- `VelocityRecoveryHumanoid-v0`

This environment:

- wraps the standard velocity task
- uses the same continuous action space as the base robot
- gives the recovery actor only the original base observation
- optionally uses the frozen nominal actor to create harder reset states via burn-in rollouts
- rewards the agent for stopping quickly without becoming unhealthy

### 4. Custom switch environment

Custom env ids:

- `VelocitySwitchAnt-v0`
- `VelocitySwitchHalfCheetah-v0`
- `VelocitySwitchHumanoid-v0`

This environment:

- wraps the same base velocity task
- freezes the nominal and recovery PPO actors
- gives the switch actor only the original base observation
- exposes a single gate action
- turns that gate into policy selection
- rewards nominal usage
- heavily penalizes constraint violations

### 5. Frozen policy loader

The script includes a generic `SavedOmniSafeActor` helper that:

- loads `config.json`
- reconstructs the actor network
- loads the latest `epoch-*.pt`
- restores observation normalization when present
- returns deterministic actions from the saved PPO actor

### 6. Stage output reuse

Runs are re-used automatically unless `--force` is passed.

This makes it easy to:

- train nominal once
- add recovery later
- retrain only the switch policy

## Design Summary

### High-level pipeline

```text
                +----------------------+
                |  Stage 1: Nominal    |
                |  PPO on base velocity |
                |  reward = go fast     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Stage 2: Recovery   |
                |  PPO on custom env   |
                |  reward = stop fast  |
                |  and stay healthy    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Stage 3: Switcher   |
                |  PPO gate policy     |
                |  action in {0,1}     |
                |  0 -> recovery       |
                |  1 -> nominal        |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Frozen execution    |
                |  nominal or recovery |
                |  action is applied   |
                +----------------------+
```

### Switch policy data flow

```text
           switch observation
                   |
                   v
        +-----------------------+
        |  PPO switch policy    |
        |  outputs scalar gate  |
        +-----------+-----------+
                    |
          threshold at 0.5
           /               \
          v                 v
 +----------------+   +----------------+
 | gate = 1       |   | gate = 0       |
 | use nominal    |   | use recovery   |
 +--------+-------+   +--------+-------+
          |                    |
          v                    v
  frozen nominal PPO    frozen recovery PPO
          |                    |
          +---------+----------+
                    |
                    v
            environment step
                    |
                    v
       reward = nominal_bonus - penalties
```

### Recovery reset strategy

```text
 base env reset
      |
      v
 choose burn-in source
      |
      +--> nominal burn-in rollout
      |
      +--> random-action burn-in rollout
      |
      v
 perturbed reachable state
      |
      v
 recovery PPO starts control
```

## Observation and Action Design

### Nominal policy

The nominal policy is trained on the original Safety-Gymnasium velocity task, so it uses:

- the original base observation
- the original base action space

### Recovery policy

The recovery actor now sees only:

```text
[ base_observation ]
```

It does not get extra access to:

- speed
- cost
- health flags
- episode progress
- an explicit speed-limit value

This keeps the recovery policy aligned with the same online observation pattern used by the baseline actors.

### Switch policy

The switch actor sees only:

- the original base observation

So the switch actor input is:

```text
[ base_observation ]
```

It does not get extra access to:

- speed
- cost
- health flags
- previous switch actions
- episode progress
- an explicit speed-limit value

This keeps the switch policy aligned with the baseline safe-velocity setup.

### Why the switch action is a 1-D Box instead of a true Discrete action

Conceptually the switch action is:

- `0 = recovery`
- `1 = nominal`

The current implementation uses:

- `Box(shape=(1,), low=0, high=1)`
- threshold at `0.5`

Reason:

- the current OmniSafe PPO path used in this repo is already wired for continuous `Box` actions
- this keeps the scenario compatible with the existing actor-building and checkpoint-loading flow

There is a code TODO to replace this with a native categorical switch policy if discrete PPO support is added to the local stack.

## Reward Design

### Stage 1: nominal PPO

Uses the base environment reward directly.

Interpretation:

- learn the fastest useful locomotion policy
- ignore safety cost during optimization

### Stage 2: recovery PPO

The recovery reward is currently:

```text
reward =
    alive_bonus * healthy
  + stable_bonus * is_below_stop_speed
  - speed_weight * speed
  - action_weight * ||a||^2
  - constraint_weight * cost
  - fall_penalty if unhealthy
  + success_bonus if stable for N consecutive steps
```

This is meant to favor:

- low speed
- staying healthy
- small unnecessary actions
- reaching a stable stopped condition quickly

### Stage 3: switch PPO

The switch reward is intentionally sparse and close to the scenario description:

```text
reward =
    nominal_reward if gate == nominal
  - violation_penalty * constraint_cost
  - unhealthy_penalty if robot becomes unhealthy
```

Interpretation:

- the agent is always tempted to stay on the nominal controller
- but it pays a large price when that causes a violation

This should encourage:

- nominal whenever possible
- recovery only when necessary

## Constraint Signal

The current implementation uses the base environment's own per-step velocity cost as the constraint signal.

That means the switch policy penalty is tied to the same cost logic already defined by the velocity task.

This is useful because:

- it avoids hard-coding a second competing speed-limit implementation
- it keeps the switch stage aligned with the environment's official constraint

## Health / Stability Signal

The script uses a best-effort shared health check:

1. if `info["is_healthy"]` exists, use it
2. else if `env.unwrapped.is_healthy` exists, use it
3. else fall back to `not terminated`

This is intentionally conservative because the three target robots do not expose a perfectly uniform posture API through one common wrapper.

Important detail:

- health is used internally by the environment reward / termination logic
- health is not exposed as an extra recovery-policy or switch-policy observation

## Recommended Equilibrium Definitions

My recommendation is to define equilibrium as a low-energy, healthy steady state that must hold for several consecutive steps, not just a single low-speed instant.

### Shared template

Use all of these together:

- healthy according to the env
- speed below a robot-specific stop threshold
- low generalized-velocity norm
- robot-specific torso posture / height bounds
- hold for `20-40` consecutive steps before success

### Ant

Recommended equilibrium:

- upright standing rest

Recommended checks:

- torso height remains within a narrow standing band
- torso pitch and roll remain small
- COM speed remains below the stop threshold
- joint / generalized velocities remain small

### Humanoid

Recommended equilibrium:

- upright standing rest

Recommended checks:

- pelvis / torso height remains within a standing band
- torso pitch and roll remain small
- COM speed remains below the stop threshold
- generalized velocities remain small

### HalfCheetah

Recommended equilibrium:

- low-energy settled rest state, not strict upright standing

Recommended checks:

- forward speed remains below the stop threshold
- torso angular rate remains small
- generalized velocities remain small
- torso height remains within a non-crashing band
- no repeated flipping / tumbling

Reason:

- HalfCheetah is not naturally a static upright balancer like Ant or Humanoid, so “stopped and settled” is a better recovery target than “standing upright”

### Practical recommendation

If I were implementing the next version of TODO 2, I would:

1. Keep the current generic `is_healthy` logic as a hard safety floor.
2. Add robot-specific equilibrium predicates.
3. Require the predicate to hold for several consecutive steps.
4. Use a low generalized-velocity norm so a robot does not count as “recovered” while briefly pausing mid-fall.

## Directory Layout

Default output root:

- `examples/benchmarks/exp-final-velocity`

Per robot:

- `exp-final-velocity/ant`
- `exp-final-velocity/halfcheetah`
- `exp-final-velocity/humanoid`

Per stage:

- `exp-final-velocity/<robot>/nominal`
- `exp-final-velocity/<robot>/recovery`
- `exp-final-velocity/<robot>/switch`
- `exp-final-velocity/<robot>/evaluation`

Inside each stage, OmniSafe writes its normal run layout:

```text
exp-final-velocity/
  ant/
    nominal/
      PPO-{SafetyAntVelocity-v1}/
        seed-000-YYYY-MM-DD-HH-MM-SS/
          config.json
          progress.csv
          tb/
          torch_save/
            epoch-0.pt
            epoch-10.pt
            ...
```

## CLI Usage

### Train everything for all three robots

```bash
python examples/benchmarks/RL_final_velocity.py
```

### Train only Ant end-to-end

```bash
python examples/benchmarks/RL_final_velocity.py --robots ant
```

### Train only the nominal policies

```bash
python examples/benchmarks/RL_final_velocity.py --stages nominal
```

### Train recovery after nominal already exists

```bash
python examples/benchmarks/RL_final_velocity.py --stages recovery
```

### Train only the switch stage after nominal and recovery exist

```bash
python examples/benchmarks/RL_final_velocity.py --stages switch
```

### Retrain recovery and switch after the recovery observation change

```bash
python examples/benchmarks/RL_final_velocity.py --stages recovery switch --force
```

### Force re-training of selected stages

```bash
python examples/benchmarks/RL_final_velocity.py --stages switch --force
```

### Evaluate and visualize saved switch checkpoints

```bash
python examples/benchmarks/RL_final_velocity.py --evaluate-switch
```

### Train switch and then evaluate it

```bash
python examples/benchmarks/RL_final_velocity.py --stages switch --evaluate-switch
```

### Example with custom budgets

```bash
python examples/benchmarks/RL_final_velocity.py \
  --robots ant halfcheetah humanoid \
  --nominal-total-steps 1000000 \
  --recovery-total-steps 500000 \
  --switch-total-steps 500000 \
  --device cuda:0
```

## Current Defaults

### Training budgets

- nominal: `1,000,000` steps
- recovery: `500,000` steps
- switch: `500,000` steps

### PPO steps per epoch

- nominal: `20,000`
- recovery: `10,000`
- switch: `10,000`

### Seeds

- default seed list: `[0]`

### Evaluation

- evaluation episodes: `5`
- trace plots saved for the first `2` episodes per policy

## Important Implementation Notes

### Sequential runtime context

The script stores the currently active nominal and recovery checkpoint paths in a module-level runtime context.

Why this is okay here:

- the script trains stages sequentially
- it sets `parallel = 1`
- the custom env only needs to know which frozen actor to load for the current robot and seed

### Recovery reset uses reachable states

The recovery env does not sample arbitrary MuJoCo state vectors directly.

Instead it creates harder initial states by:

- resetting the base env
- rolling forward with either the nominal actor or random actions
- starting recovery from the reached state

This keeps the recovery policy on the task's reachable state manifold.

### Switch evaluation and visualization

The script now includes a dedicated switch-policy evaluation path.

It evaluates three gate policies:

- `learned_switch`
- `always_nominal`
- `always_recovery`

For each robot and seed it writes:

- `episode_metrics.csv`
- `step_traces.csv`
- `summary.json`
- trace plots under `plots/` when `matplotlib` is available

The current plots show:

- speed
- cost
- gate decision
- reward

## TODOs Left in the Code

### TODO 1: native discrete switch policy

The current gate is thresholded from a continuous action.

Future improvement:

- replace with a categorical switch actor when the local PPO stack supports it cleanly

### TODO 2: explicit posture / equilibrium metrics

The current health check is intentionally generic.

Future improvement:

- add robot-specific equilibrium checks using exact torso height, pitch, roll, and generalized-velocity thresholds

### TODO 3: direct state initialization

The recovery env currently uses burn-in rollouts instead of direct `qpos/qvel` injection.

Future improvement:

- add exact low-level state sampling if we want broader coverage than reachable burn-in states

### TODO 4: expose the real speed limit as an observation feature

Future improvement:

- expose a clean `speed_margin_to_limit` feature if Safety-Gymnasium provides the threshold uniformly across all three robots

## Extension Ideas

If we keep building this scenario, the next high-value additions are:

- a switch-policy evaluation loop with baseline comparisons
- a plotter for speed and gate traces
- paired same-seed nominal/recovery/switch summary tables
- a deployment script that visualizes when switching happens

## Practical Summary

Implemented now:

- staged training flow
- custom recovery env
- custom switch env
- checkpoint loading for frozen PPO actors
- robot coverage for Ant, HalfCheetah, Humanoid
- reusable output structure
- built-in switch evaluation and trace visualization
- documented TODOs where the design is still heuristic

Not implemented yet:

- robot-specific equilibrium metrics beyond the shared health heuristic
- true categorical discrete gate
