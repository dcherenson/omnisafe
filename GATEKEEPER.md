# Gatekeeper-RL on safe-control-gym Quadrotor

This document explains the three-policy, two-stage pipeline we built on top of
OmniSafe + `safe-control-gym` to study a **gatekeeper-style forward-simulation
safety filter** around a reinforcement-learning switching policy for a 2-D
quadrotor that tracks a circle reference while avoiding a forbidden region.
The reward is shaped with a filter-override penalty that trains the RL policy
to preemptively pick the safe action itself, so the filter rarely has to
intervene at deployment.

If you have never seen this project before, read the whole file. If you just
want to train and evaluate, jump to [Running the pipeline](#running-the-pipeline).

---

## 1. The problem

A 2-D quadrotor (PyBullet, via `safe-control-gym`) must track a fixed circle
reference trajectory `X_GOAL` in the `x`–`z` plane while staying inside a safe
box:

| dim | lower | upper | meaning         |
|-----|-------|-------|-----------------|
| `x` | −2.0  | +0.3  | no-fly wall at `x = 0.3 m` |
| `z` | +0.3  | +1.7  | ground/ceiling             |
| `θ` | −0.35 | +0.35 | pitch limit (rad)          |

Each control step is at 60 Hz, each physics sub-step at 240 Hz. One full
episode is 8 s (≈ 480 control steps, two full laps of the circle).

### Why a gatekeeper?

Pure task-reward RL will happily fly into the `x = 0.3` wall if that locally
maximises tracking reward. Classical CMDP methods (CPO, FOCOPS) learn to
respect a cost budget **on average**, but still produce unsafe rollouts while
exploring. We want hard per-step safety even during training.

Our approach:

1. A **gatekeeper** forward-simulation filter (Agrawal et al., 2022) —
   forward-simulate the proposed plan for a horizon `H`; if the preview
   exits the safe set, reject it and fall back to a known-safe backup plan.
2. A **filter-override shaping reward** — a small penalty every time the
   filter rejects the RL's nominal proposal. This trains the policy to
   internalise the filter so it proposes nominal only when the proposal is
   likely to be safe, reducing filter interventions at deployment.

The safe set here is a hard axis-aligned box on `(x, z, θ)`; there is no
continuous barrier function, no Lie-derivative constraint, and no QP.

---

## 2. Three policies

```
                           +-------------------+
  circle reference ------->|  switching π_S    |  (RL, trained in Stage 2)
                           |  picks {nom, bak} |
  backup policy π_B ------>|                   |
  (frozen, trained          +---------+---------+
   in Stage 1)                        |
                                      v
                          +-----------------------+
                          |   gatekeeper filter   |  (forward-sim preview)
                          |   "verify-before-     |
                          |    commit"            |
                          +-----------+-----------+
                                      |
                                      v
                          +-----------------------+
                          |  low-level PID (π_LL) |  (classical, not trained)
                          |  tracks 6-D reference |
                          +-----------+-----------+
                                      |
                                      v
                            PyBullet quadrotor
```

### π_LL — low-level controller (classical, not trained)

`safe-control-gym`'s DSL PID position + attitude controller. Given
`(state_6d, reference_6d)` it emits physical motor thrusts. Stateful (has
integrator / derivative memory), so we `save_state` / `load_state` around
gatekeeper previews. Defined in `ClassicalLowLevel` in
`omnisafe/envs/scg_gatekeeper_env.py`.

### π_B — backup policy (learned in Stage 1, then frozen)

Learns to stabilise the quadrotor from arbitrary initial states.

- **State** (6-D): `[x, vx, z, vz, θ, θ̇]`
- **Action** (2-D): `a ∈ [−1, 1]²`, mapped to a target `(x_tgt, z_tgt)`.
  The env then builds the 6-D PID reference `[x_tgt, 0, z_tgt, 0, 0, 0]`
  — i.e. "hover at this point with zero velocity".
- **Per-step reward**:

  ```
    + alive_bonus
    − w_vel  * (vx² + vz²)
    − w_ang  * θ²
    − w_rate * θ̇²
    − w_pos  * (z − z_nom)²    (mild centering)
    + stable_bonus  if |v| < 0.15 and |θ| < 0.05
    − crash_penalty (10.0)      if state leaves the safe box (episode ends)
  ```

- **Rich initialisation.** Each `reset` samples random `x, z` inside the safe
  box and random velocities / tilt, then uses PyBullet's
  `resetBasePositionAndOrientation` + `resetBaseVelocity` to inject the state.
  This is essential — a backup trained only from near-hover cannot recover
  tumbling drones at deployment.

- **Trained via plain PPO** with default OmniSafe hyper-parameters.

Once trained, only the actor weights and observation normaliser are used;
`FrozenBackupPolicy` loads them and exposes a single method
`reference(state) -> 6-D ref`.

### π_S — switching policy (learned in Stage 2)

Outputs a per-decision binary choice: "track the nominal circle" or "track
what the backup suggests".

- **State** (7-D): `[state_6d, t / T_total]` — physical state plus the
  normalised episode clock.
- **Action** (1-D, continuous): `a ∈ [0, 1]`, thresholded at `0.5` to
  `{0: backup, 1: nominal}`. Continuous-then-thresholded gives PPO a smooth
  stochastic policy while the env consumes a clean binary decision.
- **Decision cadence.** A *decision epoch* is every `Δ = 5` control steps
  (≈ 83 ms). Between decisions, the chosen mode is held constant. This is
  receding-horizon: plan for `H`, execute `Δ`, replan.
- **Per-decision reward** (over `Δ` real steps in the committed mode):

  ```
              +R_nom   if RL chose nominal, filter accepted, no violation
  r_choice = { −R_over if RL chose nominal but filter overrode (and shaping is on)
              { 0       otherwise (RL chose backup)

  r_crash  = −R_crash · n_violations     (one per per-step violation)

  reward   = r_choice + r_crash
  ```

  Defaults: `R_nom = 1.0`, `R_over = 0.2`, `R_crash = 20.0`.

  Key property: **RL is never rewarded for picking backup**, so there is no
  "trivial return" from always hiding. But *crashes* are punished heavily
  regardless of who is to blame, so picking nominal-and-crashing is worse than
  picking backup-and-living.

---

## 3. The gatekeeper filter

Sits between π_S's proposal and the simulator. Only fires when π_S picks
nominal:

```
def step(a_rl):                           # a_rl in {0: bak, 1: nom}
    if use_filter and a_rl == 1:
        snap = pybullet.saveState(); save_pid_integrators()
        ok = simulate_keeping_mode(mode=1, steps=H=60)    # preview
        pybullet.restoreState(snap);     restore_pid_integrators()
        if not ok:
            committed = 0                 # OVERRIDE to backup
            rejected  = True
        else:
            committed = 1
    else:
        committed = a_rl

    n_violations = run_real_delta_steps(mode=committed)   # Δ = 5 real steps
    # compute r_choice, r_crash as above
```

Because `Δ ≤ H`, a previously-accepted plan is still safe for the next `Δ`
steps of real execution. Preview failures cleanly roll back thanks to
`pybullet.saveState` + PID integrator snapshots.

We deliberately do **not** preview backup proposals. The backup is trusted
by construction; if it were untrustworthy we would not be using it as a
safety fallback.

---

## 4. Ablations

Controlled by two env vars read inside the gatekeeper env.

| Variant        | `GK_USE_FILTER` | `GK_USE_SHAPING` | What it tests |
|----------------|-----------------|------------------|---------------|
| `proposed`     | True            | True             | Our full method: filter + override shaping |
| `filter_only`  | True            | False            | Hard filter but no reward signal about overrides |
| `reward_only`  | False           | True             | With the filter off, shaping *never fires*, so this degenerates to `nominal`. We leave it in for completeness — extending it would require an alternative safety signal (e.g. distance-to-boundary). |
| `nominal`      | False           | False            | Unprotected baseline; collides frequently |

---

## 5. Files

### Environments (`omnisafe/envs/`)

- `scg_backup_env.py` — Stage-1 env (`SCG-Quadrotor-Backup-v0`). Rich-init
  standalone stabiliser.
- `scg_gatekeeper_env.py` — Stage-2 env (`SCG-Quadrotor-Gatekeeper-v0`).
  Holds the `ClassicalLowLevel`, `FrozenBackupPolicy` and gatekeeper preview.
- `__init__.py` — registers both envs lazily (try/except so optional
  dependencies don't break OmniSafe's default import).

### Training scripts (`examples/benchmarks/`)

- `train_backup.py` — trains π_B (default: 2 seeds × 60 k steps, ~2 min CPU).
- `train_gatekeeper.py` — trains π_S across the four ablation variants
  (default: 2 seeds × 9.6 k steps each, ~7 min CPU) and then runs evaluation
  (baselines + trained policies with filter on/off). Writes
  `eval_results.csv` and prints a summary table.

### Visualisation

- `visualize_gatekeeper.py` — PyBullet-GUI rollout of a trained π_S with
  the filter on or off. Colour-coded cues:
  - blue circle = the nominal reference
  - red wall    = `x = 0.3` boundary
  - yellow line = reference the PID is tracking this step
  - green pulse = RL picked nominal, filter accepted
  - red pulse   = RL picked nominal, filter overrode
  - blue pulse  = RL picked backup directly

### Environment variables consumed by the envs

| Var | Default | Meaning |
|-----|---------|---------|
| `GK_H` | 60 | Preview horizon (control steps, ≈ 1 s) |
| `GK_DELTA` | 5 | Receding-horizon stride (control steps, ≈ 83 ms) |
| `GK_R_NOM` | 1.0 | Reward for a committed safe nominal decision |
| `GK_R_OVERRIDE` | 0.2 | Shaping penalty when RL's nominal is overridden |
| `GK_R_CRASH` | 20.0 | Penalty per per-step safety violation |
| `GK_USE_FILTER` | true | Run the preview filter |
| `GK_USE_SHAPING` | true | Apply the override shaping penalty |
| `GK_BACKUP_RUN_DIR` | (auto) | Path to a trained backup run dir |
| `GK_BACKUP_MODEL` | latest | Specific `epoch-XXX.pt` to load |
| `GK_BACKUP_FALLBACK` | true | If loading fails, fall back to fixed-hover reference |
| `GK_GUI` | false | Enable PyBullet GUI inside the Stage-2 env |
| `BACKUP_MAX_STEPS` | 120 | Episode length of the Stage-1 env |

---

## 6. Running the pipeline

### Prerequisites

An OmniSafe conda/venv with `safe-control-gym`, `pybullet`, `gymnasium`.
`train_backup.py` and `train_gatekeeper.py` both run on CPU.

### Stage 1 — backup policy

```bash
python examples/benchmarks/train_backup.py
```

Saves to `examples/benchmarks/exp-backup/PPO_backup/.../seed-XXX-<timestamp>/`.
Takes ~2 minutes on CPU. Inspect
`exp-backup/.../progress.csv` — a healthy run climbs from `EpRet ≈ −9`
(crash-dominated) to `EpRet ≈ +9`, with `EpLen ≈ 115/120` (survives almost
the whole episode).

### Stage 2 — switching policy + ablations + eval

```bash
# Auto-picks the most recent backup run under exp-backup/
python examples/benchmarks/train_gatekeeper.py

# Or point it explicitly
python examples/benchmarks/train_gatekeeper.py \
    --backup-run-dir examples/benchmarks/exp-backup/PPO_backup/.../seed-005-<ts>

# Only the primary variant
python examples/benchmarks/train_gatekeeper.py --variants proposed

# Re-run eval only, skip training
python examples/benchmarks/train_gatekeeper.py --eval-only
```

Saves to `examples/benchmarks/exp-gk/gatekeeper/<variant>/PPO_<variant>/...`.
The script then runs a self-contained evaluation loop against:

- `b1_always_nominal_filter` — always propose nominal, let the filter catch
  bad proposals (upper bound on return under the filter).
- `b2_always_backup` — floor on return and violation.
- `b3_random_filter` — sanity-check random baseline.
- `b1b_always_nominal_nofilter` — stress test (shows how much the filter
  matters).
- Every trained variant × seed, deployed **with** and **without** the filter.

Writes `exp-gk/gatekeeper/eval_results.csv` and prints a summary table.

### Reading the eval table

Per label, the script reports:

| col | meaning |
|-----|---------|
| `ret` | mean episode return over 5 eval episodes |
| `viol` | mean safety violations per episode |
| `rej` | mean filter rejections per episode |
| `%rl_nom` | fraction of decisions where RL proposed nominal |
| `%exec_nom` | fraction of decisions where nominal was actually executed |
| `ms` | mean time per decision (milliseconds) |

Typical story from our runs:

- `b1_always_nominal_filter` gets the highest return (`+49.6`) because it
  defers all safety to the filter — but at the cost of **24 filter
  interventions per episode**.
- A trained `proposed` policy achieves the same `0.2 viol/ep` as `b1`
    with only **~0.6 filter interventions per episode** — a 40× reduction.
    This is the filter-internalisation effect: the RL has moved the
    filter's preferences inside its own policy.
- With the filter turned off at deploy, `proposed` degrades gracefully
  (~0.6–1.0 viol/ep) versus always-nominal (1.8 viol/ep). Not zero, but
  clearly better than unprotected — and with longer training this gap is
  expected to grow.

### Visualisation

```bash
# With the filter, variant proposed, seed 0
python examples/benchmarks/visualize_gatekeeper.py --variant proposed --seed 0

# Same policy but without the filter at deploy (stress test)
python examples/benchmarks/visualize_gatekeeper.py --variant proposed --seed 0 --deploy-nofilter
```

---

## 7. Tuning knobs

All reward terms and horizons are CLI flags on `train_gatekeeper.py` and
also exposed as env vars for direct use of the env:

- **`--r-override` / `GK_R_OVERRIDE`** — lower values (e.g. `0.05`) push the
  policy to propose nominal more often; higher values make it hyper-cautious.
- **`--r-crash` / `GK_R_CRASH`** — should stay much larger than `r-override`
  and `r-nom`. If the policy ignores crashes, increase.
- **`--H` / `GK_H`** — longer previews catch more distant hazards but cost
  more sim time per decision. `H = Δ` degenerates to a myopic one-step
  check.
- **`--delta` / `GK_DELTA`** — smaller `Δ` = more frequent replans, better
  reactivity, slower training. Must satisfy `Δ ≤ H`.

---

## 8. What we deliberately do *not* do

- We are **not** solving a CMDP in the CPO / FOCOPS sense. There is no cost
  budget. Our "cost" returned by the env is always zero; safety is enforced
  by the filter and shaped into the reward instead.
- We do **not** relearn the low-level controller for the quadrotor — we use
  the classical DSL PID. This keeps Stage 1 focused on the safety-recovery
  behaviour rather than basic actuation.
- We do **not** preview backup plans. They are trusted by construction.
  A broken backup would break the whole pipeline, which is a deliberate
  design choice: the backup is the anchor of the safety guarantee.

---

## 9. Extending

Possible next steps if you want to build on this:

- **Fix `reward_only`.** Add a dense shaping term that fires without the
  filter (e.g. `−β · max(0, |x| − x_safe)²`). Then the ablation compares
  "filter only" vs "reward only" vs "both" properly.
- **Richer switching state.** Add a short history buffer or the last
  reference waypoint so π_S can anticipate upcoming wall approaches.
- **Stochastic / noisy dynamics.** `safe-control-gym` supports disturbance
  injection; see whether the learned self-filtering generalises.
- **Port to a different platform.** The split of three policies
  (classical LL + learned stabiliser + learned switch) is platform-agnostic.
  The pieces that change for a new robot are:
  - Stage-1 env: state dim, action-to-reference map, crash criterion.
  - Stage-2 env: safety box (or a Lyapunov-style set), reference signal.
  - `ClassicalLowLevel` wrapper: whatever stock controller ships with the
    platform.
