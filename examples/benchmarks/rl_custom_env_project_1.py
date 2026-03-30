# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RL Custom Env Project 1 — Paper-Faithful Velocity + Circle Benchmark.

Goal: reproduce FOCOPS paper (Zhang et al., NeurIPS 2020, arXiv:2002.06506)
velocity and circle results using OmniSafe with CPO and FOCOPS.

═══════════════════════════════════════════════════════════════════════════════
DIRECTORY LAYOUT
═══════════════════════════════════════════════════════════════════════════════
  Each env gets a parent directory.  CPO and FOCOPS each get their own
  sub-directory inside it, so OmniSafe's "already-exists" guard never fires.

  exp-x/
    RL-Custom-Env-Project-1-Hopper/
      CPO/    ← CPO grid (exp_name='CPO', parent_dir=above)
      FOCOPS/ ← FOCOPS grid (exp_name='FOCOPS', parent_dir=above)
    RL-Custom-Env-Project-1-Circle/
      CPO/
      FOCOPS/

  Analysis loads the shared parent dir so both algos appear on one plot.

═══════════════════════════════════════════════════════════════════════════════
EXPERIMENT COUNT
═══════════════════════════════════════════════════════════════════════════════
  Velocity  : 2 algos × 6 envs × 2 seeds = 24  (CPO then FOCOPS, 2 seeds parallel)
  Circle    : 2 algos × 3 envs × 2 seeds = 12  (CPO then FOCOPS, 3 seeds parallel)
  Total     : 36 experiments
  Est. time : ~20–28 hours
"""

import datetime
import os
import time
import warnings
import torch

import omnisafe.envs.paper_velocity_envs  # registers *Paper-v1 envs  # noqa: F401
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.common.statistics_tools import StatisticsTools
from omnisafe.utils.exp_grid_tools import train


def _stamped(msg: str) -> None:
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ── Velocity env config — paper Table 1 / Appendix G.1 ───────────────────────
VELOCITY_ENV_CFG = {
    'SafetyHopperVelocityPaper-v1':        {'cost_limit':  82.748},
    'SafetySwimmerVelocityPaper-v1':       {'cost_limit':  24.516},
    'SafetyWalker2dVelocityPaper-v1':      {'cost_limit':  81.886},
    'SafetyAntVelocityPaper-v1':           {'cost_limit': 103.115},
    'SafetyHalfCheetahVelocityPaper-v1':   {'cost_limit': 151.989},
    'SafetyHumanoidVelocityPaper-v1':      {'cost_limit':  20.140},
}

CIRCLE_ENVS       = ['SafetyPointCircle1-v0', 'SafetyAntCircle1-v0', 'SafetyCarCircle1-v0']
CIRCLE_COST_LIMIT = 50.0


# ── Shared helper ─────────────────────────────────────────────────────────────

def _base_grid(algo: str, env_ids: list, steps_per_epoch: int) -> ExperimentGrid:
    """Keys that exist in every on-policy algorithm's config.

    exp_name is set to the algo name ('CPO' or 'FOCOPS') so each grid saves
    to its own sub-directory inside the caller-supplied parent_dir.
    Note: cost_limit is NOT set here — CPO uses algo_cfgs:cost_limit,
    FOCOPS uses lagrange_cfgs:cost_limit. Each algo grid sets it separately.
    """
    eg = ExperimentGrid(exp_name=algo)
    eg.add('algo',   [algo])
    eg.add('env_id', env_ids)
    eg.add('seed',   [0, 5])
    eg.add('logger_cfgs:use_wandb',       [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('algo_cfgs:steps_per_epoch',   [steps_per_epoch])
    eg.add('train_cfgs:total_steps',      [1_000_000])
    eg.add('train_cfgs:vector_env_nums',  [1])
    eg.add('train_cfgs:torch_threads',    [1])
    return eg


# ── CPO grids ─────────────────────────────────────────────────────────────────

def _make_cpo_velocity(env_id: str, cost_limit: float) -> ExperimentGrid:
    eg = _base_grid('CPO', [env_id], steps_per_epoch=2048)
    eg.add('algo_cfgs:cost_limit', [cost_limit])  # CPO reads cost_limit from algo_cfgs
    eg.add('algo_cfgs:cg_damping', [0.01])
    eg.add('algo_cfgs:cg_iters',   [10])
    eg.add('model_cfgs:critic:lr', [3e-4])
    return eg


def _make_cpo_circle() -> ExperimentGrid:
    eg = _base_grid('CPO', CIRCLE_ENVS, steps_per_epoch=50_000)
    eg.add('algo_cfgs:cost_limit', [CIRCLE_COST_LIMIT])
    eg.add('algo_cfgs:gamma',      [0.995])
    eg.add('algo_cfgs:cost_gamma', [0.995])
    eg.add('algo_cfgs:cg_damping', [0.01])
    eg.add('algo_cfgs:cg_iters',   [10])
    eg.add('model_cfgs:critic:lr', [3e-4])
    return eg


# ── FOCOPS grids ──────────────────────────────────────────────────────────────

def _make_focops_velocity(env_id: str, cost_limit: float) -> ExperimentGrid:
    eg = _base_grid('FOCOPS', [env_id], steps_per_epoch=2048)
    eg.add('lagrange_cfgs:cost_limit', [cost_limit])  # FOCOPS reads cost_limit from lagrange_cfgs
    eg.add('lagrange_cfgs:lambda_lr',  [0.01])
    eg.add('algo_cfgs:update_iters',   [10])
    return eg


def _make_focops_circle() -> ExperimentGrid:
    eg = _base_grid('FOCOPS', CIRCLE_ENVS, steps_per_epoch=50_000)
    eg.add('algo_cfgs:gamma',          [0.995])
    eg.add('algo_cfgs:cost_gamma',     [0.995])
    eg.add('lagrange_cfgs:cost_limit', [CIRCLE_COST_LIMIT])
    eg.add('lagrange_cfgs:lambda_lr',  [0.01])
    eg.add('algo_cfgs:update_iters',   [10])
    eg.add('algo_cfgs:focops_lam',     [1.0])
    eg.add('algo_cfgs:focops_eta',     [0.04])
    return eg


# ── Run a CPO+FOCOPS pair concurrently inside a shared parent dir ─────────────

def _has_training_results(parent_dir: str, algo: str) -> bool:
    """Return True only if actual training data (progress.csv) exists for this algo."""
    out_dir = os.path.join(parent_dir, algo)
    if not os.path.isdir(out_dir):
        return False
    for root, dirs, files in os.walk(out_dir):
        if 'progress.csv' in files:
            return True
    return False


def _run_grid_if_needed(label: str, algo: str, eg: ExperimentGrid,
                        parent_dir: str, gpu_id, num_pool: int) -> None:
    """Run grid only if training results don't already exist (resumable).

    - If progress.csv exists  → training completed, skip.
    - If directory exists but has no progress.csv → failed/empty run,
      delete the directory so OmniSafe can create it fresh.
    - If directory doesn't exist → run normally.
    """
    import shutil
    out_dir = os.path.join(parent_dir, algo)
    if _has_training_results(parent_dir, algo):
        _stamped(f'  [{label}] {algo} already completed — skipping.')
        return
    if os.path.exists(out_dir):
        _stamped(f'  [{label}] {algo} directory exists but has no results — removing stale dir.')
        shutil.rmtree(out_dir)
    _stamped(f'  [{label}] Running {algo}...')
    eg.run(train, num_pool=num_pool, gpu_id=gpu_id, parent_dir=parent_dir)


def _run_pair(label: str, parent_dir: str, cpo_eg: ExperimentGrid,
              focops_eg: ExperimentGrid, gpu_id, num_pool: int = 2) -> None:
    """Run CPO then FOCOPS sequentially. Skips whichever already finished."""
    os.makedirs(parent_dir, exist_ok=True)

    t0 = time.time()
    _run_grid_if_needed(label, 'CPO',    cpo_eg,    parent_dir, gpu_id, num_pool)
    _run_grid_if_needed(label, 'FOCOPS', focops_eg, parent_dir, gpu_id, num_pool)

    elapsed = datetime.timedelta(seconds=int(time.time() - t0))
    _stamped(f'  [{label}] done in {elapsed}.')


# ── Analysis ──────────────────────────────────────────────────────────────────

def _analyze(parent_dir: str, cost_limit: float) -> None:
    """Plots CPO vs FOCOPS using all runs found under parent_dir."""
    if not os.path.isdir(parent_dir):
        _stamped(f'  [WARN] {parent_dir} not found, skipping.')
        return
    plot_dir = os.path.join(parent_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    orig = os.getcwd()
    os.chdir(plot_dir)
    try:
        st = StatisticsTools()
        st.load_source(parent_dir)
        st.draw_graph(
            parameter='algo',
            values=['CPO', 'FOCOPS'],
            compare_num=None,
            cost_limit=cost_limit,
            smooth=10,
            show_image=False,
        )
        plots = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        _stamped(f'  {len(plots)} plot(s) → {plot_dir}')
    finally:
        os.chdir(orig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _stamped('Starting RL Custom Env Project 1 — Paper-Faithful Benchmark...')
    _stamped('Paper: FOCOPS (Zhang et al., NeurIPS 2020, arXiv:2002.06506)')
    print('=' * 72, flush=True)

    # ── Print resume status upfront ───────────────────────────────────────────
    _stamped('Checking existing results (resume status):')
    for env_id, cfg in VELOCITY_ENV_CFG.items():
        env_short  = env_id.replace('Safety', '').replace('VelocityPaper-v1', '')
        parent_dir = os.path.join('exp-x', f'RL-Custom-Env-Project-1-{env_short}')
        for algo in ('CPO', 'FOCOPS'):
            status = 'DONE (will skip)' if _has_training_results(parent_dir, algo) else 'pending'
            print(f'    {env_short:12s} {algo:6s} — {status}', flush=True)
    circle_parent = os.path.join('exp-x', 'RL-Custom-Env-Project-1-Circle')
    for algo in ('CPO', 'FOCOPS'):
        status = 'DONE (will skip)' if _has_training_results(circle_parent, algo) else 'pending'
        print(f'    {"Circle":12s} {algo:6s} — {status}', flush=True)
    print('=' * 72, flush=True)

    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    if not set(gpu_id).issubset(available_gpus):
        warnings.warn(
            f'Requested GPU IDs {gpu_id} not available (found {available_gpus}). '
            'Falling back to CPU.',
            stacklevel=1,
        )
        gpu_id = None

    _stamped(f'Device  : {"GPU " + str(gpu_id) if gpu_id else "CPU"}')
    _stamped(f'Total   : 24 velocity + 12 circle = 36 experiments')
    _stamped(f'Order   : CPO then FOCOPS per env (2 seeds parallel each)')
    _stamped(f'Logging : TensorBoard (WandB disabled)')
    print('=' * 72, flush=True)

    t_total = time.time()
    exp_registry: list[tuple[str, float]] = []   # (parent_dir, cost_limit)

    # ── PART 1: Velocity ──────────────────────────────────────────────────────
    _stamped('PART 1 — Velocity (cost = actual speed, per-env cost limits)')
    print('-' * 72, flush=True)

    for env_id, cfg in VELOCITY_ENV_CFG.items():
        cost_limit = cfg['cost_limit']
        env_short  = env_id.replace('Safety', '').replace('VelocityPaper-v1', '')
        parent_dir = os.path.join('exp-x', f'RL-Custom-Env-Project-1-{env_short}')
        exp_registry.append((parent_dir, cost_limit))

        _stamped(f'  [{env_short}] CPO ‖ FOCOPS  cost_limit={cost_limit}')
        _run_pair(
            env_short, parent_dir,
            _make_cpo_velocity(env_id, cost_limit),
            _make_focops_velocity(env_id, cost_limit),
            gpu_id, num_pool=2,
        )
        print('-' * 72, flush=True)

    # ── PART 2: Circle ────────────────────────────────────────────────────────
    _stamped('PART 2 — Circle (binary cost, threshold=50, γ=0.995)')
    print('-' * 72, flush=True)

    circle_parent = os.path.join('exp-x', 'RL-Custom-Env-Project-1-Circle')
    exp_registry.append((circle_parent, CIRCLE_COST_LIMIT))

    _run_pair(
        'Circle', circle_parent,
        _make_cpo_circle(),
        _make_focops_circle(),
        gpu_id, num_pool=3,   # 3 envs × 2 seeds = 6, but 3 parallel per algo
    )
    print('=' * 72, flush=True)

    _stamped(f'All training done in '
             f'{datetime.timedelta(seconds=int(time.time()-t_total))}.')

    # ── Analysis ──────────────────────────────────────────────────────────────
    _stamped('Running analysis...')
    for parent_dir, cost_limit in exp_registry:
        _stamped(f'  {parent_dir}  (cost_limit={cost_limit})')
        _analyze(parent_dir, cost_limit)
    _stamped('All done.')
