# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Run CPO and FOCOPS on safe-control-gym environments (CartPole, Quadrotor).

Trains both algorithms on:
  * SCG-CartPole-Stabilization-v0    (pole-angle + cart-position constraints)
  * SCG-Quadrotor-Stabilization-v0   (altitude + lateral-position constraints)
  * SCG-Quadrotor-Tracking-v0        (circle trajectory + forbidden right zone x>0.3)

USAGE
=====
  # From the omnisafe/ directory (train all envs):
  python examples/benchmarks/run_scg_project.py

  # Train only the tracking task:
  python examples/benchmarks/run_scg_project.py --envs SCG-Quadrotor-Tracking-v0

  # Use GPU 0:
  python examples/benchmarks/run_scg_project.py --gpu 0
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import time
import warnings

import torch

# Triggers @env_register for all SCG environments.
import omnisafe.envs.safe_control_gym_env  # noqa: F401

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

ENV_CFGS: dict[str, dict] = {
    'SCG-CartPole-Stabilization-v0': {
        'cost_limit': 50.0,   # relaxed: let it learn to balance before enforcing hard limits
        'steps_per_epoch': 2000,
        'total_steps': 1_000_000,  # 2x more training
    },
    'SCG-Quadrotor-Stabilization-v0': {
        'cost_limit': 25.0,
        'steps_per_epoch': 3000,
        'total_steps': 500_000,
    },
    # 8-second episode (480 steps), forbidden zone x>0.3 covers ~23% of arc.
    # cost_limit=50 ~ 10% of 480 steps.
    'SCG-Quadrotor-Tracking-v0': {
        'cost_limit': 50.0,
        'steps_per_epoch': 4800,
        'total_steps': 600_000,
    },
}

SEEDS = [0, 5]

# Results always land next to this script, regardless of CWD.
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp-scg')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def _has_results(parent_dir: str, algo: str) -> bool:
    algo_dir = os.path.join(parent_dir, algo)
    if not os.path.isdir(algo_dir):
        return False
    for _, _, files in os.walk(algo_dir):
        if 'progress.csv' in files:
            return True
    return False


# ---------------------------------------------------------------------------
# ExperimentGrid builders
# ---------------------------------------------------------------------------

def _base_grid(algo: str, env_id: str, cfg: dict) -> ExperimentGrid:
    eg = ExperimentGrid(exp_name=algo)
    eg.add('algo',   [algo])
    eg.add('env_id', [env_id])
    eg.add('seed',   SEEDS)
    eg.add('logger_cfgs:use_wandb',       [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('algo_cfgs:steps_per_epoch',   [cfg['steps_per_epoch']])
    eg.add('train_cfgs:total_steps',      [cfg['total_steps']])
    eg.add('train_cfgs:vector_env_nums',  [1])
    eg.add('train_cfgs:torch_threads',    [1])
    return eg


def _make_cpo_grid(env_id: str, cfg: dict) -> ExperimentGrid:
    eg = _base_grid('CPO', env_id, cfg)
    eg.add('algo_cfgs:cost_limit', [cfg['cost_limit']])
    eg.add('algo_cfgs:cg_damping', [0.01])
    eg.add('algo_cfgs:cg_iters',   [10])
    eg.add('model_cfgs:critic:lr', [3e-4])
    return eg


def _make_focops_grid(env_id: str, cfg: dict) -> ExperimentGrid:
    eg = _base_grid('FOCOPS', env_id, cfg)
    eg.add('lagrange_cfgs:cost_limit', [cfg['cost_limit']])
    eg.add('lagrange_cfgs:lambda_lr',  [0.035])
    eg.add('algo_cfgs:update_iters',   [10])
    eg.add('algo_cfgs:focops_lam',     [1.5])
    eg.add('algo_cfgs:focops_eta',     [0.02])
    return eg


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_grid_if_needed(label, algo, eg, parent_dir, gpu_id, num_pool):
    if _has_results(parent_dir, algo):
        _log(f'  [{label}] {algo} already completed — skipping.')
        return

    algo_dir = os.path.join(parent_dir, algo)
    if os.path.exists(algo_dir):
        _log(f'  [{label}] {algo} has stale output — removing and re-running.')
        shutil.rmtree(algo_dir)

    _log(f'  [{label}] Running {algo}...')
    eg.run(train, num_pool=num_pool, gpu_id=gpu_id, parent_dir=parent_dir)


def _run_pair(env_id, parent_dir, cost_limit, gpu_id, num_pool=2):
    cfg = ENV_CFGS[env_id]
    os.makedirs(parent_dir, exist_ok=True)
    t0 = time.time()
    _run_grid_if_needed(env_id, 'CPO',    _make_cpo_grid(env_id, cfg),    parent_dir, gpu_id, num_pool)
    _run_grid_if_needed(env_id, 'FOCOPS', _make_focops_grid(env_id, cfg), parent_dir, gpu_id, num_pool)
    _log(f'  [{env_id}] Done in {datetime.timedelta(seconds=int(time.time()-t0))}.')


# ---------------------------------------------------------------------------
# Analysis / plotting
# ---------------------------------------------------------------------------

def _smooth(values, window=5):
    import numpy as np
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid').tolist()


def _find_progress_csvs(root):
    matches = []
    for dp, _, files in os.walk(root):
        if 'progress.csv' in files:
            matches.append(os.path.join(dp, 'progress.csv'))
    return matches


def _analyze(parent_dir, cost_limit, smooth=5):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        return

    algo_data: dict[str, list] = {'CPO': [], 'FOCOPS': []}
    for algo in algo_data:
        for csv_path in _find_progress_csvs(os.path.join(parent_dir, algo)):
            try:
                with open(csv_path, newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    algo_data[algo].append(rows)
            except Exception as exc:  # noqa: BLE001
                _log(f'  [WARN] Could not read {csv_path}: {exc}')

    metrics = [
        ('Metrics/EpRet',  'Episode Return',        False),
        ('Metrics/EpCost', 'Episode Cost',           True),
        ('Metrics/EpLen',  'Episode Length (steps)', False),
    ]

    plot_dir = os.path.join(parent_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    env_name = os.path.basename(parent_dir)
    colors = {'CPO': '#1f77b4', 'FOCOPS': '#ff7f0e'}
    n_plots = 0

    for mkey, mlabel, cost_line in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        any_data = False
        for algo, runs in algo_data.items():
            if not runs:
                continue
            step_lists, val_lists = [], []
            for r in runs:
                s = [float(x['TotalEnvSteps']) for x in r if mkey in x and x[mkey] != '']
                v = [float(x[mkey])            for x in r if mkey in x and x[mkey] != '']
                if s:
                    step_lists.append(s); val_lists.append(v)
            if not step_lists:
                continue
            mn = min(len(s) for s in step_lists)
            sx = np.array([s[:mn] for s in step_lists])
            vy = np.array([_smooth(v[:mn], smooth) for v in val_lists])
            mn2 = min(len(v) for v in vy)
            x = sx[0, :mn2]; y = np.array([v[:mn2] for v in vy])
            mean = y.mean(0); std = y.std(0)
            ax.plot(x, mean, label=algo, color=colors[algo], lw=2)
            ax.fill_between(x, mean-std, mean+std, alpha=0.2, color=colors[algo])
            any_data = True
        if not any_data:
            plt.close(fig); continue
        if cost_line and cost_limit is not None:
            ax.axhline(cost_limit, ls='--', c='red', lw=1.5, label=f'limit={cost_limit}')
        ax.set_xlabel('Total Environment Steps', fontsize=12)
        ax.set_ylabel(mlabel, fontsize=12)
        ax.set_title(f'{env_name}\n{mlabel}', fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); fig.tight_layout()
        sp = os.path.join(plot_dir, mkey.replace('/', '_') + '.png')
        fig.savefig(sp, dpi=150); plt.close(fig); n_plots += 1
    _log(f'  {n_plots} plot(s) saved -> {plot_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='*', default=None)
    parser.add_argument('--envs', type=str, nargs='*',
                        default=list(ENV_CFGS.keys()),
                        choices=list(ENV_CFGS.keys()))
    parser.add_argument('--output-dir', type=str, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = args.gpu
    if gpu_id is not None and not set(gpu_id).issubset(set(available_gpus)):
        warnings.warn(f'GPU {gpu_id} not available, falling back to CPU.', stacklevel=1)
        gpu_id = None

    _log('=' * 72)
    _log('CPO vs FOCOPS on safe-control-gym (CartPole, Quad-Stab, Quad-Tracking)')
    _log(f'Device      : {"GPU " + str(gpu_id) if gpu_id else "CPU"}')
    _log(f'Environments: {args.envs}')
    _log(f'Seeds       : {SEEDS}')
    _log(f'Output dir  : {args.output_dir}')
    _log('=' * 72)

    _log('Checking existing results:')
    for env_id in args.envs:
        parent_dir = os.path.join(args.output_dir, env_id)
        for algo in ('CPO', 'FOCOPS'):
            status = 'DONE (skip)' if _has_results(parent_dir, algo) else 'pending'
            print(f'    {env_id:42s}  {algo:6s}  {status}', flush=True)
    _log('=' * 72)

    t_total = time.time()
    exp_registry: list[tuple[str, float]] = []

    for env_id in args.envs:
        parent_dir = os.path.join(args.output_dir, env_id)
        cost_limit = ENV_CFGS[env_id]['cost_limit']
        exp_registry.append((parent_dir, cost_limit))
        _log(f'Environment: {env_id}  (cost_limit={cost_limit})')
        _run_pair(env_id=env_id, parent_dir=parent_dir, cost_limit=cost_limit,
                  gpu_id=gpu_id, num_pool=len(SEEDS))
        print('-' * 72, flush=True)

    _log(f'All training done in {datetime.timedelta(seconds=int(time.time()-t_total))}.')

    _log('Running analysis / plotting...')
    for parent_dir, cost_limit in exp_registry:
        _analyze(parent_dir, cost_limit)

    _log('All done.')
