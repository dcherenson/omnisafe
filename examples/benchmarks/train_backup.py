# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Stage-1 training: learn a BACKUP stabilizer for the SCG 2-D quadrotor.

Trains unconstrained PPO on ``SCG-Quadrotor-Backup-v0``.  The resulting actor
maps a 6-D state to a 2-D (x_target, z_target) that the DSL PID will hold.
Once trained, the run directory is consumed by ``train_gatekeeper.py`` via the
``GK_BACKUP_RUN_DIR`` environment variable (or the ``--backup-run-dir`` flag).

USAGE
=====
  # Default training (2 seeds, ~60k steps each, ~5 min on CPU)
  python examples/benchmarks/train_backup.py

  # Quick smoke test
  python examples/benchmarks/train_backup.py --total-steps 2000 --seeds 0
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import time
import warnings

import torch

import omnisafe.envs.scg_backup_env  # noqa: F401 — registers env

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


ENV_ID = 'SCG-Quadrotor-Backup-v0'
ALGO = 'PPO'

DEFAULT_SEEDS = [0, 5]
DEFAULT_TOTAL_STEPS = 60_000
DEFAULT_STEPS_PER_EPOCH = 960  # ~8 episodes per epoch (120 steps/ep)

_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'exp-backup',
)


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


def _make_ppo_grid(args: argparse.Namespace) -> ExperimentGrid:
    eg = ExperimentGrid(exp_name=f'{ALGO}_backup')
    eg.add('algo', [ALGO])
    eg.add('env_id', [ENV_ID])
    eg.add('seed', args.seeds)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('algo_cfgs:steps_per_epoch', [args.steps_per_epoch])
    eg.add('train_cfgs:total_steps', [args.total_steps])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('train_cfgs:device', ['cpu'])
    eg.add('algo_cfgs:update_iters', [10])
    eg.add('model_cfgs:actor:lr', [3e-4])
    eg.add('model_cfgs:critic:lr', [3e-4])
    return eg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='*', default=None)
    parser.add_argument('--output-dir', type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seeds', type=int, nargs='*', default=DEFAULT_SEEDS)
    parser.add_argument('--total-steps', type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument('--steps-per-epoch', type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument('--force', action='store_true',
                        help='Delete any existing output and retrain.')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = args.gpu
    if gpu_id is not None and not set(gpu_id).issubset(set(available_gpus)):
        warnings.warn(f'GPU {gpu_id} not available, falling back to CPU.', stacklevel=1)
        gpu_id = None

    _log('=' * 72)
    _log('Stage-1: BACKUP-policy training (SCG 2-D quadrotor stabilizer)')
    _log(f'Env            : {ENV_ID}')
    _log(f'Algo           : {ALGO}')
    _log(f'Seeds          : {args.seeds}')
    _log(f'Total steps    : {args.total_steps:,} per seed')
    _log(f'Steps/epoch    : {args.steps_per_epoch}')
    _log(f'Output dir     : {args.output_dir}')
    _log('=' * 72)

    parent_dir = args.output_dir
    if args.force and os.path.isdir(os.path.join(parent_dir, f'{ALGO}_backup')):
        _log('  --force set: removing existing output.')
        shutil.rmtree(os.path.join(parent_dir, f'{ALGO}_backup'))

    if _has_results(parent_dir, f'{ALGO}_backup'):
        _log('  Already trained. Use --force to retrain.')
    else:
        os.makedirs(parent_dir, exist_ok=True)
        t0 = time.time()
        eg = _make_ppo_grid(args)
        eg.run(train, num_pool=len(args.seeds), gpu_id=gpu_id, parent_dir=parent_dir)
        _log(f'Training done in {datetime.timedelta(seconds=int(time.time() - t0))}.')
        _log('Run directory layout:')
        for root, dirs, files in os.walk(parent_dir):
            depth = root.replace(parent_dir, '').count(os.sep)
            indent = '  ' * depth
            _log(f'{indent}{os.path.basename(root)}/')
            if depth >= 3:
                dirs[:] = []

    _log('Done. Pass the run dir to train_gatekeeper via GK_BACKUP_RUN_DIR '
         'or --backup-run-dir.')
