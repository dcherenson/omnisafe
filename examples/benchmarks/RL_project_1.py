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
"""RL Project 1 — Circle Environment Benchmark.

Trains 6 safe-RL algorithms across 3 circle-task robots (Point, Ant, Car)
at difficulty level 1.  The circle task requires the agent to orbit a fixed
target point; cost is incurred whenever the agent leaves the safe circular
region.

Grid summary
------------
  Algorithms  : PPO, PPOLag, TRPOLag, CPO, FOCOPS, TRPO  (6)
  Environments: SafetyPointCircle1, SafetyAntCircle1,
                SafetyCarCircle1                           (3)
  Seeds       : 0, 5                                       (2)
  ──────────────────────────────────────────────────────────────
  Total       : 6 × 3 × 2 = 36 experiments
  Parallel    : 4 at a time on GPU 0
  Est. runtime: ~6–7.5 hours
"""

import datetime
import time
import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


def _stamped(msg: str) -> None:
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


if __name__ == '__main__':
    _stamped('Starting RL Project 1 — Benchmark...')
    eg = ExperimentGrid(exp_name='RL-Project-1')

    # ── Algorithms ───────────────────────────────────────────────────────────
    # Selected to match the algorithms shown in the course slides and Table 3
    # of the CPO / FOCOPS papers: baseline (PPO, TRPO), naive Lagrangian
    # (PPOLag, TRPOLag), second-order (CPO), and first-order (FOCOPS).
    eg.add('algo', [
        'CPO',      # Constrained Policy Optimisation (second-order)
        'FOCOPS',   # First-Order Constrained Optimisation in Policy Space
    ])

    # ── Environments (Circle tasks) ───────────────────────────────────────────
    # Three robot morphologies, all at difficulty level 1.
    # Level 1 matches the difficulty used in the CPO and FOCOPS papers.
    # Racecar and Doggo dropped to keep runtime manageable (~6–7.5 hrs total).
    circle_envs = [
        'SafetyPointCircle1-v0',  # simple 2-D point mass (fastest)
        'SafetyAntCircle1-v0',    # 4-legged Ant (from course slides)
        'SafetyCarCircle1-v0',    # Car robot
    ]
    velocity_envs = [
        'SafetyAntVelocity-v1',          # Ant       — 4-legged, 8 DoF
        'SafetyHopperVelocity-v1',        # Hopper    — 1-legged, 3 DoF
        'SafetyWalker2dVelocity-v1',      # Walker 2d — bipedal, 6 DoF
        'SafetySwimmerVelocity-v1',       # Swimmer   — 2-link, 2 DoF
        'SafetyHalfCheetahVelocity-v1',   # Half-Cheetah — 6 DoF
        'SafetyHumanoidVelocity-v1',      # Humanoid  — 17 DoF (humanoid-v3)
    ]
    eg.add('env_id', circle_envs + velocity_envs)

    # ── Seeds ─────────────────────────────────────────────────────────────────
    eg.add('seed', [0, 5])

    # ── Logging ───────────────────────────────────────────────────────────────
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:wandb_project', ['RL-Project-1'])
    eg.add('logger_cfgs:use_tensorboard', [True])

    # ── Training budget ───────────────────────────────────────────────────────
    # 1 M steps total; 20 k steps per epoch → 50 epochs per run.
    # The circle paper uses 50 k batch size; keeping 20 k here for a faster
    # but still informative run.  Increase total_steps to 1e7 for paper results.
    eg.add('algo_cfgs:steps_per_epoch', [20000])
    eg.add('train_cfgs:total_steps', [1000000])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])

    # ── Device ────────────────────────────────────────────────────────────────
    # Use GPU 0 (NVIDIA RTX 4060 Laptop GPU) for faster training.
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    if not set(gpu_id).issubset(available_gpus):
        warnings.warn(
            f'Requested GPU IDs {gpu_id} not available (found {available_gpus}). '
            'Falling back to CPU.',
            stacklevel=1,
        )
        gpu_id = None

    # ── Launch ────────────────────────────────────────────────────────────────
    device_str = f'GPU {gpu_id}' if gpu_id else 'CPU'
    _stamped(
        f'Grid ready: 6 algos × 3 circle envs × 2 seeds = 36 experiments, '
        f'4 in parallel on {device_str}.'
    )
    _stamped('WandB project: RL-Project-1-Circle')
    print('-' * 72, flush=True)

    t_start = time.time()
    eg.run(train, num_pool=4, gpu_id=gpu_id)
    elapsed = str(datetime.timedelta(seconds=int(time.time() - t_start)))
    print('-' * 72, flush=True)
    _stamped(f'Training finished in {elapsed}.')

    # ── Post-run analysis ─────────────────────────────────────────────────────
    _stamped('Running analysis (reward / cost plots)...')
    eg.analyze(parameter='algo', values=None, compare_num=6, cost_limit=25)
    _stamped('Analysis done.')

    _stamped('Rendering policy videos...')
    eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    _stamped('Videos saved.')

    _stamped('Running quantitative evaluation...')
    eg.evaluate(num_episodes=1)
    _stamped('All done! Check WandB project RL-Project-1-Circle for results.')
