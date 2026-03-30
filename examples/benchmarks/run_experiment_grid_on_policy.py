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
"""On-policy benchmark runner for OmniSafe with WandB logging enabled."""

import datetime
import time
import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


def _stamped(msg: str) -> None:
    """Print a timestamped status line."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


if __name__ == '__main__':
    _stamped('Starting On-Policy-Benchmarks setup...')
    eg = ExperimentGrid(exp_name='On-Policy-Benchmarks')

    # Set up the algorithms.
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO']
    first_order_policy = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy = ['CPO', 'PCPO']
    saute_policy = ['PPOSaute', 'TRPOSaute']
    simmer_policy = ['PPOSimmerPID', 'TRPOSimmerPID']
    pid_policy = ['CPPOPID', 'TRPOPID']
    early_mdp_policy = ['PPOEarlyTerminated', 'TRPOEarlyTerminated']

    eg.add(
        'algo',
        base_policy
        + naive_lagrange_policy
        + first_order_policy
        + second_order_policy
        + saute_policy
        + simmer_policy
        + pid_policy
        + early_mdp_policy,
    )

    # Enable WandB logging — make sure you have run `wandb login` before starting.
    eg.add('logger_cfgs:use_wandb', [True])
    eg.add('logger_cfgs:wandb_project', ['On-Policy-Benchmarks'])
    # Keep TensorBoard enabled as a local backup.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # Training steps — these reproduce the 1e7-step results from the OmniSafe paper.
    # To do a shorter smoke-test run instead, swap in the two lines below:
    #   eg.add('algo_cfgs:steps_per_epoch', [2048])
    #   eg.add('train_cfgs:total_steps', [2048 * 500])
    eg.add('algo_cfgs:steps_per_epoch', [20000])
    eg.add('train_cfgs:total_steps', [10000000])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])

    # Set the environments.
    # Velocity tasks (MuJoCo locomotion with speed-limit cost).
    velocity_envs = [
        'SafetyAntVelocity-v1',
        'SafetyHopperVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetySwimmerVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
        'SafetyHumanoidVelocity-v1',
    ]
    # Circle tasks (agent must travel in a circle; cost for leaving the safe region).
    # Available robots: Point, Car, Doggo, Racecar, Ant  (no Humanoid in Safety-Gymnasium).
    # Level 1 matches the difficulty used in the CPO / FOCOPS papers.
    circle_envs = [
        'SafetyPointCircle1-v0',
        'SafetyAntCircle1-v0',
    ]
    eg.add('env_id', velocity_envs + circle_envs)

    # Five seeds so results are statistically meaningful.
    eg.add('seed', [0, 5, 10, 15, 20])

    # Device selection.
    # On-policy algorithms in OmniSafe are benchmarked on CPU for reproducibility.
    # To use specific GPUs instead, set e.g. gpu_id = [0, 1, 2, 3].
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = None
    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The requested GPU IDs are not available, falling back to CPU.', stacklevel=1)
        gpu_id = None

    # num_pool controls how many experiments run in parallel.
    # Total experiments = num_algos * num_envs * num_seeds.
    # That number must be divisible by num_pool, or adjust num_pool to fit your machine.
    # With 20 algos * 8 envs * 5 seeds = 800 total; num_pool=5 works (800 / 5 = 160 rounds).
    _stamped(f'Grid configured. Launching training with num_pool=5 on {"GPU" if gpu_id else "CPU"}...')
    _stamped('WandB project: On-Policy-Benchmarks  |  TensorBoard: examples/benchmarks/On-Policy-Benchmarks/')
    print('-' * 72, flush=True)

    t_start = time.time()
    eg.run(train, num_pool=5, gpu_id=gpu_id)
    elapsed = str(datetime.timedelta(seconds=int(time.time() - t_start)))
    print('-' * 72, flush=True)
    _stamped(f'All training finished in {elapsed}.')

    # ── Post-run analysis ────────────────────────────────────────────────────────
    _stamped('Running post-training analysis (plots)...')
    eg.analyze(parameter='algo', values=None, compare_num=6, cost_limit=25)
    _stamped('Analysis done.')

    _stamped('Rendering policy videos...')
    eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    _stamped('Videos saved.')

    _stamped('Running quantitative evaluation...')
    eg.evaluate(num_episodes=1)
    _stamped('All done! Check WandB and the On-Policy-Benchmarks/ folder for results.')
