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
"""Small smoke-test benchmark: 2 envs, selected algorithms, 2 seeds."""

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
    _stamped('Starting smoke-test benchmark...')
    eg = ExperimentGrid(exp_name='On-Policy-Test')

    # A representative selection of algorithms across all families.
    eg.add('algo', [
        'PPO',          # base
        'PPOLag',       # naive Lagrangian
        'CPO',          # second-order
        'FOCOPS',       # first-order
        'PPOSaute',     # saute
        'CPPOPID',      # PID-Lagrangian
    ])

    # Two environments: one velocity, one circle.
    eg.add('env_id', [
        'SafetyAntVelocity-v1',
        'SafetyPointCircle1-v0',
    ])

    # Two seeds — enough to catch instability without taking too long.
    eg.add('seed', [0, 5])

    # WandB + TensorBoard.
    eg.add('logger_cfgs:use_wandb', [True])
    eg.add('logger_cfgs:wandb_project', ['On-Policy-Test'])
    eg.add('logger_cfgs:use_tensorboard', [True])

    # Shorter run: 1M steps instead of 10M — enough to see learning curves.
    eg.add('algo_cfgs:steps_per_epoch', [20000])
    eg.add('train_cfgs:total_steps', [1000000])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])

    # CPU (recommended for on-policy reproducibility).
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = None
    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('GPU not available, falling back to CPU.', stacklevel=1)
        gpu_id = None

    # Total = 6 algos * 2 envs * 2 seeds = 24 experiments, num_pool=4 → 6 rounds.
    _stamped('Grid ready: 6 algos × 2 envs × 2 seeds = 24 experiments, 4 in parallel.')
    _stamped('WandB project: On-Policy-Test')
    print('-' * 72, flush=True)

    t_start = time.time()
    eg.run(train, num_pool=4, gpu_id=gpu_id)
    elapsed = str(datetime.timedelta(seconds=int(time.time() - t_start)))
    print('-' * 72, flush=True)
    _stamped(f'Training finished in {elapsed}.')

    _stamped('Running analysis...')
    eg.analyze(parameter='algo', values=None, compare_num=6, cost_limit=25)
    _stamped('Analysis done.')

    _stamped('Rendering videos...')
    eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    _stamped('Videos saved.')

    _stamped('Running evaluation...')
    eg.evaluate(num_episodes=1)
    _stamped('All done! Check wandb.ai → project On-Policy-Test for results.')
