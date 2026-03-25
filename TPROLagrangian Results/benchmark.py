import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='TPROLag_SafetyPoint')
    
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
    first_order_policy = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy = ['CPO', 'PCPO']

    mujoco_envs = [
        'SafetyPointGoal1-v0',
        'SafetyPointGoal2-v0',
        'SafetyPointButton1-v0',
        'SafetyPointButton2-v0',
        'SafetyPointPush1-v0',
        'SafetyPointPush2-v0'
    ]
    
    eg.add('env_id', mujoco_envs)

    avaliable_gpus = list(range(torch.cuda.device_count())) 
    gpu_id = [0, 1, 2, 3]

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    eg.add('algo', ['TRPOLag'])
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('train_cfgs:vector_env_nums', [4])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('algo_cfgs:steps_per_epoch', [20000])
    eg.add('train_cfgs:total_steps', [10000000])
    eg.add('seed', [0])

    eg.run(train, num_pool=12, gpu_id=gpu_id)

    eg.analyze(parameter='env_id', values=None, compare_num=6, cost_limit=25)
    eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    eg.evaluate(num_episodes=1)
