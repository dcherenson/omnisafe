import torch
import numpy as np
from omnisafe.envs.core import CMDP, env_register
from safe_control_gym.utils.registration import make as scg_make

CARTPOLE_CONFIG = {
    'ctrl_freq': 50,
    'pyb_freq': 50,
    'gui': False,
    'normalized_rl_action_space': False,
    'randomized_init': True,
    'constraints': [
        {'constraint_form': 'default_constraint', 'constrained_variable': 'input'},
        {'constraint_form': 'default_constraint', 'constrained_variable': 'state'},
    ],
    'done_on_violation': False,
    'episode_len_sec': 10,
    'task': 'traj_tracking',
    'cost': 'quadratic',
}

QUADROTOR_CONFIG = {
    'ctrl_freq': 60,
    'pyb_freq': 240,
    'gui': False,
    'physics': 'pyb',
    'quad_type': 2,
    'normalized_rl_action_space': False,
    'randomized_init': True,
    'constraints': [
        {'constraint_form': 'default_constraint', 'constrained_variable': 'input'},
        {'constraint_form': 'default_constraint', 'constrained_variable': 'state'},
    ],
    'done_on_violation': False,
    'episode_len_sec': 10,
    'task': 'traj_tracking',
    'cost': 'quadratic',
}

@env_register
class SafeControlGymWrapper(CMDP):
    _support_envs = [
        'SCG-CartPole-v0',
        'SCG-Quadrotor-v0',
    ]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    metadata = {}

    def __init__(self, env_id: str, num_envs: int = 1, **kwargs):
        super().__init__(env_id)
        self._num_envs = num_envs
        task_map = {
            'SCG-CartPole-v0':  ('cartpole',  CARTPOLE_CONFIG),
            'SCG-Quadrotor-v0': ('quadrotor', QUADROTOR_CONFIG),
        }
        task, config = task_map[env_id]
        self._env = scg_make(task, **config)
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def max_episode_steps(self):
        return getattr(self._env, 'max_episode_steps', 500)

    def step(self, action):
        result = self._env.step(action.cpu().numpy())

        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

       
        constraint_values = info.get('constraint_values', None)
        if constraint_values is not None:
            cost = float(np.sum(constraint_values > 0))
        else:
            cost = float(info.get('constraint_violation', 0.0))

        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            info,
        )

    def reset(self, seed=None, options=None):
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return torch.as_tensor(obs, dtype=torch.float32), info

    def set_seed(self, seed: int):
        self._env.seed(seed)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
