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
"""OmniSafe CMDP wrapper for safe-control-gym environments (CartPole, Quadrotor).

This module bridges safe-control-gym's BenchmarkEnv API to OmniSafe's CMDP interface
so that OmniSafe algorithms (FOCOPS, CPO, PPOLag, etc.) can train on safe-control-gym tasks.

Key translations performed in every step:
  - torch.Tensor action  →  numpy array  (before passing to SCG)
  - numpy obs/rew/done   →  torch.Tensor (after receiving from SCG)
  - info['constraint_violation'] (0/1 int)  →  cost tensor
  - single done flag + info['TimeLimit.truncated']  →  (terminated, truncated) tensors

Supported environment IDs
--------------------------
CartPole (4-D state: x, x_dot, theta, theta_dot):
  SCG-CartPole-Stabilization-v0   — pole-angle + cart-position constraint
  SCG-CartPole-NoConstraint-v0    — no safety constraint (cost always 0)

Quadrotor 2-D (6-D state: x, x_dot, z, z_dot, theta, theta_dot):
  SCG-Quadrotor-Stabilization-v0  — hover at origin with tight position/angle constraints
  SCG-Quadrotor-Tracking-v0       — circle trajectory with a forbidden right-side zone
  SCG-Quadrotor-NoConstraint-v0   — no safety constraint (cost always 0)

Trajectory tracking scenario (SCG-Quadrotor-Tracking-v0):
  The quadrotor must follow a vertical circle (radius 0.4 m, centre x=0 z=1.0 m)
  in the x-z plane.  A no-fly zone forbids the right side of the loop (x > 0.3 m):

                        * top  (z=1.4)
                       / \\
          left *------+   +------ right *  <- FORBIDDEN (x > 0.3)
          (x=-0.4)        |             (x=0.4)
                       \\ /
                        * bottom (z=0.6)

  CPO / FOCOPS must learn to track the circle while never entering x > 0.3,
  which covers ~23 % of the loop's arc length.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

import safe_control_gym  # noqa: F401  — triggers controller/env registration
from safe_control_gym.utils.registration import make as scg_make

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU


# ---------------------------------------------------------------------------
# Per-environment construction configs
# ---------------------------------------------------------------------------
# Each entry maps an OmniSafe env_id to the kwargs forwarded to safe-control-gym's
# make() call.  The 'task' key is the SCG registered task name.
#
# CartPole state indices:     0=x,  1=x_dot,  2=theta,  3=theta_dot
# Quadrotor-2D state indices: 0=x,  1=x_dot,  2=z,      3=z_dot,  4=theta, 5=theta_dot
#
# Trajectory maths for SCG-Quadrotor-Tracking-v0
# -----------------------------------------------
# trajectory_plane='zx' -> coord_a=z, coord_b=x
# scaling = -0.4, position_offset = [1.0, 0.0]  -> center at (z=1.0, x=0)
# Circle parametric (t=0 -> bottom of loop):
#   z(t) = 1.0 + (-0.4)*cos(2pi/T * t)   ->  z in [0.6, 1.4]
#   x(t) = 0   + (-0.4)*sin(2pi/T * t)   ->  x in [-0.4, 0.4]
# Forbidden zone  x > 0.3 covers roughly 23 % of the arc (right side).
# ---------------------------------------------------------------------------

_SCG_ENV_CONFIGS: dict[str, dict[str, Any]] = {
    # ── CartPole ──────────────────────────────────────────────────────────────
    'SCG-CartPole-Stabilization-v0': {
        'task': 'cartpole',
        'kwargs': {
            'task': 'stabilization',
            'cost': 'rl_reward',
            'episode_len_sec': 5,
            'ctrl_freq': 50,
            'pyb_freq': 50,
            'normalized_rl_action_space': True,
            'randomized_init': True,
            'constraints': [
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [2],
                    'lower_bounds': [-0.2],
                    'upper_bounds': [0.2],
                },
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [0],
                    'lower_bounds': [-1.0],
                    'upper_bounds': [1.0],
                },
            ],
            'done_on_violation': False,
            'done_on_out_of_bound': False,
            'verbose': False,
        },
    },
    'SCG-CartPole-NoConstraint-v0': {
        'task': 'cartpole',
        'kwargs': {
            'task': 'stabilization',
            'cost': 'rl_reward',
            'episode_len_sec': 5,
            'ctrl_freq': 50,
            'pyb_freq': 50,
            'normalized_rl_action_space': True,
            'randomized_init': True,
            'constraints': None,
            'done_on_violation': False,
            'done_on_out_of_bound': False,
            'verbose': False,
        },
    },
    # ── Quadrotor (2-D) stabilization ─────────────────────────────────────────
    'SCG-Quadrotor-Stabilization-v0': {
        'task': 'quadrotor',
        'kwargs': {
            'task': 'stabilization',
            'cost': 'rl_reward',
            'episode_len_sec': 5,
            'ctrl_freq': 60,
            'pyb_freq': 240,
            'quad_type': 2,
            'normalized_rl_action_space': True,
            'randomized_init': True,
            'constraints': [
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [0],
                    'lower_bounds': [-0.5],
                    'upper_bounds': [0.5],
                },
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [2],
                    'lower_bounds': [0.6],
                    'upper_bounds': [1.4],
                },
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [4],
                    'lower_bounds': [-0.2],
                    'upper_bounds': [0.2],
                },
            ],
            'done_on_violation': False,
            'done_on_out_of_bound': False,
            'verbose': False,
        },
    },
    # ── Quadrotor trajectory tracking with a forbidden right-side zone ─────────
    'SCG-Quadrotor-Tracking-v0': {
        'task': 'quadrotor',
        'kwargs': {
            'task': 'traj_tracking',
            'cost': 'rl_reward',
            'episode_len_sec': 8,
            'ctrl_freq': 60,
            'pyb_freq': 240,
            'quad_type': 2,
            'normalized_rl_action_space': True,
            'randomized_init': True,
            'task_info': {
                'trajectory_type': 'circle',
                'num_cycles': 2,
                'trajectory_plane': 'zx',
                'trajectory_position_offset': [1.0, 0.0],
                'trajectory_scale': -0.4,
            },
            'constraints': [
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [0],
                    'lower_bounds': [-2.0],
                    'upper_bounds': [0.3],
                },
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [2],
                    'lower_bounds': [0.3],
                    'upper_bounds': [1.7],
                },
                {
                    'constraint_form': 'bounded_constraint',
                    'constrained_variable': 'state',
                    'active_dims': [4],
                    'lower_bounds': [-0.35],
                    'upper_bounds': [0.35],
                },
            ],
            'done_on_violation': False,
            'done_on_out_of_bound': False,
            'verbose': False,
        },
    },
    'SCG-Quadrotor-NoConstraint-v0': {
        'task': 'quadrotor',
        'kwargs': {
            'task': 'stabilization',
            'cost': 'rl_reward',
            'episode_len_sec': 5,
            'ctrl_freq': 60,
            'pyb_freq': 240,
            'quad_type': 2,
            'normalized_rl_action_space': True,
            'randomized_init': False,
            'constraints': None,
            'done_on_violation': False,
            'verbose': False,
        },
    },
}


@env_register
class SafeControlGymEnv(CMDP):
    """OmniSafe CMDP wrapper around safe-control-gym BenchmarkEnv.

    Supports CartPole stabilization and Quadrotor stabilization / trajectory-tracking
    tasks with configurable safety constraints.  The cost signal is derived from
    ``info['constraint_violation']`` returned by safe-control-gym (1 if any
    constraint is violated at that step, 0 otherwise).

    Args:
        env_id (str): One of the IDs listed in ``_support_envs``.
        device (torch.device, optional): Torch device for returned tensors.
            Defaults to CPU.
        **kwargs: Additional keyword arguments (unused; kept for API compatibility).
    """

    _support_envs: ClassVar[list[str]] = list(_SCG_ENV_CONFIGS.keys())

    need_auto_reset_wrapper: bool = True
    need_time_limit_wrapper: bool = False

    _num_envs: int = 1

    def __init__(
        self,
        env_id: str,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)

        self._device = device
        self._env_id = env_id
        cfg = _SCG_ENV_CONFIGS[env_id]

        self._env = scg_make(cfg['task'], **cfg['kwargs'])
        self._env.reset()

        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

        self._max_episode_steps: int = int(
            self._env.EPISODE_LEN_SEC * self._env.CTRL_FREQ
        )
        self._metadata: dict[str, Any] = {}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        np_action = action.detach().cpu().numpy()

        obs_np, rew, done, info = self._env.step(np_action)

        time_truncated = bool(info.get('TimeLimit.truncated', False))
        terminated = done and not time_truncated
        truncated = done and time_truncated

        cost = float(info.get('constraint_violation', 0))

        return (
            torch.as_tensor(obs_np, dtype=torch.float32, device=self._device),
            torch.as_tensor(rew,    dtype=torch.float32, device=self._device),
            torch.as_tensor(cost,   dtype=torch.float32, device=self._device),
            torch.as_tensor(terminated, dtype=torch.bool, device=self._device),
            torch.as_tensor(truncated,  dtype=torch.bool, device=self._device),
            info,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)

        obs_np, info = self._env.reset()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        return obs, info

    def set_seed(self, seed: int) -> None:
        self._env.seed(seed)

    def render(self) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Any) -> None:  # noqa: ANN401
        pass
