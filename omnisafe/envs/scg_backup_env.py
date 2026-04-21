# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Stage-1 env: learn a BACKUP stabilizing policy for the SCG 2-D quadrotor.

The backup policy is trained to bring the drone to rest (zero velocity, zero
tilt) from arbitrary initial states sampled inside the safe box.  It outputs a
2-D target (x_target, z_target); the env constructs a 6-D reference
[x_target, 0, z_target, 0, 0, 0] and delegates actuation to the classical DSL
PID low-level.  Once trained, the policy is frozen and used as the "backup"
leg of the gatekeeper's nominal-or-backup switch at Stage 2.

Observation (6-D):  [x, vx, z, vz, theta, theta_dot]
Action     (2-D):   a in [-1, 1]^2, mapped to
                    x_target = mean_x + scale_x * a[0]
                    z_target = mean_z + scale_z * a[1]
Reward per step:
    +alive_bonus
    - w_vel  * (vx^2 + vz^2)
    - w_ang  * theta^2
    - w_rate * theta_dot^2
    - w_pos  * (x^2 + (z - z_nom)^2)      (mild centering)
    - crash_penalty if state leaves safe box (episode ends)
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

import safe_control_gym  # noqa: F401
from safe_control_gym.utils.registration import make as scg_make

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU


_BACKUP_ENV_ID = 'SCG-Quadrotor-Backup-v0'

# Safe box (must match the gatekeeper env's safety constraints)
_X_BOUNDS = (-2.0, 0.3)
_Z_BOUNDS = (0.3, 1.7)
_THETA_BOUNDS = (-0.35, 0.35)

# Target/reference box for the backup policy's 2-D action (slightly inset
# from the safe box so the PID reference never sits right on a boundary).
_TGT_X_RANGE = (-1.8, 0.2)
_TGT_Z_RANGE = (0.5, 1.5)

# Max episode length for backup training (60 Hz * 2 s = 120 steps).
_MAX_STEPS = 120

# Reward weights
_ALIVE_BONUS = 0.1
_W_VEL = 0.05
_W_ANG = 0.50
_W_RATE = 0.02
_W_POS = 0.01
_CRASH_PENALTY = 10.0
_STABLE_BONUS = 0.5        # extra reward if drone is effectively stopped this step
_STABLE_VEL_TOL = 0.15     # m/s
_STABLE_ANG_TOL = 0.05     # rad

_SCG_TASK = 'quadrotor'
_SCG_KWARGS: dict[str, Any] = {
    'task': 'stabilization',
    'cost': 'rl_reward',
    'episode_len_sec': int(_MAX_STEPS / 60) + 1,
    'ctrl_freq': 60,
    'pyb_freq': 240,
    'quad_type': 2,
    'normalized_rl_action_space': False,
    'randomized_init': False,
    'verbose': False,
}


@env_register
class SCGQuadrotorBackupEnv(CMDP):
    """Standalone env to train a backup stabilizer for the 2-D quadrotor.

    Args:
        env_id (str): Must be ``SCG-Quadrotor-Backup-v0``.
        device (torch.device, optional): Torch device for returned tensors.
        **kwargs: Ignored (kept for OmniSafe make() compatibility).
    """

    _support_envs: ClassVar[list[str]] = [_BACKUP_ENV_ID]

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

        self._max_steps = int(os.environ.get('BACKUP_MAX_STEPS', str(_MAX_STEPS)))

        # Inner SCG env
        scg_kwargs = dict(_SCG_KWARGS)
        self._scg_env = scg_make(_SCG_TASK, **scg_kwargs)
        init_obs, _info = self._scg_env.reset()
        self._state_dim = int(self._scg_env.observation_space.shape[0])

        # Low-level PID (same classical controller used by the gatekeeper)
        from omnisafe.envs.scg_gatekeeper_env import ClassicalLowLevel
        self._low_level = ClassicalLowLevel(quad_type=2)

        # Spaces
        obs_low = np.full((self._state_dim,), -np.inf, dtype=np.float32)
        obs_high = np.full((self._state_dim,), np.inf, dtype=np.float32)
        self._observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self._action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._state = np.asarray(init_obs, dtype=np.float32)
        self._t = 0
        self._rng = np.random.default_rng()
        self._max_episode_steps = self._max_steps

        # Centering "nominal" z (used only as mild reward shaping)
        self._z_nom = 0.5 * (_Z_BOUNDS[0] + _Z_BOUNDS[1])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_initial_state(self) -> np.ndarray:
        """Rich random init: diverse positions, velocities, and tilts inside safe box."""
        for _ in range(64):
            x = self._rng.uniform(_X_BOUNDS[0] + 0.2, _X_BOUNDS[1] - 0.1)
            z = self._rng.uniform(_Z_BOUNDS[0] + 0.2, _Z_BOUNDS[1] - 0.2)
            vx = self._rng.normal(0.0, 0.4)
            vz = self._rng.normal(0.0, 0.4)
            theta = self._rng.uniform(-0.20, 0.20)
            theta_dot = self._rng.normal(0.0, 0.3)
            s = np.array([x, vx, z, vz, theta, theta_dot], dtype=np.float32)
            if self._state_in_box(s):
                return s
        # Fallback: neutral hover-ish state near center
        return np.array([-0.8, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _state_in_box(self, s: np.ndarray) -> bool:
        return (
            _X_BOUNDS[0] <= float(s[0]) <= _X_BOUNDS[1]
            and _Z_BOUNDS[0] <= float(s[2]) <= _Z_BOUNDS[1]
            and _THETA_BOUNDS[0] <= float(s[4]) <= _THETA_BOUNDS[1]
        )

    def _action_to_reference(self, action: np.ndarray) -> np.ndarray:
        """Map a 2-D action in [-1, 1]^2 to a 6-D PID reference."""
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        mx = 0.5 * (_TGT_X_RANGE[0] + _TGT_X_RANGE[1])
        sx = 0.5 * (_TGT_X_RANGE[1] - _TGT_X_RANGE[0])
        mz = 0.5 * (_TGT_Z_RANGE[0] + _TGT_Z_RANGE[1])
        sz = 0.5 * (_TGT_Z_RANGE[1] - _TGT_Z_RANGE[0])
        x_tgt = mx + sx * float(a[0])
        z_tgt = mz + sz * float(a[1])
        return np.array([x_tgt, 0.0, z_tgt, 0.0, 0.0, 0.0], dtype=np.float32)

    def _reset_inner_to_state(self, target_state: np.ndarray) -> None:
        """Hard-reset the inner PyBullet env to the requested state."""
        import pybullet as pb
        self._scg_env.reset()
        # 2-D quad internal state layout: [x, vx, z, vz, theta, theta_dot]
        self._scg_env.state = np.asarray(target_state, dtype=np.float64).copy()
        # Reflect state into PyBullet rigid body
        x, z, th = float(target_state[0]), float(target_state[2]), float(target_state[4])
        vx, vz, th_dot = float(target_state[1]), float(target_state[3]), float(target_state[5])
        pb.resetBasePositionAndOrientation(
            self._scg_env.DRONE_ID,
            [x, 0.0, z],
            pb.getQuaternionFromEuler([0.0, th, 0.0]),
            physicsClientId=self._scg_env.PYB_CLIENT,
        )
        pb.resetBaseVelocity(
            self._scg_env.DRONE_ID,
            linearVelocity=[vx, 0.0, vz],
            angularVelocity=[0.0, th_dot, 0.0],
            physicsClientId=self._scg_env.PYB_CLIENT,
        )
        self._low_level.reset_state()

    # ------------------------------------------------------------------
    # CMDP API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self.set_seed(seed)
            self._rng = np.random.default_rng(seed)

        init_state = self._sample_initial_state()
        self._reset_inner_to_state(init_state)
        self._state = init_state.astype(np.float32)
        self._t = 0
        info = {'initial_state': init_state.tolist()}
        return (
            torch.as_tensor(self._state, dtype=torch.float32, device=self._device),
            info,
        )

    def _compute_reward(self, state: np.ndarray, crashed: bool) -> tuple[float, float]:
        """Per-step reward and a simple 'stopped' indicator."""
        x, vx, z, vz, th, th_dot = [float(v) for v in state]
        r = _ALIVE_BONUS
        r -= _W_VEL * (vx * vx + vz * vz)
        r -= _W_ANG * (th * th)
        r -= _W_RATE * (th_dot * th_dot)
        r -= _W_POS * ((z - self._z_nom) ** 2)
        stable = (
            abs(vx) < _STABLE_VEL_TOL
            and abs(vz) < _STABLE_VEL_TOL
            and abs(th) < _STABLE_ANG_TOL
            and abs(th_dot) < 2 * _STABLE_ANG_TOL
        )
        if stable:
            r += _STABLE_BONUS
        if crashed:
            r -= _CRASH_PENALTY
        return float(r), float(stable)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        a_np = action.detach().cpu().numpy().reshape(-1)
        ref = self._action_to_reference(a_np)
        cmd = self._low_level(self._state, ref)
        next_obs, _r, _done, info_step = self._scg_env.step(cmd)
        self._state = np.asarray(next_obs, dtype=np.float32)
        self._t += 1

        crashed = not self._state_in_box(self._state) or bool(
            int(info_step.get('constraint_violation', 0)) != 0,
        )

        reward, stable = self._compute_reward(self._state, crashed)

        terminated = crashed
        truncated = (self._t >= self._max_steps) and not terminated

        info = {
            'ref': ref.tolist(),
            'cmd': cmd.tolist(),
            'stable': bool(stable > 0),
            'crashed': bool(crashed),
            't': int(self._t),
        }
        return (
            torch.as_tensor(self._state, dtype=torch.float32, device=self._device),
            torch.as_tensor(reward, dtype=torch.float32, device=self._device),
            torch.as_tensor(0.0, dtype=torch.float32, device=self._device),
            torch.as_tensor(terminated, dtype=torch.bool, device=self._device),
            torch.as_tensor(truncated, dtype=torch.bool, device=self._device),
            info,
        )

    def set_seed(self, seed: int) -> None:
        self._scg_env.seed(seed)

    def render(self) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        self._scg_env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Any) -> None:  # noqa: ANN401
        pass
