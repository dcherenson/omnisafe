# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Stage-2 switching-filter gatekeeper env on SCG-Quadrotor-Tracking.

The RL policy here decides ONCE PER DECISION EPOCH whether to invoke the
LOCOMOTION controller (track the circle reference via DSL PID) or the BACKUP
controller (a frozen PPO policy trained to stabilize from any state).

A gatekeeper-style filter sits between the RL proposal and the simulator:
  - If the RL policy picks NOMINAL, we forward-simulate the candidate plan
    (keep using nominal for H control steps) inside a snapshot of PyBullet.
    If the preview ever exits the safe box, we OVERRIDE the choice to BACKUP.
  - If the RL policy picks BACKUP, no preview is required — backup is
    trusted safe by construction.

Three policies in total:
  1. Low-level tracker:     classical DSL PID (no training; tracks any 6-D ref)
  2. Backup policy:         frozen PPO actor loaded from checkpoint, outputs
                            a 2-D stabilizing target (x_tgt, z_tgt) per state
  3. Switching policy:      the RL policy trained IN THIS ENVIRONMENT

Observation (7-D): [state_6d, t / T_total]
Action     (1-D):  Box([0, 1]); threshold at 0.5 → {0: backup, 1: nominal}
Reward per decision step (over Δ real steps):
  +R_nom        if RL picked NOMINAL, filter allowed it, and no violation
  −R_override   if RL picked NOMINAL but filter overrode to BACKUP (GK_USE_SHAPING)
  0             if RL picked BACKUP (backup is reward-neutral)
  −R_crash · n  added on top, one unit per real-step violation

Env vars controlling behaviour:
  GK_BACKUP_RUN_DIR   : path to the PPO backup run dir (expects torch_save/*.pt)
  GK_BACKUP_MODEL     : model filename (default: auto-pick latest epoch-XXX.pt)
  GK_BACKUP_FALLBACK  : if 'true' and no ckpt found, fall back to fixed hover (default)
  GK_H                : preview horizon in ctrl steps (default 60)
  GK_DELTA            : decision stride in ctrl steps (default 5)
  GK_USE_FILTER       : 'true'/'false' — active filter at train/deploy (default true)
  GK_USE_SHAPING      : 'true'/'false' — include override penalty term (default true)
  GK_R_NOM            : reward for successful nominal step (default 1.0)
  GK_R_OVERRIDE       : penalty when nominal gets overridden (default 0.2)
  GK_R_CRASH          : penalty per real-step violation (default 20.0)
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pybullet as pb
import torch
from gymnasium import spaces

import safe_control_gym  # noqa: F401  — triggers controller/env registration
from safe_control_gym.controllers.pid.pid import PID
from safe_control_gym.utils.registration import make as scg_make

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU


_GATEKEEPER_ENV_ID = 'SCG-Quadrotor-Gatekeeper-v0'

_SCG_TASK = 'quadrotor'
_SCG_KWARGS: dict[str, Any] = {
    'task': 'traj_tracking',
    'cost': 'rl_reward',
    'episode_len_sec': 8,
    'ctrl_freq': 60,
    'pyb_freq': 240,
    'quad_type': 2,
    'normalized_rl_action_space': False,
    'randomized_init': True,
    'task_info': {
        'trajectory_type': 'circle',
        'num_cycles': 2,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [1.0, 0.0],
        'trajectory_scale': -0.4,
    },
    'constraints': [
        {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
         'active_dims': [0], 'lower_bounds': [-2.0], 'upper_bounds': [0.3]},
        {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
         'active_dims': [2], 'lower_bounds': [0.3], 'upper_bounds': [1.7]},
        {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
         'active_dims': [4], 'lower_bounds': [-0.35], 'upper_bounds': [0.35]},
    ],
    'done_on_violation': False,
    'done_on_out_of_bound': False,
    'verbose': False,
}

# Reference box for the backup policy's 2-D action -> 6-D PID reference.
# Must match the scg_backup_env's _TGT_X_RANGE / _TGT_Z_RANGE.
_TGT_X_RANGE = (-1.8, 0.2)
_TGT_Z_RANGE = (0.5, 1.5)


# ---------------------------------------------------------------------------
# Classical low-level controller (safe-control-gym's DSL PID)
# ---------------------------------------------------------------------------

class ClassicalLowLevel:
    """Thin wrapper around safe-control-gym's DSL PID position/attitude controller.

    We instantiate an SCG PID once and reuse its per-iteration math for each
    (state, reference) we want tracked.  The PID is stateful (integral and
    derivative terms); callers can snapshot/restore that state via
    ``save_state``/``load_state`` so that preview rollouts don't permanently
    corrupt the integrator used for real execution.
    """

    def __init__(self, quad_type: int = 2) -> None:
        self._quad_type = int(quad_type)
        env_func = partial(
            scg_make,
            _SCG_TASK,
            task='stabilization',
            quad_type=self._quad_type,
            ctrl_freq=_SCG_KWARGS['ctrl_freq'],
            pyb_freq=_SCG_KWARGS['pyb_freq'],
            normalized_rl_action_space=False,
            gui=False,
            verbose=False,
        )
        self._pid = PID(env_func=env_func)
        self._pid.reset_before_run()

    def __call__(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Map (state_6d, reference_6d) -> physical-thrust action for the inner env."""
        if self._quad_type == 2:
            cur_pos = np.array([state[0], 0.0, state[2]])
            cur_quat = np.array(pb.getQuaternionFromEuler([0, state[4], 0]))
            cur_vel = np.array([state[1], 0.0, state[3]])
            target_pos = np.array([reference[0], 0.0, reference[2]])
            target_vel = np.array([reference[1], 0.0, reference[3]])
        else:
            cur_pos = np.array([state[0], state[2], state[4]])
            cur_quat = np.array(
                pb.getQuaternionFromEuler([state[6], state[7], state[8]]),
            )
            cur_vel = np.array([state[1], state[3], state[5]])
            target_pos = np.array([reference[0], reference[2], reference[4]])
            target_vel = np.array([reference[1], reference[3], reference[5]])

        target_rpy = np.zeros(3)
        target_rpy_rates = np.zeros(3)

        thrust, target_euler, _ = self._pid._dslPIDPositionControl(  # noqa: SLF001
            cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel,
        )
        rpm = self._pid._dslPIDAttitudeControl(  # noqa: SLF001
            thrust, cur_quat, target_euler, target_rpy_rates,
        )
        action = self._pid.KF * rpm**2
        if self._quad_type == 2:
            action = np.array([action[0] + action[3], action[1] + action[2]])
        return action.astype(np.float32)

    def save_state(self) -> dict[str, np.ndarray]:
        return {
            'integral_pos_e': self._pid.integral_pos_e.copy(),
            'integral_rpy_e': self._pid.integral_rpy_e.copy(),
            'last_rpy': self._pid.last_rpy.copy(),
        }

    def load_state(self, snap: dict[str, np.ndarray]) -> None:
        self._pid.integral_pos_e = snap['integral_pos_e'].copy()
        self._pid.integral_rpy_e = snap['integral_rpy_e'].copy()
        self._pid.last_rpy = snap['last_rpy'].copy()

    def reset_state(self) -> None:
        self._pid.reset_before_run()


# ---------------------------------------------------------------------------
# Frozen backup policy (loaded from a PPO run on SCG-Quadrotor-Backup-v0)
# ---------------------------------------------------------------------------

def _find_backup_run_dir(root: str | os.PathLike) -> Path | None:
    """Find the latest PPO run directory under ``root``.

    We look for ``seed-*`` folders that contain a ``torch_save/`` subdir.
    Returns the most recently modified match, or None.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return None
    candidates: list[Path] = []
    for p in root_path.rglob('torch_save'):
        if p.is_dir():
            candidates.append(p.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_latest_checkpoint(save_dir: Path) -> str | None:
    """Find the latest ``epoch-XXX.pt`` file in ``save_dir/torch_save``."""
    torch_dir = save_dir / 'torch_save'
    if not torch_dir.exists():
        return None
    pts = sorted(torch_dir.glob('epoch-*.pt'), key=lambda p: int(p.stem.split('-')[1]))
    if not pts:
        return None
    return pts[-1].name


class FrozenBackupPolicy:
    """Frozen backup policy loaded from a saved OmniSafe PPO run.

    Given a 6-D state, returns the 6-D reference for the PID to track.
    If no checkpoint is provided (or loading fails), falls back to
    ``state → hover-at-current-(x, z)`` so the rest of the pipeline still runs.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike | None = None,
        model_name: str | None = None,
        fallback: bool = True,
    ) -> None:
        self._run_dir: Path | None = Path(run_dir).resolve() if run_dir else None
        self._model_name = model_name
        self._fallback = fallback
        self._actor = None
        self._normalizer = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if self._run_dir is None:
            if not self._fallback:
                raise FileNotFoundError('No backup run dir provided; set GK_BACKUP_RUN_DIR.')
            return
        if not self._run_dir.exists():
            if not self._fallback:
                raise FileNotFoundError(f'Backup run dir not found: {self._run_dir}')
            return

        model_name = self._model_name or _find_latest_checkpoint(self._run_dir)
        if model_name is None:
            if not self._fallback:
                raise FileNotFoundError(f'No model found under {self._run_dir}/torch_save')
            return
        model_path = self._run_dir / 'torch_save' / model_name

        cfg_path = self._run_dir / 'config.json'
        if not cfg_path.exists():
            if not self._fallback:
                raise FileNotFoundError(f'Missing config.json in {self._run_dir}')
            return
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfgs = json.load(f)

        try:
            from omnisafe.common import Normalizer
            from omnisafe.models.actor import ActorBuilder
        except ImportError as e:
            if not self._fallback:
                raise
            print(f'[FrozenBackupPolicy] Import failed ({e}); falling back.')
            return

        params = torch.load(model_path, weights_only=False, map_location='cpu')

        # Observation / action spaces for the backup env (must match the env).
        obs_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        model_cfgs = cfgs['model_cfgs']
        pi_cfg = model_cfgs['actor']
        builder = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=pi_cfg['hidden_sizes'],
            activation=pi_cfg['activation'],
            weight_initialization_mode=model_cfgs['weight_initialization_mode'],
        )
        actor = builder.build_actor(model_cfgs['actor_type'])
        actor.load_state_dict(params['pi'])
        actor.eval()
        self._actor = actor

        if cfgs.get('algo_cfgs', {}).get('obs_normalize', False):
            norm = Normalizer(shape=obs_space.shape, clip=5)
            norm.load_state_dict(params['obs_normalizer'])
            norm.eval()
            self._normalizer = norm

        self._loaded = True
        print(f'[FrozenBackupPolicy] Loaded {model_path}')

    def _action_to_reference(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        mx = 0.5 * (_TGT_X_RANGE[0] + _TGT_X_RANGE[1])
        sx = 0.5 * (_TGT_X_RANGE[1] - _TGT_X_RANGE[0])
        mz = 0.5 * (_TGT_Z_RANGE[0] + _TGT_Z_RANGE[1])
        sz = 0.5 * (_TGT_Z_RANGE[1] - _TGT_Z_RANGE[0])
        x_tgt = mx + sx * float(a[0])
        z_tgt = mz + sz * float(a[1])
        return np.array([x_tgt, 0.0, z_tgt, 0.0, 0.0, 0.0], dtype=np.float32)

    def reference(self, state: np.ndarray) -> np.ndarray:
        """Return the 6-D PID reference the backup policy wants to track."""
        if not self._loaded or self._actor is None:
            # Fallback: hover at current (x, z). Simple, reliable.
            ref = np.zeros(6, dtype=np.float32)
            ref[0] = float(state[0])
            ref[2] = float(state[2])
            return ref
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
            if self._normalizer is not None:
                s = self._normalizer.normalize(s)
            try:
                act = self._actor.predict(s, deterministic=True)
            except AttributeError:
                dist = self._actor(s)
                act = dist.mean
            act = act.reshape(-1).cpu().numpy()
        return self._action_to_reference(act)

    @property
    def loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# The gatekeeper env (binary switch between nominal and backup)
# ---------------------------------------------------------------------------

@env_register
class SCGQuadrotorGatekeeperEnv(CMDP):
    """Gatekeeper-RL env with binary locomotion/backup switch and a learned backup."""

    _support_envs: ClassVar[list[str]] = [_GATEKEEPER_ENV_ID]

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

        # --- Hyper-parameters from env vars ---
        self._H: int = int(os.environ.get('GK_H', '60'))
        self._delta: int = int(os.environ.get('GK_DELTA', '5'))
        assert self._delta <= self._H, 'delta must be <= H for recursive feasibility'

        self._r_nom: float = float(os.environ.get('GK_R_NOM', '1.0'))
        self._r_override: float = float(os.environ.get('GK_R_OVERRIDE', '0.2'))
        self._r_crash: float = float(os.environ.get('GK_R_CRASH', '20.0'))

        self._use_filter: bool = os.environ.get('GK_USE_FILTER', 'true').lower() == 'true'
        self._use_shaping: bool = os.environ.get('GK_USE_SHAPING', 'true').lower() == 'true'

        # --- Inner SCG env ---
        scg_kwargs = dict(_SCG_KWARGS)
        if os.environ.get('GK_GUI', 'false').lower() == 'true':
            scg_kwargs['gui'] = True
        self._scg_env = scg_make(_SCG_TASK, **scg_kwargs)
        init_obs, _info = self._scg_env.reset()
        self._state_dim = int(self._scg_env.observation_space.shape[0])
        self._act_dim = int(self._scg_env.action_space.shape[0])
        assert hasattr(self._scg_env, 'X_GOAL'), 'Inner SCG env has no X_GOAL.'
        self._X_GOAL: np.ndarray = np.asarray(self._scg_env.X_GOAL, dtype=np.float32)
        self._T_total: int = int(len(self._X_GOAL))
        self._max_episode_steps: int = (self._T_total // self._delta) + 1

        # Low-level (classical DSL PID)
        self._low_level = ClassicalLowLevel(quad_type=int(scg_kwargs['quad_type']))

        # Backup policy (frozen, learned)
        backup_run = os.environ.get('GK_BACKUP_RUN_DIR', '').strip()
        backup_model = os.environ.get('GK_BACKUP_MODEL', '').strip() or None
        backup_fallback = os.environ.get('GK_BACKUP_FALLBACK', 'true').lower() == 'true'
        self._backup = FrozenBackupPolicy(
            run_dir=backup_run or None,
            model_name=backup_model,
            fallback=backup_fallback,
        )

        # Safety box (matches _SCG_KWARGS constraints)
        self._x_bounds = (-2.0, 0.3)
        self._z_bounds = (0.3, 1.7)
        self._theta_bounds = (-0.35, 0.35)

        # Spaces: obs = [state_6d, t/T_total]; act = Box([0,1])
        obs_low = np.concatenate([
            np.full((self._state_dim,), -np.inf, dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.full((self._state_dim,), np.inf, dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ])
        self._observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self._action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

        self._metadata: dict[str, Any] = {
            'H': self._H,
            'delta': self._delta,
            'T_total': self._T_total,
            'r_nom': self._r_nom,
            'r_override': self._r_override,
            'r_crash': self._r_crash,
            'use_filter': self._use_filter,
            'use_shaping': self._use_shaping,
            'low_level': 'ClassicalLowLevel(DSL-PID)',
            'backup_loaded': self._backup.loaded,
            'backup_run_dir': backup_run,
        }

        self._state: np.ndarray = np.zeros(self._state_dim, dtype=np.float32)
        self._t: int = 0
        self._n_decisions: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------
    def _nominal_ref(self, t: int) -> np.ndarray:
        idx = int(np.clip(t, 0, self._T_total - 1))
        return self._X_GOAL[idx]

    def _backup_ref(self, state: np.ndarray) -> np.ndarray:
        return self._backup.reference(state)

    # ------------------------------------------------------------------
    # PyBullet snapshot / restore
    # ------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        client = self._scg_env.PYB_CLIENT
        sid = pb.saveState(physicsClientId=client)
        return {
            'pb': sid,
            'ctrl_step_counter': int(getattr(self._scg_env, 'ctrl_step_counter', 0)),
            'pyb_step_counter': int(getattr(self._scg_env, 'pyb_step_counter', 0)),
            'state': self._state.copy(),
            'pid_state': self._low_level.save_state(),
        }

    def _restore(self, snap: dict[str, Any]) -> None:
        client = self._scg_env.PYB_CLIENT
        pb.restoreState(stateId=snap['pb'], physicsClientId=client)
        if hasattr(self._scg_env, 'ctrl_step_counter'):
            self._scg_env.ctrl_step_counter = snap['ctrl_step_counter']
        if hasattr(self._scg_env, 'pyb_step_counter'):
            self._scg_env.pyb_step_counter = snap['pyb_step_counter']
        self._state = snap['state'].copy()
        self._low_level.load_state(snap['pid_state'])

    def _free_snapshot(self, snap: dict[str, Any]) -> None:
        try:
            pb.removeState(snap['pb'], physicsClientId=self._scg_env.PYB_CLIENT)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------
    def _state_in_box(self, state: np.ndarray) -> bool:
        x_ok = self._x_bounds[0] <= float(state[0]) <= self._x_bounds[1]
        z_ok = True
        if self._state_dim > 2:
            z_ok = self._z_bounds[0] <= float(state[2]) <= self._z_bounds[1]
        theta_ok = True
        if self._state_dim > 4:
            theta_ok = self._theta_bounds[0] <= float(state[4]) <= self._theta_bounds[1]
        return x_ok and z_ok and theta_ok

    # ------------------------------------------------------------------
    # Gatekeeper preview (core of the filter)
    # ------------------------------------------------------------------
    def _preview_mode(self, mode: int, t_start: int) -> bool:
        """Simulate keeping ``mode`` (0=backup, 1=nominal) for H steps.

        Returns True iff every previewed step stays inside the safe box.
        """
        snap = self._snapshot()
        safe = True
        cur_state = self._state.copy()
        for i in range(self._H):
            if mode == 1:
                ref = self._nominal_ref(t_start + i)
            else:
                ref = self._backup_ref(cur_state)
            action_np = self._low_level(cur_state, ref)
            next_state, _rew, _done, info = self._scg_env.step(action_np)
            cur_state = np.asarray(next_state, dtype=np.float32)
            violated = int(info.get('constraint_violation', 0)) != 0
            if violated or not self._state_in_box(cur_state):
                safe = False
                break
        self._restore(snap)
        self._free_snapshot(snap)
        return safe

    # ------------------------------------------------------------------
    # Build obs
    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        t_norm = float(self._t) / max(1, self._T_total)
        return np.concatenate([
            self._state.astype(np.float32),
            np.array([t_norm], dtype=np.float32),
        ])

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

        # Rejection-sample initial state inside the safe box.
        max_retries = 32
        init_obs = None
        info: dict[str, Any] = {}
        for _ in range(max_retries):
            init_obs, info = self._scg_env.reset()
            if self._state_in_box(np.asarray(init_obs, dtype=np.float32)):
                break
        self._state = np.asarray(init_obs, dtype=np.float32)
        self._t = 0
        self._n_decisions = 0
        self._low_level.reset_state()

        self._X_GOAL = np.asarray(self._scg_env.X_GOAL, dtype=np.float32)
        self._T_total = int(len(self._X_GOAL))

        info = dict(info)
        info.update({
            'low_level': 'ClassicalLowLevel(DSL-PID)',
            'backup_loaded': self._backup.loaded,
        })
        return (
            torch.as_tensor(self._build_obs(), dtype=torch.float32, device=self._device),
            info,
        )

    def _execute_delta(self, mode: int) -> int:
        """Execute ``delta`` real steps of ``mode`` (0=backup, 1=nominal).

        Returns the number of per-step safety violations that occurred.
        """
        n_violations = 0
        for _ in range(self._delta):
            if self._t >= self._T_total:
                break
            if mode == 1:
                ref = self._nominal_ref(self._t)
            else:
                ref = self._backup_ref(self._state)
            action_np = self._low_level(self._state, ref)
            next_state, _rew, _done, info = self._scg_env.step(action_np)
            self._state = np.asarray(next_state, dtype=np.float32)
            self._t += 1
            violated = int(info.get('constraint_violation', 0)) != 0
            if violated or not self._state_in_box(self._state):
                n_violations += 1
        return n_violations

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
        a_clip = float(np.clip(a_np[0], 0.0, 1.0))
        # a_rl in {0: backup, 1: nominal}
        a_rl = 1 if a_clip > 0.5 else 0

        # --- Filter: preview nominal before allowing it ---
        rejected = False
        preview_safe = True
        t_now = self._t
        if self._use_filter and a_rl == 1:
            preview_safe = self._preview_mode(mode=1, t_start=t_now)
            if not preview_safe:
                rejected = True
        committed = 0 if rejected else a_rl

        # --- Real execution ---
        n_violations = self._execute_delta(mode=committed)

        # --- Reward ---
        if committed == 1 and n_violations == 0:
            r_choice = self._r_nom
        elif rejected:
            r_choice = -self._r_override if self._use_shaping else 0.0
        else:
            r_choice = 0.0
        r_crash = -self._r_crash * float(n_violations)
        reward = r_choice + r_crash

        cost = 0.0

        terminated = n_violations > 0
        reached_end = self._t >= self._T_total
        truncated = reached_end and not terminated

        self._n_decisions += 1

        info = {
            'a_rl': int(a_rl),
            'committed': int(committed),
            'rejected': int(rejected),
            'preview_safe': int(preview_safe),
            'n_violations': int(n_violations),
            'r_choice': float(r_choice),
            'r_crash': float(r_crash),
            't': int(self._t),
            'n_decisions': int(self._n_decisions),
        }
        return (
            torch.as_tensor(self._build_obs(), dtype=torch.float32, device=self._device),
            torch.as_tensor(reward, dtype=torch.float32, device=self._device),
            torch.as_tensor(cost, dtype=torch.float32, device=self._device),
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

    @property
    def inner_env(self) -> Any:
        return self._scg_env

    @property
    def low_level(self) -> ClassicalLowLevel:
        return self._low_level

    @property
    def backup(self) -> FrozenBackupPolicy:
        return self._backup

    @property
    def t(self) -> int:
        return self._t

    @property
    def X_GOAL(self) -> np.ndarray:
        return self._X_GOAL

    @property
    def H(self) -> int:
        return self._H

    @property
    def delta(self) -> int:
        return self._delta

    def spec_log(self, logger: Any) -> None:  # noqa: ANN401
        pass
