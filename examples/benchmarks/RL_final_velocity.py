"""Three-stage velocity scenario for Ant, HalfCheetah, and Humanoid.

This script implements the scenario discussed in the project notes:

1. Train a nominal PPO policy on the standard Safety-Gymnasium velocity task.
   PPO ignores the cost signal, so this stage behaves like an unconstrained
   "go as fast as possible" locomotion policy.
2. Train a recovery PPO policy in a custom environment that starts from
   perturbed / burned-in states and rewards the robot for coming to a stable,
   low-speed equilibrium quickly.
3. Freeze both stage-1 and stage-2 actors, then train a PPO switching policy
   whose action is a binary gate between the nominal and recovery actors.

The switching policy is implemented as a 1-D Box action thresholded at 0.5
instead of a native Discrete policy head because the current OmniSafe PPO stack
used in this repository is built around continuous Box actions.

See the companion documentation file:
  examples/benchmarks/RL_final_velocity.md
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

# Allow running this script directly from the repo checkout without requiring
# `pip install -e .` first.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omnisafe.common.normalizer import Normalizer
from omnisafe.envs.core import CMDP, env_register
from omnisafe.models.actor import ActorBuilder
from omnisafe.typing import DEVICE_CPU


ALGO = 'PPO'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name('exp-final-velocity')
DEFAULT_ROBOTS = ['ant', 'halfcheetah', 'humanoid']
DEFAULT_STAGES = ['nominal', 'recovery', 'switch']
DEFAULT_SEEDS = [0]
DEFAULT_EVAL_EPISODES = 5
DEFAULT_TRACE_EPISODES = 2
DEFAULT_BASE_TASK_EVAL_EPISODES = 5
DEFAULT_VIDEO_EPISODES = 1
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_CAMERA_NAME = 'track'
DEFAULT_BASELINE_COST_LIMIT = 25.0

DEFAULT_TOTAL_STEPS = {
    'nominal': 1_000_000,
    'recovery': 1_000_000,
    'switch': 1_000_000,
}
DEFAULT_STEPS_PER_EPOCH = {
    'nominal': 20_000,
    'recovery': 10_000,
    'switch': 10_000,
}

RECOVERY_NOMINAL_RESET_PROB = 0.70
RECOVERY_MAX_RESET_ATTEMPTS = 8

RECOVERY_ALIVE_BONUS = 0.25
RECOVERY_STABLE_BONUS = 1.00
RECOVERY_SUCCESS_BONUS = 25.0
RECOVERY_FALL_PENALTY = 25.0
RECOVERY_SPEED_WEIGHT = 1.00
RECOVERY_ACTION_WEIGHT = 0.01
RECOVERY_CONSTRAINT_WEIGHT = 2.00

SWITCH_NOMINAL_REWARD = 1.00
SWITCH_VIOLATION_PENALTY = 25.0
SWITCH_UNHEALTHY_PENALTY = 10.0
SWITCH_TERMINATE_ON_VIOLATION = False


@dataclass(frozen=True)
class VelocityRobotSpec:
    """Per-robot configuration for the staged velocity scenario."""

    key: str
    title: str
    nominal_env_id: str
    recovery_env_id: str
    switch_env_id: str
    stop_speed: float
    reset_burnin_range: tuple[int, int]
    success_hold_steps: int
    recovery_horizon: int
    switch_horizon: int


ROBOT_SPECS: dict[str, VelocityRobotSpec] = {
    'ant': VelocityRobotSpec(
        key='ant',
        title='Ant',
        nominal_env_id='SafetyAntVelocity-v1',
        recovery_env_id='VelocityRecoveryAnt-v0',
        switch_env_id='VelocitySwitchAnt-v0',
        stop_speed=0.60,
        reset_burnin_range=(50, 250),
        success_hold_steps=25,
        recovery_horizon=250,
        switch_horizon=1000,
    ),
    'halfcheetah': VelocityRobotSpec(
        key='halfcheetah',
        title='HalfCheetah',
        nominal_env_id='SafetyHalfCheetahVelocity-v1',
        recovery_env_id='VelocityRecoveryHalfCheetah-v0',
        switch_env_id='VelocitySwitchHalfCheetah-v0',
        stop_speed=0.50,
        reset_burnin_range=(50, 250),
        success_hold_steps=25,
        recovery_horizon=250,
        switch_horizon=1000,
    ),
    'humanoid': VelocityRobotSpec(
        key='humanoid',
        title='Humanoid',
        nominal_env_id='SafetyHumanoidVelocity-v1',
        recovery_env_id='VelocityRecoveryHumanoid-v0',
        switch_env_id='VelocitySwitchHumanoid-v0',
        stop_speed=0.35,
        reset_burnin_range=(25, 180),
        success_hold_steps=30,
        recovery_horizon=250,
        switch_horizon=1000,
    ),
}


# Runtime checkpoint context.
# The scenario runs sequentially with parallel=1, so a simple module-level
# context is enough for the custom envs to discover the correct frozen actors.
_RUNTIME_POLICY_DIRS: dict[str, dict[str, Path]] = {
    'nominal': {},
    'recovery': {},
}


def _log(message: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {message}', flush=True)


def _resolve_robot_spec(env_id: str) -> VelocityRobotSpec:
    for spec in ROBOT_SPECS.values():
        if env_id in (spec.recovery_env_id, spec.switch_env_id):
            return spec
    raise KeyError(f'No robot spec registered for env_id={env_id!r}')


def _set_runtime_policy_dir(stage: str, robot: str, run_dir: Path) -> None:
    _RUNTIME_POLICY_DIRS[stage][robot] = Path(run_dir).expanduser().resolve()


def _get_runtime_policy_dir(stage: str, robot: str) -> Path | None:
    return _RUNTIME_POLICY_DIRS[stage].get(robot)


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _make_base_velocity_env(
    env_id: str,
    *,
    render_mode: str | None = None,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
    camera_name: str | None = None,
):
    import safety_gymnasium

    make_kwargs: dict[str, Any] = {'id': env_id, 'autoreset': False}
    if render_mode is not None:
        make_kwargs['render_mode'] = render_mode
        make_kwargs['width'] = width
        make_kwargs['height'] = height
        if camera_name:
            make_kwargs['camera_name'] = camera_name

    env = safety_gymnasium.make(**make_kwargs)
    assert isinstance(env.action_space, spaces.Box), 'This scenario only supports Box actions.'
    assert isinstance(
        env.observation_space,
        spaces.Box,
    ), 'This scenario only supports Box observations.'
    return env


def _extract_speed(info: dict[str, Any]) -> float:
    x_velocity = float(info.get('x_velocity', 0.0))
    y_velocity = float(info.get('y_velocity', 0.0))
    if 'y_velocity' in info:
        return float(math.sqrt(x_velocity**2 + y_velocity**2))
    return abs(x_velocity)


def _extract_health_flag(base_env: Any, terminated: bool, info: dict[str, Any]) -> bool:
    """Best-effort robot health signal shared across the three locomotion tasks.

    TODO: Replace this with explicit per-robot posture checks once the exact
    state conventions are pinned down for all target tasks.
    """
    if 'is_healthy' in info:
        return bool(info['is_healthy'])

    unwrapped = getattr(base_env, 'unwrapped', None)
    if unwrapped is not None:
        is_healthy_attr = getattr(unwrapped, 'is_healthy', None)
        if callable(is_healthy_attr):
            return bool(is_healthy_attr())
        if is_healthy_attr is not None:
            return bool(is_healthy_attr)

    return not bool(terminated)


def _clip_to_action_space(action: np.ndarray, action_space: spaces.Box) -> np.ndarray:
    clipped = np.clip(np.asarray(action, dtype=np.float32), action_space.low, action_space.high)
    return clipped.astype(np.float32)


def _gate_action_space() -> spaces.Box:
    return spaces.Box(
        low=np.zeros(1, dtype=np.float32),
        high=np.ones(1, dtype=np.float32),
        dtype=np.float32,
    )


def _pop_render_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        'render_mode': kwargs.pop('render_mode', None),
        'width': int(kwargs.pop('width', DEFAULT_VIDEO_WIDTH)),
        'height': int(kwargs.pop('height', DEFAULT_VIDEO_HEIGHT)),
        'camera_name': kwargs.pop('camera_name', None),
    }


def _stage_root(output_dir: Path, robot: str, stage: str) -> Path:
    return output_dir / robot / stage


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.joinpath('torch_save').glob('epoch-*.pt'),
        key=lambda path: int(path.stem.split('-')[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f'No epoch-*.pt checkpoints found under {run_dir / "torch_save"}')
    return checkpoints[-1]


def _find_seed_run_dir(stage_dir: Path, seed: int) -> Path | None:
    if not stage_dir.exists():
        return None

    seed_tag = f'seed-{seed:03d}-'
    candidates: list[Path] = []
    for torch_dir in stage_dir.rglob('torch_save'):
        run_dir = torch_dir.parent
        if seed_tag not in run_dir.name:
            continue
        if not run_dir.joinpath('config.json').exists():
            continue
        if not list(torch_dir.glob('epoch-*.pt')):
            continue
        candidates.append(run_dir)

    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _require_stage_run(output_dir: Path, robot: str, stage: str, seed: int) -> Path:
    run_dir = _find_seed_run_dir(_stage_root(output_dir, robot, stage), seed)
    if run_dir is None:
        raise FileNotFoundError(
            f'Missing {stage} checkpoint for robot={robot!r}, seed={seed}. '
            f'Expected under {_stage_root(output_dir, robot, stage)}.',
        )
    return run_dir


def _remove_stage_dir_if_requested(output_dir: Path, robot: str, stage: str, force: bool) -> None:
    stage_dir = _stage_root(output_dir, robot, stage)
    if force and stage_dir.exists():
        shutil.rmtree(stage_dir)


class SavedOmniSafeActor:
    """Frozen OmniSafe PPO actor loaded from a run directory."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str] | None,
        obs_space: spaces.Box,
        action_space: spaces.Box,
        label: str,
        required: bool,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._run_dir = Path(run_dir).expanduser().resolve() if run_dir is not None else None
        self._obs_space = obs_space
        self._action_space = action_space
        self._label = label
        self._required = required
        self._checkpoint_path_override = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None
        )

        self._actor = None
        self._obs_normalizer: Normalizer | None = None
        self._loaded = False
        self._checkpoint_path: Path | None = None

        self._load()

    def _load(self) -> None:
        if self._checkpoint_path_override is not None:
            checkpoint_path = self._checkpoint_path_override
            if self._run_dir is None:
                self._run_dir = checkpoint_path.parent.parent
        else:
            if self._run_dir is None:
                if self._required:
                    raise FileNotFoundError(f'No run_dir provided for required {self._label} actor.')
                return
            if not self._run_dir.exists():
                if self._required:
                    raise FileNotFoundError(f'{self._label} run_dir not found: {self._run_dir}')
                return
            checkpoint_path = _latest_checkpoint(self._run_dir)

        if self._run_dir is None or not self._run_dir.exists():
            if self._required:
                raise FileNotFoundError(f'{self._label} run_dir not found: {self._run_dir}')
            return
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'{self._label} checkpoint not found: {checkpoint_path}')
        config_path = self._run_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'Missing config.json in {self._run_dir}')

        with open(config_path, encoding='utf-8') as handle:
            config = json.load(handle)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_cfgs = config['model_cfgs']
        actor_cfg = model_cfgs['actor']
        builder = ActorBuilder(
            obs_space=self._obs_space,
            act_space=self._action_space,
            hidden_sizes=actor_cfg['hidden_sizes'],
            activation=actor_cfg['activation'],
            weight_initialization_mode=model_cfgs['weight_initialization_mode'],
        )
        actor = builder.build_actor(model_cfgs['actor_type'])
        actor.load_state_dict(checkpoint['pi'])
        actor.eval()
        self._actor = actor

        if config.get('algo_cfgs', {}).get('obs_normalize', False) and checkpoint.get(
            'obs_normalizer',
        ) is not None:
            obs_normalizer = Normalizer(shape=self._obs_space.shape, clip=10.0)
            obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
            obs_normalizer.eval()
            self._obs_normalizer = obs_normalizer

        self._loaded = True
        self._checkpoint_path = checkpoint_path

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    @property
    def checkpoint_path(self) -> Path | None:
        return self._checkpoint_path

    def action(self, observation: np.ndarray) -> np.ndarray:
        if not self._loaded or self._actor is None:
            raise RuntimeError(f'{self._label} actor has not been loaded.')

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if self._obs_normalizer is not None:
            std = torch.clamp(self._obs_normalizer.std, min=1e-2)
            obs_tensor = (obs_tensor - self._obs_normalizer.mean) / std
            obs_tensor = torch.clamp(obs_tensor, -10.0, 10.0)

        with torch.no_grad():
            try:
                action = self._actor.predict(obs_tensor, deterministic=True)
            except AttributeError:
                distribution = self._actor(obs_tensor)
                action = distribution.mean

        action_np = action.squeeze(0).cpu().numpy()
        return _clip_to_action_space(action_np, self._action_space)

@env_register
class VelocityRecoveryEnv(CMDP):
    """Custom PPO environment for training the recovery policy."""

    _support_envs: ClassVar[list[str]] = [
        ROBOT_SPECS['ant'].recovery_env_id,
        ROBOT_SPECS['halfcheetah'].recovery_env_id,
        ROBOT_SPECS['humanoid'].recovery_env_id,
    ]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    _num_envs = 1

    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, **kwargs: Any) -> None:
        super().__init__(env_id)
        self._device = torch.device(device)
        self._spec = _resolve_robot_spec(env_id)
        render_kwargs = _pop_render_kwargs(kwargs)
        self._base_env = _make_base_velocity_env(self._spec.nominal_env_id, **render_kwargs)
        # Match the baseline safe-velocity information pattern as closely as
        # possible: the recovery actor only sees the base environment
        # observation at decision time.
        self._observation_space = self._base_env.observation_space
        self._action_space = self._base_env.action_space

        nominal_run_dir = _get_runtime_policy_dir('nominal', self._spec.key)
        self._nominal_actor = SavedOmniSafeActor(
            run_dir=nominal_run_dir,
            obs_space=self._base_env.observation_space,
            action_space=self._action_space,
            label=f'{self._spec.title} nominal',
            required=False,
        )

        self._metadata = {
            'robot': self._spec.title,
            'base_env_id': self._spec.nominal_env_id,
            'nominal_run_dir': str(nominal_run_dir) if nominal_run_dir is not None else '',
        }
        self._max_episode_steps = self._spec.recovery_horizon

        self._rng = np.random.default_rng()
        self._base_obs = np.zeros(self._base_env.observation_space.shape, dtype=np.float32)
        self._current_speed = 0.0
        self._current_cost = 0.0
        self._current_healthy = True
        self._stable_steps = 0
        self._steps = 0
        self._last_reset_source = 'clean_reset'

    def _build_obs(self) -> np.ndarray:
        return self._base_obs.astype(np.float32)

    def _sample_burnin_action(self, use_nominal_burnin: bool) -> np.ndarray:
        if use_nominal_burnin and self._nominal_actor.loaded:
            return self._nominal_actor.action(self._base_obs)
        return _clip_to_action_space(self._base_env.action_space.sample(), self._action_space)

    def _burn_in_reset_state(self, seed: int | None) -> tuple[np.ndarray, float, float, bool, str]:
        low, high = self._spec.reset_burnin_range
        # TODO: If we need broader recovery-state coverage later, replace this
        # burn-in sampler with direct qpos/qvel state injection per robot.
        for attempt in range(RECOVERY_MAX_RESET_ATTEMPTS):
            base_obs, _info = self._base_env.reset(seed=seed if attempt == 0 else None)
            base_obs = np.asarray(base_obs, dtype=np.float32)
            speed = 0.0
            cost = 0.0
            healthy = True
            burnin_steps = int(self._rng.integers(low, high + 1))
            use_nominal_burnin = self._nominal_actor.loaded and (
                self._rng.random() < RECOVERY_NOMINAL_RESET_PROB
            )
            reset_source = 'nominal_burnin' if use_nominal_burnin else 'random_burnin'

            terminated_during_burnin = False
            for _ in range(burnin_steps):
                action = self._sample_burnin_action(use_nominal_burnin=use_nominal_burnin)
                next_obs, _reward, step_cost, terminated, truncated, step_info = self._base_env.step(
                    action,
                )
                if bool(terminated) or bool(truncated):
                    terminated_during_burnin = True
                    break
                base_obs = np.asarray(next_obs, dtype=np.float32)
                speed = _extract_speed(step_info)
                cost = float(step_cost)
                healthy = _extract_health_flag(self._base_env, bool(terminated), step_info)

            if not terminated_during_burnin:
                return base_obs, speed, cost, healthy, reset_source

        base_obs, _info = self._base_env.reset(seed=seed)
        return np.asarray(base_obs, dtype=np.float32), 0.0, 0.0, True, 'fallback_clean_reset'

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del options
        if seed is not None:
            self.set_seed(seed)
            self._rng = np.random.default_rng(seed)

        (
            self._base_obs,
            self._current_speed,
            self._current_cost,
            self._current_healthy,
            self._last_reset_source,
        ) = self._burn_in_reset_state(seed=seed)
        self._stable_steps = 0
        self._steps = 0

        info = {
            'reset_source': self._last_reset_source,
            'nominal_actor_loaded': self._nominal_actor.loaded,
            'stop_speed': self._spec.stop_speed,
        }
        return _to_tensor(self._build_obs(), self._device), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_np = _clip_to_action_space(
            action.detach().cpu().numpy().reshape(-1),
            self._action_space,
        )

        next_obs, _base_reward, base_cost, base_terminated, base_truncated, info = self._base_env.step(
            action_np,
        )
        self._base_obs = np.asarray(next_obs, dtype=np.float32)
        self._current_speed = _extract_speed(info)
        self._current_cost = float(base_cost)
        self._current_healthy = _extract_health_flag(
            self._base_env,
            bool(base_terminated),
            info,
        )
        self._steps += 1

        stable_now = (
            self._current_healthy
            and self._current_cost <= 0.0
            and self._current_speed <= self._spec.stop_speed
        )
        self._stable_steps = self._stable_steps + 1 if stable_now else 0
        success = self._stable_steps >= self._spec.success_hold_steps

        reward = (
            RECOVERY_ALIVE_BONUS * float(self._current_healthy)
            + RECOVERY_STABLE_BONUS * float(stable_now)
            - RECOVERY_SPEED_WEIGHT * self._current_speed
            - RECOVERY_ACTION_WEIGHT * float(np.square(action_np).mean())
            - RECOVERY_CONSTRAINT_WEIGHT * self._current_cost
        )
        if not self._current_healthy or bool(base_terminated):
            reward -= RECOVERY_FALL_PENALTY
        if success:
            reward += RECOVERY_SUCCESS_BONUS

        terminated = bool(base_terminated) or success
        truncated = bool(base_truncated) or (self._steps >= self._max_episode_steps and not terminated)

        info_dict = {
            'speed': self._current_speed,
            'constraint_cost': self._current_cost,
            'healthy': int(self._current_healthy),
            'stable_steps': self._stable_steps,
            'success': int(success),
            'reset_source': self._last_reset_source,
        }
        return (
            _to_tensor(self._build_obs(), self._device),
            _to_tensor(reward, self._device),
            _to_tensor(self._current_cost, self._device),
            _to_tensor(terminated, self._device, dtype=torch.bool),
            _to_tensor(truncated, self._device, dtype=torch.bool),
            info_dict,
        )

    def set_seed(self, seed: int) -> None:
        self._base_env.action_space.seed(seed)
        self._base_env.reset(seed=seed)

    def render(self) -> Any:
        return self._base_env.render()

    def close(self) -> None:
        self._base_env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Any) -> None:  # noqa: ANN401
        del logger


@env_register
class VelocitySwitchEnv(CMDP):
    """Binary switching environment over frozen nominal and recovery actors."""

    _support_envs: ClassVar[list[str]] = [
        ROBOT_SPECS['ant'].switch_env_id,
        ROBOT_SPECS['halfcheetah'].switch_env_id,
        ROBOT_SPECS['humanoid'].switch_env_id,
    ]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    _num_envs = 1

    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, **kwargs: Any) -> None:
        super().__init__(env_id)
        self._device = torch.device(device)
        self._spec = _resolve_robot_spec(env_id)
        render_kwargs = _pop_render_kwargs(kwargs)
        self._base_env = _make_base_velocity_env(self._spec.nominal_env_id, **render_kwargs)

        self._gate_action_space = _gate_action_space()
        # Match the baseline safe-velocity setup: the switch actor only sees
        # the original environment observation and must infer safety from it.
        self._observation_space = self._base_env.observation_space
        self._action_space = self._gate_action_space

        nominal_run_dir = _get_runtime_policy_dir('nominal', self._spec.key)
        recovery_run_dir = _get_runtime_policy_dir('recovery', self._spec.key)
        self._nominal_actor = SavedOmniSafeActor(
            run_dir=nominal_run_dir,
            obs_space=self._base_env.observation_space,
            action_space=self._base_env.action_space,
            label=f'{self._spec.title} nominal',
            required=True,
        )
        self._recovery_actor = SavedOmniSafeActor(
            run_dir=recovery_run_dir,
            obs_space=self._base_env.observation_space,
            action_space=self._base_env.action_space,
            label=f'{self._spec.title} recovery',
            required=True,
        )

        self._metadata = {
            'robot': self._spec.title,
            'base_env_id': self._spec.nominal_env_id,
            'nominal_run_dir': str(nominal_run_dir),
            'recovery_run_dir': str(recovery_run_dir),
        }
        self._max_episode_steps = self._spec.switch_horizon

        self._base_obs = np.zeros(self._base_env.observation_space.shape, dtype=np.float32)
        self._current_speed = 0.0
        self._current_cost = 0.0
        self._current_healthy = True
        self._last_gate = 0.0
        self._steps = 0

    def _build_switch_obs(self) -> np.ndarray:
        return self._base_obs.astype(np.float32)

    def _current_recovery_obs(self) -> np.ndarray:
        return self._base_obs.astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del options
        if seed is not None:
            self.set_seed(seed)

        base_obs, _info = self._base_env.reset(seed=seed)
        self._base_obs = np.asarray(base_obs, dtype=np.float32)
        self._current_speed = 0.0
        self._current_cost = 0.0
        self._current_healthy = True
        self._last_gate = 0.0
        self._steps = 0

        info = {
            'nominal_actor_loaded': self._nominal_actor.loaded,
            'recovery_actor_loaded': self._recovery_actor.loaded,
        }
        return _to_tensor(self._build_switch_obs(), self._device), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # TODO: Swap this thresholded Box action for a true categorical policy
        # once the local OmniSafe PPO stack supports discrete actions cleanly.
        gate_value = float(np.clip(action.detach().cpu().numpy().reshape(-1)[0], 0.0, 1.0))
        gate = 1 if gate_value > 0.5 else 0

        if gate == 1:
            env_action = self._nominal_actor.action(self._base_obs)
        else:
            env_action = self._recovery_actor.action(self._current_recovery_obs())

        next_obs, _base_reward, base_cost, base_terminated, base_truncated, info = self._base_env.step(
            env_action,
        )
        self._base_obs = np.asarray(next_obs, dtype=np.float32)
        self._current_speed = _extract_speed(info)
        self._current_cost = float(base_cost)
        self._current_healthy = _extract_health_flag(
            self._base_env,
            bool(base_terminated),
            info,
        )
        self._last_gate = float(gate)
        self._steps += 1

        reward = SWITCH_NOMINAL_REWARD if gate == 1 else 0.0
        reward -= SWITCH_VIOLATION_PENALTY * self._current_cost
        if not self._current_healthy or bool(base_terminated):
            reward -= SWITCH_UNHEALTHY_PENALTY

        terminated = bool(base_terminated)
        if SWITCH_TERMINATE_ON_VIOLATION and self._current_cost > 0.0:
            terminated = True
        truncated = bool(base_truncated) or (self._steps >= self._max_episode_steps and not terminated)

        info_dict = {
            'gate': gate,
            'gate_value': gate_value,
            'speed': self._current_speed,
            'constraint_cost': self._current_cost,
            'healthy': int(self._current_healthy),
        }
        return (
            _to_tensor(self._build_switch_obs(), self._device),
            _to_tensor(reward, self._device),
            _to_tensor(self._current_cost, self._device),
            _to_tensor(terminated, self._device, dtype=torch.bool),
            _to_tensor(truncated, self._device, dtype=torch.bool),
            info_dict,
        )

    def set_seed(self, seed: int) -> None:
        self._base_env.action_space.seed(seed)
        self._base_env.reset(seed=seed)

    def render(self) -> Any:
        return self._base_env.render()

    def close(self) -> None:
        self._base_env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Any) -> None:  # noqa: ANN401
        del logger


def _resolve_device(requested_device: str) -> str:
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        _log(f'CUDA requested ({requested_device}) but not available. Falling back to cpu.')
        return 'cpu'
    return requested_device


def _build_train_cfg(
    seed: int,
    total_steps: int,
    steps_per_epoch: int,
    log_dir: Path,
    device: str,
) -> dict[str, Any]:
    epochs = max(1, total_steps // steps_per_epoch)
    save_model_freq = max(1, epochs // 5)
    return {
        'seed': seed,
        'train_cfgs': {
            'device': device,
            'torch_threads': 1,
            'vector_env_nums': 1,
            'parallel': 1,
            'total_steps': total_steps,
        },
        'algo_cfgs': {
            'steps_per_epoch': steps_per_epoch,
            'update_iters': 10,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'save_model_freq': save_model_freq,
            'log_dir': str(log_dir),
        },
        'model_cfgs': {
            'actor': {'lr': 3e-4},
            'critic': {'lr': 3e-4},
        },
    }


def _train_stage_if_needed(
    stage_name: str,
    robot: str,
    env_id: str,
    seed: int,
    total_steps: int,
    steps_per_epoch: int,
    output_dir: Path,
    device: str,
) -> Path:
    stage_dir = _stage_root(output_dir, robot, stage_name)
    existing_run = _find_seed_run_dir(stage_dir, seed)
    if existing_run is not None:
        _log(
            f'  [{stage_name}] robot={robot} seed={seed}: using existing run {existing_run}',
        )
        return existing_run

    stage_dir.mkdir(parents=True, exist_ok=True)
    custom_cfgs = _build_train_cfg(
        seed=seed,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        log_dir=stage_dir,
        device=device,
    )

    # Imported lazily so the module can still be syntax-checked in minimal
    # environments where the full OmniSafe training stack is not installed.
    import omnisafe

    _log(
        f'  [{stage_name}] robot={robot} seed={seed}: training {ALGO} on {env_id}',
    )
    agent = omnisafe.Agent(ALGO, env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    run_dir = Path(agent.agent.logger.log_dir).resolve()
    _log(f'  [{stage_name}] robot={robot} seed={seed}: finished at {run_dir}')
    return run_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _read_progress_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing progress.csv: {path}')

    rows: list[dict[str, float]] = []
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == '':
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def _checkpoint_sort_key(path: Path) -> int:
    return int(path.stem.split('-')[1])


def _list_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = sorted(
        run_dir.joinpath('torch_save').glob('epoch-*.pt'),
        key=_checkpoint_sort_key,
    )
    if not checkpoints:
        raise FileNotFoundError(f'No epoch-*.pt checkpoints found under {run_dir / "torch_save"}')
    return checkpoints


def _extract_training_curve_rows(run_dir: Path) -> list[dict[str, float]]:
    progress_rows = _read_progress_csv(run_dir / 'progress.csv')
    return [
        {
            'total_env_steps': float(row['TotalEnvSteps']),
            'episode_reward': float(row['Metrics/EpRet']),
            'episode_cost': float(row['Metrics/EpCost']),
            'episode_length': float(row.get('Metrics/EpLen', 0.0)),
        }
        for row in progress_rows
        if 'TotalEnvSteps' in row and 'Metrics/EpRet' in row and 'Metrics/EpCost' in row
    ]


def _checkpoint_total_steps(
    checkpoint_path: Path,
    progress_rows: list[dict[str, float]],
    total_steps: int,
) -> float:
    checkpoint_epoch = _checkpoint_sort_key(checkpoint_path)
    if checkpoint_epoch < len(progress_rows) and 'TotalEnvSteps' in progress_rows[checkpoint_epoch]:
        return float(progress_rows[checkpoint_epoch]['TotalEnvSteps'])
    return float(total_steps)


def _maybe_import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _capture_render_frame(env: Any) -> tuple[np.ndarray | None, str | None]:
    try:
        frame = env.render()
    except Exception as error:  # pragma: no cover - best-effort visualization path
        return None, str(error)

    if frame is None:
        return None, 'render returned None'

    frame_np = np.asarray(frame)
    if frame_np.ndim != 3:
        return None, f'unexpected frame shape {frame_np.shape!r}'
    return frame_np.copy(), None


def _save_rollout_video(
    frames: list[np.ndarray],
    video_dir: Path,
    name_prefix: str,
    episode_index: int,
    fps: int,
) -> tuple[Path | None, str | None]:
    if not frames:
        return None, 'no frames captured'

    try:
        from gymnasium.utils.save_video import save_video
    except Exception as error:  # pragma: no cover - depends on local video deps
        return None, str(error)

    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f'{name_prefix}-episode-{episode_index}.mp4'
    try:
        save_video(
            frames,
            str(video_dir),
            fps=fps,
            episode_trigger=lambda _episode: True,
            video_length=len(frames),
            episode_index=episode_index,
            name_prefix=name_prefix,
        )
    except Exception as error:  # pragma: no cover - depends on local codecs
        return None, str(error)
    return video_path, None


def _save_trace_plot(trace_rows: list[dict[str, Any]], path: Path, title: str) -> bool:
    if not trace_rows:
        return False
    plt = _maybe_import_pyplot()
    if plt is None:
        return False

    steps = [int(row['step']) for row in trace_rows]
    speeds = [float(row['speed']) for row in trace_rows]
    costs = [float(row['constraint_cost']) for row in trace_rows]
    gates = [int(row['gate']) for row in trace_rows]
    rewards = [float(row['reward']) for row in trace_rows]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(steps, speeds, color='tab:blue')
    axes[0].set_ylabel('speed')
    axes[0].grid(alpha=0.3)

    axes[1].step(steps, costs, where='post', color='tab:red')
    axes[1].set_ylabel('cost')
    axes[1].grid(alpha=0.3)

    axes[2].step(steps, gates, where='post', color='tab:green')
    axes[2].set_ylabel('gate')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['rec', 'nom'])
    axes[2].set_ylim(-0.15, 1.15)
    axes[2].grid(alpha=0.3)

    axes[3].plot(steps, rewards, color='tab:purple')
    axes[3].set_ylabel('reward')
    axes[3].set_xlabel('step')
    axes[3].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _rollout_base_policy(
    env_id: str,
    label: str,
    episode: int,
    seed: int,
    action_fn,
) -> dict[str, float]:
    base_env = _make_base_velocity_env(env_id)
    base_obs, _info = base_env.reset(seed=seed)
    obs = np.asarray(base_obs, dtype=np.float32)

    ep_reward = 0.0
    ep_cost = 0.0
    steps = 0
    nominal_steps = 0
    terminated = False
    truncated = False
    speeds: list[float] = []
    unhealthy_steps = 0
    violation_steps = 0

    while not (terminated or truncated):
        env_action, gate = action_fn(obs)
        next_obs, reward, cost, terminated, truncated, info = base_env.step(env_action)
        obs = np.asarray(next_obs, dtype=np.float32)
        speed = _extract_speed(info)
        healthy = _extract_health_flag(base_env, bool(terminated), info)

        ep_reward += float(reward)
        ep_cost += float(cost)
        steps += 1
        nominal_steps += int(gate)
        speeds.append(speed)
        unhealthy_steps += int(not healthy)
        violation_steps += int(float(cost) > 0.0)

    base_env.close()

    return {
        'label': label,
        'episode': float(episode),
        'seed': float(seed),
        'base_episode_reward': ep_reward,
        'base_episode_cost': ep_cost,
        'base_episode_length': float(steps),
        'nominal_fraction': float(nominal_steps / steps) if steps > 0 else 0.0,
        'mean_speed': float(np.mean(speeds)) if speeds else 0.0,
        'max_speed': float(np.max(speeds)) if speeds else 0.0,
        'violation_steps': float(violation_steps),
        'unhealthy_steps': float(unhealthy_steps),
        'terminated': float(int(bool(terminated))),
        'truncated': float(int(bool(truncated))),
    }


def _rollout_base_composite_policy(
    env_id: str,
    nominal_actor: SavedOmniSafeActor,
    recovery_actor: SavedOmniSafeActor,
    switch_actor: SavedOmniSafeActor,
    episode: int,
    seed: int,
) -> dict[str, float]:
    def _composite_action(obs: np.ndarray) -> tuple[np.ndarray, int]:
        gate_action = switch_actor.action(obs)
        gate_value = float(np.clip(np.asarray(gate_action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        gate = int(gate_value > 0.5)
        env_action = nominal_actor.action(obs) if gate == 1 else recovery_actor.action(obs)
        return env_action, gate

    return _rollout_base_policy(
        env_id=env_id,
        label='composite',
        episode=episode,
        seed=seed,
        action_fn=_composite_action,
    )


def _rollout_nominal_base_policy(
    env_id: str,
    nominal_actor: SavedOmniSafeActor,
    episode: int,
    seed: int,
) -> dict[str, float]:
    return _rollout_base_policy(
        env_id=env_id,
        label='nominal',
        episode=episode,
        seed=seed,
        action_fn=lambda obs: (nominal_actor.action(obs), 1),
    )


def _save_benchmark_curve_plot(
    nominal_eval_rows: list[dict[str, float]],
    composite_rows: list[dict[str, float]],
    path: Path,
    title: str,
    stage_boundaries: tuple[float, float],
    cost_limit: float | None = None,
) -> bool:
    if not nominal_eval_rows and not composite_rows:
        return False

    plt = _maybe_import_pyplot()
    if plt is None:
        return False

    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    fig.suptitle(title)

    nominal_x = [row['cumulative_env_steps'] for row in nominal_eval_rows]
    nominal_reward = [row['avg_base_episode_reward'] for row in nominal_eval_rows]
    nominal_reward_std = [row['std_base_episode_reward'] for row in nominal_eval_rows]
    nominal_cost = [row['avg_base_episode_cost'] for row in nominal_eval_rows]
    nominal_cost_std = [row['std_base_episode_cost'] for row in nominal_eval_rows]

    composite_x = [row['cumulative_env_steps'] for row in composite_rows]
    composite_reward = [row['avg_base_episode_reward'] for row in composite_rows]
    composite_reward_std = [row['std_base_episode_reward'] for row in composite_rows]
    composite_cost = [row['avg_base_episode_cost'] for row in composite_rows]
    composite_cost_std = [row['std_base_episode_cost'] for row in composite_rows]

    if nominal_eval_rows:
        axes[0].plot(
            nominal_x,
            nominal_reward,
            color='tab:blue',
            linewidth=1.8,
            label='Nominal PPO Eval',
        )
        axes[0].fill_between(
            nominal_x,
            np.asarray(nominal_reward) - np.asarray(nominal_reward_std),
            np.asarray(nominal_reward) + np.asarray(nominal_reward_std),
            color='tab:blue',
            alpha=0.15,
        )
        axes[1].plot(
            nominal_x,
            nominal_cost,
            color='tab:blue',
            linewidth=1.8,
            label='Nominal PPO Eval',
        )
        axes[1].fill_between(
            nominal_x,
            np.asarray(nominal_cost) - np.asarray(nominal_cost_std),
            np.asarray(nominal_cost) + np.asarray(nominal_cost_std),
            color='tab:blue',
            alpha=0.15,
        )

    if composite_rows:
        axes[0].plot(
            composite_x,
            composite_reward,
            color='tab:orange',
            linewidth=2.0,
            marker='o',
            label='Composite Base Eval',
        )
        axes[0].fill_between(
            composite_x,
            np.asarray(composite_reward) - np.asarray(composite_reward_std),
            np.asarray(composite_reward) + np.asarray(composite_reward_std),
            color='tab:orange',
            alpha=0.15,
        )
        axes[1].plot(
            composite_x,
            composite_cost,
            color='tab:orange',
            linewidth=2.0,
            marker='o',
            label='Composite Base Eval',
        )
        axes[1].fill_between(
            composite_x,
            np.asarray(composite_cost) - np.asarray(composite_cost_std),
            np.asarray(composite_cost) + np.asarray(composite_cost_std),
            color='tab:orange',
            alpha=0.15,
        )

    for boundary in stage_boundaries:
        for ax in axes:
            ax.axvline(boundary, color='0.5', linestyle='--', linewidth=1.0, alpha=0.8)

    axes[0].set_title('Episode Reward')
    axes[0].set_ylabel('Return')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title('Episode Cost')
    axes[1].set_ylabel('Cost')
    axes[1].grid(alpha=0.3)
    if cost_limit is not None:
        axes[1].axhline(
            y=cost_limit,
            color='black',
            linestyle='--',
            linewidth=1.2,
            label=f'Cost Limit ({cost_limit:.0f})',
        )
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel('Cumulative Environment Steps')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f'{x/1e6:.1f}M'))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def _rollout_switch_policy(
    env: VelocitySwitchEnv,
    label: str,
    episode: int,
    seed: int,
    action_fn,
    capture_video: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray], list[str]]:
    obs_t, _info = env.reset(seed=seed)
    obs_np = obs_t.detach().cpu().numpy()

    trace_rows: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    video_errors: list[str] = []
    ep_return = 0.0
    ep_cost = 0.0
    terminated = False
    truncated = False

    if capture_video:
        frame, render_error = _capture_render_frame(env)
        if frame is not None:
            video_frames.append(frame)
        elif render_error is not None:
            video_errors.append(render_error)

    while not (terminated or truncated):
        gate_action = _clip_to_action_space(action_fn(obs_np), env.action_space)
        next_obs_t, reward_t, cost_t, term_t, trunc_t, info = env.step(
            torch.as_tensor(gate_action, dtype=torch.float32),
        )

        reward = float(reward_t.item())
        cost = float(cost_t.item())
        terminated = bool(term_t.item())
        truncated = bool(trunc_t.item())
        ep_return += reward
        ep_cost += cost

        trace_rows.append(
            {
                'label': label,
                'episode': episode,
                'seed': seed,
                'step': len(trace_rows),
                'gate': int(info['gate']),
                'gate_value': float(info['gate_value']),
                'speed': float(info['speed']),
                'constraint_cost': float(info['constraint_cost']),
                'healthy': int(info['healthy']),
                'reward': reward,
                'terminated': int(terminated),
                'truncated': int(truncated),
            },
        )

        obs_np = next_obs_t.detach().cpu().numpy()
        if capture_video:
            frame, render_error = _capture_render_frame(env)
            if frame is not None:
                video_frames.append(frame)
            elif render_error is not None:
                video_errors.append(render_error)

    nominal_fraction = float(np.mean([row['gate'] for row in trace_rows])) if trace_rows else 0.0
    mean_speed = float(np.mean([row['speed'] for row in trace_rows])) if trace_rows else 0.0
    max_speed = float(np.max([row['speed'] for row in trace_rows])) if trace_rows else 0.0
    violation_steps = int(sum(row['constraint_cost'] > 0.0 for row in trace_rows))
    unhealthy_steps = int(sum(row['healthy'] == 0 for row in trace_rows))

    episode_row = {
        'label': label,
        'episode': episode,
        'seed': seed,
        'return': ep_return,
        'episode_cost': ep_cost,
        'episode_length': len(trace_rows),
        'nominal_fraction': nominal_fraction,
        'recovery_fraction': 1.0 - nominal_fraction,
        'mean_speed': mean_speed,
        'max_speed': max_speed,
        'violation_steps': violation_steps,
        'unhealthy_steps': unhealthy_steps,
        'terminated': int(terminated),
        'truncated': int(truncated),
    }
    return episode_row, trace_rows, video_frames, video_errors


def _summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    labels = sorted({row['label'] for row in rows})
    for label in labels:
        label_rows = [row for row in rows if row['label'] == label]
        summary[label] = {
            'episodes': float(len(label_rows)),
            'avg_return': float(np.mean([row['return'] for row in label_rows])),
            'avg_episode_cost': float(np.mean([row['episode_cost'] for row in label_rows])),
            'avg_episode_length': float(np.mean([row['episode_length'] for row in label_rows])),
            'avg_nominal_fraction': float(np.mean([row['nominal_fraction'] for row in label_rows])),
            'avg_mean_speed': float(np.mean([row['mean_speed'] for row in label_rows])),
            'avg_max_speed': float(np.mean([row['max_speed'] for row in label_rows])),
            'avg_violation_steps': float(np.mean([row['violation_steps'] for row in label_rows])),
            'avg_unhealthy_steps': float(np.mean([row['unhealthy_steps'] for row in label_rows])),
        }
    return summary


def _evaluate_switch_seed(
    output_dir: Path,
    robot: str,
    seed: int,
    eval_episodes: int,
    trace_episodes: int,
    video_episodes: int,
    video_width: int,
    video_height: int,
    video_fps: int,
    video_camera_name: str,
) -> None:
    spec = ROBOT_SPECS[robot]
    nominal_run = _require_stage_run(output_dir, robot, 'nominal', seed)
    recovery_run = _require_stage_run(output_dir, robot, 'recovery', seed)
    switch_run = _require_stage_run(output_dir, robot, 'switch', seed)

    _set_runtime_policy_dir('nominal', robot, nominal_run)
    _set_runtime_policy_dir('recovery', robot, recovery_run)

    render_enabled = video_episodes > 0
    env = VelocitySwitchEnv(
        spec.switch_env_id,
        device=torch.device('cpu'),
        render_mode='rgb_array' if render_enabled else None,
        width=video_width,
        height=video_height,
        camera_name=video_camera_name,
    )
    switch_actor = SavedOmniSafeActor(
        run_dir=switch_run,
        obs_space=env.observation_space,
        action_space=_gate_action_space(),
        label=f'{spec.title} switch',
        required=True,
    )

    policy_fns = {
        'learned_switch': lambda obs: switch_actor.action(obs),
        'always_nominal': lambda _obs: np.array([1.0], dtype=np.float32),
        'always_recovery': lambda _obs: np.array([0.0], dtype=np.float32),
    }

    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    video_entries: list[dict[str, Any]] = []
    video_errors: list[str] = []
    for label, action_fn in policy_fns.items():
        for episode in range(eval_episodes):
            episode_seed = 10_000 + seed * 100 + episode
            capture_video = render_enabled and episode < min(video_episodes, eval_episodes)
            ep_row, ep_trace, video_frames, rollout_video_errors = _rollout_switch_policy(
                env=env,
                label=label,
                episode=episode,
                seed=episode_seed,
                action_fn=action_fn,
                capture_video=capture_video,
            )
            episode_rows.append(ep_row)
            trace_rows.extend(ep_trace)
            video_errors.extend(rollout_video_errors)
            if capture_video:
                saved_path, save_error = _save_rollout_video(
                    frames=video_frames,
                    video_dir=output_dir / robot / 'evaluation' / f'seed-{seed:03d}' / 'videos',
                    name_prefix=label,
                    episode_index=episode,
                    fps=video_fps,
                )
                if saved_path is not None:
                    video_entries.append(
                        {
                            'label': label,
                            'episode': episode,
                            'path': str(saved_path),
                            'frames': len(video_frames),
                            'fps': video_fps,
                            'camera_name': video_camera_name,
                            'width': video_width,
                            'height': video_height,
                        },
                    )
                elif save_error is not None:
                    video_errors.append(save_error)

    env.close()

    eval_dir = output_dir / robot / 'evaluation' / f'seed-{seed:03d}'
    _write_csv(eval_dir / 'episode_metrics.csv', episode_rows)
    _write_csv(eval_dir / 'step_traces.csv', trace_rows)

    plot_count = 0
    for label in policy_fns:
        for episode in range(min(trace_episodes, eval_episodes)):
            label_trace = [
                row for row in trace_rows if row['label'] == label and row['episode'] == episode
            ]
            saved = _save_trace_plot(
                label_trace,
                eval_dir / 'plots' / f'{label}_episode_{episode:03d}.png',
                title=f'{spec.title} | {label} | episode {episode}',
            )
            plot_count += int(saved)

    summary = {
        'robot': robot,
        'seed': seed,
        'nominal_run_dir': str(nominal_run),
        'recovery_run_dir': str(recovery_run),
        'switch_run_dir': str(switch_run),
        'switch_checkpoint': str(switch_actor.checkpoint_path),
        'eval_episodes': eval_episodes,
        'trace_episodes': min(trace_episodes, eval_episodes),
        'video_episodes': min(video_episodes, eval_episodes),
        'video_fps': video_fps,
        'video_camera_name': video_camera_name,
        'video_resolution': {'width': video_width, 'height': video_height},
        'videos_saved': len(video_entries),
        'videos': video_entries,
        'video_errors': sorted(set(video_errors)),
        'plots_saved': plot_count,
        'policies': _summarize_episode_rows(episode_rows),
    }
    with open(eval_dir / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    _log(f'  [eval] robot={robot} seed={seed}: wrote evaluation to {eval_dir}')
    for label, metrics in summary['policies'].items():
        _log(
            '    '
            f'{label:16s} ret={metrics["avg_return"]:8.2f} '
            f'cost={metrics["avg_episode_cost"]:6.2f} '
            f'nom={100.0 * metrics["avg_nominal_fraction"]:5.1f}% '
            f'viol_steps={metrics["avg_violation_steps"]:5.2f}',
        )
    if video_entries:
        _log(
            f'    saved {len(video_entries)} evaluation video(s) under {eval_dir / "videos"}',
        )
    elif render_enabled and video_errors:
        _log(
            '    video capture requested, but no videos were saved. '
            f'First error: {video_errors[0]}',
        )


def _evaluate_base_task_benchmark_curves_seed(
    output_dir: Path,
    robot: str,
    seed: int,
    eval_episodes: int,
) -> None:
    spec = ROBOT_SPECS[robot]
    nominal_run = _require_stage_run(output_dir, robot, 'nominal', seed)
    recovery_run = _require_stage_run(output_dir, robot, 'recovery', seed)
    switch_run = _require_stage_run(output_dir, robot, 'switch', seed)

    nominal_cfg = _read_json(nominal_run / 'config.json')
    recovery_cfg = _read_json(recovery_run / 'config.json')
    switch_cfg = _read_json(switch_run / 'config.json')

    nominal_total_steps = int(nominal_cfg['train_cfgs']['total_steps'])
    recovery_total_steps = int(recovery_cfg['train_cfgs']['total_steps'])
    switch_total_steps = int(switch_cfg['train_cfgs']['total_steps'])

    nominal_curve_rows = _extract_training_curve_rows(nominal_run)
    nominal_progress_rows = _read_progress_csv(nominal_run / 'progress.csv')
    nominal_checkpoints = _list_checkpoints(nominal_run)
    switch_progress_rows = _read_progress_csv(switch_run / 'progress.csv')
    switch_checkpoints = _list_checkpoints(switch_run)
    reference_env = _make_base_velocity_env(spec.nominal_env_id)
    base_obs_space = reference_env.observation_space
    base_action_space = reference_env.action_space
    reference_env.close()
    base_eval_env_id = spec.nominal_env_id

    recovery_actor = SavedOmniSafeActor(
        run_dir=recovery_run,
        obs_space=base_obs_space,
        action_space=base_action_space,
        label=f'{spec.title} recovery',
        required=True,
    )

    nominal_eval_rows: list[dict[str, float]] = []
    for checkpoint_path in nominal_checkpoints:
        nominal_actor = SavedOmniSafeActor(
            run_dir=nominal_run,
            obs_space=base_obs_space,
            action_space=base_action_space,
            label=f'{spec.title} nominal',
            required=True,
            checkpoint_path=checkpoint_path,
        )

        episode_rows: list[dict[str, float]] = []
        for episode in range(eval_episodes):
            episode_seed = 15_000 + seed * 100 + episode
            episode_rows.append(
                _rollout_nominal_base_policy(
                    env_id=base_eval_env_id,
                    nominal_actor=nominal_actor,
                    episode=episode,
                    seed=episode_seed,
                ),
            )

        nominal_steps = _checkpoint_total_steps(
            checkpoint_path=checkpoint_path,
            progress_rows=nominal_progress_rows,
            total_steps=nominal_total_steps,
        )
        nominal_eval_rows.append(
            {
                'checkpoint': float(_checkpoint_sort_key(checkpoint_path)),
                'stage_env_steps': nominal_steps,
                'cumulative_env_steps': nominal_steps,
                'avg_base_episode_reward': float(
                    np.mean([row['base_episode_reward'] for row in episode_rows]),
                ),
                'std_base_episode_reward': float(
                    np.std([row['base_episode_reward'] for row in episode_rows]),
                ),
                'avg_base_episode_cost': float(
                    np.mean([row['base_episode_cost'] for row in episode_rows]),
                ),
                'std_base_episode_cost': float(
                    np.std([row['base_episode_cost'] for row in episode_rows]),
                ),
                'avg_base_episode_length': float(
                    np.mean([row['base_episode_length'] for row in episode_rows]),
                ),
                'avg_nominal_fraction': 1.0,
                'avg_mean_speed': float(np.mean([row['mean_speed'] for row in episode_rows])),
                'avg_max_speed': float(np.mean([row['max_speed'] for row in episode_rows])),
                'avg_violation_steps': float(
                    np.mean([row['violation_steps'] for row in episode_rows]),
                ),
                'avg_unhealthy_steps': float(
                    np.mean([row['unhealthy_steps'] for row in episode_rows]),
                ),
                'eval_episodes': float(eval_episodes),
            },
        )

    nominal_actor = SavedOmniSafeActor(
        run_dir=nominal_run,
        obs_space=base_obs_space,
        action_space=base_action_space,
        label=f'{spec.title} nominal',
        required=True,
    )
    composite_curve_rows: list[dict[str, float]] = []
    for checkpoint_path in switch_checkpoints:
        switch_actor = SavedOmniSafeActor(
            run_dir=switch_run,
            obs_space=base_obs_space,
            action_space=_gate_action_space(),
            label=f'{spec.title} switch',
            required=True,
            checkpoint_path=checkpoint_path,
        )

        episode_rows: list[dict[str, float]] = []
        for episode in range(eval_episodes):
            episode_seed = 20_000 + seed * 100 + episode
            episode_rows.append(
                _rollout_base_composite_policy(
                    env_id=base_eval_env_id,
                    nominal_actor=nominal_actor,
                    recovery_actor=recovery_actor,
                    switch_actor=switch_actor,
                    episode=episode,
                    seed=episode_seed,
                ),
            )

        switch_steps = _checkpoint_total_steps(
            checkpoint_path=checkpoint_path,
            progress_rows=switch_progress_rows,
            total_steps=switch_total_steps,
        )
        composite_curve_rows.append(
            {
                'checkpoint': float(_checkpoint_sort_key(checkpoint_path)),
                'switch_stage_env_steps': switch_steps,
                'cumulative_env_steps': float(nominal_total_steps + recovery_total_steps) + switch_steps,
                'avg_base_episode_reward': float(
                    np.mean([row['base_episode_reward'] for row in episode_rows]),
                ),
                'std_base_episode_reward': float(
                    np.std([row['base_episode_reward'] for row in episode_rows]),
                ),
                'avg_base_episode_cost': float(
                    np.mean([row['base_episode_cost'] for row in episode_rows]),
                ),
                'std_base_episode_cost': float(
                    np.std([row['base_episode_cost'] for row in episode_rows]),
                ),
                'avg_base_episode_length': float(
                    np.mean([row['base_episode_length'] for row in episode_rows]),
                ),
                'avg_nominal_fraction': float(
                    np.mean([row['nominal_fraction'] for row in episode_rows]),
                ),
                'avg_mean_speed': float(np.mean([row['mean_speed'] for row in episode_rows])),
                'avg_max_speed': float(np.mean([row['max_speed'] for row in episode_rows])),
                'avg_violation_steps': float(
                    np.mean([row['violation_steps'] for row in episode_rows]),
                ),
                'avg_unhealthy_steps': float(
                    np.mean([row['unhealthy_steps'] for row in episode_rows]),
                ),
                'eval_episodes': float(eval_episodes),
            },
        )

    eval_dir = output_dir / robot / 'evaluation' / f'seed-{seed:03d}'
    nominal_curve_csv_rows = [
        {
            'total_env_steps': row['total_env_steps'],
            'episode_reward': row['episode_reward'],
            'episode_cost': row['episode_cost'],
            'episode_length': row['episode_length'],
        }
        for row in nominal_curve_rows
    ]
    _write_csv(eval_dir / 'base_task_nominal_train_curve.csv', nominal_curve_csv_rows)
    nominal_eval_csv_rows = [
        {
            'checkpoint': int(row['checkpoint']),
            'stage_env_steps': row['stage_env_steps'],
            'cumulative_env_steps': row['cumulative_env_steps'],
            'avg_base_episode_reward': row['avg_base_episode_reward'],
            'std_base_episode_reward': row['std_base_episode_reward'],
            'avg_base_episode_cost': row['avg_base_episode_cost'],
            'std_base_episode_cost': row['std_base_episode_cost'],
            'avg_base_episode_length': row['avg_base_episode_length'],
            'avg_nominal_fraction': row['avg_nominal_fraction'],
            'avg_mean_speed': row['avg_mean_speed'],
            'avg_max_speed': row['avg_max_speed'],
            'avg_violation_steps': row['avg_violation_steps'],
            'avg_unhealthy_steps': row['avg_unhealthy_steps'],
            'eval_episodes': int(row['eval_episodes']),
        }
        for row in nominal_eval_rows
    ]
    _write_csv(eval_dir / 'base_task_nominal_eval_curve.csv', nominal_eval_csv_rows)

    composite_curve_csv_rows = [
        {
            'checkpoint': int(row['checkpoint']),
            'switch_stage_env_steps': row['switch_stage_env_steps'],
            'cumulative_env_steps': row['cumulative_env_steps'],
            'avg_base_episode_reward': row['avg_base_episode_reward'],
            'std_base_episode_reward': row['std_base_episode_reward'],
            'avg_base_episode_cost': row['avg_base_episode_cost'],
            'std_base_episode_cost': row['std_base_episode_cost'],
            'avg_base_episode_length': row['avg_base_episode_length'],
            'avg_nominal_fraction': row['avg_nominal_fraction'],
            'avg_mean_speed': row['avg_mean_speed'],
            'avg_max_speed': row['avg_max_speed'],
            'avg_violation_steps': row['avg_violation_steps'],
            'avg_unhealthy_steps': row['avg_unhealthy_steps'],
            'eval_episodes': int(row['eval_episodes']),
        }
        for row in composite_curve_rows
    ]
    _write_csv(eval_dir / 'base_task_composite_curve.csv', composite_curve_csv_rows)

    plot_saved = _save_benchmark_curve_plot(
        nominal_eval_rows=nominal_eval_rows,
        composite_rows=composite_curve_rows,
        path=eval_dir / 'base_task_reward_cost_vs_steps.png',
        title=f'{spec.title} | Base-Task Reward/Cost vs Steps',
        stage_boundaries=(
            float(nominal_total_steps),
            float(nominal_total_steps + recovery_total_steps),
        ),
        cost_limit=DEFAULT_BASELINE_COST_LIMIT,
    )

    summary = {
        'robot': robot,
        'seed': seed,
        'base_env_id': base_eval_env_id,
        'cost_limit': DEFAULT_BASELINE_COST_LIMIT,
        'nominal_run_dir': str(nominal_run),
        'recovery_run_dir': str(recovery_run),
        'switch_run_dir': str(switch_run),
        'eval_episodes': eval_episodes,
        'nominal_total_steps': nominal_total_steps,
        'recovery_total_steps': recovery_total_steps,
        'switch_total_steps': switch_total_steps,
        'stage_boundaries': {
            'nominal_end': nominal_total_steps,
            'recovery_end': nominal_total_steps + recovery_total_steps,
            'switch_end': nominal_total_steps + recovery_total_steps + switch_total_steps,
        },
        'checkpoints_evaluated': [path.name for path in switch_checkpoints],
        'plot_saved': int(plot_saved),
    }
    with open(eval_dir / 'base_task_curve_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    _log(
        f'  [base-curve] robot={robot} seed={seed}: wrote benchmark-style curves to {eval_dir}',
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Three-stage velocity scenario: nominal, recovery, and switch PPO training.',
    )
    parser.add_argument(
        '--robots',
        nargs='*',
        default=DEFAULT_ROBOTS,
        choices=DEFAULT_ROBOTS,
        help='Robots to train.',
    )
    parser.add_argument(
        '--stages',
        nargs='*',
        default=DEFAULT_STAGES,
        choices=DEFAULT_STAGES,
        help='Scenario stages to execute. They still run in nominal -> recovery -> switch order.',
    )
    parser.add_argument('--seeds', nargs='*', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--force', action='store_true', help='Delete requested stage outputs first.')
    parser.add_argument(
        '--nominal-total-steps',
        type=int,
        default=DEFAULT_TOTAL_STEPS['nominal'],
    )
    parser.add_argument(
        '--recovery-total-steps',
        type=int,
        default=DEFAULT_TOTAL_STEPS['recovery'],
    )
    parser.add_argument(
        '--switch-total-steps',
        type=int,
        default=DEFAULT_TOTAL_STEPS['switch'],
    )
    parser.add_argument(
        '--nominal-steps-per-epoch',
        type=int,
        default=DEFAULT_STEPS_PER_EPOCH['nominal'],
    )
    parser.add_argument(
        '--recovery-steps-per-epoch',
        type=int,
        default=DEFAULT_STEPS_PER_EPOCH['recovery'],
    )
    parser.add_argument(
        '--switch-steps-per-epoch',
        type=int,
        default=DEFAULT_STEPS_PER_EPOCH['switch'],
    )
    parser.add_argument(
        '--evaluate-switch',
        action='store_true',
        help='Run switch-policy evaluation and visualization. This also happens automatically '
             'after switch training in the current run.',
    )
    parser.add_argument('--eval-episodes', type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument('--trace-episodes', type=int, default=DEFAULT_TRACE_EPISODES)
    parser.add_argument(
        '--video-episodes',
        type=int,
        default=DEFAULT_VIDEO_EPISODES,
        help='Number of evaluation episodes per policy to save as rendered videos. Set 0 to disable.',
    )
    parser.add_argument('--video-width', type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument('--video-height', type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument('--video-fps', type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        '--video-camera-name',
        type=str,
        default=DEFAULT_VIDEO_CAMERA_NAME,
        help='Safety-Gymnasium camera used for evaluation videos.',
    )
    parser.add_argument(
        '--base-task-eval-episodes',
        type=int,
        default=DEFAULT_BASE_TASK_EVAL_EPISODES,
        help='Number of episodes per saved switch checkpoint when building benchmark-style '
             'base-task reward/cost curves.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    requested_stages = set(args.stages)
    collected_runs: dict[tuple[str, str, int], Path] = {}

    _log('=' * 72)
    _log('Final Velocity Scenario')
    _log(f'Robots         : {args.robots}')
    _log(f'Stages         : {args.stages}')
    _log(f'Seeds          : {args.seeds}')
    _log(f'Output dir     : {output_dir}')
    _log(f'Device         : {device}')
    _log('Switch action  : 1-D Box thresholded at 0.5 -> {0: recovery, 1: nominal}')
    _log('=' * 72)

    for robot in args.robots:
        spec = ROBOT_SPECS[robot]
        _log(f'Robot: {spec.title}')

        for stage_name in ('nominal', 'recovery', 'switch'):
            if stage_name in requested_stages:
                _remove_stage_dir_if_requested(output_dir, robot, stage_name, args.force)

        if 'nominal' in requested_stages:
            for seed in args.seeds:
                run_dir = _train_stage_if_needed(
                    stage_name='nominal',
                    robot=robot,
                    env_id=spec.nominal_env_id,
                    seed=seed,
                    total_steps=args.nominal_total_steps,
                    steps_per_epoch=args.nominal_steps_per_epoch,
                    output_dir=output_dir,
                    device=device,
                )
                collected_runs[(robot, 'nominal', seed)] = run_dir

        if 'recovery' in requested_stages:
            for seed in args.seeds:
                nominal_run = collected_runs.get((robot, 'nominal', seed))
                if nominal_run is None:
                    nominal_run = _require_stage_run(output_dir, robot, 'nominal', seed)
                _set_runtime_policy_dir('nominal', robot, nominal_run)

                run_dir = _train_stage_if_needed(
                    stage_name='recovery',
                    robot=robot,
                    env_id=spec.recovery_env_id,
                    seed=seed,
                    total_steps=args.recovery_total_steps,
                    steps_per_epoch=args.recovery_steps_per_epoch,
                    output_dir=output_dir,
                    device=device,
                )
                collected_runs[(robot, 'recovery', seed)] = run_dir

        if 'switch' in requested_stages:
            for seed in args.seeds:
                nominal_run = collected_runs.get((robot, 'nominal', seed))
                if nominal_run is None:
                    nominal_run = _require_stage_run(output_dir, robot, 'nominal', seed)

                recovery_run = collected_runs.get((robot, 'recovery', seed))
                if recovery_run is None:
                    recovery_run = _require_stage_run(output_dir, robot, 'recovery', seed)

                _set_runtime_policy_dir('nominal', robot, nominal_run)
                _set_runtime_policy_dir('recovery', robot, recovery_run)

                run_dir = _train_stage_if_needed(
                    stage_name='switch',
                    robot=robot,
                    env_id=spec.switch_env_id,
                    seed=seed,
                    total_steps=args.switch_total_steps,
                    steps_per_epoch=args.switch_steps_per_epoch,
                    output_dir=output_dir,
                    device=device,
                )
                collected_runs[(robot, 'switch', seed)] = run_dir

        _log('-' * 72)

    _log('Run summary:')
    for (robot, stage, seed), run_dir in sorted(collected_runs.items()):
        _log(f'  robot={robot:12s} stage={stage:8s} seed={seed:03d} -> {run_dir}')

    if 'switch' in requested_stages or args.evaluate_switch:
        _log('Running switch evaluation + visualization...')
        for robot in args.robots:
            for seed in args.seeds:
                try:
                    _evaluate_switch_seed(
                        output_dir=output_dir,
                        robot=robot,
                        seed=seed,
                        eval_episodes=args.eval_episodes,
                        trace_episodes=args.trace_episodes,
                        video_episodes=max(0, args.video_episodes),
                        video_width=args.video_width,
                        video_height=args.video_height,
                        video_fps=args.video_fps,
                        video_camera_name=args.video_camera_name,
                    )
                    _evaluate_base_task_benchmark_curves_seed(
                        output_dir=output_dir,
                        robot=robot,
                        seed=seed,
                        eval_episodes=args.base_task_eval_episodes,
                    )
                except FileNotFoundError as error:
                    _log(f'  [eval] skipping robot={robot} seed={seed}: {error}')


if __name__ == '__main__':
    main()
