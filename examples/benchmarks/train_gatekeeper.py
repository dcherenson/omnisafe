# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Stage-2: switching-filter gatekeeper-RL training + ablation + baselines.

Trains unconstrained PPO on ``SCG-Quadrotor-Gatekeeper-v0`` with a binary
per-decision action (nominal vs backup), a FROZEN learned backup policy from
Stage-1, and the gatekeeper forward-sim filter.

Filter + shaping ablations (controlled by GK_USE_FILTER / GK_USE_SHAPING):

  Variant         use_filter  use_shaping (override penalty)
  --------------  ----------  ------------------------------
  proposed         True        True     (primary: filter + shaping, our full method)
  filter_only      True        False    (filter only, no override penalty)
  reward_only      False       True     (no filter; shaping never triggers, so
                                         this is equivalent to ``nominal``)
  nominal          False       False    (neither; the policy is unprotected)

Non-learned baselines:
  b1  always-nominal            (a == 1 every step; filter will still intercept
                                 when --baseline-filter)
  b2  always-backup              (a == 0 every step)
  b3  random                     (uniform binary)

USAGE
=====
  # Prereq: train the backup first
  python examples/benchmarks/train_backup.py

  # Then train all four variants + run baselines + eval:
  python examples/benchmarks/train_gatekeeper.py \
      --backup-run-dir examples/benchmarks/exp-backup/PPO_backup/SCG-Quadrotor-Backup-v0/seed-000-XXXX

  # Or let the script auto-pick the latest backup run:
  python examples/benchmarks/train_gatekeeper.py

  # Only the primary variant:
  python examples/benchmarks/train_gatekeeper.py --variants proposed

  # Skip training, only baselines + eval of existing checkpoints:
  python examples/benchmarks/train_gatekeeper.py --eval-only
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

import omnisafe.envs.scg_gatekeeper_env  # noqa: F401 — registers env

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.common.normalizer import Normalizer
from omnisafe.envs.core import make as omnisafe_make
from omnisafe.envs.scg_gatekeeper_env import (
    SCGQuadrotorGatekeeperEnv,
    _find_backup_run_dir,
)
from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config import Config
from omnisafe.utils.exp_grid_tools import train


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_ID = 'SCG-Quadrotor-Gatekeeper-v0'
ALGO = 'PPO'

SEEDS = [0, 5]

# Hyper-params for the gatekeeper env
DEFAULT_H = 60
DEFAULT_DELTA = 5
DEFAULT_R_NOM = 1.0
DEFAULT_R_OVERRIDE = 0.2
DEFAULT_R_CRASH = 20.0

STEPS_PER_EPOCH = 192  # ~2 episodes (~96 decisions/ep) per epoch
TOTAL_STEPS = 9_600    # ~50 episodes per seed per variant

VARIANTS: dict[str, dict[str, Any]] = {
    'proposed':    {'use_filter': True,  'use_shaping': True},
    'filter_only': {'use_filter': True,  'use_shaping': False},
    'reward_only': {'use_filter': False, 'use_shaping': True},
    'nominal':     {'use_filter': False, 'use_shaping': False},
}

_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'exp-gk',
    'gatekeeper',
)
_DEFAULT_BACKUP_SEARCH_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'exp-backup',
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Env variable handling for variants
# ---------------------------------------------------------------------------

def _set_variant_env(
    variant: dict[str, Any],
    H: int,
    delta: int,
    r_nom: float,
    r_override: float,
    r_crash: float,
    backup_run_dir: str | None,
) -> None:
    os.environ['GK_USE_FILTER'] = str(variant['use_filter']).lower()
    os.environ['GK_USE_SHAPING'] = str(variant['use_shaping']).lower()
    os.environ['GK_H'] = str(H)
    os.environ['GK_DELTA'] = str(delta)
    os.environ['GK_R_NOM'] = str(r_nom)
    os.environ['GK_R_OVERRIDE'] = str(r_override)
    os.environ['GK_R_CRASH'] = str(r_crash)
    if backup_run_dir:
        os.environ['GK_BACKUP_RUN_DIR'] = str(backup_run_dir)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_ppo_grid(variant_name: str) -> ExperimentGrid:
    eg = ExperimentGrid(exp_name=f'{ALGO}_{variant_name}')
    eg.add('algo', [ALGO])
    eg.add('env_id', [ENV_ID])
    eg.add('seed', SEEDS)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('algo_cfgs:steps_per_epoch', [STEPS_PER_EPOCH])
    eg.add('train_cfgs:total_steps', [TOTAL_STEPS])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('train_cfgs:device', ['cpu'])
    eg.add('algo_cfgs:update_iters', [10])
    eg.add('model_cfgs:actor:lr', [3e-4])
    eg.add('model_cfgs:critic:lr', [3e-4])
    return eg


def _train_variant(
    variant_name: str,
    output_dir: str,
    args: argparse.Namespace,
    gpu_id: list[int] | None,
) -> None:
    parent_dir = os.path.join(output_dir, variant_name)
    if _has_results(parent_dir, f'{ALGO}_{variant_name}') or _has_results(parent_dir, ALGO):
        _log(f'  [{variant_name}] already trained — skipping.')
        return
    algo_dir = os.path.join(parent_dir, f'{ALGO}_{variant_name}')
    if os.path.exists(algo_dir):
        _log(f'  [{variant_name}] stale output — removing.')
        shutil.rmtree(algo_dir)

    _set_variant_env(
        variant=VARIANTS[variant_name],
        H=args.H,
        delta=args.delta,
        r_nom=args.r_nom,
        r_override=args.r_override,
        r_crash=args.r_crash,
        backup_run_dir=args.backup_run_dir,
    )
    os.makedirs(parent_dir, exist_ok=True)
    _log(f'  [{variant_name}] Training (use_filter={VARIANTS[variant_name]["use_filter"]}, '
         f'use_shaping={VARIANTS[variant_name]["use_shaping"]})...')
    eg = _make_ppo_grid(variant_name)
    eg.run(train, num_pool=len(SEEDS), gpu_id=gpu_id, parent_dir=parent_dir)


# ---------------------------------------------------------------------------
# Evaluation: baselines + trained policies
# ---------------------------------------------------------------------------

def _find_trained_run(parent_dir: str, variant_name: str, seed: int) -> str | None:
    seed_tag = f'seed-{seed:03d}'
    for root, _dirs, files in os.walk(parent_dir):
        if 'config.json' in files and os.path.isdir(os.path.join(root, 'torch_save')):
            if seed_tag in root:
                return root
    return None


def _load_gk_actor(run_dir: str, obs_dim: int, act_dim: int, device: str = 'cpu'):
    from gymnasium import spaces as gym_spaces

    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        cfg = Config.dict2config(json.load(f))
    model_cfgs = cfg.model_cfgs

    save_dir = os.path.join(run_dir, 'torch_save')
    ckpts = glob.glob(os.path.join(save_dir, 'epoch-*.pt'))
    ckpt_path = max(
        ckpts,
        key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0]),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    obs_space = gym_spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym_spaces.Box(0.0, 1.0, shape=(act_dim,), dtype=np.float32)
    actor = ActorBuilder(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=model_cfgs.actor.hidden_sizes,
        activation=model_cfgs.actor.activation,
        weight_initialization_mode=model_cfgs.weight_initialization_mode,
    ).build_actor(model_cfgs.actor_type)
    actor.load_state_dict(ckpt['pi'])
    actor.eval()

    obs_norm = None
    if ckpt.get('obs_normalizer') is not None:
        obs_norm = Normalizer(shape=(obs_dim,), clip=10.0)
        obs_norm.load_state_dict(ckpt['obs_normalizer'])
    return actor, obs_norm, ckpt_path


def _actor_action(actor, obs_norm, obs_np: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    if obs_norm is not None:
        std = torch.clamp(obs_norm.std, min=1e-2)
        obs_t = (obs_t - obs_norm.mean) / std
        obs_t = torch.clamp(obs_t, -10.0, 10.0)
    with torch.no_grad():
        action = actor.predict(obs_t, deterministic=True)
    return action.squeeze(0).numpy()


def _rollout_episode(
    env: SCGQuadrotorGatekeeperEnv,
    action_fn,
    max_decisions: int,
    seed: int,
) -> dict[str, Any]:
    obs_t, _info = env.reset(seed=seed)
    obs_np = obs_t.detach().cpu().numpy()

    total_return = 0.0
    n_violations = 0
    n_rejections = 0
    a_rl_list: list[int] = []
    committed_list: list[int] = []
    decision_times: list[float] = []
    terminated = False
    for _ in range(max_decisions):
        t0 = time.time()
        a = action_fn(obs_np, env)
        a = float(np.clip(a, 0.0, 1.0))
        dt = time.time() - t0

        a_tensor = torch.as_tensor([a], dtype=torch.float32)
        obs_t, rew, _cost, term, trunc, info = env.step(a_tensor)
        obs_np = obs_t.detach().cpu().numpy()

        total_return += float(rew)
        n_violations += int(info['n_violations'])
        n_rejections += int(info['rejected'])
        a_rl_list.append(int(info['a_rl']))
        committed_list.append(int(info['committed']))
        decision_times.append(dt)

        if bool(term):
            terminated = True
            break
        if bool(trunc):
            break
    n_dec = len(a_rl_list)
    nom_frac_rl = float(np.mean(a_rl_list)) if a_rl_list else 0.0
    nom_frac_exec = float(np.mean(committed_list)) if committed_list else 0.0
    return {
        'return': total_return,
        'n_violations': n_violations,
        'n_rejections': n_rejections,
        'n_decisions': n_dec,
        'nom_frac_rl': nom_frac_rl,
        'nom_frac_executed': nom_frac_exec,
        'terminated': terminated,
        'decision_ms_mean': float(np.mean(decision_times) * 1000) if decision_times else 0.0,
    }


# --- Baseline action functions ---

def _baseline_always_nominal():
    def fn(_obs, _env):
        return 1.0
    return fn


def _baseline_always_backup():
    def fn(_obs, _env):
        return 0.0
    return fn


def _baseline_random(rng: np.random.Generator):
    def fn(_obs, _env):
        return float(rng.random())
    return fn


def _trained_action(actor, obs_norm):
    def fn(obs, _env):
        a = _actor_action(actor, obs_norm, obs)
        return float(a[0])
    return fn


# --- Evaluation driver ---

def _eval_variant(
    label: str,
    args: argparse.Namespace,
    use_filter: bool,
    use_shaping: bool,
    action_fn_builder,
    n_episodes: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    _set_variant_env(
        variant={'use_filter': use_filter, 'use_shaping': use_shaping},
        H=args.H,
        delta=args.delta,
        r_nom=args.r_nom,
        r_override=args.r_override,
        r_crash=args.r_crash,
        backup_run_dir=args.backup_run_dir,
    )
    env = omnisafe_make(ENV_ID, device=torch.device('cpu'))
    action_fn = action_fn_builder(env)
    max_decisions = (env._T_total // env.delta) + 1  # noqa: SLF001

    results: list[dict[str, Any]] = []
    for ep_idx in range(n_episodes):
        seed = seeds[ep_idx % len(seeds)] + ep_idx
        m = _rollout_episode(env, action_fn, max_decisions, seed=seed)
        m['label'] = label
        m['use_filter'] = use_filter
        m['use_shaping'] = use_shaping
        m['episode'] = ep_idx
        m['seed'] = seed
        results.append(m)
    env.close()
    return results


def _run_all_evals(
    args: argparse.Namespace,
    output_dir: str,
    n_episodes: int,
) -> None:
    all_results: list[dict[str, Any]] = []
    eval_seeds = [100, 101, 102, 103, 104]

    # ---- Baselines with filter on ----
    for label, builder in [
        ('b1_always_nominal_filter',  lambda _env: _baseline_always_nominal()),
        ('b2_always_backup',          lambda _env: _baseline_always_backup()),
        ('b3_random_filter',          lambda _env: _baseline_random(np.random.default_rng(42))),
    ]:
        _log(f'  [eval] baseline {label}')
        res = _eval_variant(
            label=label,
            args=args,
            use_filter=True,
            use_shaping=False,
            action_fn_builder=builder,
            n_episodes=n_episodes,
            seeds=eval_seeds,
        )
        all_results.extend(res)

    # Always-nominal WITHOUT filter — stress test that the filter matters
    _log('  [eval] baseline b1b_always_nominal_nofilter (stress test)')
    res = _eval_variant(
        label='b1b_always_nominal_nofilter',
        args=args,
        use_filter=False,
        use_shaping=False,
        action_fn_builder=lambda _env: _baseline_always_nominal(),
        n_episodes=n_episodes,
        seeds=eval_seeds,
    )
    all_results.extend(res)

    # ---- Trained policies: deploy with filter on, and with filter off ----
    for variant_name in VARIANTS:
        parent_dir = os.path.join(output_dir, variant_name)
        for seed in SEEDS:
            run_dir = _find_trained_run(parent_dir, variant_name, seed)
            if run_dir is None:
                _log(f'  [eval] skipping {variant_name} seed={seed}: no trained run found')
                continue
            for deploy_filter in (True, False):
                suffix = 'filter' if deploy_filter else 'nofilter'
                label = f'{variant_name}_seed{seed}_deploy_{suffix}'
                _log(f'  [eval] trained {label}')

                def make_fn(run_dir=run_dir):
                    def builder(env: SCGQuadrotorGatekeeperEnv):
                        actor, obs_norm, _ = _load_gk_actor(
                            run_dir,
                            obs_dim=env.observation_space.shape[0],
                            act_dim=env.action_space.shape[0],
                        )
                        return _trained_action(actor, obs_norm)
                    return builder

                res = _eval_variant(
                    label=label,
                    args=args,
                    use_filter=deploy_filter,
                    use_shaping=False,
                    action_fn_builder=make_fn(),
                    n_episodes=n_episodes,
                    seeds=eval_seeds,
                )
                all_results.extend(res)

    _write_eval_results(all_results, output_dir)


def _write_eval_results(results: list[dict[str, Any]], output_dir: str) -> None:
    if not results:
        _log('  [eval] no results to write.')
        return
    csv_path = os.path.join(output_dir, 'eval_results.csv')
    keys = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    _log(f'  [eval] wrote {len(results)} rows to {csv_path}')

    summary: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        summary.setdefault(row['label'], []).append(row)

    print()
    print('=' * 108)
    print(f'{"label":42s} {"ret":>8s} {"viol":>5s} {"rej":>5s} '
          f'{"%rl_nom":>8s} {"%exec_nom":>10s} {"ep":>3s} {"ms":>6s}')
    print('-' * 108)
    for label, rows in summary.items():
        n = len(rows)
        ret = np.mean([r['return'] for r in rows])
        viol = np.mean([r['n_violations'] for r in rows])
        rej = np.mean([r['n_rejections'] for r in rows])
        rl_nom = 100.0 * np.mean([r['nom_frac_rl'] for r in rows])
        exec_nom = 100.0 * np.mean([r['nom_frac_executed'] for r in rows])
        ms = np.mean([r['decision_ms_mean'] for r in rows])
        print(f'{label:42s} {ret:8.2f} {viol:5.1f} {rej:5.1f} '
              f'{rl_nom:7.1f}% {exec_nom:9.1f}% {n:3d} {ms:6.2f}')
    print('=' * 108)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_backup_run_dir(cli_value: str | None) -> str | None:
    if cli_value:
        return str(Path(cli_value).expanduser().resolve())
    # Auto-discover under the default search root.
    found = _find_backup_run_dir(_DEFAULT_BACKUP_SEARCH_ROOT)
    if found is not None:
        return str(found)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='*', default=None)
    parser.add_argument('--output-dir', type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument('--variants', type=str, nargs='*', default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, only run baselines + evaluate existing checkpoints.')
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--backup-run-dir', type=str, default=None,
                        help='Path to a trained backup run dir. Defaults to the latest '
                             'under examples/benchmarks/exp-backup/.')
    parser.add_argument('--H', type=int, default=DEFAULT_H)
    parser.add_argument('--delta', type=int, default=DEFAULT_DELTA)
    parser.add_argument('--r-nom', type=float, default=DEFAULT_R_NOM)
    parser.add_argument('--r-override', type=float, default=DEFAULT_R_OVERRIDE)
    parser.add_argument('--r-crash', type=float, default=DEFAULT_R_CRASH)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    args.backup_run_dir = _resolve_backup_run_dir(args.backup_run_dir)

    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = args.gpu
    if gpu_id is not None and not set(gpu_id).issubset(set(available_gpus)):
        warnings.warn(f'GPU {gpu_id} not available, falling back to CPU.', stacklevel=1)
        gpu_id = None

    _log('=' * 72)
    _log('Stage-2: switching-filter gatekeeper-RL training + ablation + baselines')
    _log(f'Env            : {ENV_ID}')
    _log(f'Algo           : {ALGO}')
    _log(f'Seeds          : {SEEDS}')
    _log(f'Total steps    : {TOTAL_STEPS:,} per seed per variant')
    _log(f'H / delta      : {args.H} / {args.delta}')
    _log(f'r_nom/over/cr  : {args.r_nom} / {args.r_override} / {args.r_crash}')
    _log('Low-level      : safe-control-gym DSL PID (classical)')
    if args.backup_run_dir:
        _log(f'Backup run dir : {args.backup_run_dir}')
    else:
        _log('Backup run dir : NONE — using fixed-hover fallback. Run train_backup.py first.')
    _log(f'Output dir     : {args.output_dir}')
    _log(f'Variants       : {args.variants}')
    _log(f'Eval-only      : {args.eval_only}')
    _log('=' * 72)

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    if not args.eval_only:
        for variant_name in args.variants:
            _train_variant(variant_name, args.output_dir, args, gpu_id)
            print('-' * 72)
        _log(f'All training done in {datetime.timedelta(seconds=int(time.time() - t0))}.')

    _log('Running baselines + policy evaluations...')
    _run_all_evals(args, args.output_dir, args.eval_episodes)
    _log(f'All done in {datetime.timedelta(seconds=int(time.time() - t0))}.')
