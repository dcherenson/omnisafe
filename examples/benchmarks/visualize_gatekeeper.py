"""Visualize a trained gatekeeper-RL policy on the SCG Quadrotor circle task.

Loads a trained Stage-2 PPO checkpoint and runs one episode inside the
gatekeeper env with the PyBullet GUI on.  The GUI shows:

  * Blue circle  : full nominal X_GOAL reference
  * Red wall     : forbidden zone boundary  (x > 0.3 m)
  * Yellow line  : current reference being tracked by the PID this step
                   (from nominal when executed_mode=NOMINAL, from the backup
                   policy when executed_mode=BACKUP)
  * Green pulse  : RL picked NOMINAL and the filter allowed it
  * Red pulse    : RL picked NOMINAL but the filter OVERRODE to backup
  * Blue pulse   : RL picked BACKUP directly
  * HUD          : step, a_rl, committed, rejection rate, violations

USAGE
-----
  python examples/benchmarks/visualize_gatekeeper.py --variant proposed --seed 0
  python examples/benchmarks/visualize_gatekeeper.py --variant proposed --seed 0 --deploy-nofilter
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import omnisafe.envs.scg_gatekeeper_env  # noqa: F401

from omnisafe.common.normalizer import Normalizer
from omnisafe.envs.core import make as omnisafe_make
from omnisafe.envs.scg_gatekeeper_env import (
    SCGQuadrotorGatekeeperEnv,
    _find_backup_run_dir,
)
from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config import Config


_DEFAULT_GK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'exp-gk',
    'gatekeeper',
)
_DEFAULT_BACKUP_SEARCH_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'exp-backup',
)


def _find_run(parent_dir: str, variant: str, seed: int) -> str:
    seed_tag = f'seed-{seed:03d}'
    variant_root = os.path.join(parent_dir, variant)
    if not os.path.isdir(variant_root):
        raise FileNotFoundError(f'No trained dir {variant_root}')
    for root, _dirs, files in os.walk(variant_root):
        if 'config.json' in files and os.path.isdir(os.path.join(root, 'torch_save')):
            if seed_tag in root:
                return root
    raise FileNotFoundError(
        f'No trained run found for variant={variant} seed={seed} under {variant_root}',
    )


def _load_actor(run_dir: str, obs_dim: int, act_dim: int):
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
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

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


def _actor_action(actor, obs_norm, obs_np: np.ndarray) -> float:
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    if obs_norm is not None:
        std = torch.clamp(obs_norm.std, min=1e-2)
        obs_t = (obs_t - obs_norm.mean) / std
        obs_t = torch.clamp(obs_t, -10.0, 10.0)
    with torch.no_grad():
        a = actor.predict(obs_t, deterministic=True)
    return float(a.squeeze(0).numpy()[0])


def _draw_static(env: SCGQuadrotorGatekeeperEnv) -> dict:
    handles: dict = {}
    try:
        import pybullet as p
    except ImportError:
        return handles
    inner = env.inner_env
    if not getattr(inner, 'GUI', False):
        return handles
    client = inner.PYB_CLIENT

    p.resetDebugVisualizerCamera(
        cameraDistance=2.5, cameraPitch=-5, cameraYaw=0,
        cameraTargetPosition=[0.0, 0.0, 1.0],
        physicsClientId=client,
    )

    # Nominal trajectory (blue)
    traj = env.X_GOAL
    for i in range(len(traj) - 1):
        x1, z1 = float(traj[i, 0]), float(traj[i, 2])
        x2, z2 = float(traj[i + 1, 0]), float(traj[i + 1, 2])
        p.addUserDebugLine([x1, 0, z1], [x2, 0, z2],
                           lineColorRGB=[0.0, 0.55, 1.0],
                           lineWidth=2, physicsClientId=client)

    # Forbidden zone (x > 0.3)
    p.addUserDebugLine([0.3, 0, 0.2], [0.3, 0, 1.8],
                       lineColorRGB=[1.0, 0.0, 0.0], lineWidth=4,
                       physicsClientId=client)
    for xh in [0.5, 0.7, 0.9, 1.1]:
        p.addUserDebugLine([xh, 0, 0.2], [xh, 0, 1.8],
                           lineColorRGB=[1.0, 0.4, 0.4], lineWidth=1,
                           physicsClientId=client)
    p.addUserDebugText('NO-FLY  x > 0.3 m', [0.5, 0, 1.88],
                       textColorRGB=[1.0, 0.0, 0.0], textSize=1.2,
                       physicsClientId=client)

    p.addUserDebugText('x ->', [1.0, 0, 0.1], textColorRGB=[1, 1, 1],
                       textSize=1.0, physicsClientId=client)
    p.addUserDebugText('^ z', [-0.7, 0, 1.8], textColorRGB=[1, 1, 1],
                       textSize=1.0, physicsClientId=client)

    dummy = [0, 0, 0.01]
    handles['ref_line'] = p.addUserDebugLine(
        dummy, dummy, lineColorRGB=[1.0, 1.0, 0.0], lineWidth=3,
        physicsClientId=client)
    handles['pulse'] = p.addUserDebugLine(
        dummy, dummy, lineColorRGB=[0.5, 0.5, 0.5], lineWidth=6,
        physicsClientId=client)
    handles['status_text'] = p.addUserDebugText(
        '...', [-1.4, 0, 2.1], textColorRGB=[1, 1, 1], textSize=1.2,
        physicsClientId=client)
    handles['client'] = client
    return handles


def _update_dyn(
    handles: dict,
    env: SCGQuadrotorGatekeeperEnv,
    obs_np: np.ndarray,
    info_step: dict,
    n_viol_total: int,
    n_dec: int,
    n_rej: int,
) -> None:
    if not handles:
        return
    try:
        import pybullet as p
    except ImportError:
        return
    client = handles['client']

    state = obs_np[:6]
    drone_x, drone_z = float(state[0]), float(state[2])

    a_rl = int(info_step.get('a_rl', 0))
    committed = int(info_step.get('committed', 0))
    rejected = int(info_step.get('rejected', 0))

    # Compute the reference actually being tracked this step.
    if committed == 1:
        ref = env._nominal_ref(env.t)  # noqa: SLF001
    else:
        ref = env._backup_ref(state)  # noqa: SLF001
    ref_x, ref_z = float(ref[0]), float(ref[2])
    handles['ref_line'] = p.addUserDebugLine(
        [drone_x, 0, drone_z], [ref_x, 0, ref_z],
        lineColorRGB=[1.0, 1.0, 0.0], lineWidth=3,
        replaceItemUniqueId=handles['ref_line'],
        physicsClientId=client,
    )

    # Pulse color:
    #   green  = RL picked NOMINAL and filter accepted it
    #   red    = RL picked NOMINAL but filter rejected -> BACKUP
    #   blue   = RL picked BACKUP
    if a_rl == 1 and not rejected:
        color = [0.2, 1.0, 0.2]
    elif a_rl == 1 and rejected:
        color = [1.0, 0.2, 0.2]
    else:
        color = [0.3, 0.6, 1.0]
    handles['pulse'] = p.addUserDebugLine(
        [drone_x, 0, drone_z + 0.15], [drone_x, 0, drone_z + 0.25],
        lineColorRGB=color, lineWidth=6,
        replaceItemUniqueId=handles['pulse'],
        physicsClientId=client,
    )

    rej_rate = n_rej / max(1, n_dec)
    safe_flag = 'SAFE' if env._state_in_box(state) else '** VIOLATING **'  # noqa: SLF001
    hud_color = [0.2, 1.0, 0.2] if env._state_in_box(state) else [1.0, 0.2, 0.2]  # noqa: SLF001
    mode_str = 'NOM' if committed == 1 else 'BAK'
    rl_str = 'NOM' if a_rl == 1 else 'BAK'
    hud = (
        f't={env.t:3d}/{env._T_total:3d}  '  # noqa: SLF001
        f'rl={rl_str}  exec={mode_str}  '
        f'rej={n_rej:2d}/{n_dec:3d}={rej_rate:.2f}  '
        f'viol={n_viol_total:2d}  {safe_flag}'
    )
    handles['status_text'] = p.addUserDebugText(
        hud, [-1.4, 0, 2.1], textColorRGB=hud_color, textSize=1.1,
        replaceItemUniqueId=handles['status_text'],
        physicsClientId=client,
    )


def run_visual(
    variant: str,
    seed: int,
    deploy_nofilter: bool,
    gk_dir: str,
    backup_run_dir: str | None,
    H: int,
    delta: int,
    slow: float,
    num_episodes: int,
) -> None:
    run_dir = _find_run(gk_dir, variant, seed)
    print(f'\nLoading trained run: {run_dir}')

    os.environ['GK_USE_FILTER'] = 'false' if deploy_nofilter else 'true'
    os.environ['GK_USE_SHAPING'] = 'false'
    os.environ['GK_H'] = str(H)
    os.environ['GK_DELTA'] = str(delta)
    os.environ['GK_GUI'] = 'true'
    if backup_run_dir:
        os.environ['GK_BACKUP_RUN_DIR'] = str(backup_run_dir)
        print(f'Backup policy: {backup_run_dir}')
    else:
        print('Backup policy: fallback (fixed hover) — pass --backup-run-dir for learned backup.')

    env = omnisafe_make('SCG-Quadrotor-Gatekeeper-v0', device=torch.device('cpu'))
    actor, obs_norm, ckpt_path = _load_actor(
        run_dir,
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
    )
    print(f'Actor ckpt: {ckpt_path}')
    print('Low-level: safe-control-gym DSL PID (classical)')
    print(
        f'use_filter = {env._use_filter}  '  # noqa: SLF001
        f'use_shaping = {env._use_shaping}  '  # noqa: SLF001
        f'backup_loaded = {env.backup.loaded}',
    )

    for ep in range(num_episodes):
        obs_t, _info = env.reset(seed=seed + ep)
        obs_np = obs_t.detach().cpu().numpy()
        handles = _draw_static(env)

        n_dec = 0
        n_rej = 0
        n_viol = 0
        n_nom_rl = 0
        n_nom_exec = 0
        total_ret = 0.0
        max_decisions = env._T_total // env.delta + 1  # noqa: SLF001

        for _ in range(max_decisions):
            a = _actor_action(actor, obs_norm, obs_np)
            a = float(np.clip(a, 0.0, 1.0))
            a_tensor = torch.as_tensor([a], dtype=torch.float32)
            obs_t, rew, _cost, term, trunc, info_step = env.step(a_tensor)
            obs_np = obs_t.detach().cpu().numpy()

            total_ret += float(rew)
            n_dec += 1
            n_rej += int(info_step['rejected'])
            n_viol += int(info_step['n_violations'])
            n_nom_rl += int(info_step['a_rl'])
            n_nom_exec += int(info_step['committed'])

            _update_dyn(
                handles, env, obs_np, info_step,
                n_viol_total=n_viol, n_dec=n_dec, n_rej=n_rej,
            )
            if slow > 0:
                time.sleep(slow)

            if bool(term):
                print(f'  Episode {ep + 1}: terminated on violation at t={env.t}')
                break
            if bool(trunc):
                break

        print(
            f'  Episode {ep + 1:2d}/{num_episodes}  '
            f'Return={total_ret:+.2f}  '
            f'Decisions={n_dec}  '
            f'Rejections={n_rej} ({n_rej / max(1, n_dec):.1%})  '
            f'RL-nom={n_nom_rl / max(1, n_dec):.1%}  '
            f'Exec-nom={n_nom_exec / max(1, n_dec):.1%}  '
            f'Violations={n_viol}',
        )

    env.close()
    print('\nDone.')


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize a trained gatekeeper-RL policy.')
    p.add_argument('--variant', default='proposed',
                   choices=['proposed', 'filter_only', 'reward_only', 'nominal'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--gk-dir', default=_DEFAULT_GK_DIR)
    p.add_argument('--backup-run-dir', default=None,
                   help='Path to a trained backup run dir. Defaults to the latest '
                        'under examples/benchmarks/exp-backup/.')
    p.add_argument('--deploy-nofilter', action='store_true',
                   help='Run without the gatekeeper filter at deploy (tests internalization).')
    p.add_argument('--H', type=int, default=60)
    p.add_argument('--delta', type=int, default=5)
    p.add_argument('--slow', type=float, default=0.05)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    backup_dir = args.backup_run_dir
    if backup_dir is None:
        found = _find_backup_run_dir(_DEFAULT_BACKUP_SEARCH_ROOT)
        backup_dir = str(found) if found is not None else None
    run_visual(
        variant=args.variant,
        seed=args.seed,
        deploy_nofilter=args.deploy_nofilter,
        gk_dir=args.gk_dir,
        backup_run_dir=backup_dir,
        H=args.H,
        delta=args.delta,
        slow=args.slow,
        num_episodes=args.episodes,
    )
