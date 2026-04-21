"""Visualize a trained CPO or FOCOPS policy on safe-control-gym environments.

Loads a saved OmniSafe checkpoint and runs it inside safe-control-gym with
PyBullet GUI enabled.  For the trajectory-tracking task the GUI shows:
  - Blue circle  : full reference trajectory
  - Green arrow  : travel direction on the trajectory
  - Red wall     : forbidden zone boundary  (x > 0.3 m)
  - Red hatching : forbidden region
  - Yellow line  : drone -> current reference waypoint  (short = tracking well)
  - HUD text     : step, tracking error, cost, SAFE / VIOLATING status

USAGE
-----
  python visualize_policy.py --env quadrotor_tracking --algo CPO --seed 0
  python visualize_policy.py --env quadrotor_tracking --algo FOCOPS --seed 0
  python visualize_policy.py --env quadrotor  --algo CPO --seed 0
  python visualize_policy.py --env cartpole   --algo FOCOPS --seed 0 --no-gui
  python visualize_policy.py --env quadrotor_tracking --algo CPO --seed 0 --epoch 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import omnisafe.envs.safe_control_gym_env  # noqa: F401

from omnisafe.common.normalizer import Normalizer
from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config import Config
from safe_control_gym.utils.registration import make as scg_make


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

SCG_TASK_NAME = {
    'cartpole':           'cartpole',
    'quadrotor':          'quadrotor',
    'quadrotor_tracking': 'quadrotor',
}

SCG_KWARGS = {
    'cartpole': {
        'task': 'stabilization',
        'cost': 'rl_reward',
        'episode_len_sec': 10,
        'ctrl_freq': 50,
        'pyb_freq': 50,
        'normalized_rl_action_space': True,
        'randomized_init': True,
        'constraints': [
            {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
             'active_dims': [2], 'lower_bounds': [-0.2], 'upper_bounds': [0.2]},
            {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
             'active_dims': [0], 'lower_bounds': [-1.0], 'upper_bounds': [1.0]},
        ],
        'done_on_violation': False,
        'done_on_out_of_bound': False,
        'verbose': False,
    },
    'quadrotor': {
        'task': 'stabilization',
        'cost': 'rl_reward',
        'episode_len_sec': 5,
        'ctrl_freq': 60,
        'pyb_freq': 240,
        'quad_type': 2,
        'normalized_rl_action_space': True,
        'randomized_init': True,
        'constraints': [
            {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
             'active_dims': [0], 'lower_bounds': [-0.5], 'upper_bounds': [0.5]},
            {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
             'active_dims': [2], 'lower_bounds': [0.6], 'upper_bounds': [1.4]},
            {'constraint_form': 'bounded_constraint', 'constrained_variable': 'state',
             'active_dims': [4], 'lower_bounds': [-0.2], 'upper_bounds': [0.2]},
        ],
        'done_on_violation': False,
        'done_on_out_of_bound': False,
        'verbose': False,
    },
    'quadrotor_tracking': {
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
    },
}

# Results are always stored next to this script.
EXP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp-scg')

OMNISAFE_ENV_ID = {
    'cartpole':           'SCG-CartPole-Stabilization-v0',
    'quadrotor':          'SCG-Quadrotor-Stabilization-v0',
    'quadrotor_tracking': 'SCG-Quadrotor-Tracking-v0',
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_run_dir(env: str, algo: str, seed: int) -> str:
    env_dir = os.path.join(EXP_ROOT, OMNISAFE_ENV_ID[env], algo)
    seed_str = f'seed-{seed:03d}'
    matches = [m for m in glob.glob(os.path.join(env_dir, '**', f'{seed_str}*'), recursive=True)
               if os.path.isdir(m)]
    if not matches:
        raise FileNotFoundError(
            f'No run found for env={env}, algo={algo}, seed={seed} under {env_dir}\n'
            f'Make sure you have run run_scg_project.py first.'
        )
    return matches[0]


def _find_checkpoint(run_dir: str, epoch: int | None) -> str:
    save_dir = os.path.join(run_dir, 'torch_save')
    if epoch is not None:
        path = os.path.join(save_dir, f'epoch-{epoch}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path
    pts = glob.glob(os.path.join(save_dir, 'epoch-*.pt'))
    if not pts:
        raise FileNotFoundError(f'No checkpoints in {save_dir}')
    return max(pts, key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0]))


def _load_actor(run_dir: str, ckpt_path: str, obs_dim: int, act_dim: int) -> tuple:
    from gymnasium import spaces as gym_spaces

    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        cfg = Config.dict2config(json.load(f))
    model_cfgs = cfg.model_cfgs

    ckpt = torch.load(ckpt_path, map_location='cpu')

    obs_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym_spaces.Box(low=-1.0,    high=1.0,    shape=(act_dim,), dtype=np.float32)

    actor = ActorBuilder(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=model_cfgs.actor.hidden_sizes,
        activation=model_cfgs.actor.activation,
        weight_initialization_mode=model_cfgs.weight_initialization_mode,
    ).build_actor(model_cfgs.actor_type)

    actor.load_state_dict(ckpt['pi'])
    actor.eval()

    obs_normalizer = None
    if 'obs_normalizer' in ckpt and ckpt['obs_normalizer'] is not None:
        obs_normalizer = Normalizer(shape=(obs_dim,), clip=10.0)
        obs_normalizer.load_state_dict(ckpt['obs_normalizer'])

    return actor, obs_normalizer


def _select_action(actor, obs_np: np.ndarray, obs_normalizer) -> np.ndarray:
    """Normalise obs with frozen checkpoint stats, then run the actor."""
    obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    if obs_normalizer is not None:
        obs = obs.to(obs_normalizer.mean.device)
        std = torch.clamp(obs_normalizer.std, min=1e-2)
        obs = (obs - obs_normalizer.mean) / std
        obs = torch.clamp(obs, -10.0, 10.0)
    with torch.no_grad():
        action = actor.predict(obs, deterministic=True)
    return action.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# PyBullet debug-drawing helpers
# ---------------------------------------------------------------------------

def _draw_scene(env, env_name: str) -> dict:
    """Draw static scene elements once after reset.  Returns mutable handle dict."""
    handles: dict = {}
    if not getattr(env, 'GUI', False):
        return handles
    try:
        import pybullet as p
        client = env.PYB_CLIENT
    except Exception:
        return handles

    # ── 0. Camera: look straight at the x-z plane ────────────────────────
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraPitch=-5,
        cameraYaw=0,
        cameraTargetPosition=[0.0, 0.0, 1.0],
        physicsClientId=client,
    )

    # ── 1. Reference trajectory (blue connected segments + direction arrow) ─
    if hasattr(env, 'X_GOAL') and env.X_GOAL.ndim == 2:
        traj = env.X_GOAL
        x_col, z_col = (0, 2) if traj.shape[1] >= 4 else (None, None)
        if x_col is not None:
            n = len(traj)
            for i in range(n - 1):
                x1, z1 = float(traj[i,   x_col]), float(traj[i,   z_col])
                x2, z2 = float(traj[i+1, x_col]), float(traj[i+1, z_col])
                p.addUserDebugLine([x1, 0, z1], [x2, 0, z2],
                                   lineColorRGB=[0.0, 0.55, 1.0],
                                   lineWidth=2, physicsClientId=client)
            # Green arrow showing travel direction
            ai = n // 8
            p.addUserDebugLine(
                [float(traj[ai,   x_col]), 0, float(traj[ai,   z_col])],
                [float(traj[ai+3, x_col]), 0, float(traj[ai+3, z_col])],
                lineColorRGB=[0.0, 1.0, 0.0], lineWidth=5, physicsClientId=client)
            p.addUserDebugText(
                'START',
                [float(traj[0, x_col]) - 0.12, 0, float(traj[0, z_col]) - 0.12],
                textColorRGB=[0.0, 1.0, 0.5], textSize=1.0, physicsClientId=client)

    # ── 2. Environment-specific overlays ─────────────────────────────────
    if env_name == 'quadrotor_tracking':
        # Forbidden zone: red wall + hatching at x = 0.3
        x_lim, z_lo, z_hi = 0.3, 0.2, 1.8
        p.addUserDebugLine([x_lim, 0, z_lo], [x_lim, 0, z_hi],
                           lineColorRGB=[1.0, 0.0, 0.0], lineWidth=4,
                           physicsClientId=client)
        for x_h in [0.5, 0.7, 0.9, 1.1, 1.3]:
            p.addUserDebugLine([x_h, 0, z_lo], [x_h, 0, z_hi],
                               lineColorRGB=[1.0, 0.4, 0.4], lineWidth=1,
                               physicsClientId=client)
        p.addUserDebugText('NO-FLY  x > 0.3 m', [0.5, 0, 1.88],
                           textColorRGB=[1.0, 0.0, 0.0], textSize=1.2,
                           physicsClientId=client)

    elif env_name == 'quadrotor':
        # Goal marker: yellow cross at hover target (x=0, z=1.0)
        gx, gz, arm = 0.0, 1.0, 0.08
        for (a, b) in [([gx-arm,0,gz],[gx+arm,0,gz]),
                       ([gx,0,gz-arm],[gx,0,gz+arm])]:
            p.addUserDebugLine(a, b, lineColorRGB=[1.0,1.0,0.0],
                               lineWidth=4, physicsClientId=client)
        p.addUserDebugText('GOAL (hover)', [gx+0.1, 0, gz+0.12],
                           textColorRGB=[1.0,1.0,0.0], textSize=1.0,
                           physicsClientId=client)
        # Constraint box: |x|<=0.5, z in [0.6,1.4]
        corners = [(-0.5,0,0.6),( 0.5,0,0.6),( 0.5,0,1.4),(-0.5,0,1.4)]
        for i in range(4):
            c1, c2 = list(corners[i]), list(corners[(i+1)%4])
            p.addUserDebugLine(c1, c2, lineColorRGB=[1.0,0.5,0.0],
                               lineWidth=2, physicsClientId=client)
        p.addUserDebugText('safe zone', [-0.48, 0, 1.42],
                           textColorRGB=[1.0,0.5,0.0], textSize=0.9,
                           physicsClientId=client)

    elif env_name == 'cartpole':
        # Camera: side view looking along y-axis so cart track is visible
        p.resetDebugVisualizerCamera(
            cameraDistance=3.5, cameraPitch=-10, cameraYaw=0,
            cameraTargetPosition=[0.0, 0.0, 0.5], physicsClientId=client)
        # Cart position constraint walls: |x| <= 1.0 m  (orange verticals)
        for sx in [-1.0, 1.0]:
            p.addUserDebugLine([sx, 0, 0.0], [sx, 0, 1.5],
                               lineColorRGB=[1.0, 0.5, 0.0], lineWidth=3,
                               physicsClientId=client)
        p.addUserDebugText('|x|<=1.0', [ 1.05, 0, 1.55],
                           textColorRGB=[1.0, 0.5, 0.0], textSize=1.0,
                           physicsClientId=client)
        p.addUserDebugText('|x|<=1.0', [-1.3,  0, 1.55],
                           textColorRGB=[1.0, 0.5, 0.0], textSize=1.0,
                           physicsClientId=client)
        # Goal: balanced upright at x=0 — small yellow cross on the track
        p.addUserDebugLine([-0.08, 0, 0], [0.08, 0, 0],
                           lineColorRGB=[1.0,1.0,0.0], lineWidth=4,
                           physicsClientId=client)
        p.addUserDebugText('GOAL (balance)', [0.1, 0, 0.08],
                           textColorRGB=[1.0,1.0,0.0], textSize=1.0,
                           physicsClientId=client)
        # Pole-angle constraint label
        p.addUserDebugText('|theta|<=0.2 rad constraint active',
                           [-1.4, 0, 1.75],
                           textColorRGB=[1.0, 0.5, 0.0], textSize=0.9,
                           physicsClientId=client)

    # ── 3. Axis labels ────────────────────────────────────────────────────
    p.addUserDebugText('x ->', [1.0, 0, 0.1], textColorRGB=[1,1,1],
                       textSize=1.0, physicsClientId=client)
    p.addUserDebugText('^ z', [-0.7, 0, 1.8], textColorRGB=[1,1,1],
                       textSize=1.0, physicsClientId=client)

    # ── 4. Mutable handles (updated every step) ───────────────────────────
    dummy = [0, 0, 0.01]
    handles['ref_line'] = p.addUserDebugLine(
        dummy, dummy, lineColorRGB=[1.0, 1.0, 0.0], lineWidth=3,
        physicsClientId=client)
    handles['status_text'] = p.addUserDebugText(
        '...', [-1.2, 0, 2.1], textColorRGB=[1,1,1], textSize=1.2,
        physicsClientId=client)
    handles['client'] = client
    return handles


def _update_dynamic_visuals(handles: dict, env, obs: np.ndarray,
                             step: int, ep_cost: float,
                             env_name: str = '') -> None:
    """Move the yellow reference-marker line and update the HUD text."""
    if not handles:
        return
    try:
        import pybullet as p
    except Exception:
        return

    client = handles['client']
    drone_x = float(obs[0]) if len(obs) > 2 else 0.0
    drone_z = float(obs[2]) if len(obs) > 2 else 0.0

    # Reference point: trajectory waypoint (tracking) or static goal (stab)
    ref_x, ref_z = drone_x, drone_z
    if hasattr(env, 'X_GOAL'):
        if env.X_GOAL.ndim == 2:                   # traj_tracking
            idx = min(step, len(env.X_GOAL) - 1)
            ref_x = float(env.X_GOAL[idx, 0])
            ref_z = float(env.X_GOAL[idx, 2])
        else:                                       # stabilization
            ref_x = float(env.X_GOAL[0])
            ref_z = float(env.X_GOAL[2])

    handles['ref_line'] = p.addUserDebugLine(
        [drone_x, 0, drone_z], [ref_x, 0, ref_z],
        lineColorRGB=[1.0, 1.0, 0.0], lineWidth=3,
        replaceItemUniqueId=handles['ref_line'], physicsClientId=client)

    err = np.sqrt((drone_x - ref_x)**2 + (drone_z - ref_z)**2)

    if env_name == 'quadrotor_tracking':
        safe   = drone_x <= 0.3
        status = 'SAFE' if safe else '** VIOLATING x>0.3 **'
        color  = [0.2, 1.0, 0.2] if safe else [1.0, 0.2, 0.2]
        hud = f'Step {step:3d}  |  track_err={err:.3f}m  |  cost={ep_cost:.0f}  |  {status}'
    elif env_name == 'quadrotor':
        in_box = (abs(drone_x) <= 0.5 and 0.6 <= drone_z <= 1.4)
        status = 'in safe box' if in_box else '** outside box **'
        color  = [0.2, 1.0, 0.2] if in_box else [1.0, 0.5, 0.0]
        hud = f'Step {step:3d}  |  dist_to_goal={err:.3f}m  |  cost={ep_cost:.0f}  |  {status}'
    else:  # cartpole
        # obs = [x, x_dot, theta, theta_dot]
        cart_x = float(obs[0])
        theta  = float(obs[2]) if len(obs) > 2 else 0.0
        safe   = abs(cart_x) <= 1.0 and abs(theta) <= 0.2
        status = 'balanced' if safe else '** constraint violated **'
        color  = [0.2, 1.0, 0.2] if safe else [1.0, 0.2, 0.2]
        hud = (f'Step {step:3d}  |  x={cart_x:+.3f}m  '
               f'theta={theta:+.3f}rad  |  cost={ep_cost:.0f}  |  {status}')

    handles['status_text'] = p.addUserDebugText(
        hud, [-1.2, 0, 2.1], textColorRGB=color, textSize=1.2,
        replaceItemUniqueId=handles['status_text'], physicsClientId=client)


# ---------------------------------------------------------------------------
# Main visualisation loop
# ---------------------------------------------------------------------------

def run_visual(env_name: str, algo: str, seed: int, num_episodes: int,
               gui: bool, epoch: int | None, slow: float) -> None:

    run_dir  = _find_run_dir(env_name, algo, seed)
    ckpt     = _find_checkpoint(run_dir, epoch)
    epoch_id = os.path.basename(ckpt).replace('.pt', '')

    # Headless temp env to get obs/act dims
    kwargs = dict(SCG_KWARGS[env_name])
    kwargs['gui'] = False
    tmp_env = scg_make(SCG_TASK_NAME[env_name], **kwargs)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    print(f'\n{"="*60}')
    print(f'  Env    : {OMNISAFE_ENV_ID[env_name]}  (obs={obs_dim}, act={act_dim})')
    print(f'  Algo   : {algo}   Seed: {seed}   Checkpoint: {epoch_id}')
    print(f'  GUI    : {gui}')
    print(f'{"="*60}\n')

    actor, obs_normalizer = _load_actor(run_dir, ckpt, obs_dim, act_dim)

    kwargs['gui'] = gui
    env = scg_make(SCG_TASK_NAME[env_name], **kwargs)

    for ep in range(num_episodes):
        obs, info = env.reset()
        handles = _draw_scene(env, env_name) if gui else {}
        ep_ret, ep_cost, ep_steps = 0.0, 0.0, 0
        done = False

        while not done:
            action = _select_action(actor, obs, obs_normalizer)
            obs, rew, done, info = env.step(action)
            ep_ret  += float(rew)
            ep_cost += float(info.get('constraint_violation', 0))
            ep_steps += 1
            if gui:
                _update_dynamic_visuals(handles, env, obs, ep_steps, ep_cost, env_name)
                if slow > 0:
                    time.sleep(slow)

        print(f'  Episode {ep+1:2d}/{num_episodes}  '
              f'Return={ep_ret:8.2f}  Cost={ep_cost:5.0f}  Steps={ep_steps}')

    env.close()
    print('\nDone.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description='Visualize a trained SCG policy.')
    p.add_argument('--env',  choices=['cartpole', 'quadrotor', 'quadrotor_tracking'],
                   default='cartpole')
    p.add_argument('--algo', choices=['CPO', 'FOCOPS'], default='CPO')
    p.add_argument('--seed',     type=int,   default=0)
    p.add_argument('--episodes', type=int,   default=3)
    p.add_argument('--epoch',    type=int,   default=None)
    p.add_argument('--no-gui',   action='store_true')
    p.add_argument('--slow',     type=float, default=0.01,
                   help='Sleep (s) between steps when GUI is on.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    run_visual(
        env_name=args.env,
        algo=args.algo,
        seed=args.seed,
        num_episodes=args.episodes,
        gui=not args.no_gui,
        epoch=args.epoch,
        slow=args.slow,
    )
