"""Plot CPO vs FOCOPS for every environment in RL-Custom-Env-Project-1.

Reads progress.csv files directly — no dependency on StatisticsTools.
Produces one PNG per environment with two subplots: EpRet and EpCost.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), 'exp-x')

ENVS = {
    'Hopper':       ('RL-Custom-Env-Project-1-Hopper',       82.748),
    'Swimmer':      ('RL-Custom-Env-Project-1-Swimmer',      24.516),
    'Walker2d':     ('RL-Custom-Env-Project-1-Walker2d',     81.886),
    'Ant':          ('RL-Custom-Env-Project-1-Ant',         103.115),
    'HalfCheetah':  ('RL-Custom-Env-Project-1-HalfCheetah', 151.989),
    'Humanoid':     ('RL-Custom-Env-Project-1-Humanoid',     20.140),
    'PointCircle':  ('RL-Custom-Env-Project-1-Circle',       50.0),
    'AntCircle':    ('RL-Custom-Env-Project-1-Circle',       50.0),
    'CarCircle':    ('RL-Custom-Env-Project-1-Circle',       50.0),
}

ALGO_COLORS = {'CPO': '#1f77b4', 'FOCOPS': '#ff7f0e'}
SMOOTH = 10


def load_runs(exp_dir: str, algo: str, env_filter: str = None) -> list[pd.DataFrame]:
    """Return list of DataFrames, one per seed, for the given algo."""
    pattern = os.path.join(exp_dir, algo, '**', 'progress.csv')
    files = glob.glob(pattern, recursive=True)
    dfs = []
    for f in sorted(files):
        if env_filter and env_filter not in f:
            continue
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    return dfs


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode='same')


def plot_metric(ax, runs: list[pd.DataFrame], metric: str, label: str,
                color: str, x_col: str = 'TotalEnvSteps') -> None:
    if not runs:
        return
    # Align all runs to the same x-grid using the shortest run
    min_len = min(len(df) for df in runs)
    xs = runs[0][x_col].values[:min_len]
    ys = np.stack([df[metric].values[:min_len] for df in runs], axis=0)

    mean = smooth(np.mean(ys, axis=0), SMOOTH)
    std  = smooth(np.std(ys,  axis=0), SMOOTH)

    ax.plot(xs, mean, label=label, color=color, linewidth=1.8)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)


def plot_env(env_label: str, exp_subdir: str, cost_limit: float,
             env_filter: str = None, out_dir: str = None) -> str:
    exp_dir = os.path.join(BASE, exp_subdir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle(env_label, fontsize=13, fontweight='bold')

    for algo, color in ALGO_COLORS.items():
        runs = load_runs(exp_dir, algo, env_filter)
        if not runs:
            print(f'  [{env_label}] WARNING: no runs found for {algo}')
            continue
        plot_metric(axes[0], runs, 'Metrics/EpRet',  algo, color)
        plot_metric(axes[1], runs, 'Metrics/EpCost', algo, color)

    # EpRet subplot
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Environment Steps')
    axes[0].set_ylabel('Return')
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.1f}M'))

    # EpCost subplot
    axes[1].set_title('Episode Cost')
    axes[1].set_xlabel('Environment Steps')
    axes[1].set_ylabel('Cost')
    axes[1].axhline(cost_limit, color='red', linestyle='--',
                    linewidth=1.2, label=f'Limit ({cost_limit})')
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.1f}M'))

    plt.tight_layout()

    if out_dir is None:
        out_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{env_label}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    out_root = os.path.join(BASE, 'RL-Custom-Env-Project-1-Plots')
    os.makedirs(out_root, exist_ok=True)

    print('Generating plots...')

    # Velocity envs (one env per exp dir)
    for label, (subdir, cl) in list(ENVS.items())[:6]:
        path = plot_env(label, subdir, cl, out_dir=out_root)
        print(f'  Saved: {path}')

    # Circle envs (3 envs share one exp dir — filter by env name)
    circle_filters = {
        'PointCircle': 'SafetyPoint',
        'AntCircle':   'SafetyAnt',
        'CarCircle':   'SafetyCar',
    }
    for label, env_filter in circle_filters.items():
        subdir, cl = ENVS[label]
        path = plot_env(label, subdir, cl, env_filter=env_filter, out_dir=out_root)
        print(f'  Saved: {path}')

    print(f'\nAll plots saved to:\n  {out_root}')
