import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BASE = os.path.expanduser('~/saferl_project/results')
COST_LIMIT = 25.0

EXPERIMENTS = {
    # Safety-Gymnasium
    'PointGoal1': [
        'PointGoal1_s0',
        'PointGoal1_s10',
    ],
    'PointButton1': [
        'PointButton1_s0',
        'PointButton1_s10',
    ],
    'CarGoal1': [
        'CarGoal1_s0',
        'CarGoal1_s10',
    ],
    # Safe-Control-Gym
    'SCG-CartPole': [
        'SCGCartPole_s0',
        'SCGCartPole_s10',
    ],
    'SCG-Quadrotor': [
        'SCGQuadrotor_s0',
        'SCGQuadrotor_s10',
    ],
}

def load_csv(folder):
    """Find and load progress.csv from a result folder, return latest run."""
    pattern = os.path.join(BASE, folder)
    csvs = []
    for root, dirs, files in os.walk(pattern):
        for f in files:
            if f == 'progress.csv':
                csvs.append(os.path.join(root, f))
    if not csvs:
        print(f"WARNING: no progress.csv found in {folder}")
        return None
    # pick the most recently modified
    csvs.sort(key=os.path.getmtime)
    df = pd.read_csv(csvs[-1])
    return df

def compute_cost_rate(df):
    """Cost rate = cumulative cost / total steps (matches paper definition)."""
    cumulative_cost = (df['Metrics/EpCost'] * df['Metrics/EpLen']).cumsum()
    total_steps = df['TotalEnvSteps']
    return cumulative_cost / total_steps

def smooth(y, window=3):
    """Simple moving average smoothing."""
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values

def plot_experiment(name, folders, ax_ret, ax_cost, ax_rate):
    """Plot mean ± std across seeds for one environment."""
    dfs = []
    for folder in folders:
        df = load_csv(folder)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print(f"Skipping {name} — no data")
        return

    # Align on TotalEnvSteps
    min_len = min(len(df) for df in dfs)
    steps = dfs[0]['TotalEnvSteps'].values[:min_len]

    rets  = np.array([df['Metrics/EpRet'].values[:min_len] for df in dfs])
    costs = np.array([df['Metrics/EpCost'].values[:min_len] for df in dfs])
    rates = np.array([compute_cost_rate(df).values[:min_len] for df in dfs])

    mean_ret  = smooth(rets.mean(axis=0))
    std_ret   = rets.std(axis=0)
    mean_cost = smooth(costs.mean(axis=0))
    std_cost  = costs.std(axis=0)
    mean_rate = smooth(rates.mean(axis=0))
    std_rate  = rates.std(axis=0)

    color = plt.cm.tab10(list(EXPERIMENTS.keys()).index(name) / len(EXPERIMENTS))

    for ax, mean, std, ylabel in [
        (ax_ret,  mean_ret,  std_ret,  'AverageEpRet'),
        (ax_cost, mean_cost, std_cost, 'AverageEpCost'),
        (ax_rate, mean_rate, std_rate, 'CostRate'),
    ]:
        ax.plot(steps, mean, label='PPO-Lag', color=color)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    # Cost limit dashed line
    ax_cost.axhline(COST_LIMIT, color='red', linestyle='--', linewidth=1.0, label='Cost Limit')
    cost_rate_limit = COST_LIMIT / dfs[0]['Metrics/EpLen'].mean()
    ax_rate.axhline(cost_rate_limit, color='red', linestyle='--', linewidth=1.0)

sg_envs  = {k: v for k, v in EXPERIMENTS.items() if not k.startswith('SCG')}
scg_envs = {k: v for k, v in EXPERIMENTS.items() if k.startswith('SCG')}

for title, envs, fname in [
    ('PPO-Lagrangian — Safety-Gymnasium',  sg_envs,  'plot_safety_gymnasium.png'),
    ('PPO-Lagrangian — Safe-Control-Gym',  scg_envs, 'plot_safe_control_gym.png'),
]:
    n = len(envs)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    col_titles = ['AverageEpRet', 'AverageEpCost', 'CostRate']
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=11)

    for row, (name, folders) in enumerate(envs.items()):
        ax_ret, ax_cost, ax_rate = axes[row]

        plot_experiment(name, folders, ax_ret, ax_cost, ax_rate)

        ax_ret.set_ylabel(name, fontsize=9, rotation=90, labelpad=10)
        for ax in [ax_ret, ax_cost, ax_rate]:
            ax.set_xlabel('TotalEnvSteps')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.grid(True, alpha=0.3)

        ax_cost.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.expanduser(f'~/saferl_project/results/{fname}')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

print("All plots saved!")
