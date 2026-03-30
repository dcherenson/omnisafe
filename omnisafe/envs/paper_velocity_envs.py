# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Paper-faithful velocity environments for OmniSafe.

These environments replicate the cost function used in the original FOCOPS paper
(Zhang et al., NeurIPS 2020, arXiv:2002.06506).

Key difference from the standard Safety-Gymnasium velocity environments:
  - Standard Safety-Gymnasium : cost = float(speed > threshold)  →  0 or 1 per step
  - These environments        : cost = actual speed value         →  continuous per step

Cost thresholds (from FOCOPS paper, Appendix G.1 / environment.py):
  Calculated as 50% of the discounted cost return of an unconstrained PPO agent
  trained for 1 million samples.

  Environment              Paper cost limit (episode return)
  ─────────────────────────────────────────────────────────
  SafetyAntVelocityPaper         103.115
  SafetyHalfCheetahVelocityPaper 151.989
  SafetyHopperVelocityPaper       82.748
  SafetyHumanoidVelocityPaper     20.140
  SafetySwimmerVelocityPaper      24.516
  SafetyWalker2dVelocityPaper     81.886

Usage in a benchmark script:
  eg.add('env_id', [
      'SafetyHopperVelocityPaper-v1',
      'SafetyAntVelocityPaper-v1',
      ...
  ])
  # Set cost limits matching the paper:
  eg.add('algo_cfgs:cost_limit', [82.748])  # per-env; or use per-env grids

Original environments are completely unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import gymnasium

# ── Import parent classes from Safety-Gymnasium ──────────────────────────────
from safety_gymnasium.tasks.safety_velocity.safety_hopper_velocity_v0 import (
    SafetyHopperVelocityEnv as _HopperBase,
)
from safety_gymnasium.tasks.safety_velocity.safety_swimmer_velocity_v0 import (
    SafetySwimmerVelocityEnv as _SwimmerBase,
)
from safety_gymnasium.tasks.safety_velocity.safety_walker2d_velocity_v0 import (
    SafetyWalker2dVelocityEnv as _Walker2dBase,
)
from safety_gymnasium.tasks.safety_velocity.safety_half_cheetah_velocity_v0 import (
    SafetyHalfCheetahVelocityEnv as _HalfCheetahBase,
)
from safety_gymnasium.tasks.safety_velocity.safety_ant_velocity_v0 import (
    SafetyAntVelocityEnv as _AntBase,
)
from safety_gymnasium.tasks.safety_velocity.safety_humanoid_velocity_v0 import (
    SafetyHumanoidVelocityEnv as _HumanoidBase,
)


# ── Paper cost limits (from focops-main/environment.py) ──────────────────────
PAPER_COST_LIMITS: dict[str, float] = {
    'SafetyAntVelocityPaper-v1':          103.115,
    'SafetyHalfCheetahVelocityPaper-v1':  151.989,
    'SafetyHopperVelocityPaper-v1':        82.748,
    'SafetyHumanoidVelocityPaper-v1':      20.140,
    'SafetySwimmerVelocityPaper-v1':       24.516,
    'SafetyWalker2dVelocityPaper-v1':      81.886,
}


# ── 1D environments (cost = |x_velocity|) ────────────────────────────────────

class SafetyHopperVelocityPaperEnv(_HopperBase):
    """Hopper with paper-faithful cost: cost = |x_velocity| (continuous).

    Paper cost limit: 82.748
    """

    def step(self, action):
        obs, reward, _binary_cost, terminated, truncated, info = super().step(action)
        # Replace binary cost with actual forward speed (paper Appendix G.1)
        cost = float(abs(info['x_velocity']))
        return obs, reward, cost, terminated, truncated, info


class SafetyHalfCheetahVelocityPaperEnv(_HalfCheetahBase):
    """HalfCheetah with paper-faithful cost: cost = |x_velocity| (continuous).

    Paper cost limit: 151.989
    """

    def step(self, action):
        obs, reward, _binary_cost, terminated, truncated, info = super().step(action)
        cost = float(abs(info['x_velocity']))
        return obs, reward, cost, terminated, truncated, info


class SafetyWalker2dVelocityPaperEnv(_Walker2dBase):
    """Walker2d with paper-faithful cost: cost = |x_velocity| (continuous).

    Paper cost limit: 81.886
    """

    def step(self, action):
        obs, reward, _binary_cost, terminated, truncated, info = super().step(action)
        cost = float(abs(info['x_velocity']))
        return obs, reward, cost, terminated, truncated, info


# ── 2D environments (cost = sqrt(vx² + vy²)) ─────────────────────────────────

class SafetySwimmerVelocityPaperEnv(_SwimmerBase):
    """Swimmer with paper-faithful cost: cost = sqrt(vx² + vy²) (continuous).

    Paper cost limit: 24.516
    """

    def step(self, action):
        obs, reward, _binary_cost, terminated, truncated, info = super().step(action)
        cost = math.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
        return obs, reward, cost, terminated, truncated, info


class SafetyAntVelocityPaperEnv(_AntBase):
    """Ant with paper-faithful cost: cost = sqrt(vx² + vy²) (continuous).

    Paper cost limit: 103.115
    """

    def step(self, action):
        obs, reward, _binary_cost, terminated, truncated, info = super().step(action)
        cost = math.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
        return obs, reward, cost, terminated, truncated, info


class SafetyHumanoidVelocityPaperEnv(_HumanoidBase):
    """Humanoid with paper-faithful cost: cost = sqrt(vx² + vy²) (continuous).

    Paper cost limit: 20.140
    """

    def step(self, action):
        obs, reward, _binary_cost, terminated, truncated, info = super().step(action)
        cost = math.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
        return obs, reward, cost, terminated, truncated, info


# ── Register with safety_gymnasium ───────────────────────────────────────────
# safety_gymnasium.register() calls gymnasium.register() internally AND adds
# the env to the safe_registry, so safety_gymnasium.make() applies the correct
# SafePassiveEnvChecker / SafeTimeLimit wrappers that expect a 6-element step
# tuple (obs, reward, cost, terminated, truncated, info).

from safety_gymnasium.utils.registration import register as _sg_register

_REGISTRY = [
    (
        'SafetyHopperVelocityPaper-v1',
        'omnisafe.envs.paper_velocity_envs:SafetyHopperVelocityPaperEnv',
        1000,
    ),
    (
        'SafetyHalfCheetahVelocityPaper-v1',
        'omnisafe.envs.paper_velocity_envs:SafetyHalfCheetahVelocityPaperEnv',
        1000,
    ),
    (
        'SafetyWalker2dVelocityPaper-v1',
        'omnisafe.envs.paper_velocity_envs:SafetyWalker2dVelocityPaperEnv',
        1000,
    ),
    (
        'SafetySwimmerVelocityPaper-v1',
        'omnisafe.envs.paper_velocity_envs:SafetySwimmerVelocityPaperEnv',
        1000,
    ),
    (
        'SafetyAntVelocityPaper-v1',
        'omnisafe.envs.paper_velocity_envs:SafetyAntVelocityPaperEnv',
        1000,
    ),
    (
        'SafetyHumanoidVelocityPaper-v1',
        'omnisafe.envs.paper_velocity_envs:SafetyHumanoidVelocityPaperEnv',
        1000,
    ),
]

for _env_id, _entry_point, _max_steps in _REGISTRY:
    if _env_id not in gymnasium.envs.registry:
        _sg_register(
            id=_env_id,
            entry_point=_entry_point,
            max_episode_steps=_max_steps,
        )
