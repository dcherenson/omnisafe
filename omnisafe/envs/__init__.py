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
"""Environment API for OmniSafe."""

from omnisafe.envs import classic_control
from omnisafe.envs.core import CMDP, env_register, make, support_envs
from omnisafe.envs.crabs_env import CRABSEnv
from omnisafe.envs.custom_env import CustomEnv
from omnisafe.envs.meta_drive_env import SafetyMetaDriveEnv
from omnisafe.envs.mujoco_env import MujocoEnv
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from omnisafe.envs.safety_gymnasium_modelbased import SafetyGymnasiumModelBased
from omnisafe.envs.safety_isaac_gym_env import SafetyIsaacGymEnv
import omnisafe.envs.paper_velocity_envs  # registers Paper-faithful envs with gymnasium

# Safe-control-gym environments (CartPole, Quadrotor stabilization/tracking)
# used by the earlier CPO/FOCOPS benchmark work.
try:
    from omnisafe.envs.safe_control_gym_env import SafeControlGymEnv  # noqa: F401
except ImportError:
    pass

# Gatekeeper-RL project, Stage-1 env: train a stabilizing backup policy from
# rich random initial states (see examples/benchmarks/train_backup.py).
try:
    from omnisafe.envs.scg_backup_env import SCGQuadrotorBackupEnv  # noqa: F401
except ImportError:
    pass

# Gatekeeper-RL project, Stage-2 env: RL switching policy that picks, per
# decision step, between the nominal circle-tracking reference and the frozen
# Stage-1 backup policy, with a forward-sim safety filter and shaping reward.
# Low-level tracking = safe-control-gym's classical DSL PID (not trained).
try:
    from omnisafe.envs.scg_gatekeeper_env import SCGQuadrotorGatekeeperEnv  # noqa: F401
except ImportError:
    pass
