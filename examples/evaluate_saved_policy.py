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
"""One example for evaluate saved policy."""

import os

import omnisafe


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
LOG_DIR = '/home/kbn/rl_ece567/omnisafe/examples/benchmarks/exp-x/On-Policy-Test/SafetyAntVelocity-v1---2ed752d93226bee88a1f93f711e3706928644b87755a9d84d790778351e915cb/CPO-{SafetyAntVelocity-v1}/seed-000-2026-03-24-11-28-59'
if __name__ == '__main__':
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name='track',
                width=256,
                height=256,
            )
            evaluator.render(num_episodes=1)
            evaluator.evaluate(num_episodes=1)
    scan_dir.close()
