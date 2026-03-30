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
"""Analyze results from RL Project 1 (CPO vs FOCOPS, circle + velocity envs).

Trained algorithms : CPO, FOCOPS  (2)
Environments       : SafetyPointCircle1-v0, SafetyAntCircle1-v0,
                     SafetyCarCircle1-v0,
                     SafetyAntVelocity-v1,  SafetyHopperVelocity-v1,
                     SafetyWalker2dVelocity-v1, SafetySwimmerVelocity-v1,
                     SafetyHalfCheetahVelocity-v1, SafetyHumanoidVelocity-v1
Seeds              : 0, 5

Plots are saved to:
  examples/benchmarks/exp-x/RL-Project-1/plots/
"""

import os

from omnisafe.common.statistics_tools import StatisticsTools


EXP_PATH = '/home/kbn/rl_ece567/omnisafe/examples/benchmarks/exp-x/RL-Project-1'
PLOT_DIR = os.path.join(EXP_PATH, 'plots')

if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)

    # StatisticsTools saves plots relative to CWD, so we change into the
    # output directory before calling draw_graph.
    os.chdir(PLOT_DIR)

    st = StatisticsTools()
    st.load_source(EXP_PATH)

    # Compare CPO vs FOCOPS (2 algorithms trained).
    # smooth=10 averages over a 10-epoch window to make curves easier to read.
    #
    # Cost limit notes (from FOCOPS paper Table 1 & Appendix G):
    #   - Velocity tasks: paper uses actual velocity as cost (not binary), thresholds
    #     are env-specific (Ant=103, HalfCheetah=152, Hopper=83, Humanoid=20,
    #     Swimmer=25, Walker2d=82). Safety-Gymnasium uses binary cost (0/1 per step),
    #     so scales are NOT directly comparable.
    #   - Circle tasks: paper uses same binary cost but threshold = 50 (not 25).
    #     Set cost_limit=50 to match paper for circle envs.
    #
    # Using cost_limit=25 here (OmniSafe default) as a general reference line.
    # Change to 50 for circle-only analysis or env-specific values for velocity.
    st.draw_graph(
        parameter='algo',
        values=['CPO', 'FOCOPS'],
        compare_num=None,
        cost_limit=25,
        smooth=10,
        show_image=False,
    )

    plots = [f for f in os.listdir(PLOT_DIR) if f.endswith('.png')]
    print(f'\n{len(plots)} plot(s) saved to: {PLOT_DIR}')
    for p in sorted(plots):
        print(f'  {p}')
