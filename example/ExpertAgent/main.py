# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import grid2op
from evaluate import evaluate
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend

# Example usage
if __name__ == "__main__":

    # 使用标准环境名称
    env = grid2op.make(
        "l2rpn_neurips_2020_track2_small",
        reward_class=RedispReward,
        backend=LightSimBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    logs_path = "./result/neurips-2020/expert-agent-baseline"
    # Evaluate agent
    res = evaluate(env, nb_episode=7, verbose=True, save_gif=True, grid="IEEE118",logs_path=logs_path)

