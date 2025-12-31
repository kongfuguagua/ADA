# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import grid2op
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend
from evaluate import evaluate

# Example usage
if __name__ == "__main__":
    # Create environment
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        reward_class=RedispReward,
        backend=LightSimBackend(),
        other_rewards={
            "bridge": BridgeReward,
            "overflow": CloseToOverflowReward,
            "distance": DistanceReward
        }
    )
    
    # Evaluate agent
    res = evaluate(env, nb_episode=7,
                   margin_th_limit=0.9,
                   rho_safe=0.9,
                   rho_danger=0.95,
                   penalty_storage_unsafe=0.01,
                   penalty_curtailment_unsafe=0.01,
                   penalty_redispatching_unsafe=0.01,
                #    alpha_por_error=0.5,
                #    weight_redisp_target=0.3,
                   verbose=True, save_gif=True)

