# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
from grid2op.Environment import Environment
from .expertAgent import ExpertAgent


def make_agent(env: Environment,
               dir_path: os.PathLike,
               gridName: str = "IEEE14",
               **kwargs) -> ExpertAgent:
    """Create an ExpertAgent instance for l2rpn competitions or grid2game.

    Parameters
    ----------
    env : Environment
        The grid2op environment instance.
        
    dir_path : os.PathLike
        Path to directory (not used for ExpertAgent, kept for interface consistency).
        
    gridName : str, optional
        Grid identifier name. Used for local optimization of choices.
        Options: "IEEE14", "IEEE118", "IEEE118_R2". Default: "IEEE14".

    Returns
    -------
    ExpertAgent
        An instance of the ExpertAgent.
    """
    agent = ExpertAgent(env.action_space,
                       env.observation_space,
                       name="ExpertAgent",
                       gridName=gridName,
                       **kwargs)
    
    return agent

