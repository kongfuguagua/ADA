import grid2op
from l2rpn_baselines.DoNothing import evaluate
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend

# 使用标准环境名称
env = grid2op.make(
    "l2rpn_wcci_2022",
    reward_class=RedispReward,
    backend=LightSimBackend(),
    other_rewards={
        "bridge": BridgeReward,
        "overflow": CloseToOverflowReward,
        "distance": DistanceReward
    }
)
logs_path = "./result/wcci-2022/all_do-nothing-baseline"
res = evaluate(env, nb_episode=10000, verbose=True, save_gif=True,logs_path=logs_path)