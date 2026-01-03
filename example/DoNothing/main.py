import grid2op
from l2rpn_baselines.DoNothing import evaluate
from grid2op.Reward import RedispReward, BridgeReward, CloseToOverflowReward, DistanceReward
from lightsim2grid import LightSimBackend

# 使用标准环境名称
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
logs_path = "./result/ijcnn2019/DoNothing"
res = evaluate(env, nb_episode=7, verbose=True, save_gif=True,logs_path=logs_path)