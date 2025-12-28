import grid2op
from l2rpn_baselines.DoNothing import evaluate

# 使用标准环境名称
env = grid2op.make("l2rpn_case14_sandbox")
res = evaluate(env, nb_episode=7, verbose=True, save_gif=True)