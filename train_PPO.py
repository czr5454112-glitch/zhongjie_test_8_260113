# -*- coding: utf-8 -*-
# @author: Jay-Bling
# @email: gzj22@mails.tsinghua.edu.cn
# @date: 2024/12/15
import copy

import pandas as pd

from ccbsenv import *
from stable_baselines3.common.vec_env import DummyVecEnv
from ccbs import *
from config import *

config = Config()
map = Map("instances/roadmaps/sparse/map.xml")
training_file = [
    os.path.join("instances", "roadmaps", "sparse", "train", fname)
    for fname in os.listdir(os.path.join("instances", "roadmaps", "sparse", "train"))
    if fname != ".DS_Store"
]
origin_ccbs = CCBS(map)
all_envs = []
i = 1
for f in training_file:
    task = Task()
    task.load_from_file(f)
    ccbs = copy.deepcopy(origin_ccbs)

    if config.use_precalculated_heuristic:
        # Initialize the heuristic based on reverse Dijkstra
        heuristic_start_time = time.time()
        ccbs.map.init_heuristic(task.agents)
        print(f"heuristic process time {time.time() - heuristic_start_time}s")

    ccbs.solution = Solution()
    start_time = time.time()

    cardinal_solved = 0
    semicardinal_solved = 0

    # 初始化根节点
    if not ccbs.init_root(task):
        print("No root solution possible, cannot continue")
        continue

    expanded = 1
    time_elapsed = 0
    low_level_searches = 0
    low_level_expanded = 0
    agent_id = 2

    if len(ccbs.tree.container) == 0:
        print('No Solution')
        continue
    else:
        parent = ccbs.tree.get_front()  # Get frontal node from the tree
        if parent.conflicts_num == 0:
            continue
        env = CCBSEnv(task, parent, map, expanded, time_elapsed, low_level_searches, low_level_expanded, ccbs.tree)
        all_envs.append(env)
    print("存储第{}/{}个算例".format(i, len(training_file)))
    i += 1

model = PPO(
    "MultiInputPolicy",
    env=all_envs[0],
    learning_rate=2e-4,
    gamma=0.98,
    verbose=1,
    ent_coef=0.01,
    gae_lambda=0.95,
    clip_range=0.2)
reward_callback = RewardCallback(max_episodes=1000)
start_time = time.time()
e_num = 1
for env in all_envs:
    print("第{}/{}个算例训练开始".format(e_num, len(training_file)))
    model.set_env(env)
    model.learn(total_timesteps=10000, callback=reward_callback)
    print("第{}/{}个算例训练结束".format(e_num, len(training_file)))
    e_num += 1

learning_time = time.time() - start_time
print(f"PPO训练时间: {learning_time:.4f}s")
model.save("ppo_road-sparse")
rewards = reward_callback.rewards
df = pd.DataFrame(rewards)
df.to_csv("rewards_road-sparse.csv", index=False)
# plt.plot(reward_callback.rewards)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.savefig("PPO_converge_fig.png", dpi=300)
# plt.show()

