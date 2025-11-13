# -*- coding: utf-8 -*-
# @author: Jay-Bling
# @email: gzj22@mails.tsinghua.edu.cn
# @date: 2024/10/15
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import spaces
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from sipp import SIPP
from ccbs import CCBS
import random
from structs import *
from map import *
import time
import copy
from stable_baselines3.common.callbacks import BaseCallback
import heapq


class CCBSEnv(gym.Env):
    def __init__(self, task, ini_node, map: Map, expanded, time_elapsed, low_level_searches, low_level_expanded, tree):
        super(CCBSEnv, self).__init__()
        self.state = {}  # 状态初始化
        self.reward = 0
        self.task = task
        self.ini_node = ini_node
        self.node = ini_node  # 当前CCBS节点
        self.current_step = 1
        self.planner = SIPP(map)
        self.expanded = expanded
        self.time_elapsed = time_elapsed
        self.low_level_searches = low_level_searches
        self.low_level_expanded = low_level_expanded
        self.alg = CCBS(map)
        self.max_process_agent = 100
        self.max_step = 1024
        self.reward_1 = 10  # 当前节点满足约束且无其它冲突
        self.reward_2 = 2  # 分支数量权重系数
        self.reward_3 = -5  # 当前节点不满足约束
        self.cardinal_conflicts_weight = 1  # cardinal_conflict权重
        self.semicard_conflicts_weight = 1  # semicard_conflict权重
        self.non_cardinal_conflict_weight = 1  # non_cardinal冲突数量权重
        self.reward_iter_pos = -0.5
        self.cost_weight = 0.01  # 目标函数权重
        self.high_level_generated = 1
        self.cur_id = 2
        self.lb = 0
        self.ub = float('inf')
        self.tree = tree
        self.final_res = None
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 状态空间：定义CBS状态空间
        self.observation_space = Dict({
            'cardinal_conflict': Discrete(100),  # cardinal冲突数量
            'semi_cardinal_conflict': Discrete(100),  # semi_cardinal冲突数量
            'non_cardinal_conflict': Discrete(100),  # non_cardinal冲突数量
            "agents_number": Discrete(100),  # 涉及冲突的智能体数量
            'cost': Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 当前路径总和
            'cur_depth': Discrete(1025)  # 当前搜索树深度
        })

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        # 重置环境
        self.done = False
        self.reward = 0
        self.current_step = 1
        self.low_level_searches = 0
        self.low_level_expanded = 0
        self.expanded = 1
        self.time_elapsed = 0
        self.high_level_generated = 1
        self.cur_id = 2
        self.final_res = None
        self.node = self.ini_node
        paths = self.alg.get_paths(self.node, len(self.task.agents))

        self.state = {
            'cardinal_conflict': np.array([len(self.node.cardinal_conflicts)]).astype(np.int8),
            'semi_cardinal_conflict': np.array([len(self.node.semicard_conflicts)]).astype(np.int8),
            'non_cardinal_conflict': np.array([len(self.node.conflicts)]).astype(np.int8),
            "agents_number": np.array([self.alg.cal_conflict_agent(paths)]).astype(np.int8),
            'cost': np.array([self.alg.get_spath_cost(paths)]).astype(np.float32),
            'cur_depth': np.array([self.current_step]).astype(np.int32)
        }
        info = {}
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        return self.state, info

    def step(self, action):
        node = self.node.create_node_move_conflicts()
        paths = self.alg.get_paths(node, len(self.task.agents))

        if not node.conflicts and not node.semicard_conflicts and not node.cardinal_conflicts:
            self.final_res = node
            print("No conflicts, solution found successfully")
            self.done = True

        elif self.current_step >= self.max_step or (len(self.tree.container) == 0 and self.current_step > 1):
            print('No Solution')
            self.done = True

        else:
            node.cost -= node.h
            all_conflicts = node.cardinal_conflicts + node.semicard_conflicts + node.conflicts
            all_conflicts.sort(key=lambda x: x.t)
            print(f"current all conflicts: {all_conflicts}")

            conflict_index = 0
            if action == 1:
                conflict_index = -1
            else:
                conflict_index = int(np.ceil(len(all_conflicts) * action)) - 1
            conflict = all_conflicts[conflict_index]
            print(f"selected conflict is {conflict}")

            corridor_constraints = []
            if self.alg.config.use_corridor_symmetry:
                corridor = self.alg.check_corridor_conflict(conflict, self.task)
                corridor_constraints = self.alg.get_corridor_constraint(conflict, corridor)
            target_constraints = []
            if self.alg.config.use_target_symmetry:
                target_constraints = self.alg.get_target_constraint(conflict, self.task)

            constraintsA = self.alg.get_constraints(node, conflict.agent1)
            if len(corridor_constraints) > 0:
                constraintA = corridor_constraints[0]
            elif len(target_constraints) > 0:
                constraintA = target_constraints[0]
            else:
                constraintA = self.alg.get_constraint(conflict.agent1, conflict.move1, conflict.move2)
            constraintsA.append(constraintA)
            pathA = self.planner.find_path(self.task.get_agent(conflict.agent1), constraintsA)
            self.low_level_searches += 1
            self.low_level_expanded += pathA.expanded

            constraintsB = self.alg.get_constraints(node, conflict.agent2)
            if len(corridor_constraints) > 0:
                constraintB = corridor_constraints[1]
            elif len(target_constraints) > 0:
                constraintB = target_constraints[1]
            else:
                constraintB = self.alg.get_constraint(conflict.agent2, conflict.move2, conflict.move1)
            constraintsB.append(constraintB)
            pathB = self.planner.find_path(self.task.get_agent(conflict.agent2), constraintsB)
            self.low_level_searches += 1
            self.low_level_expanded += pathB.expanded

            left_node = CBS_Node([pathA], self.node, constraintA,
                                 node.cost + pathA.cost - self.alg.get_cost(node, conflict.agent1), 0, node.total_cons + 1)
            left_node.id = self.cur_id
            self.cur_id += 1

            right_node = CBS_Node([pathB], self.node, constraintB,
                                  node.cost + pathB.cost - self.alg.get_cost(node, conflict.agent2), 0, node.total_cons + 1)
            right_node.id = self.cur_id
            self.cur_id += 1

            if pathA.cost > 0 and self.alg.validate_constraints(constraintsA, pathA.agentID):
                time_now = time.time()
                self.low_level_searches, self.low_level_expanded = self.alg.find_new_conflicts(self.task, left_node, paths, pathA,
                                                                                               node.conflicts,
                                                                                               node.semicard_conflicts,
                                                                                               node.cardinal_conflicts,
                                                                                               self.low_level_searches,
                                                                                               self.low_level_expanded)
                time_spent = time.time() - time_now
                self.time_elapsed += time_spent
                self.expanded += 1

                if left_node.cost > 0:
                    left_node.h = self.alg.get_hl_heuristic(left_node.cardinal_conflicts)
                    left_node.cost += left_node.h
                    self.tree.add_node(left_node)

                    if left_node.conflicts_num == 0:
                        l_global_reward = self.reward_1 + self.reward_2 / self.current_step
                        self.ub = min(self.ub, left_node.cost)
                    else:
                        # left_index = self.node_list.index((left_node.cost, left_node.id, left_node))
                        # l_global_reward = self.reward_iter_pos * (1 - left_index / len(self.node_list))
                        l_global_reward = self.reward_iter_pos
                        self.lb = max(self.lb, left_node.cost)
                    l_reward_1 = self.semicard_conflicts_weight * (len(node.semicard_conflicts) - len(left_node.semicard_conflicts)) + \
                                 self.cardinal_conflicts_weight * (len(node.cardinal_conflicts) - len(left_node.cardinal_conflicts)) + \
                                 self.non_cardinal_conflict_weight * (len(node.conflicts) - len(left_node.conflicts))

                    l_reward_2 = self.cost_weight * min(left_node.cost - self.lb, self.ub-left_node.cost)
                    l_total_reward = l_global_reward + l_reward_1 + l_reward_2
                else:
                    l_total_reward = self.reward_3
            else:
                l_total_reward = self.reward_3

            if pathB.cost > 0 and self.alg.validate_constraints(constraintsB, pathB.agentID):
                time_now = time.time()
                self.low_level_searches, self.low_level_expanded = self.alg.find_new_conflicts(self.task, right_node, paths, pathB,
                                                                                               node.conflicts,
                                                                                               node.semicard_conflicts,
                                                                                               node.cardinal_conflicts,
                                                                                               self.low_level_searches,
                                                                                               self.low_level_expanded)
                time_spent = time.time() - time_now
                self.time_elapsed += time_spent
                self.expanded += 1

                if right_node.cost > 0:
                    right_node.h = self.alg.get_hl_heuristic(right_node.cardinal_conflicts)
                    right_node.cost += right_node.h
                    self.tree.add_node(right_node)

                    if right_node.conflicts_num == 0:
                        r_global_reward = self.reward_1 + self.reward_2 / self.current_step
                        self.ub = min(self.ub, right_node.cost)
                    else:
                        # right_index = self.node_list.index((right_node.cost, right_node.id, right_node))
                        # r_global_reward = self.reward_iter_pos * (1 - right_index / len(self.node_list))
                        r_global_reward = self.reward_iter_pos
                        self.lb = max(self.lb, right_node.cost)
                    r_reward_1 = self.semicard_conflicts_weight * (len(node.semicard_conflicts) - len(right_node.semicard_conflicts)) + \
                                 self.cardinal_conflicts_weight * (len(node.cardinal_conflicts) - len(right_node.cardinal_conflicts)) + \
                                 self.non_cardinal_conflict_weight * (len(node.conflicts) - len(right_node.conflicts))

                    r_reward_2 = self.cost_weight * min(right_node.cost - self.lb, self.ub - right_node.cost)
                    r_total_reward = r_global_reward + r_reward_1 + r_reward_2

                else:
                    r_total_reward = self.reward_3
            else:
                r_total_reward = self.reward_3

            self.reward += (l_total_reward + r_total_reward) / 2

            self.current_step += 1

            # 下一状态节点
            if len(self.tree.container) > 0:
                self.node = self.tree.get_front()

                paths = self.alg.get_paths(node=self.node, agents_size=len(self.task.agents))
                soc = self.alg.get_spath_cost(paths)
                self.state['cardinal_conflict'] = np.array([len(self.node.cardinal_conflicts)]).astype(np.int8)
                self.state['semi_cardinal_conflict'] = np.array([len(self.node.semicard_conflicts)]).astype(np.int8)
                self.state['non_cardinal_conflict'] = np.array([len(self.node.conflicts)]).astype(np.int8)
                self.state['agents_number'] = np.array([self.alg.cal_conflict_agent(paths)]).astype(np.int8)
                self.state['cost'] = np.array([soc]).astype(np.float32)
                self.state['cur_depth'] = np.array([self.current_step]).astype(np.int32)
            else:
                self.done = True

        truncated = False
        info = {}
        print("current step:", self.current_step)
        print("current cardinal:", self.state['cardinal_conflict'])
        print("current semi_cardinal:", self.state['semi_cardinal_conflict'])
        print("current non_cardinal:", self.state['non_cardinal_conflict'])
        print("current reward:", self.reward)
        print("is done:", self.done)

        return self.state, self.reward, self.done, truncated, info


# 定义回调类
class RewardCallback(BaseCallback):
    def __init__(self, max_episodes, *args, **kwargs):
        super(RewardCallback, self).__init__(*args, **kwargs)
        self.rewards = []
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 每步记录一次奖励
        if self.locals['dones'].any():
            self.rewards.append(self.locals['rewards'])
            self.episode_count += 1
            print(f"已完成 {self.episode_count} 个回合")
        if self.episode_count >= self.max_episodes:
            print(f"已完成 {self.max_episodes} 个回合，停止训练")
            return False  # 停止训练
        return True




