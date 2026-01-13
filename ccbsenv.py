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
        self.episode_reward = 0.0  # episode累计奖励（仅用于记录，step()返回单步奖励）
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
        self.max_step = 4096  # 从2048增加到4096，给算法更多时间找到解
        self.reward_1 = 15  # 当前节点满足约束且无其它冲突（从10提高到15）
        self.reward_2 = 2  # 分支数量权重系数
        self.reward_3 = -2  # 当前节点不满足约束（从-5改为-2，减少无效路径的累积惩罚）
        self.cardinal_conflicts_weight = 2  # cardinal_conflict权重（从1提高到2）
        self.semicard_conflicts_weight = 1.5  # semicard_conflict权重（从1提高到1.5）
        self.non_cardinal_conflict_weight = 1.2  # non_cardinal冲突数量权重（从1提高到1.2）
        self.reward_iter_pos = -0.05  # 每次迭代惩罚（从-0.2降低到-0.05，避免长episode累积负回报过多）
        self.reward_fail = -100  # episode失败惩罚（未找到解时的惩罚，从-15提高到-100避免失败也能拿高分）
        self.cost_weight = 0.0  # 目标函数权重（暂时关闭，避免反向激励问题）
        self.high_level_generated = 1
        self.cur_id = 2
        self.lb = 0
        self.ub = float('inf')
        self.tree = tree
        self.final_res = None
        # action_space改为[-1,1]，避免硬clip导致梯度语义差
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 课程学习参数（用于降低难度，让success_rate先>0）
        self.curriculum_mode = True  # 是否启用课程学习
        self.curriculum_max_agents = 25  # 当前阶段允许的最大agent数量（初始25，通过set_attr动态更新）
        self._agent_count_cache = {}  # 缓存任务文件的agent数量，避免重复解析

        # 状态空间：定义CBS状态空间（统一使用Box，避免Discrete与np.array不匹配）
        self.observation_space = Dict({
            'cardinal_conflict': Box(low=0, high=1000, shape=(1,), dtype=np.float32),  # cardinal冲突数量
            'semi_cardinal_conflict': Box(low=0, high=1000, shape=(1,), dtype=np.float32),  # semi_cardinal冲突数量
            'non_cardinal_conflict': Box(low=0, high=5000, shape=(1,), dtype=np.float32),  # non_cardinal冲突数量
            "agents_number": Box(low=0, high=1000, shape=(1,), dtype=np.float32),  # 涉及冲突的智能体数量
            'cost': Box(low=0, high=1e9, shape=(1,), dtype=np.float32),  # 当前路径总和
            'cur_depth': Box(low=0, high=4096, shape=(1,), dtype=np.float32)  # 当前搜索树深度（与max_step同步）
        })

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        
        # 如果环境有_map_path和_train_task_dir属性，说明需要重新选择任务
        if hasattr(self, '_map_path') and hasattr(self, '_train_task_dir') and hasattr(self, '_ccbs_config'):
            # 重新选择任务（实现跨任务训练）
            import random
            import os
            from structs import Task, Solution
            
            max_retries = 30  # 增加重试次数，以便跳过有问题的任务文件
            last_error = None
            tried_files = set()  # 记录已尝试的文件，避免重复选择有问题的文件
            task_file = None  # 初始化task_file，避免在except分支中未定义
            curriculum_skipped = False  # 标记是否是课程学习跳过的episode
            
            for retry in range(max_retries):
                try:
                    # 重新扫描任务文件目录（确保每次reset都能获取最新任务列表）
                    if not os.path.exists(self._train_task_dir):
                        raise ValueError(f"训练任务目录不存在: {self._train_task_dir}")
                    
                    training_files = [
                        os.path.join(self._train_task_dir, fname)
                        for fname in os.listdir(self._train_task_dir)
                        if fname.endswith('.xml') and fname != ".DS_Store"
                    ]
                    
                    if len(training_files) == 0:
                        raise ValueError(f"训练任务目录中没有可用的任务文件: {self._train_task_dir}")
                    
                    # 排除已尝试过的有问题的文件
                    available_files = [f for f in training_files if f not in tried_files]
                    if len(available_files) == 0:
                        # 如果所有文件都试过了，重置记录（可能是新的一轮）
                        tried_files.clear()
                        available_files = training_files
                    
                    # 课程学习：根据agent数量过滤任务（降低难度）
                    # 关键：如果过滤后无可用任务，返回空episode（立即done），而不是报错
                    # 这样hard map的env在早期会快速done，不会真正训练，也不会干扰训练
                    curriculum_skipped = False
                    if self.curriculum_mode and self.curriculum_max_agents > 0:
                        filtered_files = []
                        for f in available_files:
                            # 获取或缓存agent数量
                            agent_count = self._get_agent_count_from_file(f)
                            if agent_count is not None and agent_count <= self.curriculum_max_agents:
                                filtered_files.append(f)
                        
                        # 如果过滤后有可用文件，使用过滤后的列表
                        if len(filtered_files) > 0:
                            available_files = filtered_files
                        else:
                            # 过滤后为空：说明当前课程阶段不允许训练此地图
                            # 设置标志，返回空episode（立即done）
                            curriculum_skipped = True
                            break  # 跳出重试循环，进入空episode处理逻辑
                    
                    # 随机选择一个任务文件
                    task_file = random.choice(available_files)
                    
                    # 重用地图对象（如果已存在）
                    if hasattr(self, '_world_map') and self._world_map is not None:
                        world_map = self._world_map
                    else:
                        from map import Map
                        world_map = Map(self._map_path)
                        self._world_map = world_map
                    
                    # 创建新的CCBS对象
                    from ccbs import CCBS
                    ccbs = CCBS(world_map)
                    
                    # 应用配置
                    ccbs.config.agent_size = self._ccbs_config["agent_size"]
                    ccbs.config.hlh_type = self._ccbs_config["hlh_type"]
                    ccbs.config.precision = self._ccbs_config["precision"]
                    ccbs.config.timelimit = self._ccbs_config["timelimit"]
                    ccbs.config.use_precalculated_heuristic = self._ccbs_config["use_precalculated_heuristic"]
                    ccbs.config.use_disjoint_splitting = self._ccbs_config["use_disjoint_splitting"]
                    ccbs.config.use_cardinal = self._ccbs_config["use_cardinal"]
                    ccbs.config.use_corridor_symmetry = self._ccbs_config["use_corridor_symmetry"]
                    ccbs.config.use_target_symmetry = self._ccbs_config["use_target_symmetry"]
                    ccbs.config.use_rl = self._ccbs_config["use_rl"]
                    ccbs.verbose = self._ccbs_config["verbose"]
                    
                    # 加载任务
                    task = Task()
                    try:
                        task.load_from_file(task_file)
                    except (ValueError, KeyError, AttributeError) as e:
                        # 如果任务文件格式错误，记录并重试
                        tried_files.add(task_file)
                        if "invalid literal" in str(e) or "任务文件格式错误" in str(e):
                            last_error = f"任务文件格式错误: {os.path.basename(task_file)} - {str(e)}"
                            if retry < max_retries - 1:
                                continue
                        raise
                    
                    if self._ccbs_config["use_precalculated_heuristic"]:
                        ccbs.map.init_heuristic(task.agents)
                    
                    ccbs.solution = Solution()
                    
                    # 初始化根节点
                    if not ccbs.init_root(task):
                        if retry < max_retries - 1:
                            continue
                        raise ValueError(f"无法找到根节点解: {os.path.basename(task_file)}")
                    
                    if len(ccbs.tree.container) == 0:
                        if retry < max_retries - 1:
                            continue
                        raise ValueError(f"无解: {os.path.basename(task_file)}")
                    
                    parent = ccbs.tree.get_front()
                    if parent.conflicts_num == 0:
                        if retry < max_retries - 1:
                            continue
                        raise ValueError(f"根节点无冲突: {os.path.basename(task_file)}")
                    
                    # 更新环境状态
                    self.task = task
                    self.ini_node = parent
                    self.node = parent
                    self.alg = ccbs
                    self.tree = ccbs.tree
                    self.planner = ccbs.planner
                    break
                    
                except Exception as e:
                    last_error = str(e)
                    # 如果是任务文件格式错误，记录文件并重试
                    if "invalid literal" in str(e) or "任务文件格式错误" in str(e):
                        if task_file:
                            tried_files.add(task_file)
                    if retry < max_retries - 1:
                        continue
                    # 如果所有重试都失败
                    raise RuntimeError(f"reset时重新选择任务失败，已重试{max_retries}次。最后错误: {last_error}")
        
        # 检查是否是课程学习跳过的episode（当前阶段不允许训练此地图）
        if curriculum_skipped:
            # 返回空episode（立即done，reward=0）
            # 这样hard map的env在早期会快速done，不会真正训练，也不会干扰训练
            self.done = True
            self.episode_reward = 0.0
            self.current_step = 1
            self.final_res = None
            
            # 返回一个有效的初始状态（全零状态，表示"跳过"）
            self.state = {
                'cardinal_conflict': np.array([0.0], dtype=np.float32),
                'semi_cardinal_conflict': np.array([0.0], dtype=np.float32),
                'non_cardinal_conflict': np.array([0.0], dtype=np.float32),
                'agents_number': np.array([0.0], dtype=np.float32),
                'cost': np.array([0.0], dtype=np.float32),
                'cur_depth': np.array([0.0], dtype=np.float32)
            }
            
            info = {
                'done_reason': 4,  # 4表示课程学习跳过（区别于其他done原因）
                'curriculum_skipped': True  # 标记这是课程学习跳过的episode
                # 注意：不包含is_success，避免空episode稀释success_rate统计
            }
            
            # 设置标志，以便step()也能识别
            self.curriculum_skipped = True
            
            return self.state, info
        
        # 重置环境状态（正常情况）
        self.done = False
        self.episode_reward = 0.0  # 重置episode累计奖励
        self.current_step = 1
        self.low_level_searches = 0
        self.low_level_expanded = 0
        self.expanded = 1
        self.time_elapsed = 0
        self.high_level_generated = 1
        self.cur_id = 2
        self.final_res = None
        self.node = self.ini_node
        self.lb = 0
        self.ub = float('inf')
        paths = self.alg.get_paths(self.node, len(self.task.agents))

        self.state = {
            'cardinal_conflict': np.array([float(len(self.node.cardinal_conflicts))], dtype=np.float32),
            'semi_cardinal_conflict': np.array([float(len(self.node.semicard_conflicts))], dtype=np.float32),
            'non_cardinal_conflict': np.array([float(len(self.node.conflicts))], dtype=np.float32),
            "agents_number": np.array([float(self.alg.cal_conflict_agent(paths))], dtype=np.float32),
            'cost': np.array([float(self.alg.get_spath_cost(paths))], dtype=np.float32),
            'cur_depth': np.array([float(self.current_step)], dtype=np.float32)
        }
        info = {}
        # action_space改为[-1,1]，避免硬clip导致梯度语义差
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        return self.state, info

    def step(self, action):
        # 如果这是课程学习跳过的episode（在reset()时已设置done=True），直接返回
        if self.done and hasattr(self, 'curriculum_skipped') and getattr(self, 'curriculum_skipped', False):
            info = {
                'done_reason': 4,  # 4表示课程学习跳过
                'curriculum_skipped': True
                # 注意：不包含is_success，避免空episode稀释success_rate统计
            }
            return self.state, 0.0, True, False, info
        
        # 修复action处理：从[-1,1]线性映射到[0,1]，避免硬clip
        # action通常是np.ndarray(shape=(1,))，在[-1,1]范围内
        if isinstance(action, np.ndarray):
            action_raw = float(action[0])
        elif isinstance(action, (list, tuple)):
            action_raw = float(action[0])
        else:
            action_raw = float(action)
        
        # 线性映射：[-1,1] -> [0,1]
        action_value = (action_raw + 1.0) / 2.0
        # 理论上已经在[0,1]范围内，但为了安全可以加一个很小的clip
        action_value = np.clip(action_value, 0.0, 1.0)
        
        # 初始化单步奖励和truncated标志（step()必须返回单步奖励，不是累计奖励）
        step_reward = 0.0
        truncated = False  # 必须：保证所有路径都定义，避免UnboundLocalError
        
        node = self.node.create_node_move_conflicts()
        paths = self.alg.get_paths(node, len(self.task.agents))

        if not node.conflicts and not node.semicard_conflicts and not node.cardinal_conflicts:
            self.final_res = node
            # 找到解时给予成功奖励（单步奖励）
            step_reward += self.reward_1 + self.reward_2 / self.current_step
            self.ub = min(self.ub, node.cost)  # 更新上界
            # print("No conflicts, solution found successfully")  # 移除print以提高性能
            self.done = True

        elif self.current_step >= self.max_step:
            # 超步截断：中等惩罚（比无解惩罚更温和），使用truncated语义更正确
            self.done = True
            truncated = True
            step_reward += -30  # 比无解惩罚(-100)更温和，鼓励探索
        elif len(self.tree.container) == 0 and self.current_step > 1:
            # 无解/树空：重罚
            self.done = True
            # 添加失败惩罚（单步奖励）
            step_reward += self.reward_fail

        else:
            node.cost -= node.h
            all_conflicts = node.cardinal_conflicts + node.semicard_conflicts + node.conflicts
            all_conflicts.sort(key=lambda x: x.t)
            # print(f"current all conflicts: {all_conflicts}")  # 移除print以提高性能

            conflict_index = 0
            if len(all_conflicts) == 0:
                conflict_index = 0
            elif action_value >= 1.0 or action_value == 1.0:
                conflict_index = -1  # 选择最后一个冲突
            else:
                conflict_index = int(np.ceil(len(all_conflicts) * action_value)) - 1
                conflict_index = max(0, min(conflict_index, len(all_conflicts) - 1))  # 确保索引有效
            
            conflict = all_conflicts[conflict_index]
            # print(f"selected conflict is {conflict}")  # 移除print以提高性能

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
                    
                    # 冲突差分奖励归一化（按父节点冲突总数，避免跨地图尺度差异）
                    parent_total = len(node.cardinal_conflicts) + len(node.semicard_conflicts) + len(node.conflicts)
                    scale = max(1.0, float(parent_total))
                    l_reward_1 = (
                        self.semicard_conflicts_weight * (len(node.semicard_conflicts) - len(left_node.semicard_conflicts)) +
                        self.cardinal_conflicts_weight * (len(node.cardinal_conflicts) - len(left_node.cardinal_conflicts)) +
                        self.non_cardinal_conflict_weight * (len(node.conflicts) - len(left_node.conflicts))
                    ) / scale

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
                    
                    # 冲突差分奖励归一化（按父节点冲突总数，避免跨地图尺度差异）
                    parent_total = len(node.cardinal_conflicts) + len(node.semicard_conflicts) + len(node.conflicts)
                    scale = max(1.0, float(parent_total))
                    r_reward_1 = (
                        self.semicard_conflicts_weight * (len(node.semicard_conflicts) - len(right_node.semicard_conflicts)) +
                        self.cardinal_conflicts_weight * (len(node.cardinal_conflicts) - len(right_node.cardinal_conflicts)) +
                        self.non_cardinal_conflict_weight * (len(node.conflicts) - len(right_node.conflicts))
                    ) / scale

                    r_reward_2 = self.cost_weight * min(right_node.cost - self.lb, self.ub - right_node.cost)
                    r_total_reward = r_global_reward + r_reward_1 + r_reward_2

                else:
                    r_total_reward = self.reward_3
            else:
                r_total_reward = self.reward_3

            step_reward += (l_total_reward + r_total_reward) / 2

            self.current_step += 1

            # 下一状态节点
            if len(self.tree.container) > 0:
                self.node = self.tree.get_front()

                paths = self.alg.get_paths(node=self.node, agents_size=len(self.task.agents))
                soc = self.alg.get_spath_cost(paths)
                self.state['cardinal_conflict'] = np.array([float(len(self.node.cardinal_conflicts))], dtype=np.float32)
                self.state['semi_cardinal_conflict'] = np.array([float(len(self.node.semicard_conflicts))], dtype=np.float32)
                self.state['non_cardinal_conflict'] = np.array([float(len(self.node.conflicts))], dtype=np.float32)
                self.state['agents_number'] = np.array([float(self.alg.cal_conflict_agent(paths))], dtype=np.float32)
                self.state['cost'] = np.array([float(soc)], dtype=np.float32)
                self.state['cur_depth'] = np.array([float(self.current_step)], dtype=np.float32)
            else:
                self.done = True
                # 如果树为空且未找到解，添加失败惩罚（单步奖励）
                if self.final_res is None:
                    step_reward += self.reward_fail

        # 累计episode奖励（仅用于记录，step()返回单步奖励）
        self.episode_reward += step_reward

        info = {}
        
        # VecMonitor会自动生成episode信息，不需要手动设置
        # 添加自定义字段（成功标志和done_reason统计）
        if self.done:
            info['is_success'] = (self.final_res is not None)
            # done_reason: 0=running, 1=success, 2=timeout, 3=fail
            if self.final_res is not None:
                info['done_reason'] = 1  # success
            elif truncated:
                info['done_reason'] = 2  # timeout (max_step截断)
            else:
                info['done_reason'] = 3  # fail (树空/无解)
            
            # 添加CCBS算法运行状态信息（仅在episode结束时输出，避免刷屏）
            info['ccbs_stats'] = {
                'step': self.current_step,
                'tree_size': len(self.tree.container),
                'cardinal_conflicts': len(self.node.cardinal_conflicts) if hasattr(self, 'node') else 0,
                'semi_cardinal_conflicts': len(self.node.semicard_conflicts) if hasattr(self, 'node') else 0,
                'non_cardinal_conflicts': len(self.node.conflicts) if hasattr(self, 'node') else 0,
                'low_level_searches': self.low_level_searches,
                'low_level_expanded': self.low_level_expanded,
                'final_res': self.final_res is not None
            }
        
        # 移除print以提高性能（20个并行环境会疯狂刷屏）
        # print("current step:", self.current_step)
        # print("current cardinal:", self.state['cardinal_conflict'])
        # print("current semi_cardinal:", self.state['semi_cardinal_conflict'])
        # print("current non_cardinal:", self.state['non_cardinal_conflict'])
        # print("current reward:", step_reward)
        # print("is done:", self.done)

        # 返回单步奖励（不是累计奖励）
        return self.state, float(step_reward), self.done, truncated, info
    
    def _get_agent_count_from_file(self, task_file):
        """从任务文件中获取agent数量（带缓存）"""
        # 检查缓存
        if task_file in self._agent_count_cache:
            return self._agent_count_cache[task_file]
        
        # 解析XML获取agent数量
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(task_file)
            root = tree.getroot()
            agents = root.findall('.//agent')
            agent_count = len(agents)
            # 缓存结果
            self._agent_count_cache[task_file] = agent_count
            return agent_count
        except Exception as e:
            # 解析失败，返回None（会在后续重试逻辑中处理）
            return None


# 定义回调类
class RewardCallback(BaseCallback):
    def __init__(self, max_episodes, convergence_window=100, convergence_threshold=0.01, *args, **kwargs):
        super(RewardCallback, self).__init__(*args, **kwargs)
        self.rewards = []
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.convergence_window = convergence_window  # 用于检测收敛的窗口大小
        self.convergence_threshold = convergence_threshold  # 收敛阈值（奖励变化率）
        self.best_reward = float('-inf')
        self.converged = False
        self.last_checkpoint_episode = 0
        self.checkpoint_interval = 50  # 每50个episode保存一次检查点

    def _on_step(self) -> bool:
        # 每步记录一次奖励
        try:
            # 安全地获取 locals 中的信息
            dones = self.locals.get('dones', None)
            if dones is None:
                return True  # 如果没有 dones 信息，继续训练
            
            # 处理单个环境的情况（dones 可能是标量或数组）
            if hasattr(dones, 'any'):
                if not dones.any():
                    return True
            elif not dones:
                return True
            
            # 获取奖励
            rewards = self.locals.get('rewards', None)
            if rewards is None:
                return True  # 如果没有奖励信息，继续训练
            
            # 处理单个环境的情况（rewards 可能是标量或数组）
            if hasattr(rewards, '__iter__') and not isinstance(rewards, str):
                # 如果是数组，取最后一个完成的episode的奖励
                reward = float(rewards[-1]) if len(rewards) > 0 else 0.0
            else:
                reward = float(rewards)
            
            self.rewards.append(reward)
            self.episode_count += 1
            
            # 更新最佳奖励
            if reward > self.best_reward:
                self.best_reward = reward
            
            # 检测收敛
            if len(self.rewards) >= self.convergence_window:
                recent_rewards = self.rewards[-self.convergence_window:]
                reward_std = np.std(recent_rewards)
                reward_mean = np.mean(recent_rewards)
                
                # 如果最近窗口内的奖励标准差很小，认为收敛
                if reward_std < abs(reward_mean) * self.convergence_threshold:
                    if not self.converged:
                        self.converged = True
                        print(f"[收敛检测] Episode {self.episode_count}: 检测到收敛! "
                              f"最近{self.convergence_window}个episode的平均奖励={reward_mean:.4f}, "
                              f"标准差={reward_std:.4f}")
                else:
                    self.converged = False
            
            # 定期打印信息
            if self.episode_count % 10 == 0:
                avg_reward = np.mean(self.rewards[-10:]) if len(self.rewards) >= 10 else np.mean(self.rewards)
                print(f"Episode {self.episode_count}/{self.max_episodes}: "
                      f"最近平均奖励={avg_reward:.4f}, 最佳奖励={self.best_reward:.4f}, "
                      f"收敛状态={'已收敛' if self.converged else '训练中'}")
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            # 如果获取信息失败，记录错误但继续训练
            if self.episode_count % 100 == 0:  # 每100个episode才打印一次，避免刷屏
                print(f"[警告] RewardCallback获取episode信息时出错: {e}，将跳过此步骤")
            return True
        
        if self.episode_count >= self.max_episodes:
            print(f"已完成 {self.max_episodes} 个回合，停止训练")
            return False  # 停止训练
        return True




