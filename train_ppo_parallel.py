# -*- coding: utf-8 -*-
"""
PPO提升CCBS冲突选择的并行训练脚本
支持多地图并行训练，使用SubprocVecEnv实现环境并行化
"""
import os
import sys
import time
import copy
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ccbs import CCBS
from ccbsenv import CCBSEnv, RewardCallback
from config import Config
from map import Map
from structs import Task, Solution
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
# CheckpointCallback已删除，改用TimeLimitedRewardCallback的保存逻辑
# from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# ============================================================================
# ============================ 配置区域 ======================================
# ============================================================================

# 训练时间限制（小时）
TRAINING_TIME_LIMIT_HOURS = 100

# 地图配置列表
MAP_CONFIGS = [
    {
        "name": "empty-16-16-random",
        "map_path": "instances/empty-16-16-random/map.xml",
        "train_task_dir": "instances/empty-16-16-random/train",
    },
    {
        "name": "room-64-64-8_random",
        "map_path": "instances/room-64-64-8_random/map.xml",
        "train_task_dir": "instances/room-64-64-8_random/train",
    },
    {
        "name": "den520d_random",
        "map_path": "instances/den520d_random/map.xml",
        "train_task_dir": "instances/den520d_random/train",
    },
    {
        "name": "warehouse-10-20-random",
        "map_path": "instances/warehouse-10-20-random/map.xml",
        "train_task_dir": "instances/warehouse-10-20-random/train",
    },
    {
        "name": "roadmap-sparse",
        "map_path": "instances/roadmaps/sparse/map.xml",
        "train_task_dir": "instances/roadmaps/sparse/train",
    },
]

# 优化的PPO训练配置（根据文档要求）
PPO_CONFIG = {
    "learning_rate": 3e-4,  # 初始学习率
    "gamma": 0.999,  # 折扣因子（提高到0.999）
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # 熵系数
    "n_steps": 256,  # 每个环境采样步数
    "batch_size": 1024,  # 批大小
    "n_epochs": 10,  # 每批优化轮数
    "max_episodes": 80000,  # 最大训练回合数
    "total_timesteps": int(2e6),  # 总训练步数（200万步）
    "checkpoint_freq": 200000,  # 检查点保存频率（步数）
}

# 并行环境配置
PARALLEL_CONFIG = {
    "num_envs_per_map": 4,  # 每种地图的并行环境数量
    "total_num_envs": 20,  # 总并行环境数（5种地图 * 4）
}

# CCBS算法配置
CCBS_CONFIG = {
    "agent_size": 0.5,
    "hlh_type": 2,
    "precision": 0.1,
    "timelimit": 300,
    "use_precalculated_heuristic": False,
    "use_disjoint_splitting": True,
    "use_cardinal": True,
    "use_corridor_symmetry": False,
    "use_target_symmetry": False,
    "use_rl": False,  # 训练时不需要RL模型
    "verbose": False,  # 并行环境时关闭详细输出
}

# 环境配置（奖励参数已在ccbsenv.py中优化）
RL_ENV_CONFIG = {
    "max_step": 1024,
    "max_process_agent": 100,
}

# 全局变量用于时间控制
training_start_time = None
training_time_limit = None

# 全局变量存储地图配置和任务文件
map_configs_cache = {}
task_files_cache = {}


def get_abs_path(rel_path):
    """将相对路径转换为绝对路径"""
    if rel_path is None:
        return None
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(SCRIPT_DIR, rel_path)


def load_map_configs():
    """加载所有地图配置和任务文件列表"""
    global map_configs_cache, task_files_cache
    
    for map_config in MAP_CONFIGS:
        map_name = map_config["name"]
        map_path = get_abs_path(map_config["map_path"])
        train_task_dir = get_abs_path(map_config["train_task_dir"])
        
        if not os.path.exists(map_path):
            print(f"[警告] 地图文件未找到: {map_path}，跳过该地图")
            continue
        
        if not os.path.exists(train_task_dir):
            print(f"[警告] 训练任务目录未找到: {train_task_dir}，跳过该地图")
            continue
        
        # 加载任务文件列表
        training_files = [
            os.path.join(train_task_dir, fname)
            for fname in os.listdir(train_task_dir)
            if fname.endswith('.xml') and fname != ".DS_Store"
        ]
        
        if len(training_files) == 0:
            print(f"[警告] 地图 {map_name} 没有可用的训练任务，跳过")
            continue
        
        map_configs_cache[map_name] = {
            "map_path": map_path,
            "train_task_dir": train_task_dir,
            "training_files": training_files
        }
        task_files_cache[map_name] = training_files
        
        print(f"[信息] 加载地图 {map_name}: {len(training_files)} 个训练任务")
    
    return map_configs_cache


def apply_config_to_ccbs(ccbs, config):
    """将配置应用到CCBS对象"""
    ccbs.config.agent_size = config["agent_size"]
    ccbs.config.hlh_type = config["hlh_type"]
    ccbs.config.precision = config["precision"]
    ccbs.config.timelimit = config["timelimit"]
    ccbs.config.use_precalculated_heuristic = config["use_precalculated_heuristic"]
    ccbs.config.use_disjoint_splitting = config["use_disjoint_splitting"]
    ccbs.config.use_cardinal = config["use_cardinal"]
    ccbs.config.use_corridor_symmetry = config["use_corridor_symmetry"]
    ccbs.config.use_target_symmetry = config["use_target_symmetry"]
    ccbs.config.use_rl = config["use_rl"]
    ccbs.verbose = config["verbose"]


def apply_config_to_env(env, config):
    """将配置应用到环境"""
    env.max_step = config["max_step"]
    env.max_process_agent = config["max_process_agent"]


def create_single_env(map_path, train_task_dir, env_id=0):
    """为指定地图创建一个环境实例（每次调用创建新任务）
    
    参数:
        map_path: 地图文件路径（绝对路径）
        train_task_dir: 训练任务目录（绝对路径，用于在子进程中重新扫描）
        env_id: 环境ID（用于日志）
    """
    try:
        # 在子进程中重新扫描任务文件（避免缓存问题）
        if not os.path.exists(train_task_dir):
            raise ValueError(f"训练任务目录不存在: {train_task_dir}")
        
        training_files = [
            os.path.join(train_task_dir, fname)
            for fname in os.listdir(train_task_dir)
            if fname.endswith('.xml') and fname != ".DS_Store"
        ]
        
        if len(training_files) == 0:
            raise ValueError(f"训练任务目录中没有可用的任务文件: {train_task_dir}")
        
        # 随机选择一个任务文件
        task_file = random.choice(training_files)
        
        # 加载地图
        world_map = Map(map_path)
        
        # 创建CCBS对象
        ccbs = CCBS(world_map)
        apply_config_to_ccbs(ccbs, CCBS_CONFIG)
        
        # 加载任务
        task = Task()
        try:
            task.load_from_file(task_file)
        except (ValueError, KeyError, AttributeError) as e:
            raise ValueError(f"任务文件格式错误: {os.path.basename(task_file)} - {str(e)}")
        
        # 验证任务有效性
        if not task.agents or len(task.agents) == 0:
            raise ValueError(f"任务文件无有效智能体: {os.path.basename(task_file)}")
        
        if CCBS_CONFIG["use_precalculated_heuristic"]:
            ccbs.map.init_heuristic(task.agents)
        
        ccbs.solution = Solution()
        
        # 初始化根节点
        if not ccbs.init_root(task):
            raise ValueError(f"无法找到根节点解: {os.path.basename(task_file)}")
        
        if len(ccbs.tree.container) == 0:
            raise ValueError(f"无解: {os.path.basename(task_file)}")
        
        parent = ccbs.tree.get_front()
        if parent.conflicts_num == 0:
            raise ValueError(f"根节点无冲突: {os.path.basename(task_file)}")
        
        # 创建环境
        env = CCBSEnv(
            task, parent, world_map,
            expanded=1,
            time_elapsed=0,
            low_level_searches=0,
            low_level_expanded=0,
            tree=ccbs.tree
        )
        apply_config_to_env(env, RL_ENV_CONFIG)
        
        # 保存地图路径和任务目录，用于reset时重新选择任务
        env._map_path = map_path
        env._train_task_dir = train_task_dir  # 保存目录路径，在reset时重新扫描
        env._ccbs_config = CCBS_CONFIG.copy()
        env._rl_env_config = RL_ENV_CONFIG.copy()
        env._world_map = world_map  # 保存地图对象，避免重复加载
        
        return env
        
    except Exception as e:
        print(f"[环境 {env_id}] 创建环境失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_env_for_map(map_path, train_task_dir, env_id=0):
    """为指定地图创建环境的工厂函数（用于SubprocVecEnv）
    
    参数:
        map_path: 地图文件路径（绝对路径，在闭包中捕获）
        train_task_dir: 训练任务目录（绝对路径，在闭包中捕获）
        env_id: 环境ID
    """
    def _init():
        # 创建环境，如果失败则重试（最多20次，增加重试次数以跳过有问题的任务）
        max_retries = 20
        last_error = None
        for retry in range(max_retries):
            try:
                env = create_single_env(map_path, train_task_dir, env_id)
                if env is not None:
                    return env
            except Exception as e:
                last_error = str(e)
                # 如果是任务文件格式错误，继续重试（会随机选择其他任务）
                if "任务文件格式错误" in str(e) or "invalid literal" in str(e):
                    if retry < max_retries - 1:
                        continue
                # 其他错误也重试
                if retry < max_retries - 1:
                    continue
        
        # 如果所有重试都失败，抛出异常
        raise RuntimeError(f"环境 {env_id} 创建失败，已重试 {max_retries} 次。最后错误: {last_error}")
    
    return _init


class VecEnvRewardCallback(RewardCallback):
    """VecEnv兼容的奖励回调，修复VecEnv下的episode统计问题"""
    def __init__(self, max_episodes, convergence_window=100, convergence_threshold=0.01, 
                 patience=2000, min_delta=0.1, mean_min=1.0, *args, **kwargs):
        super().__init__(max_episodes=max_episodes, 
                        convergence_window=convergence_window,
                        convergence_threshold=convergence_threshold, 
                        *args, **kwargs)
        # patience机制：best_reward长时间无提升时停止
        self.patience = patience
        self.min_delta = min_delta  # 奖励提升的最小阈值
        self.last_improve_ep = 0  # 上次提升的episode
        self.mean_min = mean_min  # 收敛判定的最小均值门槛
        # done_reason统计：1=success, 2=timeout, 3=fail
        self.done_reasons = []  # 记录最近episode的done_reason
        self.done_reason_window = 100  # 统计窗口大小
    
    def _on_step(self) -> bool:
        # VecEnv兼容的episode统计
        # 从infos中获取每个环境的episode信息（需要VecMonitor）
        try:
            if 'infos' in self.locals:
                infos = self.locals.get('infos', [])
                dones = self.locals.get('dones', [])
                
                # 确保infos和dones是列表/数组
                if not isinstance(infos, (list, np.ndarray)):
                    infos = []
                if not isinstance(dones, (list, np.ndarray)):
                    dones = []
                
                # 统计本step中done的环境数量
                num_dones = int(np.sum(dones)) if isinstance(dones, np.ndarray) else sum(dones) if dones else 0
                
                # 从每个done的环境的infos中提取episode reward
                for i in range(min(len(infos), len(dones))):
                    try:
                        info = infos[i] if i < len(infos) else {}
                        done = dones[i] if i < len(dones) else False
                        
                        if done and isinstance(info, dict):
                            # 关键修复：忽略课程学习跳过的episode（避免episode计数炸穿）
                            if info.get('curriculum_skipped', False):
                                # 这是课程学习跳过的空episode，不计入统计
                                # 避免hard map env在早期产生大量空episode导致训练提前停止
                                continue
                            
                            # VecMonitor会在infos中添加episode信息
                            # 注意：episode信息可能不存在，需要安全获取
                            episode_info = info.get('episode', None)
                            if episode_info is None:
                                # 如果episode信息不存在，尝试从info中直接获取
                                # 某些情况下VecMonitor可能将episode信息直接放在info中
                                if 'r' in info and 'l' in info:
                                    episode_info = {'r': info['r'], 'l': info['l']}
                            
                            if episode_info and isinstance(episode_info, dict):
                                episode_reward = episode_info.get('r', None)
                                episode_length = episode_info.get('l', None)
                                
                                if episode_reward is not None:
                                    self.rewards.append(float(episode_reward))
                                    self.episode_count += 1
                                    
                                    # 统计done_reason（从info中获取，VecMonitor会保留env的info字段）
                                    done_reason = info.get('done_reason', 0)
                                    if done_reason > 0:  # 只记录有效的done_reason
                                        self.done_reasons.append(done_reason)
                                    
                                    # 更新最佳奖励（patience机制）
                                    if episode_reward > self.best_reward + self.min_delta:
                                        self.best_reward = episode_reward
                                        self.last_improve_ep = self.episode_count
                                    
                                    # 检测收敛（加epsilon + mean_min门槛，避免均值接近0时误判）
                                    if len(self.rewards) >= self.convergence_window:
                                        recent_rewards = self.rewards[-self.convergence_window:]
                                        reward_std = float(np.std(recent_rewards))
                                        reward_mean = float(np.mean(recent_rewards))
                                        eps = 1e-6
                                        
                                        # 收敛条件：均值大于门槛 且 标准差小于阈值
                                        if abs(reward_mean) > self.mean_min and reward_std < max(eps, abs(reward_mean)) * self.convergence_threshold:
                                            if not self.converged:
                                                self.converged = True
                                                print(f"[收敛检测] Episode {self.episode_count}: 检测到收敛! "
                                                      f"最近{self.convergence_window}个episode的平均奖励={reward_mean:.4f}, "
                                                      f"标准差={reward_std:.4f}")
                                        else:
                                            self.converged = False
                                    
                                    # 定期打印信息（包含done_reason统计）
                                    if self.episode_count % 10 == 0:
                                        avg_reward = np.mean(self.rewards[-10:]) if len(self.rewards) >= 10 else np.mean(self.rewards)
                                        
                                        # 统计最近窗口的done_reason分布
                                        done_reason_stats = ""
                                        if len(self.done_reasons) >= 10:
                                            recent_reasons = self.done_reasons[-self.done_reason_window:]
                                            total = len(recent_reasons)
                                            success_count = recent_reasons.count(1)
                                            timeout_count = recent_reasons.count(2)
                                            fail_count = recent_reasons.count(3)
                                            done_reason_stats = f", done_reason: success={success_count}/{total}, timeout={timeout_count}/{total}, fail={fail_count}/{total}"
                                        
                                        print(f"Episode {self.episode_count}/{self.max_episodes}: "
                                              f"最近平均奖励={avg_reward:.4f}, 最佳奖励={self.best_reward:.4f}, "
                                              f"收敛状态={'已收敛' if self.converged else '训练中'}{done_reason_stats}")
                    except (KeyError, IndexError, TypeError, AttributeError) as e:
                        # 忽略单个环境的错误，继续处理其他环境
                        continue
        except Exception as e:
            # 如果发生任何错误，记录但不中断训练
            # 只在第一次遇到错误时打印，避免日志过多
            if not hasattr(self, '_error_logged'):
                print(f"[警告] VecEnvRewardCallback处理episode信息时出错: {e}，将跳过此步骤")
                self._error_logged = True
        
        # 检查最大回合数限制
        if self.episode_count >= self.max_episodes:
            print(f"已完成 {self.max_episodes} 个回合，停止训练")
            return False
        
        # 检查patience机制：best_reward长时间无提升时停止
        if self.episode_count - self.last_improve_ep >= self.patience:
            print(f"[Patience停止] Episode {self.episode_count}: 最佳奖励{self.patience}个episode无提升，停止训练。"
                  f"最佳奖励={self.best_reward:.4f}, 上次提升在Episode {self.last_improve_ep}")
            return False
        
        return True


class TimeLimitedRewardCallback(VecEnvRewardCallback):
    """增强的回调，支持时间限制和检查点保存，以及课程学习进度更新"""
    def __init__(self, model, checkpoint_path, checkpoint_freq, max_episodes, 
                 convergence_window=100, convergence_threshold=0.01,
                 patience=2000, min_delta=0.1, mean_min=1.0, 
                 curriculum_start_max_agents=25, curriculum_final_max_agents=100,
                 curriculum_timesteps=600000, *args, **kwargs):
        super().__init__(max_episodes=max_episodes, 
                        convergence_window=convergence_window,
                        convergence_threshold=convergence_threshold,
                        patience=patience, min_delta=min_delta, mean_min=mean_min,
                        *args, **kwargs)
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.last_checkpoint_step = 0
        
        # 课程学习参数
        self.curriculum_start_max_agents = curriculum_start_max_agents  # 初始最大agent数
        self.curriculum_final_max_agents = curriculum_final_max_agents  # 最终最大agent数
        self.curriculum_timesteps = curriculum_timesteps  # 课程学习爬坡周期（600k步）
        self.last_curriculum_update_step = 0  # 上次更新课程进度时的步数
        self.curriculum_update_freq = 50000  # 每50k步更新一次课程进度
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # 检查时间限制
        global training_time_limit
        if training_time_limit and time.time() >= training_time_limit:
            print(f"\n[时间限制] 已达到训练时间限制 ({TRAINING_TIME_LIMIT_HOURS}小时)，停止训练")
            return False
        
        # 从模型获取当前步数
        if hasattr(self.model, 'num_timesteps'):
            num_timesteps = self.model.num_timesteps
        else:
            num_timesteps = self.n_calls * PARALLEL_CONFIG["total_num_envs"]  # 使用调用次数估算
        
        # 定期保存检查点（基于步数）
        if num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq:
            checkpoint_file = f"{self.checkpoint_path}_checkpoint_step{num_timesteps}"
            try:
                self.model.save(checkpoint_file)
                print(f"[检查点] 保存检查点到: {checkpoint_file} (步数: {num_timesteps})")
                self.last_checkpoint_step = num_timesteps
            except Exception as e:
                print(f"[检查点] 保存失败: {e}")
        
        # 更新课程学习进度（基于训练进度线性增加max_agents）
        if num_timesteps - self.last_curriculum_update_step >= self.curriculum_update_freq:
            self._update_curriculum(num_timesteps)
            self.last_curriculum_update_step = num_timesteps
        
        return result
    
    def _update_curriculum(self, num_timesteps):
        """更新所有环境的课程学习进度（基于课程学习周期线性增加max_agents）"""
        # 计算课程学习进度（0.0-1.0），基于单独的curriculum_timesteps，而不是total_timesteps
        progress = min(1.0, float(num_timesteps) / float(self.curriculum_timesteps))
        
        # 线性插值计算当前max_agents
        current_max_agents = int(
            self.curriculum_start_max_agents + 
            (self.curriculum_final_max_agents - self.curriculum_start_max_agents) * progress
        )
        
        # 更新所有环境的curriculum_max_agents
        # 使用VecEnv的set_attr方法（这是SubprocVecEnv下唯一有效的方式）
        try:
            if hasattr(self.model, 'env') and hasattr(self.model.env, 'set_attr'):
                try:
                    self.model.env.set_attr('curriculum_max_agents', current_max_agents)
                    print(f"[课程学习] 课程进度 {progress*100:.1f}% ({num_timesteps}/{self.curriculum_timesteps}步), "
                          f"更新max_agents: {self.curriculum_start_max_agents} → {current_max_agents} (已同步到所有环境)")
                except Exception as set_attr_error:
                    print(f"[课程学习] 课程进度 {progress*100:.1f}% ({num_timesteps}/{self.curriculum_timesteps}步), "
                          f"更新max_agents: {self.curriculum_start_max_agents} → {current_max_agents} (set_attr失败: {set_attr_error})")
            else:
                print(f"[课程学习] 课程进度 {progress*100:.1f}% ({num_timesteps}/{self.curriculum_timesteps}步), "
                      f"更新max_agents: {self.curriculum_start_max_agents} → {current_max_agents} (env不支持set_attr)")
        except Exception as e:
            # 如果更新失败，不影响训练
            if not hasattr(self, '_curriculum_error_logged'):
                print(f"[警告] 更新课程学习进度时出错: {e}，将跳过此步骤")
                self._curriculum_error_logged = True


def create_parallel_envs(initial_max_agents=25):
    """创建并行环境（修复子进程缓存问题）
    
    参数:
        initial_max_agents: 初始课程学习阶段的最大agent数，只创建有符合条件任务的地图的env
    """
    env_constructors = []
    env_id = 0
    
    # 按地图类型分配环境
    # 在闭包中捕获map_path和training_files，避免子进程缓存问题
    for map_config in MAP_CONFIGS:
        map_name = map_config["name"]
        map_path = get_abs_path(map_config["map_path"])
        train_task_dir = get_abs_path(map_config["train_task_dir"])
        
        # 检查路径是否存在
        if not os.path.exists(map_path):
            print(f"[警告] 地图文件未找到: {map_path}，跳过该地图")
            continue
        
        if not os.path.exists(train_task_dir):
            print(f"[警告] 训练任务目录未找到: {train_task_dir}，跳过该地图")
            continue
        
        # 加载任务文件列表（在子进程中也会重新扫描，但这里先加载用于验证）
        training_files = [
            os.path.join(train_task_dir, fname)
            for fname in os.listdir(train_task_dir)
            if fname.endswith('.xml') and fname != ".DS_Store"
        ]
        
        if len(training_files) == 0:
            print(f"[警告] 地图 {map_name} 没有可用的训练任务，跳过")
            continue
        
        # 课程学习：预先创建所有地图的env（包括hard map）
        # 在reset()时，如果当前课程阶段不允许，会返回空episode（立即done），不会真正训练
        # 当课程进度允许时，正常采样任务并训练
        import xml.etree.ElementTree as ET
        agent_counts = []
        for f in training_files:
            try:
                tree = ET.parse(f)
                root = tree.getroot()
                agents = root.findall('.//agent')
                agent_count = len(agents)
                agent_counts.append(agent_count)
            except:
                continue
        
        # 统计符合初始阶段的任务数量（用于日志）
        if agent_counts:
            valid_count = sum(1 for c in agent_counts if c <= initial_max_agents)
            min_agents = min(agent_counts)
            max_agents = max(agent_counts)
            if valid_count == 0:
                print(f"[课程学习] 地图 {map_name}: agent范围 {min_agents}-{max_agents}, "
                      f"所有任务都超过初始max_agents({initial_max_agents})，"
                      f"env已创建但会在早期返回空episode（课程进度允许时自动激活）")
            else:
                print(f"[课程学习] 地图 {map_name}: agent范围 {min_agents}-{max_agents}, "
                      f"{valid_count}/{len(training_files)}个任务符合初始阶段")
        
        num_envs = PARALLEL_CONFIG["num_envs_per_map"]
        
        for i in range(num_envs):
            # 在闭包中捕获map_path和train_task_dir，确保子进程可以访问
            env_constructors.append(create_env_for_map(map_path, train_task_dir, env_id))
            env_id += 1
        
        valid_count = sum(1 for c in agent_counts if c <= initial_max_agents) if agent_counts else len(training_files)
        print(f"[信息] 地图 {map_name}: {num_envs} 个环境，{len(training_files)} 个任务文件，"
              f"其中{valid_count}个符合初始课程阶段(<= {initial_max_agents} agents)")
    
    if len(env_constructors) == 0:
        raise RuntimeError("没有可用的环境构造函数，请检查地图配置")
    
    print(f"\n[信息] 创建 {len(env_constructors)} 个并行环境")
    
    # 创建并行环境（使用fork方法以确保子进程可以访问主进程的变量）
    # 注意：如果fork不可用，会fallback到spawn，此时需要确保闭包中捕获了所有必要信息
    try:
        env = SubprocVecEnv(env_constructors, start_method='fork')
    except RuntimeError:
        # 如果fork不可用（例如在Windows上），使用spawn
        print("[警告] fork方法不可用，使用spawn方法（闭包中已捕获必要信息）")
        env = SubprocVecEnv(env_constructors, start_method='spawn')
    
    # 使用VecMonitor包装环境，以便正确统计episode信息
    # 保存monitor.csv以便后续分析（可选，如果不需要可以设为None）
    monitor_file = os.path.join(SCRIPT_DIR, "logs", "monitor.csv")
    os.makedirs(os.path.dirname(monitor_file), exist_ok=True)
    # episode是VecMonitor自动生成的dict字段，不应放在info_keywords中
    env = VecMonitor(env, filename=monitor_file, info_keywords=())
    
    return env


def train_parallel():
    """执行并行训练"""
    global training_start_time, training_time_limit
    
    training_start_time = time.time()
    training_time_limit = training_start_time + TRAINING_TIME_LIMIT_HOURS * 3600
    
    print("\n" + "=" * 80)
    print("PPO提升CCBS冲突选择的并行训练")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练时间限制: {TRAINING_TIME_LIMIT_HOURS}小时")
    print(f"结束时间限制: {datetime.fromtimestamp(training_time_limit).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 加载地图配置
    print("\n[步骤1] 加载地图配置...")
    load_map_configs()
    
    if len(map_configs_cache) == 0:
        print("[错误] 没有可用的地图配置，训练终止")
        return False
    
    print(f"[信息] 成功加载 {len(map_configs_cache)} 种地图类型")
    
    # 创建并行环境
    print("\n[步骤2] 创建并行环境...")
    env = create_parallel_envs()
    
    # 创建PPO模型
    print("\n[步骤3] 创建PPO模型...")
    # 确保TensorBoard日志目录存在
    tensorboard_log_dir = os.path.join(SCRIPT_DIR, "logs", "tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    model = PPO(
        "MultiInputPolicy",  # 使用MultiInputPolicy因为观察空间是Dict
        env=env,
        learning_rate=PPO_CONFIG["learning_rate"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        verbose=1,
        tensorboard_log=tensorboard_log_dir
    )
    
    print(f"[信息] PPO模型创建成功")
    print(f"[信息] TensorBoard日志目录: {tensorboard_log_dir}")
    print(f"[信息] Monitor CSV文件: {os.path.join(SCRIPT_DIR, 'logs', 'monitor.csv')}")
    print(f"  - 学习率: {PPO_CONFIG['learning_rate']}")
    print(f"  - 折扣因子: {PPO_CONFIG['gamma']}")
    print(f"  - 每环境采样步数: {PPO_CONFIG['n_steps']}")
    print(f"  - 批大小: {PPO_CONFIG['batch_size']}")
    print(f"  - 优化轮数: {PPO_CONFIG['n_epochs']}")
    
    # 准备回调
    print("\n[步骤4] 准备训练回调...")
    model_save_path = os.path.join(SCRIPT_DIR, "ppo_ccbs_multi_map")
    
    # 使用TimeLimitedRewardCallback处理回合数限制、时间限制和检查点保存
    # 注意：删除了CheckpointCallback，因为它的save_freq是按callback调用次数（在20个并行env下200000次≈400万timesteps，超过总步数200万）
    # TimeLimitedRewardCallback已实现按model.num_timesteps的保存逻辑，更准确
    reward_cb = TimeLimitedRewardCallback(
        model=model,
        checkpoint_path=model_save_path,
        checkpoint_freq=PPO_CONFIG["checkpoint_freq"],
        max_episodes=PPO_CONFIG["max_episodes"],
        convergence_window=100,
        convergence_threshold=0.01,
        curriculum_start_max_agents=25,  # 初始最大agent数
        curriculum_final_max_agents=100,  # 最终最大agent数（覆盖所有任务）
        curriculum_timesteps=600000  # 课程学习爬坡周期（600k步内完成，剩余1.4M步全量训练）
    )
    
    callbacks = reward_cb  # 只使用TimeLimitedRewardCallback，它已经包含了检查点保存功能
    
    # 开始训练
    print("\n[步骤5] 开始训练...")
    print("=" * 80)
    print(f"[提示] 训练日志将同时输出到控制台和TensorBoard")
    print(f"[提示] 实时监控方法：")
    print(f"  1. TensorBoard: tensorboard --logdir {tensorboard_log_dir} --host 127.0.0.1 --port 6006")
    print(f"  2. 日志文件: tail -f logs/train_stdout.log (如果使用tee重定向)")
    print(f"  3. Monitor CSV: tail -f logs/monitor.csv")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=PPO_CONFIG["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[中断] 收到中断信号，保存进度...")
    except Exception as e:
        print(f"\n[错误] 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存最终模型
    print("\n[步骤6] 保存最终模型...")
    try:
        final_model_path = f"{model_save_path}_final"
        model.save(final_model_path)
        print(f"[信息] 最终模型已保存到: {final_model_path}")
    except Exception as e:
        print(f"[错误] 保存最终模型失败: {e}")
    
    # 保存奖励记录
    if hasattr(reward_cb, 'rewards') and len(reward_cb.rewards) > 0:
        print("\n[步骤7] 保存训练记录...")
        try:
            import pandas as pd
            rewards_path = os.path.join(SCRIPT_DIR, "rewards_ppo_multi_map.csv")
            df = pd.DataFrame({
                'episode': range(1, len(reward_cb.rewards) + 1),
                'reward': reward_cb.rewards
            })
            df.to_csv(rewards_path, index=False)
            print(f"[信息] 奖励记录已保存到: {rewards_path}")
            print(f"  - 总回合数: {len(reward_cb.rewards)}")
            if len(reward_cb.rewards) > 0:
                print(f"  - 最佳奖励: {reward_cb.best_reward:.4f}")
                print(f"  - 最终平均奖励: {np.mean(reward_cb.rewards[-100:]):.4f}")
        except Exception as e:
            print(f"[错误] 保存奖励记录失败: {e}")
    
    # 训练总结
    total_time = time.time() - training_start_time
    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    if hasattr(reward_cb, 'episode_count'):
        print(f"总回合数: {reward_cb.episode_count}")
    print("=" * 80)
    
    # 关闭环境
    env.close()
    
    return True


def main():
    """主函数"""
    try:
        train_parallel()
    except KeyboardInterrupt:
        print("\n[中断] 收到中断信号，退出")
    except Exception as e:
        print(f"\n[错误] 程序异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

