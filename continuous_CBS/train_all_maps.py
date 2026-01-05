# -*- coding: utf-8 -*-
"""
多地图类型训练脚本
支持对所有地图类型进行训练，包含收敛检测、检查点保存和时间限制
"""
import os
import sys
import time
import copy
import pandas as pd
from datetime import datetime, timedelta
import signal

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ccbs import CCBS
from ccbsenv import CCBSEnv, RewardCallback
from config import Config
from map import Map
from structs import Task, Solution
from stable_baselines3 import PPO

# ============================================================================
# ============================ 配置区域 ======================================
# ============================================================================

# 训练时间限制（小时）
TRAINING_TIME_LIMIT_HOURS = 88

# 地图配置列表
MAP_CONFIGS = [
    {
        "name": "empty-16-16-random",
        "map_path": "instances/empty-16-16-random/map.xml",
        "train_task_dir": "instances/empty-16-16-random/train",
        "model_save_path": "ppo_empty-16-16-random",
        "rewards_save_path": "rewards_empty-16-16-random.csv",
        "skip": False,  # 如果已经有模型，可以设置为True跳过
    },
    {
        "name": "room-64-64-8_random",
        "map_path": "instances/room-64-64-8_random/map.xml",
        "train_task_dir": "instances/room-64-64-8_random/train",
        "model_save_path": "ppo_room-64-64-8_random",
        "rewards_save_path": "rewards_room-64-64-8_random.csv",
        "skip": False,
    },
    {
        "name": "den520d_random",
        "map_path": "instances/den520d_random/map.xml",
        "train_task_dir": "instances/den520d_random/train",
        "model_save_path": "ppo_den520d_random",
        "rewards_save_path": "rewards_den520d_random.csv",
        "skip": False,
    },
    {
        "name": "warehouse-10-20-random",
        "map_path": "instances/warehouse-10-20-random/map.xml",
        "train_task_dir": "instances/warehouse-10-20-random/train",
        "model_save_path": "ppo_warehouse-10-20-random",
        "rewards_save_path": "rewards_warehouse-10-20-random.csv",
        "skip": False,
    },
]

# PPO训练配置
PPO_CONFIG = {
    "learning_rate": 2e-4,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps_per_env": 10000,
    "max_episodes": 1000,
    "convergence_window": 100,  # 收敛检测窗口
    "convergence_threshold": 0.01,  # 收敛阈值
    "checkpoint_interval": 50,  # 检查点保存间隔（episode数）
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
    "verbose": True,
}

# 环境配置
RL_ENV_CONFIG = {
    "max_step": 1024,
    "reward_1": 10,
    "reward_2": 2,
    "reward_3": -5,
    "reward_iter_pos": -0.5,
    "cost_weight": 0.01,
    "cardinal_conflicts_weight": 1,
    "semicard_conflicts_weight": 1,
    "non_cardinal_conflict_weight": 1,
    "max_process_agent": 100,
}

# 全局变量用于时间控制
training_start_time = None
training_time_limit = None


def get_abs_path(rel_path):
    """将相对路径转换为绝对路径"""
    if rel_path is None:
        return None
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(SCRIPT_DIR, rel_path)


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
    env.reward_1 = config["reward_1"]
    env.reward_2 = config["reward_2"]
    env.reward_3 = config["reward_3"]
    env.reward_iter_pos = config["reward_iter_pos"]
    env.cost_weight = config["cost_weight"]
    env.cardinal_conflicts_weight = config["cardinal_conflicts_weight"]
    env.semicard_conflicts_weight = config["semicard_conflicts_weight"]
    env.non_cardinal_conflict_weight = config["non_cardinal_conflict_weight"]
    env.max_process_agent = config["max_process_agent"]


class CheckpointCallback(RewardCallback):
    """增强的回调，支持检查点保存"""
    def __init__(self, model, checkpoint_path, checkpoint_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # 定期保存检查点
        if (self.episode_count - self.last_checkpoint_episode) >= self.checkpoint_interval:
            checkpoint_file = f"{self.checkpoint_path}_checkpoint_ep{self.episode_count}"
            try:
                self.model.save(checkpoint_file)
                print(f"[检查点] 保存检查点到: {checkpoint_file}")
                self.last_checkpoint_episode = self.episode_count
            except Exception as e:
                print(f"[检查点] 保存失败: {e}")
        
        # 检查时间限制
        if training_time_limit and time.time() >= training_time_limit:
            print(f"\n[时间限制] 已达到训练时间限制 ({TRAINING_TIME_LIMIT_HOURS}小时)，停止训练")
            return False
        
        return result


def train_single_map(map_config, log_file=None):
    """训练单个地图类型"""
    map_name = map_config["name"]
    print("\n" + "=" * 80)
    print(f"开始训练地图: {map_name}")
    print("=" * 80)
    
    if log_file:
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"开始训练地图: {map_name}\n")
        log_file.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*80}\n")
        log_file.flush()
    
    # 加载地图
    map_path = get_abs_path(map_config["map_path"])
    if not os.path.exists(map_path):
        error_msg = f"地图文件未找到: {map_path}"
        print(f"[错误] {error_msg}")
        if log_file:
            log_file.write(f"[错误] {error_msg}\n")
            log_file.flush()
        return False
    
    print(f"加载地图: {map_path}")
    world_map = Map(map_path)
    
    # 创建CCBS对象
    origin_ccbs = CCBS(world_map)
    apply_config_to_ccbs(origin_ccbs, CCBS_CONFIG)
    
    # 加载训练任务
    train_dir = get_abs_path(map_config["train_task_dir"])
    if not os.path.exists(train_dir):
        error_msg = f"训练任务目录未找到: {train_dir}"
        print(f"[错误] {error_msg}")
        if log_file:
            log_file.write(f"[错误] {error_msg}\n")
            log_file.flush()
        return False
    
    training_files = [
        os.path.join(train_dir, fname)
        for fname in os.listdir(train_dir)
        if fname.endswith('.xml') and fname != ".DS_Store"
    ]
    print(f"找到 {len(training_files)} 个训练任务")
    
    # 创建环境列表
    all_envs = []
    config = Config()
    config.use_precalculated_heuristic = CCBS_CONFIG["use_precalculated_heuristic"]
    
    for i, task_file in enumerate(training_files, 1):
        try:
            task = Task()
            task.load_from_file(task_file)
            ccbs = copy.deepcopy(origin_ccbs)
            
            if CCBS_CONFIG["use_precalculated_heuristic"]:
                ccbs.map.init_heuristic(task.agents)
            
            ccbs.solution = Solution()
            
            # 初始化根节点
            if not ccbs.init_root(task):
                print(f"任务 {i}: 无法找到根节点解，跳过")
                continue
            
            if len(ccbs.tree.container) == 0:
                print(f"任务 {i}: 无解，跳过")
                continue
            
            parent = ccbs.tree.get_front()
            if parent.conflicts_num == 0:
                print(f"任务 {i}: 根节点无冲突，跳过")
                continue
            
            env = CCBSEnv(
                task, parent, world_map,
                expanded=1,
                time_elapsed=0,
                low_level_searches=0,
                low_level_expanded=0,
                tree=ccbs.tree
            )
            apply_config_to_env(env, RL_ENV_CONFIG)
            all_envs.append(env)
            print(f"任务 {i}/{len(training_files)}: 环境创建完成")
        except Exception as e:
            print(f"任务 {i}: 创建环境失败: {e}")
            continue
    
    if len(all_envs) == 0:
        error_msg = "错误: 没有可用的训练环境"
        print(f"[错误] {error_msg}")
        if log_file:
            log_file.write(f"[错误] {error_msg}\n")
            log_file.flush()
        return False
    
    print(f"\n总共创建 {len(all_envs)} 个训练环境")
    
    # 创建PPO模型
    print("\n创建PPO模型...")
    model = PPO(
        "MultiInputPolicy",
        env=all_envs[0],
        learning_rate=PPO_CONFIG["learning_rate"],
        gamma=PPO_CONFIG["gamma"],
        verbose=PPO_CONFIG.get("verbose", 1),
        ent_coef=PPO_CONFIG["ent_coef"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"]
    )
    
    # 创建增强的回调（包含收敛检测和检查点保存）
    model_save_path = get_abs_path(map_config["model_save_path"])
    callback = CheckpointCallback(
        model=model,
        checkpoint_path=model_save_path,
        checkpoint_interval=PPO_CONFIG["checkpoint_interval"],
        max_episodes=PPO_CONFIG["max_episodes"],
        convergence_window=PPO_CONFIG["convergence_window"],
        convergence_threshold=PPO_CONFIG["convergence_threshold"]
    )
    
    # 开始训练
    print("\n开始训练...")
    map_start_time = time.time()
    e_num = 1
    
    for env in all_envs:
        if training_time_limit and time.time() >= training_time_limit:
            print(f"\n[时间限制] 已达到训练时间限制，停止训练地图 {map_name}")
            break
        
        print(f"\n{'='*60}")
        print(f"训练环境 {e_num}/{len(all_envs)} (地图: {map_name})")
        print(f"{'='*60}")
        
        if log_file:
            log_file.write(f"\n训练环境 {e_num}/{len(all_envs)}\n")
            log_file.flush()
        
        try:
            model.set_env(env)
            model.learn(
                total_timesteps=PPO_CONFIG["total_timesteps_per_env"],
                callback=callback
            )
        except Exception as e:
            print(f"[错误] 训练环境 {e_num} 时出错: {e}")
            if log_file:
                log_file.write(f"[错误] 训练环境 {e_num} 时出错: {e}\n")
                log_file.flush()
            continue
        
        e_num += 1
    
    map_training_time = time.time() - map_start_time
    print(f"\n{'='*60}")
    print(f"地图 {map_name} 训练完成!")
    print(f"训练耗时: {map_training_time:.2f}秒 ({map_training_time/60:.2f}分钟)")
    print(f"总回合数: {callback.episode_count}")
    print(f"最终收敛状态: {'已收敛' if callback.converged else '未收敛'}")
    if callback.rewards:
        print(f"最佳奖励: {callback.best_reward:.4f}")
        print(f"最终平均奖励: {np.mean(callback.rewards[-100:]):.4f}")
    print(f"{'='*60}")
    
    if log_file:
        log_file.write(f"\n地图 {map_name} 训练完成!\n")
        log_file.write(f"训练耗时: {map_training_time:.2f}秒 ({map_training_time/60:.2f}分钟)\n")
        log_file.write(f"总回合数: {callback.episode_count}\n")
        log_file.write(f"最终收敛状态: {'已收敛' if callback.converged else '未收敛'}\n")
        if callback.rewards:
            log_file.write(f"最佳奖励: {callback.best_reward:.4f}\n")
            log_file.write(f"最终平均奖励: {np.mean(callback.rewards[-100:]):.4f}\n")
        log_file.flush()
    
    # 保存模型
    print(f"\n保存模型到: {model_save_path}")
    try:
        model.save(model_save_path)
        print(f"模型保存成功")
    except Exception as e:
        print(f"[错误] 模型保存失败: {e}")
        if log_file:
            log_file.write(f"[错误] 模型保存失败: {e}\n")
            log_file.flush()
        return False
    
    # 保存奖励记录
    if callback.rewards:
        rewards_path = get_abs_path(map_config["rewards_save_path"])
        print(f"保存奖励记录到: {rewards_path}")
        try:
            df = pd.DataFrame({
                'episode': range(1, len(callback.rewards) + 1),
                'reward': callback.rewards
            })
            df.to_csv(rewards_path, index=False)
            print(f"奖励统计: 最小={df['reward'].min():.2f}, 最大={df['reward'].max():.2f}, "
                  f"平均={df['reward'].mean():.2f}, 标准差={df['reward'].std():.2f}")
        except Exception as e:
            print(f"[错误] 奖励记录保存失败: {e}")
            if log_file:
                log_file.write(f"[错误] 奖励记录保存失败: {e}\n")
                log_file.flush()
    
    return True


def main():
    """主函数"""
    global training_start_time, training_time_limit
    
    training_start_time = time.time()
    training_time_limit = training_start_time + TRAINING_TIME_LIMIT_HOURS * 3600
    
    # 创建日志文件
    log_file_path = os.path.join(SCRIPT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    print("\n" + "=" * 80)
    print("多地图类型训练脚本")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练时间限制: {TRAINING_TIME_LIMIT_HOURS}小时")
    print(f"结束时间限制: {datetime.fromtimestamp(training_time_limit).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file_path}")
    print("=" * 80)
    
    log_file.write("多地图类型训练脚本\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"训练时间限制: {TRAINING_TIME_LIMIT_HOURS}小时\n")
    log_file.write(f"结束时间限制: {datetime.fromtimestamp(training_time_limit).strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.flush()
    
    # 训练每个地图
    results = {}
    for map_config in MAP_CONFIGS:
        if map_config.get("skip", False):
            print(f"\n跳过地图: {map_config['name']} (配置为跳过)")
            continue
        
        elapsed_time = time.time() - training_start_time
        remaining_time = training_time_limit - time.time()
        
        print(f"\n[进度] 已用时间: {elapsed_time/3600:.2f}小时, 剩余时间: {remaining_time/3600:.2f}小时")
        
        if time.time() >= training_time_limit:
            print(f"\n[时间限制] 已达到训练时间限制，停止训练")
            break
        
        try:
            success = train_single_map(map_config, log_file)
            results[map_config["name"]] = success
        except KeyboardInterrupt:
            print("\n[中断] 收到中断信号，保存进度并退出...")
            log_file.write("\n[中断] 收到中断信号\n")
            log_file.flush()
            break
        except Exception as e:
            print(f"\n[错误] 训练地图 {map_config['name']} 时发生异常: {e}")
            log_file.write(f"\n[错误] 训练地图 {map_config['name']} 时发生异常: {e}\n")
            log_file.flush()
            results[map_config["name"]] = False
            continue
    
    # 总结
    total_time = time.time() - training_start_time
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    print(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    print(f"完成的地图:")
    for map_name, success in results.items():
        status = "成功" if success else "失败"
        print(f"  - {map_name}: {status}")
    print("=" * 80)
    
    log_file.write("\n训练总结\n")
    log_file.write(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)\n")
    log_file.write(f"完成的地图:\n")
    for map_name, success in results.items():
        status = "成功" if success else "失败"
        log_file.write(f"  - {map_name}: {status}\n")
    log_file.close()
    
    print(f"\n日志已保存到: {log_file_path}")


if __name__ == "__main__":
    import numpy as np
    main()

