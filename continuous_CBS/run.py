# -*- coding: utf-8 -*-
"""
CCBS 项目统一运行脚本
支持训练强化学习模型、使用模型求解、评估性能等多种模式
可在脚本中配置所有超参数和设定
"""
import os
import time
import copy
import pandas as pd
from typing import List, Optional

# 获取脚本所在目录，用于路径解析
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from ccbs import CCBS
from ccbsenv import CCBSEnv, RewardCallback
from config import Config
from map import Map
from structs import Task, Solution, Agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Animation import GraphRender


# ============================================================================
# ============================ 配置区域 ======================================
# ============================================================================

# -------------------- 运行模式配置 --------------------
RUN_MODE = "evaluate"  # 可选: "train", "solve", "evaluate", "batch_test"

# -------------------- CCBS 算法配置 --------------------
CCBS_CONFIG = {
    # 智能体参数
    "agent_size": 0.5,                    # 智能体尺寸 (0, 0.5]
    
    # 高级启发式参数
    "hlh_type": 2,                        # 0-无hlh, 1-使用simplex求解lpp, 2-贪婪选择不相交冲突
    "precision": 0.1,                     # 等待时间确定精度
    "timelimit": 300,                     # 时间限制（秒）- 可以修改此值来改变时间限制
    
    # 启发式配置
    "use_precalculated_heuristic": False, # True: 使用反向Dijkstra, False: 使用欧几里得距离
    
    # 冲突处理配置
    "use_disjoint_splitting": True,       # 使用不相交分割
    "use_cardinal": True,                 # 优先处理cardinal冲突，然后是semi-cardinal，最后是普通冲突
    "use_corridor_symmetry": False,       # 使用走廊对称性
    "use_target_symmetry": False,         # 使用目标对称性
    
    # 强化学习配置
    "use_rl": True,                       # True: 使用PPO进行CCBS分支策略, False: 使用传统方法
    "rl_model_path": None,                # RL模型路径（None表示自动根据地图选择，也可以手动指定）
    
    # 输出配置
    "verbose": True,                      # 是否显示详细信息
}

# -------------------- 地图和任务配置 --------------------
MAP_AND_TASK_CONFIG = {
    # 地图路径（相对于脚本目录）
    "map_path": "instances/roadmaps/sparse/map.xml",
    
    # 单个任务路径（solve模式使用，相对于脚本目录）
    "task_path": "instances/roadmaps/sparse/test/10/1_task-10.xml",
    
    # 训练任务目录（train模式使用，相对于脚本目录）
    "train_task_dir": "instances/roadmaps/sparse/train",
    
    # 测试任务目录（evaluate/batch_test模式使用，相对于脚本目录）
    "test_task_dir": "instances/roadmaps/sparse/test",
    
    # 测试任务子目录（如果test_task_dir下有子目录，如test/10/，设置为["10"]）
    "test_subdirs": None,  # None表示测试test_task_dir下所有xml文件，或指定如["10", "20"]
}

# -------------------- PPO 训练配置 --------------------
PPO_TRAIN_CONFIG = {
    # 学习率参数
    "learning_rate": 2e-4,                # 学习率
    
    # 折扣因子
    "gamma": 0.98,                        # 折扣因子
    
    # GAE参数
    "gae_lambda": 0.95,                   # GAE lambda参数
    
    # PPO裁剪参数
    "clip_range": 0.2,                    # PPO裁剪范围
    
    # 熵系数
    "ent_coef": 0.01,                     # 熵系数（鼓励探索）
    
    # 训练步数
    "total_timesteps_per_env": 10000,     # 每个环境的训练步数
    
    # 最大回合数
    "max_episodes": 1000,                 # 最大训练回合数
    
    # 模型保存路径
    "model_save_path": "ppo_road-sparse", # 保存的模型名称（不含扩展名）
    
    # 奖励保存路径
    "rewards_save_path": "rewards_road-sparse.csv",  # 奖励记录CSV文件路径
    
    # 详细输出
    "verbose": 1,                         # 0-无输出, 1-基本信息, 2-详细输出
}

# -------------------- 强化学习环境配置 --------------------
RL_ENV_CONFIG = {
    # 最大步数
    "max_step": 1024,                     # 最大搜索步数
    
    # 奖励参数
    "reward_1": 10,                       # 当前节点满足约束且无其他冲突的奖励
    "reward_2": 2,                        # 分支数量权重系数
    "reward_3": -5,                       # 当前节点不满足约束的惩罚
    "reward_iter_pos": -0.5,              # 迭代位置奖励
    "cost_weight": 0.01,                  # 目标函数权重
    
    # 冲突权重
    "cardinal_conflicts_weight": 1,       # cardinal冲突权重
    "semicard_conflicts_weight": 1,       # semi-cardinal冲突权重
    "non_cardinal_conflict_weight": 1,    # non-cardinal冲突权重
    
    # 最大处理的智能体数
    "max_process_agent": 100,
}

# -------------------- 求解配置 --------------------
SOLVE_CONFIG = {
    # 是否保存解决方案
    "save_solution": False,               # 是否保存解决方案到文件
    "solution_output_path": "test_out.xml",  # 解决方案输出路径
    
    # 是否可视化
    "visualize": True,                   # 是否图形化显示结果
    
    # 是否打印详细路径信息
    "print_paths": True,                  # 是否打印每个智能体的路径信息
}

# -------------------- 批量测试配置 --------------------
BATCH_TEST_CONFIG = {
    # 是否使用多进程
    "use_multiprocessing": False,         # 是否使用多进程加速
    "num_processes": 10,                  # 进程数（如果使用多进程）
    
    # 结果保存
    "save_results": True,                 # 是否保存结果
    "results_output_path": "batch_test_results.csv",  # 结果输出CSV路径
}

# ============================================================================
# ============================ 工具函数 ======================================
# ============================================================================

# 地图到模型的映射表（自动匹配）
MAP_TO_MODEL_MAPPING = {
    "roadmaps/sparse": "ppo_road-sparse.zip",
    "roadmaps/dense": "ppo_road-sparse.zip",  # 如果没有专门的dense模型，使用sparse的
    "roadmaps/super-dense": "ppo_road-sparse.zip",  # 如果没有专门的super-dense模型，使用sparse的
    "empty-16-16-random": "ppo_empty-16-16-random.zip",
    "room-64-64-8_random": None,  # 如果没有训练好的模型，设置为None
    "den520d_random": None,
    "warehouse-10-20-random": None,
}


def get_abs_path(rel_path: Optional[str]) -> Optional[str]:
    """将相对路径转换为基于脚本目录的绝对路径"""
    if rel_path is None:
        return None
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(SCRIPT_DIR, rel_path)


def auto_select_model_path(map_path: str, manual_model_path: Optional[str] = None) -> Optional[str]:
    """
    根据地图路径自动选择合适的模型路径
    
    Args:
        map_path: 地图路径（如 "instances/roadmaps/sparse/map.xml"）
        manual_model_path: 手动指定的模型路径（如果提供了，优先使用）
    
    Returns:
        模型路径（相对路径），如果找不到返回None
    """
    # 如果手动指定了模型路径，优先使用
    if manual_model_path:
        return manual_model_path
    
    # 从地图路径中提取地图类型
    # 例如: "instances/roadmaps/sparse/map.xml" -> "roadmaps/sparse"
    path_parts = map_path.replace("\\", "/").split("/")
    
    # 查找地图标识符
    map_key = None
    for key in MAP_TO_MODEL_MAPPING:
        # 检查路径中是否包含地图键
        if key in map_path:
            map_key = key
            break
    
    # 如果在映射表中找到，返回对应的模型路径
    if map_key and MAP_TO_MODEL_MAPPING[map_key]:
        return MAP_TO_MODEL_MAPPING[map_key]
    
    # 如果找不到，尝试根据路径结构推断
    # 提取最后两个目录层级作为地图标识
    if len(path_parts) >= 2:
        inferred_key = "/".join(path_parts[-3:-1]) if len(path_parts) >= 3 else path_parts[-2]
        for key in MAP_TO_MODEL_MAPPING:
            if key in inferred_key:
                if MAP_TO_MODEL_MAPPING[key]:
                    return MAP_TO_MODEL_MAPPING[key]
    
    # 如果都找不到，返回None
    return None


def resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    """解析模型路径，支持.zip和文件夹两种格式"""
    if model_path is None:
        return None
    
    abs_model_path = get_abs_path(model_path)
    if abs_model_path is None:
        return None
    
    # 尝试多个候选路径
    candidates = [abs_model_path]
    base, ext = os.path.splitext(abs_model_path)
    if ext == "":
        # 如果没有扩展名，尝试添加.zip
        candidates.append(f"{abs_model_path}.zip")
    else:
        # 如果有扩展名，尝试去掉扩展名（可能是文件夹）
        candidates.append(base)
    
    # 查找存在的路径
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    
    # 如果都没找到，返回原始路径（让后续代码报错）
    raise FileNotFoundError(
        f"RL模型未找到。尝试的路径: {', '.join(candidates)}\n"
        f"请检查模型路径是否正确: {model_path}"
    )


# ============================================================================
# ============================ 主程序 ========================================
# ============================================================================

def apply_config_to_ccbs(ccbs: CCBS, config: dict, map_path: Optional[str] = None):
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
    
    # 自动选择或解析模型路径
    if config["use_rl"]:
        # 如果没有手动指定模型路径，尝试根据地图自动选择
        if not config["rl_model_path"] and map_path:
            auto_selected = auto_select_model_path(map_path)
            if auto_selected:
                config["rl_model_path"] = auto_selected
                if config["verbose"]:
                    print(f"根据地图自动选择模型: {map_path} -> {auto_selected}")
            else:
                # 找不到匹配的模型，报错
                raise FileNotFoundError(
                    f"错误: 无法为地图 '{map_path}' 自动选择合适的RL模型。\n"
                    f"请手动在 CCBS_CONFIG 中设置 'rl_model_path'，或确保对应的模型文件存在。"
                )
        
        # 解析模型路径为绝对路径
        if config["rl_model_path"]:
            try:
                resolved_path = resolve_model_path(config["rl_model_path"])
                if resolved_path:
                    ccbs.config.rl_model_path = resolved_path
                    if config["verbose"]:
                        print(f"解析模型路径: {config['rl_model_path']} -> {resolved_path}")
                else:
                    raise FileNotFoundError(
                        f"错误: 无法解析RL模型路径 '{config['rl_model_path']}'"
                    )
            except FileNotFoundError as e:
                # 模型文件不存在，报错并停止
                raise FileNotFoundError(
                    f"错误: RL模型未找到或无法加载。\n"
                    f"模型路径: {config['rl_model_path']}\n"
                    f"详细信息: {str(e)}\n"
                    f"请检查模型文件是否存在，或修改 CCBS_CONFIG['rl_model_path'] 配置。"
                )
        else:
            # 未指定模型路径，报错
            raise ValueError(
                f"错误: use_rl=True 但未指定RL模型路径。\n"
                f"请在 CCBS_CONFIG 中设置 'rl_model_path'，或设置为 None 以自动选择。"
            )
    else:
        ccbs.config.rl_model_path = None
    ccbs.verbose = config["verbose"]


def apply_config_to_env(env: CCBSEnv, config: dict):
    """将配置应用到强化学习环境"""
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


def train_model():
    """训练PPO模型"""
    print("=" * 60)
    print("开始训练PPO模型")
    print("=" * 60)
    
    # 加载地图
    map_path = get_abs_path(MAP_AND_TASK_CONFIG["map_path"])
    print(f"加载地图: {map_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"地图文件未找到: {map_path}")
    world_map = Map(map_path)
    
    # 创建CCBS对象
    origin_ccbs = CCBS(world_map)
    apply_config_to_ccbs(origin_ccbs, CCBS_CONFIG, map_path)
    
    # 加载训练任务
    train_dir = get_abs_path(MAP_AND_TASK_CONFIG["train_task_dir"])
    print(f"加载训练任务目录: {train_dir}")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练任务目录未找到: {train_dir}")
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
        task = Task()
        task.load_from_file(task_file)
        ccbs = copy.deepcopy(origin_ccbs)
        
        if CCBS_CONFIG["use_precalculated_heuristic"]:
            heuristic_start_time = time.time()
            ccbs.map.init_heuristic(task.agents)
            print(f"任务 {i}: 启发式计算耗时 {time.time() - heuristic_start_time:.2f}s")
        
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
    
    if len(all_envs) == 0:
        print("错误: 没有可用的训练环境")
        return
    
    print(f"\n总共创建 {len(all_envs)} 个训练环境")
    
    # 创建PPO模型
    print("\n创建PPO模型...")
    model = PPO(
        "MultiInputPolicy",
        env=all_envs[0],
        learning_rate=PPO_TRAIN_CONFIG["learning_rate"],
        gamma=PPO_TRAIN_CONFIG["gamma"],
        verbose=PPO_TRAIN_CONFIG["verbose"],
        ent_coef=PPO_TRAIN_CONFIG["ent_coef"],
        gae_lambda=PPO_TRAIN_CONFIG["gae_lambda"],
        clip_range=PPO_TRAIN_CONFIG["clip_range"]
    )
    
    # 训练回调
    reward_callback = RewardCallback(max_episodes=PPO_TRAIN_CONFIG["max_episodes"])
    
    # 开始训练
    print("\n开始训练...")
    start_time = time.time()
    e_num = 1
    for env in all_envs:
        print(f"\n{'='*60}")
        print(f"训练环境 {e_num}/{len(all_envs)}")
        print(f"{'='*60}")
        model.set_env(env)
        model.learn(
            total_timesteps=PPO_TRAIN_CONFIG["total_timesteps_per_env"],
            callback=reward_callback
        )
        e_num += 1
    
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成! 总耗时: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
    print(f"{'='*60}")
    
    # 保存模型
    model_save_path = get_abs_path(PPO_TRAIN_CONFIG["model_save_path"])
    print(f"\n保存模型到: {model_save_path}")
    model.save(model_save_path)
    
    # 保存奖励记录
    if reward_callback.rewards:
        rewards_path = get_abs_path(PPO_TRAIN_CONFIG["rewards_save_path"])
        print(f"保存奖励记录到: {rewards_path}")
        df = pd.DataFrame(reward_callback.rewards)
        df.to_csv(rewards_path, index=False)
        print(f"奖励统计: 最小={df.min().values[0]:.2f}, 最大={df.max().values[0]:.2f}, "
              f"平均={df.mean().values[0]:.2f}")


def solve_task():
    """使用训练好的模型求解单个任务"""
    print("=" * 60)
    print("使用CCBS求解任务")
    print("=" * 60)
    
    # 加载地图
    map_path = get_abs_path(MAP_AND_TASK_CONFIG["map_path"])
    print(f"加载地图: {map_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"地图文件未找到: {map_path}")
    world_map = Map(map_path)
    
    # 创建CCBS求解器
    solver = CCBS(world_map)
    apply_config_to_ccbs(solver, CCBS_CONFIG, map_path)
    solver.solution = Solution()
    
    # 加载任务
    task_path = get_abs_path(MAP_AND_TASK_CONFIG["task_path"])
    print(f"加载任务: {task_path}")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"任务文件未找到: {task_path}")
    task = Task()
    task.load_from_file(task_path)
    print(f"任务包含 {len(task.agents)} 个智能体")
    
    # 求解
    print("\n开始求解...")
    start_time = time.time()
    solution = solver.find_solution(task)
    elapsed = time.time() - start_time
    
    # 输出结果
    print(f"\n{'='*60}")
    print("求解结果")
    print(f"{'='*60}")
    print(f"是否找到解: {solution.found}")
    # 检查是否有有效的解（found=True 且有路径且flowtime>0）
    has_valid_solution = (solution.found and 
                         solution.paths and 
                         len(solution.paths) > 0 and 
                         solution.flowtime > 0)
    
    if has_valid_solution:
        print(f"总路径长度 (flowtime): {solution.flowtime:.2f}")
        print(f"最大路径长度 (makespan): {solution.makespan:.2f}")
        print(f"求解耗时: {elapsed:.2f}秒")
        print(f"扩展的高层节点数: {solution.high_level_expanded}")
        print(f"低层搜索次数: {solution.low_level_expansions}")
        print(f"低层扩展节点数: {solution.low_level_expanded:.2f}")
        
        if SOLVE_CONFIG["print_paths"] and solution.paths:
            print(f"\n各智能体路径详情:")
            for idx, path in enumerate(solution.paths):
                print(f"  智能体 {idx}: 代价={path.cost:.2f}, 节点数={len(path.nodes)}")
                if CCBS_CONFIG["verbose"]:
                    for node in path.nodes:
                        print(f"    节点={node.id}, 时间={node.g:.2f}")
    else:
        print("未找到有效解")
        if solution.found:
            print("  注意: solution.found=True 但路径无效或flowtime<=0")
    
    # 保存解决方案
    if SOLVE_CONFIG["save_solution"] and solution.found:
        output_path = get_abs_path(SOLVE_CONFIG["solution_output_path"])
        solver.write_to_log_path(output_path)
        print(f"\n解决方案已保存到: {output_path}")
    
    # 可视化
    if SOLVE_CONFIG["visualize"] and has_valid_solution:
        print("\n显示可视化结果...")
        try:
            animation = GraphRender(world_map, task, solution.paths)
            animation.show()
        except Exception as e:
            print(f"可视化失败: {e}")
            print("提示: 如果是在无GUI环境，可以设置 visualize=False 或使用非交互式后端")
    elif SOLVE_CONFIG["visualize"] and not has_valid_solution:
        print("\n跳过可视化: 未找到有效解，无法显示")
    
    return solution


def evaluate_model():
    """评估模型在测试集上的性能"""
    print("=" * 60)
    print("评估模型性能")
    print("=" * 60)
    
    # 加载地图
    map_path = get_abs_path(MAP_AND_TASK_CONFIG["map_path"])
    print(f"加载地图: {map_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"地图文件未找到: {map_path}")
    world_map = Map(map_path)
    
    # 加载测试任务
    test_dir = get_abs_path(MAP_AND_TASK_CONFIG["test_task_dir"])
    test_subdirs = MAP_AND_TASK_CONFIG["test_subdirs"]
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试任务目录未找到: {test_dir}")
    
    if test_subdirs:
        test_files = []
        for subdir in test_subdirs:
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                test_files.extend([
                    os.path.join(subdir_path, fname)
                    for fname in os.listdir(subdir_path)
                    if fname.endswith('.xml')
                ])
    else:
        test_files = [
            os.path.join(test_dir, fname)
            for fname in os.listdir(test_dir)
            if fname.endswith('.xml')
        ]
        # 如果test_dir下有子目录，递归查找
        if not test_files:
            for root, dirs, files in os.walk(test_dir):
                test_files.extend([
                    os.path.join(root, fname)
                    for fname in files
                    if fname.endswith('.xml')
                ])
    
    print(f"找到 {len(test_files)} 个测试任务")
    
    # 评估
    results = []
    for i, task_file in enumerate(test_files, 1):
        print(f"\n处理任务 {i}/{len(test_files)}: {os.path.basename(task_file)}")
        
        solver = CCBS(world_map)
        apply_config_to_ccbs(solver, CCBS_CONFIG, map_path)
        solver.solution = Solution()
        
        task = Task()
        task.load_from_file(task_file)
        
        start_time = time.time()
        solution = solver.find_solution(task)
        elapsed = time.time() - start_time
        
        result = {
            "task_file": task_file,
            "num_agents": len(task.agents),
            "found": solution.found,
            "flowtime": solution.flowtime if solution.found and solution.flowtime > 0 else None,
            "makespan": solution.makespan if solution.found and solution.makespan > 0 else None,
            "elapsed_time": elapsed,
            "high_level_expanded": solution.high_level_expanded,
            "low_level_expansions": solution.low_level_expansions,
            "low_level_expanded": solution.low_level_expanded,
        }
        results.append(result)
        
        print(f"  结果: 找到解={solution.found}, "
              f"耗时={elapsed:.2f}s, "
              f"flowtime={solution.flowtime if solution.found and solution.flowtime > 0 else 'N/A'}")
    
    # 统计结果
    print(f"\n{'='*60}")
    print("评估统计")
    print(f"{'='*60}")
    found_count = sum(1 for r in results if r["found"])
    print(f"总任务数: {len(results)}")
    print(f"成功求解: {found_count} ({found_count/len(results)*100:.1f}%)")
    if found_count > 0:
        # 过滤掉None值再计算平均值
        flowtimes = [r["flowtime"] for r in results if r["found"] and r["flowtime"] is not None]
        makespans = [r["makespan"] for r in results if r["found"] and r["makespan"] is not None]
        elapsed_times = [r["elapsed_time"] for r in results if r["found"]]
        
        if flowtimes:
            avg_flowtime = sum(flowtimes) / len(flowtimes)
            print(f"平均 flowtime: {avg_flowtime:.2f}")
        if makespans:
            avg_makespan = sum(makespans) / len(makespans)
            print(f"平均 makespan: {avg_makespan:.2f}")
        if elapsed_times:
            avg_time = sum(elapsed_times) / len(elapsed_times)
            print(f"平均求解时间: {avg_time:.2f}秒")
    
    # 保存结果
    if BATCH_TEST_CONFIG["save_results"]:
        output_path = get_abs_path(BATCH_TEST_CONFIG["results_output_path"])
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
    
    return results


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("CCBS 项目运行脚本")
    print("=" * 60)
    print(f"脚本目录: {SCRIPT_DIR}")
    print(f"运行模式: {RUN_MODE}")
    print(f"地图路径: {MAP_AND_TASK_CONFIG['map_path']}")
    print(f"地图绝对路径: {get_abs_path(MAP_AND_TASK_CONFIG['map_path'])}")
    print(f"使用RL: {CCBS_CONFIG['use_rl']}")
    if CCBS_CONFIG['use_rl']:
        # 如果没有手动指定模型路径，尝试自动选择
        if CCBS_CONFIG['rl_model_path'] is None:
            map_path = MAP_AND_TASK_CONFIG['map_path']
            auto_selected = auto_select_model_path(map_path)
            if auto_selected:
                CCBS_CONFIG['rl_model_path'] = auto_selected
                print(f"自动选择模型路径: {auto_selected}")
            else:
                # 找不到匹配的模型，停止并报错
                raise FileNotFoundError(
                    f"错误: 无法为地图 '{map_path}' 自动选择合适的RL模型。\n"
                    f"请手动在 CCBS_CONFIG 中设置 'rl_model_path'，或确保对应的模型文件存在。\n"
                    f"可用模型: {', '.join([k for k, v in MAP_TO_MODEL_MAPPING.items() if v])}"
                )
        
        # 验证模型路径是否存在
        if CCBS_CONFIG['rl_model_path']:
            print(f"RL模型路径: {CCBS_CONFIG['rl_model_path']}")
            try:
                resolved_model = resolve_model_path(CCBS_CONFIG['rl_model_path'])
                if resolved_model:
                    print(f"RL模型绝对路径: {resolved_model}")
                else:
                    # 解析失败，报错
                    raise FileNotFoundError(
                        f"错误: 无法解析RL模型路径 '{CCBS_CONFIG['rl_model_path']}'"
                    )
            except (FileNotFoundError, TypeError) as e:
                # 模型文件不存在，报错
                raise FileNotFoundError(
                    f"错误: RL模型未找到或无法加载。\n"
                    f"模型路径: {CCBS_CONFIG['rl_model_path']}\n"
                    f"详细信息: {str(e)}\n"
                    f"请检查模型文件是否存在，或修改 CCBS_CONFIG['rl_model_path'] 配置。"
                )
        else:
            # 未指定模型路径，报错
            raise ValueError(
                f"错误: use_rl=True 但未指定RL模型路径。\n"
                f"请在 CCBS_CONFIG 中设置 'rl_model_path'，或设置为 None 以自动选择。"
            )
    print("=" * 60 + "\n")
    
    if RUN_MODE == "train":
        train_model()
    elif RUN_MODE == "solve":
        solve_task()
    elif RUN_MODE == "evaluate":
        evaluate_model()
    elif RUN_MODE == "batch_test":
        evaluate_model()  # 与evaluate相同
    else:
        print(f"错误: 未知的运行模式 '{RUN_MODE}'")
        print("可用模式: train, solve, evaluate, batch_test")
        return
    
    print("\n" + "=" * 60)
    print("运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
