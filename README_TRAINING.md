# PPO并行训练实时监控指南

## 快速开始

### 方法1: TensorBoard（推荐，可视化最清晰）

#### 步骤1: 在tmux中启动训练
```bash
cd /root/zhongjie_test_1/continuous_CBS
python train_ppo_parallel.py 2>&1 | tee -a logs/train_stdout.log
```

#### 步骤2: 在tmux新pane中启动TensorBoard
```bash
# 按 Ctrl+b 然后按 " 或 % 创建新pane
cd /root/zhongjie_test_1/continuous_CBS
tensorboard --logdir ./logs/tensorboard --host 127.0.0.1 --port 6006
```

#### 步骤3: 在本地电脑建立SSH端口转发
```bash
# 在本地终端执行（保持连接）
ssh -L 6006:127.0.0.1:6006 user@your-server
```

#### 步骤4: 在本地浏览器打开
```
http://127.0.0.1:6006
```

### 方法2: 直接查看日志（轻量级）

#### 在tmux新pane中监控日志
```bash
tail -f logs/train_stdout.log
```

或者使用监控脚本：
```bash
bash monitor_training.sh
```

### 方法3: 查看Monitor CSV
```bash
tail -f logs/monitor.csv
```

## 推荐的tmux布局

### 布局1: 三pane布局（推荐）
```
┌─────────────┬─────────────┐
│             │             │
│   训练      │  TensorBoard│
│  (Pane 1)   │  (Pane 2)   │
│             │             │
├─────────────┴─────────────┤
│                           │
│      日志监控 (Pane 3)     │
│                           │
└───────────────────────────┘
```

创建方法：
```bash
# 启动训练
python train_ppo_parallel.py 2>&1 | tee -a logs/train_stdout.log

# Ctrl+b, "  (横向分屏)
# 启动TensorBoard
tensorboard --logdir ./logs/tensorboard --host 127.0.0.1 --port 6006

# Ctrl+b, %  (在训练pane中竖向分屏)
# 监控日志
tail -f logs/train_stdout.log
```

## TensorBoard中重点关注的指标

### 收敛判断关键指标：

1. **rollout/ep_rew_mean** (回报均值)
   - 应该整体上升并趋于稳定
   - 如果长期为负或波动大，说明训练有问题

2. **rollout/ep_len_mean** (回合长度)
   - 应该下降或稳定
   - 说明策略越来越高效

3. **train/approx_kl** (KL散度)
   - 应该稳定在合理范围（通常0.01-0.1）
   - 过大说明策略更新太激进

4. **train/entropy_loss** (熵损失)
   - 应该逐步下降
   - 说明探索逐渐减少，策略趋于确定

5. **train/clip_fraction** (裁剪比例)
   - 应该<0.2
   - 长期过高说明更新太猛

### 日志中的收敛提示

训练脚本会每10个episode打印一次：
```
Episode 100/80000: 最近平均奖励=5.2341, 最佳奖励=8.1234, 收敛状态=训练中
```

当检测到收敛时会显示：
```
[收敛检测] Episode 500: 检测到收敛! 最近100个episode的平均奖励=6.1234, 标准差=0.0123
```

## 常见问题排查

### TensorBoard看不到曲线

1. 检查日志目录是否存在：
```bash
ls -lh ./logs/tensorboard
find ./logs/tensorboard -type f | head -20
```

2. 确认TensorBoard路径正确：
```bash
# 应该能看到类似这样的目录结构
./logs/tensorboard/PPO_1/
```

3. 检查PPO是否在写日志：
```bash
# 训练开始后，应该能看到新文件
watch -n 5 'ls -lht ./logs/tensorboard/PPO_*/events.out.tfevents.* | head -5'
```

### 端口转发失败

如果6006端口被占用，可以换端口：
```bash
# 服务器端
tensorboard --logdir ./logs/tensorboard --host 127.0.0.1 --port 6007

# 本地
ssh -L 6007:127.0.0.1:6007 user@server

# 浏览器
http://127.0.0.1:6007
```

### 曲线更新很慢

这是正常的！SB3的TensorBoard日志不是每步都写，通常按rollout周期刷新（每256步×20环境=5120步一次）。

### 查看实时统计

如果想看更频繁的更新，可以：
```bash
# 每5秒刷新一次TensorBoard页面
# 或者直接tail日志文件看实时输出
tail -f logs/train_stdout.log | grep -E "(Episode|收敛|奖励)"
```

## 训练输出示例

正常训练时，你应该看到：
```
Episode 10/80000: 最近平均奖励=-2.3456, 最佳奖励=5.1234, 收敛状态=训练中
Episode 20/80000: 最近平均奖励=-1.2345, 最佳奖励=5.1234, 收敛状态=训练中
...
Episode 100/80000: 最近平均奖励=2.3456, 最佳奖励=8.1234, 收敛状态=训练中
[收敛检测] Episode 500: 检测到收敛! 最近100个episode的平均奖励=6.1234, 标准差=0.0123
```

## 检查点保存

训练会自动保存检查点：
- 每200000步保存一次到 `checkpoints/` 目录
- 最终模型保存为 `ppo_ccbs_multi_map_final.zip`

## 训练完成后

训练结束后会生成：
- `logs/train_stdout.log` - 完整训练日志
- `logs/monitor.csv` - 环境监控数据
- `logs/tensorboard/` - TensorBoard日志
- `rewards_ppo_multi_map.csv` - 奖励记录


