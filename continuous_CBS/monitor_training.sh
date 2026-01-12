#!/bin/bash
# 训练监控脚本 - 在tmux中使用
# 使用方法：
#   1. 在tmux中启动训练: python train_ppo_parallel.py 2>&1 | tee -a logs/train_stdout.log
#   2. 在另一个pane运行此脚本: bash monitor_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 训练监控工具 ==="
echo "选择监控方式："
echo "1. TensorBoard (推荐，需要SSH端口转发)"
echo "2. 查看训练日志 (tail -f)"
echo "3. 查看Monitor CSV"
echo "4. 查看所有"
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo "启动TensorBoard..."
        echo "在本地执行: ssh -L 6006:127.0.0.1:6006 user@server"
        echo "然后浏览器打开: http://127.0.0.1:6006"
        tensorboard --logdir ./logs/tensorboard --host 127.0.0.1 --port 6006
        ;;
    2)
        echo "监控训练日志 (Ctrl+C退出)..."
        tail -f logs/train_stdout.log
        ;;
    3)
        echo "监控Monitor CSV (Ctrl+C退出)..."
        tail -f logs/monitor.csv
        ;;
    4)
        echo "启动TensorBoard (后台)..."
        tensorboard --logdir ./logs/tensorboard --host 127.0.0.1 --port 6006 > /dev/null 2>&1 &
        TB_PID=$!
        echo "TensorBoard PID: $TB_PID"
        echo ""
        echo "监控训练日志 (Ctrl+C退出)..."
        tail -f logs/train_stdout.log
        kill $TB_PID 2>/dev/null
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac


