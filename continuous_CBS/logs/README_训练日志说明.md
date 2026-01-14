# 训练日志说明

## 已上传到GitHub的文件

以下文件已成功上传到GitHub仓库：
- `ppo_ccbs_multi_map_checkpoint_step200000.zip` (147KB) - 训练checkpoint（200,000步）
- `ppo_ccbs_multi_map_checkpoint_step400000.zip` (147KB) - 训练checkpoint（400,000步）
- `logs/monitor.csv` (1.3MB) - 训练监控数据

## 未上传的文件

### `train_stdout.log` (118MB)

**原因**: 该文件大小（118MB）超过了GitHub的单个文件大小限制（100MB）。

**文件位置**: `continuous_CBS/logs/train_stdout.log`

**文件内容**: 完整的训练输出日志，包含所有训练过程的详细信息。

**如需上传，有以下选项**：

1. **使用Git LFS（Git Large File Storage）**
   ```bash
   # 安装Git LFS（如果未安装）
   git lfs install
   
   # 跟踪大文件
   git lfs track "*.log"
   
   # 添加并提交
   git add .gitattributes
   git add continuous_CBS/logs/train_stdout.log
   git commit -m "添加训练日志（使用Git LFS）"
   git push new_origin main
   ```

2. **手动下载到本地**
   ```bash
   scp user@server:/root/zhongjie_test_1/continuous_CBS/logs/train_stdout.log ./
   ```

3. **如果日志不重要，可以忽略**
   - 模型文件和监控数据已经上传，这些是最重要的数据
   - 日志文件主要用于调试和分析，如果需要可以稍后下载

## 数据完整性

虽然`train_stdout.log`未上传，但关键的训练数据已经完整保存：
- ✅ 模型checkpoint文件（可用于恢复训练）
- ✅ 监控数据（monitor.csv，包含训练指标）
- ✅ 代码和配置（已上传到GitHub）

如果需要完整的训练日志，请从服务器下载`train_stdout.log`文件。

