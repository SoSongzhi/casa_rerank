# 顺序处理脚本使用指南

## 📋 脚本功能

`sequential_test_high_nine.py` 是一个逐个谱图处理的脚本，具有以下特点：

- **🔄 逐个处理**: 按谱图顺序，一个一个处理
- **⚡ 实时显示**: 每个谱图处理完立即显示结果
- **💾 断点续传**: 支持中断后继续处理
- **📊 详细统计**: 显示每个步骤的时间和准确率

## 🚀 使用方法

### 基本运行
```bash
cd high_nine_standalone
python sequential_test_high_nine.py
```

### 断点续传
如果脚本中断，直接重新运行即可：
```bash
python sequential_test_high_nine.py
```
脚本会自动检测已处理的谱图，从中断处继续。

## 📁 输出文件

运行后会生成以下文件：

```
high_nine_results_sequential/
├── sequential_results.csv          # 最终结果（实时更新）
├── processing_state.json           # 处理状态（用于断点续传）
└── temp_spectra/                   # 临时文件目录（自动清理）
```

## 📊 输出格式

### 实时进度显示
```
[1/1000] 0.1% | Spectrum 0
  ├─ De Novo: 2.31s | Rerank: 0.84s | Total: 3.15s
  ├─ True: PEPTIDEK
  ├─ Pred: PEPTIDEK
  ├─ Correct: ✓
  └─ ETA: 3148.5s | Avg: 3.15s/spectrum
```

### 结果文件格式 (sequential_results.csv)
包含每个谱图的详细结果：
- `spectrum_index`: 谱图索引
- `peptide`: 预测的肽段序列
- `true_sequence`: 真实序列
- `similarity_score`: 相似度得分
- `denovo_time`: De Novo耗时
- `rerank_time`: 重排序耗时
- `total_time`: 总耗时

## ⚙️ 配置选项

### 修改模型路径
在脚本中修改：
```python
model_path="casanovo_v5_0_0_v5_0_0.ckpt"
```

### 修改配置文件
在脚本中修改：
```python
config_path="beam50.yaml"
```

### 修改输出目录
在脚本中修改：
```python
output_dir = Path("high_nine_results_sequential")
```

## 🔄 断点续传机制

### 状态保存
- 每处理完一个谱图，状态会自动保存到 `processing_state.json`
- 包含已处理的谱图索引、时间统计等信息

### 恢复处理
- 重新运行脚本时，会自动读取状态文件
- 从上次中断的位置继续处理
- 跳过已处理的谱图

### 重新开始
如果要完全重新开始，删除状态文件：
```bash
rm high_nine_results_sequential/processing_state.json
```

## 📈 性能对比

| 模式 | 优势 | 劣势 |
|------|------|------|
| 批量处理 | 总时间短，GPU利用率高 | 无法实时看到结果 |
| 顺序处理 | 实时进度，支持断点续传 | 总时间可能较长 |

## 🐛 故障排除

### 常见问题

1. **Casanovo超时**
   - 单个谱图处理超时（5分钟）
   - 自动跳过该谱图，继续处理下一个

2. **内存不足**
   - 逐个处理模式内存使用较低
   - 如仍有问题，检查系统内存

3. **模型文件不存在**
   - 确保模型文件在正确路径
   - 检查 `casanovo_v5_0_0_v5_0_0.ckpt` 是否存在

4. **索引文件缺失**
   - 脚本会自动构建索引
   - 确保有足够磁盘空间（~4GB）

### 调试模式
如需详细错误信息，可以修改脚本中的日志级别：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 技术支持

如有问题，请检查：
1. 数据文件是否存在
2. 依赖是否正确安装
3. 磁盘空间是否充足
4. GPU内存是否足够（如使用GPU）

---

**提示**: 首次运行建议先用少量数据测试，确认流程正常后再处理完整数据集。
