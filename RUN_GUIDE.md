# 运行指南 - Sequential Test with Modifications

## ⚠️ 重要：首次运行或修饰格式更新后

如果你是首次运行，或者之前的索引没有使用Unimod格式，**必须先重建索引**：

```bash
# 1. 激活casa环境
conda activate casa

# 2. 重建索引（将database转换为Unimod格式）
python rebuild_index_with_unimod.py

# 3. 删除旧的处理结果
rm -rf high_nine_results_sequential_progressive/

# 4. 运行程序
python sequential_test_high_nine_progressive.py
```

### Windows系统
```powershell
conda activate casa
python rebuild_index_with_unimod.py
Remove-Item -Recurse -Force high_nine_results_sequential_progressive
python sequential_test_high_nine_progressive.py
```

## 为什么需要重建索引？

**问题**: Database中的肽段格式是 `SISC(+57.02)TYDDDTYR`，但预测结果是 `SLSC[UNIMOD:4]TYDDDTYR`，导致无法匹配。

**解决**: 重建索引时将database中所有肽段转换为Unimod格式，统一格式后就能正确匹配了。

**效果**:
- ✅ Database: `SISC[UNIMOD:4]TYDDDTYR`
- ✅ Prediction: `SLSC[UNIMOD:4]TYDDDTYR`
- ✅ 可以正确匹配和比较

---

## 如何重新运行程序

### 方法1: 完全重新开始（删除之前的状态）

```bash
# 1. 激活casa环境
conda activate casa

# 2. 删除之前的处理状态和结果
rm -rf high_nine_results_sequential_progressive/

# 3. 运行程序
python sequential_test_high_nine_progressive.py
```

### 方法2: 从断点继续（保留已处理的结果）

```bash
# 1. 激活casa环境
conda activate casa

# 2. 直接运行（会自动从上次中断的地方继续）
python sequential_test_high_nine_progressive.py
```

程序会自动：
- 读取 `high_nine_results_sequential_progressive/processing_state.json`
- 从上次处理的位置继续
- 跳过已处理的谱图

### 方法3: 重置特定部分

#### 只删除状态文件（重新处理，但保留结果）
```bash
conda activate casa
rm high_nine_results_sequential_progressive/processing_state.json
python sequential_test_high_nine_progressive.py
```

#### 只删除结果文件（保留进度状态）
```bash
conda activate casa
rm high_nine_results_sequential_progressive/sequential_progressive_results.csv
python sequential_test_high_nine_progressive.py
```

## Windows系统命令

如果你在Windows上，使用以下命令：

### PowerShell
```powershell
# 完全重新开始
conda activate casa
Remove-Item -Recurse -Force high_nine_results_sequential_progressive
python sequential_test_high_nine_progressive.py

# 或者只删除状态文件
conda activate casa
Remove-Item high_nine_results_sequential_progressive/processing_state.json
python sequential_test_high_nine_progressive.py
```

### CMD
```cmd
# 完全重新开始
conda activate casa
rmdir /s /q high_nine_results_sequential_progressive
python sequential_test_high_nine_progressive.py

# 或者只删除状态文件
conda activate casa
del high_nine_results_sequential_progressive\processing_state.json
python sequential_test_high_nine_progressive.py
```

## 输出文件说明

运行后会生成以下文件：

```
high_nine_results_sequential_progressive/
├── processing_state.json              # 处理状态（用于断点续传）
├── sequential_progressive_results.csv # 最终结果（Unimod格式）
└── temp_spectra/                      # 临时谱图文件（自动清理）
```

### 结果文件格式

`sequential_progressive_results.csv` 包含：
- `spectrum_index`: 谱图索引
- `peptide`: 预测的肽段（**Unimod格式**: `M[UNIMOD:35]PEPTIDE`）
- `true_sequence`: 真实序列
- `similarity`: 相似度分数
- `source`: 来源（database/prosit/denovo）
- `denovo_time`: De Novo预测时间
- `rerank_time`: 重排序时间
- 其他元数据...

## 检查运行状态

### 查看处理进度
```bash
# 查看状态文件
cat high_nine_results_sequential_progressive/processing_state.json

# 或在Python中
python -c "import json; print(json.load(open('high_nine_results_sequential_progressive/processing_state.json')))"
```

### 查看已处理的谱图数量
```bash
# Linux/Mac
wc -l high_nine_results_sequential_progressive/sequential_progressive_results.csv

# Windows PowerShell
(Get-Content high_nine_results_sequential_progressive/sequential_progressive_results.csv).Count
```

### 查看最新结果
```bash
# 查看最后10行
tail -n 10 high_nine_results_sequential_progressive/sequential_progressive_results.csv

# Windows PowerShell
Get-Content high_nine_results_sequential_progressive/sequential_progressive_results.csv -Tail 10
```

## 常见问题

### Q1: 程序说"Loaded existing state"，但我想重新开始
**A**: 删除整个输出目录
```bash
rm -rf high_nine_results_sequential_progressive/
```

### Q2: 如何只处理前N个谱图进行测试？
**A**: 修改代码或在运行一段时间后手动停止（Ctrl+C），下次运行会继续

### Q3: 如何验证输出是Unimod格式？
**A**: 检查结果文件
```bash
# 查找UNIMOD标记
grep "UNIMOD:" high_nine_results_sequential_progressive/sequential_progressive_results.csv | head -5
```

### Q4: 程序运行很慢怎么办？
**A**: 
- 检查GPU是否可用：`nvidia-smi`
- 减少beam size（修改`beam_schedule`）
- 使用更小的测试数据集

### Q5: 如何处理带修饰的肽段？
**A**: 当前代码已经支持！
- 第234行：处理所有有序列的谱图（包括带修饰的）
- 第350行：自动转换为Unimod格式
- 不需要额外配置

## 性能优化建议

### 1. 使用GPU加速
```bash
# 检查GPU
nvidia-smi

# 确保PyTorch能识别GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 调整Beam Schedule
```python
# 在sequential_test_high_nine_progressive.py中修改
beam_schedule = {0: 5, 1: 25, 2: 125, 3: 100}  # 默认
# 改为更快的配置
beam_schedule = {0: 5, 1: 10, 2: 25, 3: 50}    # 更快但可能准确率略低
```

### 3. 批处理大小
```python
# 在config中调整
predict_batch_size: 1024  # 默认
# 如果内存充足可以增加
predict_batch_size: 2048
```

## 监控运行

### 实时查看输出
```bash
# 运行并保存日志
python sequential_test_high_nine_progressive.py 2>&1 | tee run.log

# 在另一个终端查看
tail -f run.log
```

### 估算完成时间
程序会自动显示：
```
ETA: 1234.5s | Avg: 2.5s/spectrum
```

## 结果分析

### 统计准确率
```python
import pandas as pd

df = pd.read_csv('high_nine_results_sequential_progressive/sequential_progressive_results.csv')

# 计算准确率
correct = (df['peptide'] == df['true_sequence']).sum()
total = len(df)
accuracy = correct / total * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")
```

### 检查修饰格式
```python
# 检查是否所有输出都是Unimod格式
import re

for peptide in df['peptide']:
    if '[' in peptide or '(' in peptide:
        # 检查是否包含UNIMOD
        if 'UNIMOD:' not in peptide:
            print(f"Non-Unimod format found: {peptide}")
```

## 下一步

运行成功后：
1. ✅ 检查输出文件格式
2. ✅ 验证修饰转换正确
3. ✅ 分析准确率
4. ✅ 如有问题，查看日志文件

如需修改代码以处理特定问题（如Prosit fallback、database匹配），请参考 [`MODIFICATION_HANDLING_PLAN.md`](MODIFICATION_HANDLING_PLAN.md:1)