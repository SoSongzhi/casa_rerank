# 使用 Python API 独立运行说明

## 已完成的工作

1. ✅ 创建了 `casanovo_predictor.py` - Python API预测器
2. ✅ 所有必需的 Casanovo 模块已拷贝
3. ✅ 模型权重 `casanovo_v4_2_0.ckpt` 已包含
4. ✅ 数据文件已链接

## 🚀 快速测试独立预测器

### 测试 casanovo_predictor.py

```bash
cd C:\Users\research\Desktop\casanovo\high_nine_standalone
conda activate casa

# 测试单独的预测器（使用较小的beam）
python casanovo_predictor.py casanovo_v4_2_0.ckpt test_data/high_nine/high_nine_validation_1000.mgf test_output.txt 5 10
```

参数说明：
- `casanovo_v4_2_0.ckpt` - 模型文件
- `test_data/high_nine/high_nine_validation_1000.mgf` - 输入MGF
- `test_output.txt` - 输出文件
- `5` - n_beams (beam width)
- `10` - top_match (每个谱图返回多少个候选)

## 📝 修改原脚本步骤

如果你想修改 `batch_test_high_nine_efficient.py` 使用 Python API：

### 方法1: 使用 casanovo_predictor.py

在脚本中找到这两处 `subprocess.run` 调用（约142行和190行），替换为：

```python
# 原代码：
# result = subprocess.run(["casanovo", "sequence", ...])

# 新代码：
from casanovo_predictor import CasanovoPredictor

predictor = CasanovoPredictor("casanovo_v4_2_0.ckpt", use_config)
success = predictor.predict(
    str(test_all_mgf),
    str(denovo_output / "casanovo_predictions.txt"),
    n_beams=50,
    top_match=50
)

if not success:
    print("Casanovo prediction failed!")
    # 处理错误...
```

### 方法2: 直接运行原目录的脚本

最简单的方法仍然是在原始目录运行：

```bash
cd C:\Users\research\Desktop\casanovo
conda activate casa
python batch_test_high_nine_efficient.py
```

## 📦 这个文件夹的用途

当前这个 `high_nine_standalone` 文件夹包含：

✅ **可以独立使用**:
- `casanovo_predictor.py` - 独立的预测器
- `efficient_reranker.py` - 重排序器
- `build_efficient_index.py` - 索引构建器
- 所有 Casanovo 模块代码
- 模型权重文件

⚠️ **需要手动修改**:
- `batch_test_high_nine_efficient.py` - 仍使用命令行调用，需要按上述方法修改

## 🔧 下一步

选择一个：

1. **测试独立预测器** - 运行上面的测试命令
2. **手动修改脚本** - 按方法1修改 batch_test 脚本
3. **使用原脚本** - 在原目录运行（最简单）

建议先测试独立预测器，确认可以工作后再修改batch脚本。
