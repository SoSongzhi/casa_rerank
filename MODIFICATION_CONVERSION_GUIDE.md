# Modification Format Conversion Guide

## ✅ 确认：输出已经是Unimod格式

**是的！** [`sequential_test_high_nine_progressive.py`](sequential_test_high_nine_progressive.py:1) 生成的输出**已经是Unimod ID格式**。

### 转换位置

修饰格式转换在以下位置自动执行：

1. **De Novo预测输出** ([第350行](sequential_test_high_nine_progressive.py:350))
   ```python
   converted_peptide = self.convert_modification_format(peptide)
   ```
   所有从Casanovo beam search输出的肽段都会立即转换为Unimod格式

2. **保存到结果文件** ([第442行](sequential_test_high_nine_progressive.py:442))
   通过`save_single_result()`保存到CSV文件的肽段都是Unimod格式

3. **进度显示** ([第478-518行](sequential_test_high_nine_progressive.py:478))
   屏幕上显示的所有肽段都是Unimod格式

### 输出文件格式

输出文件 `high_nine_results_sequential_progressive/sequential_progressive_results.csv` 中的肽段格式：

- ✅ `M[UNIMOD:35]PEPTIDE` - 氧化
- ✅ `C[UNIMOD:4]PEPTIDE` - 烷基化
- ✅ `N[UNIMOD:7]PEPTIDE` - 脱酰胺
- ❌ ~~`M[Oxidation]PEPTIDE`~~ - 不会出现
- ❌ ~~`M(+15.99)PEPTIDE`~~ - 不会出现

---


## 概述

已成功实现将Casanovo输出的修饰格式转换为Unimod标准格式的功能。

## 修饰格式对照

### 数据中的修饰格式
- `C(+57.02)` - 半胱氨酸烷基化
- `M(+15.99)` - 甲硫氨酸氧化
- `N(+.98)` 或 `Q(+.98)` - 天冬酰胺/谷氨酰胺脱酰胺

### Casanovo输出格式
- `C[Carbamidomethyl]` 或 `C[+57.021464]`
- `M[Oxidation]` 或 `M[+15.994915]`
- `N[Deamidated]` 或 `N[+0.984016]`

### Unimod标准格式（转换后）
- `C[UNIMOD:4]` - Carbamidomethyl (烷基化)
- `M[UNIMOD:35]` - Oxidation (氧化)
- `N[UNIMOD:7]` - Deamidation (脱酰胺)
- `Q[UNIMOD:7]` - Deamidation (脱酰胺)

## 支持的Unimod修饰

| Unimod ID | 修饰名称 | 质量偏移 | 氨基酸 | 说明 |
|-----------|---------|---------|--------|------|
| UNIMOD:4 | Carbamidomethyl | +57.021464 | C | 半胱氨酸烷基化 |
| UNIMOD:7 | Deamidation | +0.984016 | N, Q | 脱酰胺 |
| UNIMOD:35 | Oxidation | +15.994915 | M | 甲硫氨酸氧化 |
| UNIMOD:1 | Acetyl | +42.010565 | N-term | N端乙酰化 |
| UNIMOD:5 | Carbamyl | +43.005814 | N-term | N端氨甲酰化 |
| UNIMOD:385 | Ammonia-loss | -17.026549 | N-term | 氨损失 |
| UNIMOD:27 | Glu->pyro-Glu | -18.010565 | E | 谷氨酸环化 |
| UNIMOD:28 | Gln->pyro-Glu | -17.026549 | Q | 谷氨酰胺环化 |

## 转换示例

### 单个修饰
```
输入: M[Oxidation]PEPTIDE
输出: M[UNIMOD:35]PEPTIDE

输入: C(+57.02)PEPTIDE
输出: C[UNIMOD:4]PEPTIDE

输入: N[+.98]PEPTIDE
输出: N[UNIMOD:7]PEPTIDE
```

### 多个修饰
```
输入: M[Oxidation]C[Carbamidomethyl]PEPTIDE
输出: M[UNIMOD:35]C[UNIMOD:4]PEPTIDE

输入: DC(+57.02)PAHSIC(+57.02)HNHR
输出: DC[UNIMOD:4]PAHSIC[UNIMOD:4]HNHR
```

### 混合格式
```
输入: M[Oxidation]C(+57.02)PEPTIDE
输出: M[UNIMOD:35]C[UNIMOD:4]PEPTIDE
```

## 实现细节

### 转换函数位置
文件: `sequential_test_high_nine_progressive.py`
函数: `convert_modification_format(peptide: str) -> str`

### 转换逻辑
1. **命名修饰转换**: 识别Casanovo的命名修饰（如`[Oxidation]`），直接映射到对应的Unimod ID
2. **数值修饰转换**: 识别质量偏移（如`[+15.99]`或`(+57.02)`），通过质量范围匹配到Unimod ID
3. **容差处理**: 使用±0.15 Da的容差来匹配不同精度的质量值

### 支持的输入格式
- 方括号: `M[Oxidation]`, `M[+15.994915]`
- 圆括号: `M(+15.99)`, `C(+57.02)`
- 简化格式: `N(+.98)` (小数点前无0)
- 已转换格式: `M[UNIMOD:35]` (保持不变)

## 测试结果

### 真实数据测试
- 测试文件: `test_real_peptides.py`
- 数据来源: `test_data/high_nine/high_nine_database.mgf`
- 测试肽段数: 30个带修饰的肽段
- **转换成功率: 100%**

### 测试示例
```
✓ NMN(+.98)RHDVIFPGFIK → NMN[UNIMOD:7]RHDVIFPGFIK
✓ GHSC(+57.02)YRPR → GHSC[UNIMOD:4]YRPR
✓ M(+15.99)KQEPVKPEEGR → M[UNIMOD:35]KQEPVKPEEGR
✓ DC(+57.02)PAHSIC(+57.02)HNHR → DC[UNIMOD:4]PAHSIC[UNIMOD:4]HNHR
```

## 使用方法

### 在代码中使用
```python
from sequential_test_high_nine_progressive import ProgressiveSequentialProcessor

processor = ProgressiveSequentialProcessor(...)

# 转换单个肽段
peptide = "M[Oxidation]PEPTIDE"
converted = processor.convert_modification_format(peptide)
print(converted)  # 输出: M[UNIMOD:35]PEPTIDE
```

### 运行测试
```bash
# 测试基本转换功能
python test_modification_conversion.py

# 测试真实数据库肽段
python test_real_peptides.py
```

## 集成到处理流程

修饰转换已集成到以下位置：

1. **De Novo预测输出** (第252-260行)
   - 在beam search解码后立即转换候选肽段

2. **结果保存** (第337-349行)
   - 保存到CSV文件前转换肽段格式

3. **进度显示** (第351-437行)
   - 显示时使用Unimod格式

## 注意事项

1. **未知修饰**: 如果遇到未在映射表中的修饰，会保留原始质量值（格式化为6位小数）
2. **精度**: 质量匹配使用±0.15 Da容差，足以区分常见修饰
3. **兼容性**: 支持方括号和圆括号两种输入格式
4. **幂等性**: 已经是Unimod格式的肽段保持不变

## 扩展修饰支持

如需添加新的修饰，在`convert_modification_format`函数中更新两个映射表：

```python
# 1. 命名修饰映射
mod_name_to_unimod = {
    'NewMod': 'UNIMOD:XXX',
    ...
}

# 2. 质量范围映射
mass_to_unimod = [
    (min_mass, max_mass, 'UNIMOD:XXX', 'ModName'),
    ...
]
```

## 参考资源

- [Unimod数据库](http://www.unimod.org/)
- [PSI-MOD](https://www.ebi.ac.uk/ols/ontologies/mod)
- [ProForma规范](https://www.psidev.info/proforma)