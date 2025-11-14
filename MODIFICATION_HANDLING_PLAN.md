# 修饰处理方案

## 问题分析

### 1. Prosit生成失败的情况
**问题**: Prosit可能无法为带修饰的肽段生成理论谱图
**原因**: 
- Prosit只支持特定的修饰类型
- 某些修饰组合可能不被支持
- 修饰格式可能不兼容

**解决方案**:
```python
# 在efficient_reranker.py中添加fallback机制
def rerank_with_efficient_index(...):
    try:
        # 尝试使用Prosit
        prosit_spectrum = generate_prosit_spectrum(peptide)
    except Exception as e:
        logger.warning(f"Prosit failed for {peptide}: {e}")
        # Fallback 1: 使用De Novo分数
        return use_denovo_score_only(peptide)
        # 或 Fallback 2: 跳过Prosit，只用数据库匹配
```

### 2. Database匹配时的修饰格式问题

**当前情况**:
- Casanovo输出 (转换后): `M[UNIMOD:35]PEPTIDE`
- Database格式: `M(+15.99)PEPTIDE`
- **格式不匹配，无法直接比较！**

**解决方案A: 统一转换Database格式**
```python
# 在build_efficient_index.py中
def normalize_peptide_for_index(peptide):
    """将database肽段转换为Unimod格式"""
    # M(+15.99) -> M[UNIMOD:35]
    # C(+57.02) -> C[UNIMOD:4]
    # N(+.98) -> N[UNIMOD:7]
    return convert_modification_format(peptide)
```

**解决方案B: 创建双向映射**
```python
# 在匹配时同时支持两种格式
def create_peptide_variants(peptide):
    """为一个肽段创建所有可能的格式变体"""
    variants = set()
    variants.add(peptide)  # 原始格式
    variants.add(convert_to_unimod(peptide))  # Unimod格式
    variants.add(convert_to_mass(peptide))  # 质量格式
    return variants
```

**解决方案C: 标准化比较函数**
```python
def normalize_for_comparison(peptide):
    """标准化肽段用于比较"""
    # 1. 移除所有修饰标记
    # 2. 只保留氨基酸序列
    # 3. 分别比较修饰位置和类型
    
    seq = remove_modifications(peptide)
    mods = extract_modifications(peptide)
    return (seq, normalize_mods(mods))
```

### 3. 推荐方案：修饰感知的匹配系统

```python
class ModificationAwareIndex:
    """支持修饰的索引系统"""
    
    def __init__(self):
        self.unmodified_index = {}  # 无修饰序列索引
        self.modification_index = {}  # 修饰信息索引
    
    def add_peptide(self, peptide):
        """添加肽段到索引"""
        # 分离序列和修饰
        seq, mods = parse_peptide(peptide)
        
        # 标准化修饰为Unimod格式
        normalized_mods = normalize_modifications(mods)
        
        # 存储
        if seq not in self.unmodified_index:
            self.unmodified_index[seq] = []
        self.unmodified_index[seq].append({
            'original': peptide,
            'modifications': normalized_mods
        })
    
    def search(self, query_peptide):
        """搜索肽段"""
        query_seq, query_mods = parse_peptide(query_peptide)
        query_mods_norm = normalize_modifications(query_mods)
        
        # 1. 先匹配序列
        candidates = self.unmodified_index.get(query_seq, [])
        
        # 2. 再匹配修饰
        matches = []
        for candidate in candidates:
            if modifications_match(query_mods_norm, candidate['modifications']):
                matches.append(candidate)
        
        return matches
```

## 实施步骤

### Step 1: 更新索引构建
```python
# build_efficient_index.py
def build_index(mgf_file):
    index = ModificationAwareIndex()
    
    for peptide in read_peptides(mgf_file):
        # 转换为Unimod格式
        normalized = convert_modification_format(peptide)
        index.add_peptide(normalized)
    
    return index
```

### Step 2: 更新重排序器
```python
# efficient_reranker.py
def rerank_with_efficient_index(...):
    # 1. 标准化候选肽段
    normalized_candidates = [
        convert_modification_format(c['peptide']) 
        for c in candidates
    ]
    
    # 2. 在索引中查找
    for candidate in normalized_candidates:
        matches = self.index.search(candidate)
        
        if matches:
            # 找到匹配，使用数据库谱图
            try:
                similarity = calculate_similarity(...)
            except Exception:
                # Prosit失败，使用De Novo分数
                similarity = candidate['score']
        else:
            # 未找到匹配，使用Prosit
            try:
                prosit_spectrum = generate_prosit(candidate)
                similarity = calculate_similarity(...)
            except Exception:
                # Prosit也失败，只用De Novo分数
                similarity = candidate['score']
```

### Step 3: 处理Prosit失败
```python
def safe_prosit_prediction(peptide):
    """安全的Prosit预测，带fallback"""
    try:
        # 检查修饰是否被Prosit支持
        if not is_prosit_compatible(peptide):
            logger.warning(f"Peptide {peptide} has unsupported modifications")
            return None
        
        # 尝试生成
        spectrum = prosit.predict(peptide)
        return spectrum
        
    except Exception as e:
        logger.error(f"Prosit failed for {peptide}: {e}")
        return None

def rerank_with_fallback(candidates, ...):
    """带fallback的重排序"""
    results = []
    
    for candidate in candidates:
        # 尝试1: Database匹配
        db_match = search_in_database(candidate)
        if db_match:
            score = calculate_similarity(spectrum, db_match)
            results.append({
                'peptide': candidate,
                'score': score,
                'source': 'database'
            })
            continue
        
        # 尝试2: Prosit
        prosit_spectrum = safe_prosit_prediction(candidate)
        if prosit_spectrum:
            score = calculate_similarity(spectrum, prosit_spectrum)
            results.append({
                'peptide': candidate,
                'score': score,
                'source': 'prosit'
            })
            continue
        
        # Fallback: De Novo分数
        results.append({
            'peptide': candidate,
            'score': candidate['denovo_score'],
            'source': 'denovo_only'
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)
```

## 修饰标准化函数

```python
def normalize_modifications(mods):
    """标准化修饰为统一格式"""
    normalized = []
    
    for position, mod_mass in mods:
        # 转换为Unimod ID
        unimod_id = mass_to_unimod(mod_mass)
        normalized.append((position, unimod_id))
    
    return tuple(sorted(normalized))

def modifications_match(mods1, mods2, tolerance=0.1):
    """比较两个修饰是否匹配"""
    if len(mods1) != len(mods2):
        return False
    
    for (pos1, mod1), (pos2, mod2) in zip(mods1, mods2):
        if pos1 != pos2:
            return False
        
        # 如果都是Unimod ID，直接比较
        if mod1 == mod2:
            continue
        
        # 如果是质量，比较质量差异
        mass1 = unimod_to_mass(mod1) if 'UNIMOD' in str(mod1) else mod1
        mass2 = unimod_to_mass(mod2) if 'UNIMOD' in str(mod2) else mod2
        
        if abs(mass1 - mass2) > tolerance:
            return False
    
    return True
```

## 测试计划

### 1. 单元测试
```python
def test_modification_normalization():
    assert normalize("M(+15.99)") == "M[UNIMOD:35]"
    assert normalize("C(+57.02)") == "C[UNIMOD:4]"
    assert normalize("N(+.98)") == "N[UNIMOD:7]"

def test_modification_matching():
    assert modifications_match(
        [(1, "UNIMOD:35")],
        [(1, 15.994915)]
    ) == True

def test_prosit_fallback():
    # 测试Prosit失败时的fallback
    result = rerank_with_fallback([unsupported_peptide])
    assert result[0]['source'] == 'denovo_only'
```

### 2. 集成测试
```bash
# 在casa环境下运行
conda activate casa
python sequential_test_high_nine_progressive.py
```

### 3. 验证输出
```python
# 检查输出文件中的修饰格式
def verify_output():
    df = pd.read_csv('results.csv')
    for peptide in df['peptide']:
        assert 'UNIMOD:' in peptide or no_modifications(peptide)
```

## 配置建议

```yaml
# config.yaml
modification_handling:
  # 修饰格式
  output_format: "unimod"  # unimod, mass, or name
  
  # Prosit设置
  prosit_fallback: true
  prosit_timeout: 5.0  # seconds
  
  # Database匹配
  modification_tolerance: 0.1  # Da
  normalize_database: true
  
  # 重排序策略
  rerank_strategy: "hybrid"  # database_first, prosit_first, or hybrid
```

## 下一步行动

1. ✅ 修改`convert_modification_format`确保输出Unimod格式
2. 🔄 更新`build_efficient_index.py`标准化database
3. 🔄 更新`efficient_reranker.py`添加fallback机制
4. 🔄 实现修饰感知的匹配函数
5. 🔄 在casa环境测试
6. 🔄 验证带修饰肽段的处理

## 预期结果

- ✅ 所有输出都是Unimod格式
- ✅ Database和预测结果可以正确匹配
- ✅ Prosit失败时有合理的fallback
- ✅ 支持带修饰的肽段处理
- ✅ 在casa环境下稳定运行