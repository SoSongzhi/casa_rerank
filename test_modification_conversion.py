#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test modification format conversion
"""

import sys
import re
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_modification_conversion():
    """测试各种修饰格式的转换"""
    
    # 创建一个临时处理器实例（只用于测试转换函数）
    class MockProcessor:
        def convert_modification_format(self, peptide: str) -> str:
            """Unimod ID格式转换函数"""
            if not peptide:
                return peptide
            
            # Casanovo修饰名称到Unimod ID
            mod_name_to_unimod = {
                'Oxidation': 'UNIMOD:35',
                'Deamidated': 'UNIMOD:7',
                'Carbamidomethyl': 'UNIMOD:4',
                'Acetyl': 'UNIMOD:1',
                'Carbamyl': 'UNIMOD:5',
                'Ammonia-loss': 'UNIMOD:385',
                'Gln->pyro-Glu': 'UNIMOD:28',
                'Glu->pyro-Glu': 'UNIMOD:27',
            }
            
            # 质量范围到Unimod ID
            mass_to_unimod = [
                (15.85, 16.15, 'UNIMOD:35', 'Oxidation'),
                (0.83, 1.13, 'UNIMOD:7', 'Deamidation'),
                (56.87, 57.17, 'UNIMOD:4', 'Carbamidomethyl'),
                (41.86, 42.16, 'UNIMOD:1', 'Acetyl'),
                (42.86, 43.16, 'UNIMOD:5', 'Carbamyl'),
                (-17.18, -16.88, 'UNIMOD:385', 'Ammonia-loss'),
                (-18.16, -17.86, 'UNIMOD:27', 'Glu->pyro-Glu'),
                (25.83, 26.13, 'UNIMOD:526', 'Carbamyl+Ammonia'),
                (79.96, 80.06, 'UNIMOD:21', 'Phospho'),
                (14.01, 14.03, 'UNIMOD:34', 'Methyl'),
            ]
            
            converted = peptide
            
            # 处理命名修饰
            for mod_name, unimod_id in mod_name_to_unimod.items():
                converted = converted.replace(f'[{mod_name}]', f'[{unimod_id}]')
            
            # 处理数值修饰
            import re
            pattern = r'[\[\(]([+\-]?\d*\.?\d+)[\]\)]'
            
            def replace_numeric_mod(match):
                mass_str = match.group(1)
                try:
                    mass_val = float(mass_str)
                    
                    for min_mass, max_mass, unimod_id, mod_name in mass_to_unimod:
                        if min_mass <= mass_val <= max_mass:
                            return f'[{unimod_id}]'
                    
                    # 未知修饰保留质量
                    if mass_val >= 0:
                        return f'[+{mass_val:.6f}]'
                    else:
                        return f'[{mass_val:.6f}]'
                        
                except ValueError:
                    return f'[{mass_str}]'
            
            converted = re.sub(pattern, replace_numeric_mod, converted)
            return converted
    
    processor = MockProcessor()
    
    # 测试用例 - Unimod ID格式
    test_cases = [
        # (输入, 期望输出, 描述)
        ("PEPTIDE", "PEPTIDE", "无修饰肽段"),
        
        # Casanovo命名修饰 -> Unimod ID
        ("M[Oxidation]PEPTIDE", "M[UNIMOD:35]PEPTIDE", "命名修饰: 氧化 -> UNIMOD:35"),
        ("PEPTIDEM[Oxidation]", "PEPTIDEM[UNIMOD:35]", "末端氧化 -> UNIMOD:35"),
        ("N[Deamidated]PEPTIDE", "N[UNIMOD:7]PEPTIDE", "命名修饰: 脱酰胺N -> UNIMOD:7"),
        ("Q[Deamidated]PEPTIDE", "Q[UNIMOD:7]PEPTIDE", "命名修饰: 脱酰胺Q -> UNIMOD:7"),
        ("C[Carbamidomethyl]PEPTIDE", "C[UNIMOD:4]PEPTIDE", "命名修饰: 烷基化 -> UNIMOD:4"),
        
        # Casanovo数值修饰 -> Unimod ID
        ("M[+15.994915]PEPTIDE", "M[UNIMOD:35]PEPTIDE", "Casanovo精确氧化质量 -> UNIMOD:35"),
        ("N[+0.984016]PEPTIDE", "N[UNIMOD:7]PEPTIDE", "Casanovo精确脱酰胺质量 -> UNIMOD:7"),
        ("C[+57.021464]PEPTIDE", "C[UNIMOD:4]PEPTIDE", "Casanovo精确烷基化质量 -> UNIMOD:4"),
        
        # 数据中的简化格式 -> Unimod ID
        ("M[+15.99]PEPTIDE", "M[UNIMOD:35]PEPTIDE", "简化氧化 -> UNIMOD:35"),
        ("N[+.98]PEPTIDE", "N[UNIMOD:7]PEPTIDE", "简化脱酰胺 -> UNIMOD:7"),
        ("C[+57.02]PEPTIDE", "C[UNIMOD:4]PEPTIDE", "简化烷基化 -> UNIMOD:4"),
        
        # 真实数据示例（圆括号格式）-> Unimod ID
        ("SISC(+57.02)TYDDDTYR", "SISC[UNIMOD:4]TYDDDTYR", "数据示例1: 烷基化 -> UNIMOD:4"),
        ("AQ(+.98)IIMAANPFNRPDIFVK", "AQ[UNIMOD:7]IIMAANPFNRPDIFVK", "数据示例2: 脱酰胺 -> UNIMOD:7"),
        ("IDNIC(+57.02)AIFDINR", "IDNIC[UNIMOD:4]AIFDINR", "数据示例3: 烷基化 -> UNIMOD:4"),
        ("HM(+15.99)THGDIQNR", "HM[UNIMOD:35]THGDIQNR", "数据示例4: 氧化 -> UNIMOD:35"),
        
        # 多个修饰
        ("M[Oxidation]C[Carbamidomethyl]PEPTIDE", "M[UNIMOD:35]C[UNIMOD:4]PEPTIDE", "多个命名修饰"),
        ("M[+15.99]C[+57.02]PEPTIDE", "M[UNIMOD:35]C[UNIMOD:4]PEPTIDE", "多个数值修饰"),
        ("M(+15.99)C(+57.02)PEPTIDE", "M[UNIMOD:35]C[UNIMOD:4]PEPTIDE", "多个圆括号修饰"),
        
        # 已经是Unimod ID格式的应该保持不变
        ("M[UNIMOD:35]PEPTIDE", "M[UNIMOD:35]PEPTIDE", "已是Unimod ID格式"),
        ("C[UNIMOD:4]PEPTIDE", "C[UNIMOD:4]PEPTIDE", "已是Unimod ID格式"),
        
        # 混合格式
        ("M[Oxidation]C(+57.02)PEPTIDE", "M[UNIMOD:35]C[UNIMOD:4]PEPTIDE", "混合命名和数值"),
    ]
    
    print("="*80)
    print("Modification Format Conversion Test")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for input_seq, expected, description in test_cases:
        result = processor.convert_modification_format(input_seq)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
            print(f"{status} {description}")
            print(f"  输入:   {input_seq}")
            print(f"  输出:   {result}")
        else:
            failed += 1
            print(f"{status} {description} [失败]")
            print(f"  输入:   {input_seq}")
            print(f"  期望:   {expected}")
            print(f"  实际:   {result}")
        print()
    
    print("="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0

if __name__ == "__main__":
    success = test_modification_conversion()
    sys.exit(0 if success else 1)