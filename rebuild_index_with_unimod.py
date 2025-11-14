#!/usr/bin/env python
"""
重新构建索引 - 将database转换为Unimod格式

使用方法:
python rebuild_index_with_unimod.py
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from build_efficient_index import EfficientIndexBuilder

def main():
    """重新构建索引，统一为Unimod格式"""
    
    # 配置
    reference_mgf = "test_data/high_nine/high_nine_database.mgf"
    index_file = f"{reference_mgf}.efficient_index.pkl"
    backup_file = f"{index_file}.backup"
    
    print("="*70)
    print("Rebuilding Index with Unimod Format Conversion")
    print("="*70)
    print()
    
    # 备份旧索引
    if Path(index_file).exists():
        print(f"Backing up existing index to: {backup_file}")
        import shutil
        shutil.copy(index_file, backup_file)
        print("✓ Backup complete")
        print()
    
    # 构建新索引
    print("Building new index with Unimod format...")
    print("This will convert all modifications to Unimod ID format:")
    print("  - M(+15.99) -> M[UNIMOD:35]")
    print("  - C(+57.02) -> C[UNIMOD:4]")
    print("  - N(+.98) -> N[UNIMOD:7]")
    print()
    
    builder = EfficientIndexBuilder()
    index = builder.build_index(reference_mgf)
    
    # 保存索引
    builder.save_index(index, index_file)
    
    print()
    print("="*70)
    print("✓ Index Rebuild Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. The new index uses Unimod format for all modifications")
    print("2. Database peptides will now match prediction results")
    print("3. Run your sequential test again:")
    print("   python sequential_test_high_nine_progressive.py")
    print()
    print(f"Note: Old index backed up to: {backup_file}")
    print("="*70)

if __name__ == "__main__":
    main()