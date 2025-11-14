#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析Casanovo和Rerank结果对比
"""

import pandas as pd
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def normalize_peptide(peptide):
    """标准化肽段用于比较（去除修饰，L->I）"""
    import re
    if not peptide:
        return ''
    # 去除修饰
    clean = re.sub(r'\[.*?\]', '', peptide)
    clean = re.sub(r'\(.*?\)', '', clean)
    # L->I
    clean = clean.replace('L', 'I')
    return clean

def analyze_results(results_file):
    """分析结果文件"""
    
    print("="*80)
    print("Casanovo vs Rerank Accuracy Analysis")
    print("="*80)
    print(f"Results file: {results_file}\n")
    
    # 读取结果
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    total = len(df)
    print(f"Total spectra: {total}\n")
    
    # 标准化序列
    df['true_norm'] = df['true_sequence'].apply(normalize_peptide)
    df['casanovo_norm'] = df['casanovo_peptide'].apply(normalize_peptide)
    df['rerank_norm'] = df['rerank_peptide'].apply(normalize_peptide)
    
    # 计算准确率
    df['casanovo_correct'] = (df['true_norm'] == df['casanovo_norm']) & (df['true_norm'] != '')
    df['rerank_correct'] = (df['true_norm'] == df['rerank_norm']) & (df['true_norm'] != '')
    
    casanovo_correct = df['casanovo_correct'].sum()
    rerank_correct = df['rerank_correct'].sum()
    
    casanovo_acc = casanovo_correct / total * 100
    rerank_acc = rerank_correct / total * 100
    improvement = rerank_acc - casanovo_acc
    
    # 显示结果
    print("="*80)
    print("Accuracy Results:")
    print("="*80)
    print(f"Casanovo (De Novo Top 1): {casanovo_correct}/{total} = {casanovo_acc:.2f}%")
    print(f"Rerank (After Reranking):  {rerank_correct}/{total} = {rerank_acc:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")
    print()
    
    if improvement > 0:
        print(f"✓ Reranking improved accuracy by {improvement:.2f}%!")
    elif improvement < 0:
        print(f"✗ Reranking decreased accuracy by {abs(improvement):.2f}%")
    else:
        print(f"= No change in accuracy")
    print()
    
    # 分析重排序效果
    print("="*80)
    print("Reranking Effect Analysis:")
    print("="*80)
    
    # 情况1: Casanovo错误 -> Rerank正确
    fixed = df[(~df['casanovo_correct']) & df['rerank_correct']]
    print(f"Fixed by reranking: {len(fixed)} ({len(fixed)/total*100:.1f}%)")
    
    # 情况2: Casanovo正确 -> Rerank错误
    broken = df[df['casanovo_correct'] & (~df['rerank_correct'])]
    print(f"Broken by reranking: {len(broken)} ({len(broken)/total*100:.1f}%)")
    
    # 情况3: 都正确
    both_correct = df[df['casanovo_correct'] & df['rerank_correct']]
    print(f"Both correct: {len(both_correct)} ({len(both_correct)/total*100:.1f}%)")
    
    # 情况4: 都错误
    both_wrong = df[(~df['casanovo_correct']) & (~df['rerank_correct'])]
    print(f"Both wrong: {len(both_wrong)} ({len(both_wrong)/total*100:.1f}%)")
    print()
    
    # 按来源分析
    if 'source' in df.columns:
        print("="*80)
        print("Results by Source:")
        print("="*80)
        source_stats = df.groupby('source').agg({
            'rerank_correct': ['sum', 'count']
        })
        source_stats.columns = ['Correct', 'Total']
        source_stats['Accuracy'] = source_stats['Correct'] / source_stats['Total'] * 100
        print(source_stats)
        print()
    
    # 显示一些示例
    print("="*80)
    print("Sample Results:")
    print("="*80)
    
    if len(fixed) > 0:
        print("\nFixed by Reranking (first 3):")
        for idx, row in fixed.head(3).iterrows():
            print(f"  Spectrum {row['spectrum_index']}:")
            print(f"    True:     {row['true_sequence']}")
            print(f"    Casanovo: {row['casanovo_peptide']} (score: {row['casanovo_score']:.3f})")
            print(f"    Rerank:   {row['rerank_peptide']} (similarity: {row['rerank_similarity']:.4f})")
            print()
    
    if len(broken) > 0:
        print("\nBroken by Reranking (first 3):")
        for idx, row in broken.head(3).iterrows():
            print(f"  Spectrum {row['spectrum_index']}:")
            print(f"    True:     {row['true_sequence']}")
            print(f"    Casanovo: {row['casanovo_peptide']} (score: {row['casanovo_score']:.3f}) ✓")
            print(f"    Rerank:   {row['rerank_peptide']} (similarity: {row['rerank_similarity']:.4f}) ✗")
            print()
    
    print("="*80)

if __name__ == "__main__":
    results_file = "high_nine_results_sequential_progressive/sequential_progressive_results.csv"
    analyze_results(results_file)