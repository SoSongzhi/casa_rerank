#!/usr/bin/env python
"""
High-Nine 数据集顺序测试 - 逐个谱图处理

特点:
1. 按谱图顺序逐个处理
2. De Novo预测后立即重排序
3. 实时显示进度和结果
4. 支持断点续传
5. 每个谱图处理完立即保存结果
"""

import pandas as pd
import subprocess
import time
import pickle
import json
from pathlib import Path
import re
from pyteomics import mgf
import sys
import torch
import numpy as np
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from efficient_reranker import EfficientReranker
from build_efficient_index import EfficientIndexBuilder

class SequentialProcessor:
    """逐个谱图处理器"""
    
    def __init__(self, test_mgf, reference_mgf, index_file, output_dir, 
                 model_path="casanovo_v5_0_0_v5_0_0.ckpt", config_path="beam50.yaml"):
        self.test_mgf = test_mgf
        self.reference_mgf = reference_mgf
        self.index_file = index_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化重排序器
        print("Initializing reranker...")
        self.reranker = EfficientReranker(model_path=model_path, config_path=config_path)
        self.reranker.load_precomputed_index(index_file)
        
        # 状态文件
        self.state_file = self.output_dir / "processing_state.json"
        self.results_file = self.output_dir / "sequential_results.csv"
        self.temp_mgf_dir = self.output_dir / "temp_spectra"
        self.temp_mgf_dir.mkdir(exist_ok=True)
        
        # 加载或初始化状态
        self.state = self.load_state()
        
    def load_state(self):
        """加载处理状态，支持断点续传"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                print(f"Loaded existing state: processed {state['processed_count']}/{state['total_spectra']} spectra")
                return state
            except Exception as e:
                print(f"Failed to load state file, starting fresh: {e}")
        
        # 初始化新状态
        with mgf.MGF(self.test_mgf) as reader:
            total_spectra = sum(1 for _ in reader)
        
        state = {
            'total_spectra': total_spectra,
            'processed_count': 0,
            'processed_indices': [],
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'total_denovo_time': 0.0,
            'total_rerank_time': 0.0
        }
        
        # 保存初始状态
        self.save_state(state)
        return state
    
    def save_state(self, state=None):
        """保存处理状态"""
        if state is None:
            state = self.state
        state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def extract_single_spectrum(self, spec_idx, output_file):
        """提取单个谱图到临时MGF文件"""
        try:
            with mgf.MGF(self.test_mgf) as reader:
                spectra_list = list(reader)
                if spec_idx >= len(spectra_list):
                    return None
                
                spectrum = spectra_list[spec_idx]
                
                # 提取ground truth
                true_seq = spectrum['params'].get('seq', '')
                if not true_seq and 'title' in spectrum['params']:
                    match = re.search(r'[Ss]eq[=:]([A-Z\[\]0-9\-\+\.]+)', spectrum['params']['title'])
                    if match:
                        true_seq = match.group(1)
                
                # 保存单个谱图
                with open(output_file, 'w') as writer:
                    mgf.write([spectrum], writer)
                
                return {
                    'spectrum_index': spec_idx,
                    'true_sequence': true_seq,
                    'precursor_mz': spectrum['params'].get('pepmass', [0])[0] if isinstance(spectrum['params'].get('pepmass'), (list, tuple)) else spectrum['params'].get('pepmass', 0),
                    'charge': spectrum['params'].get('charge', [2])[0] if isinstance(spectrum['params'].get('charge'), (list, tuple)) else spectrum['params'].get('charge', 2)
                }
        except Exception as e:
            print(f"Error extracting spectrum {spec_idx}: {e}")
            return None
    
    def run_denovo_single(self, spectrum_file, spec_idx):
        """对单个谱图运行De Novo预测"""
        temp_output = self.temp_mgf_dir / f"denovo_{spec_idx}_{int(time.time())}"
        
        # 创建临时配置文件
        temp_config = self.temp_mgf_dir / f"beam50_single_{spec_idx}.yaml"
        try:
            with open("beam50.yaml", "r", encoding="utf-8") as cf:
                config_content = cf.read()
            
            # 确保单谱图处理设置
            config_content = re.sub(r'(?m)^\s*predict_batch_size\s*:\s*\d+\s*$', 'predict_batch_size: 1', config_content)
            if 'predict_batch_size' not in config_content:
                config_content += "\npredict_batch_size: 1\n"
            
            with open(temp_config, "w", encoding="utf-8") as sf:
                sf.write(config_content)
        except Exception as e:
            print(f"Warning: Could not create optimized config: {e}")
            temp_config = "beam50.yaml"
        
        # 运行Casanovo
        start_time = time.time()
        try:
            result = subprocess.run(
                [
                    "casanovo", "sequence",
                    str(spectrum_file),
                    "--config", str(temp_config),
                    "--output_root", str(temp_output)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            denovo_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"Casanovo failed for spectrum {spec_idx}: {result.stderr}")
                return None, denovo_time
            
            # 解析mzTab结果
            mztab_file = f"{temp_output}.mztab"
            candidates = self.parse_mztab_single(mztab_file, spec_idx)
            
            return candidates, denovo_time
            
        except subprocess.TimeoutExpired:
            print(f"Casanovo timeout for spectrum {spec_idx}")
            return None, time.time() - start_time
        except Exception as e:
            print(f"Error running denovo for spectrum {spec_idx}: {e}")
            return None, time.time() - start_time
        finally:
            # 清理临时文件
            if temp_config.exists() and str(temp_config) != "beam50.yaml":
                temp_config.unlink()
            for temp_file in self.temp_mgf_dir.glob(f"denovo_{spec_idx}_*"):
                if temp_file.is_file():
                    temp_file.unlink()
    
    def parse_mztab_single(self, mztab_file, spec_idx):
        """解析单个谱图的mzTab结果"""
        candidates = []
        try:
            with open(mztab_file, 'r', encoding="utf-8") as f:
                header = None
                for line in f:
                    if line.startswith('PSH'):
                        header = line.strip().split('\t')[1:]
                    elif line.startswith('PSM'):
                        values = line.strip().split('\t')[1:]
                        if header:
                            row = dict(zip(header, values))
                            peptide = row.get('sequence', '').strip()
                            score = float(row.get('search_engine_score[1]', 0))
                            
                            if peptide:  # 只添加非空的肽段序列
                                candidates.append({
                                    'peptide': peptide,
                                    'score': score
                                })
            
            print(f"  ├─ De Novo candidates found: {len(candidates)}")
            if candidates:
                print(f"  ├─ Top candidate: {candidates[0]['peptide']} (score: {candidates[0]['score']:.3f})")
            
            # 限制候选数量
            return candidates[:50]
            
        except Exception as e:
            print(f"Error parsing mztab for spectrum {spec_idx}: {e}")
            return []
    
    def rerank_single_spectrum(self, spec_idx, candidates, spectrum_info):
        """对单个谱图进行重排序"""
        if not candidates:
            return None
        
        start_time = time.time()
        try:
            # 创建临时谱图文件用于重排序
            temp_spectrum_file = self.temp_mgf_dir / f"spectrum_{spec_idx}.mgf"
            spectrum_data = self.extract_single_spectrum(spec_idx, temp_spectrum_file)
            
            if spectrum_data is None:
                return None
            
            # 关键修复：对于单个谱图的临时文件，索引应该是0
            result = self.reranker.rerank_with_efficient_index(
                str(temp_spectrum_file),
                0,  # 单个谱图文件中的索引总是0
                candidates,
                use_prosit=True,
                top_k=3
            )
            
            rerank_time = time.time() - start_time
            
            # 添加额外信息
            result.update({
                'spectrum_index': spec_idx,
                'true_sequence': spectrum_info['true_sequence'],
                'precursor_mz': spectrum_info['precursor_mz'],
                'charge': spectrum_info['charge'],
                'denovo_time': spectrum_info.get('denovo_time', 0),
                'rerank_time': rerank_time,
                'total_time': spectrum_info.get('denovo_time', 0) + rerank_time
            })
            
            return result
            
        except Exception as e:
            print(f"Error reranking spectrum {spec_idx}: {e}")
            return None
        finally:
            # 清理临时文件
            temp_file = self.temp_mgf_dir / f"spectrum_{spec_idx}.mgf"
            if temp_file.exists():
                temp_file.unlink()
    
    def save_single_result(self, result):
        """保存单个谱图的结果"""
        if result is None:
            return
        
        # 转换为DataFrame
        df = pd.DataFrame([result])
        
        # 如果结果文件不存在，创建新文件；否则追加
        if not self.results_file.exists():
            df.to_csv(self.results_file, index=False)
        else:
            df.to_csv(self.results_file, mode='a', header=False, index=False)
    
    def print_progress(self, spec_idx, spectrum_info):
        """打印详细的进度信息"""
        processed = self.state['processed_count']
        total = self.state['total_spectra']
        progress = (processed / total) * 100
        
        # 计算时间统计
        total_elapsed = time.time() - self._get_start_time()
        avg_time = total_elapsed / processed if processed > 0 else 0
        remaining = (total - processed) * avg_time
        
        # 当前谱图信息
        denovo_time = spectrum_info.get('denovo_time', 0)
        rerank_time = spectrum_info.get('rerank_time', 0)
        total_time = denovo_time + rerank_time
        
        # 判断是否正确
        is_correct = False
        pred_sequence = "N/A"
        
        if result := spectrum_info.get('result'):
            pred_sequence = result.get('peptide', 'N/A')
            true_seq = spectrum_info.get('true_sequence', '')
            if true_seq:
                pred_seq = self.reranker.normalize_peptide(pred_sequence)
                true_seq_clean = self.reranker.normalize_peptide(true_seq)
                is_correct = pred_seq == true_seq_clean and true_seq_clean != ''
        elif candidates := spectrum_info.get('candidates'):
            # 如果重排序失败，显示De Novo的top候选
            pred_sequence = candidates[0]['peptide'] if candidates else "N/A"
            true_seq = spectrum_info.get('true_sequence', '')
            if true_seq and pred_sequence != "N/A":
                pred_seq = self.reranker.normalize_peptide(pred_sequence)
                true_seq_clean = self.reranker.normalize_peptide(true_seq)
                is_correct = pred_seq == true_seq_clean and true_seq_clean != ''
        
        print(f"\n[{processed+1}/{total}] {progress:.1f}% | Spectrum {spec_idx}")
        print(f"  ├─ De Novo: {denovo_time:.2f}s | Rerank: {rerank_time:.2f}s | Total: {total_time:.2f}s")
        print(f"  ├─ True: {spectrum_info.get('true_sequence', 'N/A')}")
        print(f"  ├─ Pred: {pred_sequence}")
        print(f"  ├─ Correct: {'✓' if is_correct else '✗'}")
        print(f"  └─ ETA: {remaining:.1f}s | Avg: {avg_time:.2f}s/spectrum")
    
    def _get_start_time(self):
        """获取开始时间戳"""
        try:
            from datetime import datetime
            return datetime.fromisoformat(self.state['start_time']).timestamp()
        except:
            return time.time()
    
    def process_all(self):
        """处理所有谱图"""
        print(f"Starting sequential processing of {self.state['total_spectra']} spectra...")
        print(f"Resume from spectrum {self.state['processed_count']}")
        print("="*70)
        
        correct_count = 0
        total_processed = 0
        
        for spec_idx in range(self.state['processed_count'], self.state['total_spectra']):
            if spec_idx in self.state['processed_indices']:
                continue  # 跳过已处理的
            
            # 提取谱图信息
            temp_spectrum_file = self.temp_mgf_dir / f"temp_{spec_idx}.mgf"
            spectrum_info = self.extract_single_spectrum(spec_idx, temp_spectrum_file)
            
            if spectrum_info is None:
                print(f"Skipping spectrum {spec_idx} (extraction failed)")
                continue
            
            # De Novo预测
            candidates, denovo_time = self.run_denovo_single(temp_spectrum_file, spec_idx)
            spectrum_info['denovo_time'] = denovo_time
            spectrum_info['candidates'] = candidates  # 保存候选信息
            
            if candidates is None:
                print(f"Skipping spectrum {spec_idx} (denovo failed)")
                temp_spectrum_file.unlink()
                continue
            
            # 重排序
            result = self.rerank_single_spectrum(spec_idx, candidates, spectrum_info)
            spectrum_info['result'] = result
            spectrum_info['rerank_time'] = result.get('rerank_time', 0) if result else 0
            
            # 保存结果
            if result:
                self.save_single_result(result)
                
                # 统计正确率
                pred_seq = self.reranker.normalize_peptide(result.get('peptide', ''))
                true_seq = self.reranker.normalize_peptide(spectrum_info['true_sequence'])
                if true_seq != '' and pred_seq == true_seq:
                    correct_count += 1
                total_processed += 1
            
            # 更新状态
            self.state['processed_count'] += 1
            self.state['processed_indices'].append(spec_idx)
            self.state['total_denovo_time'] += denovo_time
            self.state['total_rerank_time'] += spectrum_info.get('rerank_time', 0)
            self.save_state()
            
            # 显示进度
            self.print_progress(spec_idx, spectrum_info)
            
            # 清理临时文件
            if temp_spectrum_file.exists():
                temp_spectrum_file.unlink()
        
        # 最终统计
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"Total spectra processed: {total_processed}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {correct_count/total_processed*100:.2f}%" if total_processed > 0 else "Accuracy: N/A")
        print(f"Total De Novo time: {self.state['total_denovo_time']:.1f}s")
        print(f"Total rerank time: {self.state['total_rerank_time']:.1f}s")
        print(f"Average time per spectrum: {(self.state['total_denovo_time'] + self.state['total_rerank_time'])/total_processed:.2f}s" if total_processed > 0 else "N/A")
        print(f"Results saved to: {self.results_file}")
        print("="*70)

def main():
    """主函数"""
    # 配置路径
    test_mgf = "test_data/high_nine/high_nine_validation_1000.mgf"
    reference_mgf = "test_data/high_nine/high_nine_database.mgf"
    index_file = f"{reference_mgf}.efficient_index.pkl"
    output_dir = Path("high_nine_results_sequential")
    
    # 检查索引文件
    if not Path(index_file).exists():
        print(f"Index file not found: {index_file}")
        print("Building index first...")
        builder = EfficientIndexBuilder()
        index = builder.build_index(reference_mgf)
        builder.save_index(index, index_file)
    
    # 创建处理器并开始处理
    processor = SequentialProcessor(
        test_mgf=test_mgf,
        reference_mgf=reference_mgf,
        index_file=index_file,
        output_dir=output_dir
    )
    
    processor.process_all()

if __name__ == "__main__":
    main()
