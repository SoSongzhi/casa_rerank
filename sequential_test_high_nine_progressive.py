#!/usr/bin/env python
"""
High-Nine 数据集顺序测试 - 使用渐进式Beam Search

特点:
1. 按谱图顺序逐个处理
2. 使用渐进式Beam Search (5→25→125→100)
3. De Novo预测后立即重排序
4. 实时显示进度和结果
5. 支持断点续传
6. 每个谱图处理完立即保存结果

渐进策略: 第1步5个beam → 第2步25个beam → 第3步125个beam → 第4步+100个beam
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
import logging
import einops
import collections
import tempfile
import shutil

from casanovo.denovo.model import Spec2Pep
from casanovo.denovo.dataloaders import DeNovoDataModule
from casanovo.config import Config

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from efficient_reranker import EfficientReranker
from build_efficient_index import EfficientIndexBuilder
from progressive_beam_search import ProgressiveBeamSpec2Pep

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ProgressiveSequentialProcessor:
    """使用渐进式Beam Search的逐个谱图处理器"""

    def __init__(self, test_mgf, reference_mgf, index_file, output_dir,
                 model_path="casanovo_v5_0_0_v5_0_0.ckpt", config_path=None,
                 beam_schedule=None):
        """
        初始化处理器

        Parameters:
        -----------
        beam_schedule : dict, optional
            渐进beam演化策略，如 {0: 5, 1: 25, 2: 125, 3: 100}
            默认使用 {0: 5, 1: 25, 2: 125, 3: 100}
        """
        self.test_mgf = test_mgf
        self.reference_mgf = reference_mgf
        self.index_file = index_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 设置beam schedule
        if beam_schedule is None:
            beam_schedule = {0: 5, 1: 25, 2: 125, 3: 100}
        self.beam_schedule = beam_schedule
        self.max_beam = max(beam_schedule.values())

        print(f"Initializing Progressive Beam Search Processor...")
        print(f"Progressive Strategy: {beam_schedule}")
        print(f"Max candidates: {self.max_beam}")

        # 初始化重排序器（使用渐进式beam）
        print("Initializing reranker with progressive beam...")
        self.reranker = EfficientReranker(
            model_path=model_path,
            config_path=config_path,
            use_progressive_beam=True,
            beam_schedule=beam_schedule
        )
        self.reranker.load_precomputed_index(index_file)

        # 状态文件
        self.state_file = self.output_dir / "processing_state.json"
        self.results_file = self.output_dir / "sequential_progressive_results.csv"
        self.temp_mgf_dir = self.output_dir / "temp_spectra"
        self.temp_mgf_dir.mkdir(exist_ok=True)

        # 加载或初始化状态
        self.state = self.load_state()

        # 获取已加载的渐进式模型
        self.progressive_model = self.reranker.model
        self.device = self.reranker.device
        self.config = self.reranker.config

        print(f"Progressive model initialized on device: {self.device}")

    def convert_modification_format(self, peptide: str) -> str:
        """
        将Casanovo的修饰格式转换为Unimod ID标准格式
        
        Unimod标准格式: 氨基酸[UNIMOD:ID]
        
        常见修饰的Unimod ID:
        - UNIMOD:4  - Carbamidomethyl (C+57.021464)
        - UNIMOD:7  - Deamidation (N/Q+0.984016)
        - UNIMOD:35 - Oxidation (M+15.994915)
        - UNIMOD:1  - Acetyl (N-term+42.010565)
        - UNIMOD:5  - Carbamyl (N-term+43.005814)
        - UNIMOD:28 - Gln->pyro-Glu (Q-17.026549)
        - UNIMOD:27 - Glu->pyro-Glu (E-18.010565)
        - UNIMOD:385 - Ammonia-loss (N-term-17.026549)
        
        转换示例:
        - M[Oxidation] -> M[UNIMOD:35]
        - M[+15.994915] -> M[UNIMOD:35]
        - M(+15.99) -> M[UNIMOD:35]
        - C[Carbamidomethyl] -> C[UNIMOD:4]
        - C[+57.021464] -> C[UNIMOD:4]
        - C(+57.02) -> C[UNIMOD:4]
        - N[Deamidated] -> N[UNIMOD:7]
        - N(+.98) -> N[UNIMOD:7]
        """
        if not peptide:
            return peptide
        
        # Casanovo修饰名称到Unimod ID的映射
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
        
        # 质量范围到Unimod ID的映射（容差±0.15 Da）
        mass_to_unimod = [
            # (最小值, 最大值, Unimod ID, 修饰名称)
            (15.85, 16.15, 'UNIMOD:35', 'Oxidation'),           # M氧化
            (0.83, 1.13, 'UNIMOD:7', 'Deamidation'),            # N/Q脱酰胺
            (56.87, 57.17, 'UNIMOD:4', 'Carbamidomethyl'),      # C烷基化
            (41.86, 42.16, 'UNIMOD:1', 'Acetyl'),               # N端乙酰化
            (42.86, 43.16, 'UNIMOD:5', 'Carbamyl'),             # N端氨甲酰化
            (-17.18, -16.88, 'UNIMOD:385', 'Ammonia-loss'),     # 氨损失
            (-18.16, -17.86, 'UNIMOD:27', 'Glu->pyro-Glu'),     # Glu环化
            (25.83, 26.13, 'UNIMOD:526', 'Carbamyl+Ammonia'),   # 组合修饰
            # 添加更多常见修饰
            (79.96, 80.06, 'UNIMOD:21', 'Phospho'),             # 磷酸化
            (14.01, 14.03, 'UNIMOD:34', 'Methyl'),              # 甲基化
            (42.04, 42.06, 'UNIMOD:1', 'Acetyl'),               # 乙酰化
        ]
        
        converted = peptide
        
        # 第一步: 处理Casanovo命名修饰 (如 M[Oxidation])
        for mod_name, unimod_id in mod_name_to_unimod.items():
            converted = converted.replace(f'[{mod_name}]', f'[{unimod_id}]')
        
        # 第二步: 处理数值修饰 (如 M[+15.994915] 或 M(+15.99) 或 M(+.98))
        import re
        # 匹配方括号或圆括号中的数值（但不匹配已经是UNIMOD格式的）
        # 支持 +15.99, +.98, +0.98 等格式
        pattern = r'[\[\(]([+\-]?\d*\.?\d+)[\]\)]'
        
        def replace_numeric_mod(match):
            mass_str = match.group(1)
            try:
                mass_val = float(mass_str)
                
                # 查找匹配的Unimod ID
                for min_mass, max_mass, unimod_id, mod_name in mass_to_unimod:
                    if min_mass <= mass_val <= max_mass:
                        logger.debug(f"Converting mass {mass_val:.6f} to {unimod_id} ({mod_name})")
                        return f'[{unimod_id}]'
                
                # 如果没有找到匹配的Unimod ID，保留原始质量但用方括号
                logger.warning(f"Unknown modification mass: {mass_val:.6f}, keeping as mass")
                if mass_val >= 0:
                    return f'[+{mass_val:.6f}]'
                else:
                    return f'[{mass_val:.6f}]'
                    
            except ValueError:
                # 如果转换失败，保持原样
                return f'[{mass_str}]'
        
        converted = re.sub(pattern, replace_numeric_mod, converted)
        
        return converted
    
    def is_unmodified(self, seq: str) -> bool:
        """判断肽段是否不带修饰（不包含括号/方括号/加号/数字）"""
        if seq is None:
            return False
        return re.search(r"[\[\]\(\)\+\d]", seq) is None

    def _extract_true_seq_from_params(self, params: dict) -> str:
        seq = params.get('seq', '')
        if not seq and 'title' in params:
            match = re.search(r'[Ss]eq[=:]([A-Z\[\]\(\)0-9\-\+\.]+)', params['title'])
            if match:
                seq = match.group(1)
        return seq

    def load_state(self):
        """加载处理状态，支持断点续传"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                print(f"Loaded existing state: processed {state['processed_count']}/{state['total_eligible']} spectra")
                return state
            except Exception as e:
                print(f"Failed to load state file, starting fresh: {e}")

        # 初始化新状态：扫描所有谱图（包括带修饰的）
        eligible_indices = []
        total_spectra = 0
        with mgf.MGF(self.test_mgf) as reader:
            for idx, spec in enumerate(reader):
                total_spectra += 1
                true_seq = self._extract_true_seq_from_params(spec.get('params', {}))
                # 现在处理所有谱图，包括带修饰的
                if true_seq:  # 只要有序列就处理
                    eligible_indices.append(idx)

        state = {
            'total_spectra': total_spectra,
            'eligible_indices': eligible_indices,
            'total_eligible': len(eligible_indices),
            'processed_count': 0,
            'processed_indices': [],
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'total_denovo_time': 0.0,
            'total_rerank_time': 0.0,
            'beam_schedule': self.beam_schedule
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
                    match = re.search(r'[Ss]eq[=:]([A-Z\[\]\(\)0-9\-\+\.]+)', spectrum['params']['title'])
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

    def _progressive_decode_single(self, spectrum_file: Path, top_match: int = 125):
        """
        使用真正的渐进式Beam Search解码单个谱图

        使用ProgressiveBeamSpec2Pep进行预测，策略: 5→25→125→100
        """
        start_time = time.time()
        try:
            # 创建临时lance目录
            lance_dir = tempfile.mkdtemp(prefix="casanovo_lance_")

            try:
                # 使用数据模块做预处理
                dm = DeNovoDataModule(
                    lance_dir=lance_dir,
                    test_paths=[str(spectrum_file)],
                    eval_batch_size=1,
                    min_peaks=self.config.min_peaks,
                    max_peaks=self.config.max_peaks,
                    min_mz=self.config.min_mz,
                    max_mz=self.config.max_mz,
                    min_intensity=self.config.min_intensity,
                    remove_precursor_tol=self.config.remove_precursor_tol,
                    max_charge=self.config.max_charge,
                    n_workers=0
                )
                dm.setup(stage="test", annotated=False)
                loader = dm.predict_dataloader()

                candidates = []

                with torch.no_grad():
                    for batch in loader:
                        # 移动数据到设备
                        mzs = batch["mz_array"].to(self.device)
                        intensities = batch["intensity_array"].to(self.device)
                        precursor_mz = batch["precursor_mz"].to(self.device)
                        precursor_charge = batch["precursor_charge"].to(self.device)

                        # 确保维度正确
                        if len(precursor_mz.shape) == 2:
                            precursor_mz = precursor_mz.squeeze(-1)
                        if len(precursor_charge.shape) == 2:
                            precursor_charge = precursor_charge.squeeze(-1)

                        # 计算precursor mass
                        precursor_masses = (precursor_mz - 1.007276) * precursor_charge
                        precursors = torch.stack([precursor_masses, precursor_charge, precursor_mz], dim=-1)

                        # 使用渐进式beam search预测
                        logger.info(f"Running progressive beam search: {self.beam_schedule}")
                        predictions = self.progressive_model.beam_search_decode(mzs, intensities, precursors)

                        # 处理结果
                        for spec_results in predictions:
                            for score, aa_scores, peptide in spec_results[:top_match]:
                                # 转换修饰格式：方括号 -> 圆括号
                                converted_peptide = self.convert_modification_format(peptide)
                                candidates.append({
                                    'peptide': converted_peptide,
                                    'score': float(score)
                                })

                # 排序并返回
                candidates.sort(key=lambda x: x['score'], reverse=True)
                return candidates[:top_match]

            finally:
                # 清理lance目录
                try:
                    shutil.rmtree(lance_dir, ignore_errors=True)
                except Exception:
                    pass

        except Exception as e:
            print(f"Progressive decode failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def run_denovo_single(self, spectrum_file, spec_idx):
        """对单个谱图运行渐进式De Novo预测"""
        start_time = time.time()
        try:
            candidates = self._progressive_decode_single(Path(spectrum_file), top_match=self.max_beam)
            return candidates, time.time() - start_time
        except Exception as e:
            print(f"Progressive decode failed for spectrum {spec_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None, time.time() - start_time

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

            # 单个谱图文件中的索引总是0
            result = self.reranker.rerank_with_efficient_index(
                str(temp_spectrum_file),
                0,
                candidates,
                use_prosit=True,
                top_k=3
            )

            rerank_time = time.time() - start_time

            # 添加Casanovo原始结果（De Novo Top 1）
            denovo_top1 = candidates[0] if candidates else {'peptide': '', 'score': 0}
            
            # 添加额外信息
            result.update({
                'spectrum_index': spec_idx,
                'true_sequence': spectrum_info['true_sequence'],
                'casanovo_peptide': denovo_top1['peptide'],  # Casanovo原始Top 1
                'casanovo_score': denovo_top1['score'],
                'rerank_peptide': result.get('peptide', ''),  # 重排序后Top 1
                'rerank_similarity': result.get('similarity', -1.0),
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
        total = self.state.get('total_eligible', self.state.get('total_spectra', 0))
        progress = (processed / total) * 100

        # 计算时间统计
        total_elapsed = time.time() - self._get_start_time()
        avg_time = total_elapsed / processed if processed > 0 else 0
        remaining = (total - processed) * avg_time

        # 当前谱图信息
        denovo_time = spectrum_info.get('denovo_time', 0)
        rerank_time = spectrum_info.get('rerank_time', 0)
        total_time = denovo_time + rerank_time

        # 获取候选信息
        candidates = spectrum_info.get('candidates', [])
        result = spectrum_info.get('result', {})
        true_seq = spectrum_info.get('true_sequence', '')

        print(f"\n{'='*70}")
        print(f"[{processed+1}/{total}] {progress:.1f}% | Spectrum {spec_idx}")
        print(f"{'='*70}")
        print(f"Progressive Strategy: {self.beam_schedule}")
        print(f"Timing: De Novo {denovo_time:.2f}s | Rerank {rerank_time:.2f}s | Total {total_time:.2f}s")
        print(f"True Sequence: {true_seq}")

        # 显示Progressive Beam Top 5
        if candidates:
            print(f"\nProgressive Beam Top 5 Candidates:")
            for i, candidate in enumerate(candidates[:5], 1):
                peptide = candidate['peptide']
                score = candidate['score']
                is_correct = False
                if true_seq:
                    pred_seq = self.reranker.normalize_peptide(peptide)
                    # 将True Sequence也转换为Unimod格式再比较
                    true_seq_unimod = self.convert_modification_format(true_seq)
                    true_seq_clean = self.reranker.normalize_peptide(true_seq_unimod)
                    is_correct = pred_seq == true_seq_clean and true_seq_clean != ''
                check = '[OK]' if is_correct else '    '
                print(f"   {i}. {peptide:20} (score: {score:.3f}) {check}")

        # 显示重排序结果
        if result and result.get('peptide'):
            pred_source = result.get('source', 'Unknown')
            similarity_score = result.get('similarity', -1.0)
            denovo_score = result.get('denovo_score', -1.0)

            print(f"\nRerank Top 1:")
            print(f"   Peptide: {result['peptide']} ({pred_source})")
            if similarity_score >= 0:
                print(f"   Similarity: {similarity_score:.4f}")
            if denovo_score >= 0:
                print(f"   De Novo Score: {denovo_score:.3f}")

            # 判断重排序结果是否正确
            is_rerank_correct = False
            if true_seq and result['peptide']:
                pred_seq = self.reranker.normalize_peptide(result['peptide'])
                # 将True Sequence也转换为Unimod格式再比较
                true_seq_unimod = self.convert_modification_format(true_seq)
                true_seq_clean = self.reranker.normalize_peptide(true_seq_unimod)
                is_rerank_correct = pred_seq == true_seq_clean and true_seq_clean != ''

            # 对比Progressive Beam Top 1 vs Rerank Top 1
            if candidates:
                progressive_top1 = candidates[0]
                progressive_correct = False
                if true_seq and progressive_top1['peptide']:
                    pred_seq = self.reranker.normalize_peptide(progressive_top1['peptide'])
                    # 将True Sequence也转换为Unimod格式再比较
                    true_seq_unimod = self.convert_modification_format(true_seq)
                    true_seq_clean = self.reranker.normalize_peptide(true_seq_unimod)
                    progressive_correct = pred_seq == true_seq_clean and true_seq_clean != ''

                print(f"\nComparison:")
                print(f"   Progressive Top 1: {progressive_top1['peptide']:20} (score: {progressive_top1['score']:.3f}) {'[OK]' if progressive_correct else '[X]'}")
                print(f"   Rerank Top 1:      {result['peptide']:20} (similarity: {similarity_score:.4f}) {'[OK]' if is_rerank_correct else '[X]'}")

                # 显示重排序效果
                if progressive_top1['peptide'] == result['peptide']:
                    print(f"   Reranking: No change (kept original top 1)")
                else:
                    if progressive_correct and not is_rerank_correct:
                        print(f"   Result: Reranking made it worse")
                    elif not progressive_correct and is_rerank_correct:
                        print(f"   Result: Reranking fixed it! (improved accuracy)")
        else:
            print(f"\nReranking failed - using Progressive Beam Top 1")

        print(f"\nETA: {remaining:.1f}s | Avg: {avg_time:.2f}s/spectrum")

    def _get_start_time(self):
        """获取开始时间戳"""
        try:
            from datetime import datetime
            return datetime.fromisoformat(self.state['start_time']).timestamp()
        except:
            return time.time()

    def process_all(self):
        """处理所有谱图"""
        total_eligible = self.state.get('total_eligible', 0)
        print(f"Starting progressive beam search processing of {total_eligible} spectra...")
        print(f"Progressive Strategy: {self.beam_schedule}")
        print(f"Resume from eligible position {self.state['processed_count']}")
        print("="*70)

        # 分别统计Casanovo和Rerank的准确率
        casanovo_correct = 0
        rerank_correct = 0
        total_processed = 0

        eligible = self.state.get('eligible_indices', list(range(self.state.get('total_spectra', 0))))
        for pos in range(self.state['processed_count'], len(eligible)):
            spec_idx = eligible[pos]
            if spec_idx in self.state['processed_indices']:
                # 已处理则推进计数指针
                self.state['processed_count'] = pos + 1
                self.save_state()
                continue

            # 提取谱图信息
            temp_spectrum_file = self.temp_mgf_dir / f"temp_{spec_idx}.mgf"
            spectrum_info = self.extract_single_spectrum(spec_idx, temp_spectrum_file)

            if spectrum_info is None:
                print(f"Skipping spectrum {spec_idx} (extraction failed)")
                continue

            # Progressive Beam De Novo预测
            candidates, denovo_time = self.run_denovo_single(temp_spectrum_file, spec_idx)
            spectrum_info['denovo_time'] = denovo_time
            spectrum_info['candidates'] = candidates

            if candidates is None:
                print(f"Skipping spectrum {spec_idx} (progressive decode failed)")
                temp_spectrum_file.unlink()
                continue

            # 重排序
            result = self.rerank_single_spectrum(spec_idx, candidates, spectrum_info)
            spectrum_info['result'] = result
            spectrum_info['rerank_time'] = result.get('rerank_time', 0) if result else 0

            # 保存结果
            if result:
                self.save_single_result(result)

                # 统计Casanovo和Rerank的准确率
                # 将True Sequence转换为Unimod格式再比较
                true_seq_unimod = self.convert_modification_format(spectrum_info['true_sequence'])
                true_seq_norm = self.reranker.normalize_peptide(true_seq_unimod)
                
                # Casanovo准确率
                if candidates:
                    casanovo_pred = self.reranker.normalize_peptide(candidates[0]['peptide'])
                    if true_seq_norm != '' and casanovo_pred == true_seq_norm:
                        casanovo_correct += 1
                
                # Rerank准确率
                rerank_pred = self.reranker.normalize_peptide(result.get('peptide', ''))
                if true_seq_norm != '' and rerank_pred == true_seq_norm:
                    rerank_correct += 1
                
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
        print(f"Progressive Strategy: {self.beam_schedule}")
        print(f"Total spectra processed: {total_processed}")
        print()
        
        # 显示Casanovo和Rerank的准确率对比
        if total_processed > 0:
            casanovo_acc = casanovo_correct / total_processed * 100
            rerank_acc = rerank_correct / total_processed * 100
            improvement = rerank_acc - casanovo_acc
            
            print("Accuracy Comparison:")
            print(f"  Casanovo (De Novo Top 1): {casanovo_correct}/{total_processed} = {casanovo_acc:.2f}%")
            print(f"  Rerank (After Reranking):  {rerank_correct}/{total_processed} = {rerank_acc:.2f}%")
            print(f"  Improvement: {improvement:+.2f}%")
            
            if improvement > 0:
                print(f"  ✓ Reranking improved accuracy!")
            elif improvement < 0:
                print(f"  ✗ Reranking decreased accuracy")
            else:
                print(f"  = No change in accuracy")
        else:
            print("No spectra processed")
        
        print()
        print(f"Timing:")
        print(f"  Total De Novo time: {self.state['total_denovo_time']:.1f}s")
        print(f"  Total rerank time: {self.state['total_rerank_time']:.1f}s")
        print(f"  Average time per spectrum: {(self.state['total_denovo_time'] + self.state['total_rerank_time'])/total_processed:.2f}s" if total_processed > 0 else "N/A")
        print()
        print(f"Results saved to: {self.results_file}")
        print("="*70)

def main():
    """主函数"""
    # 配置路径
    test_mgf = "test_data/high_nine/high_nine_validation_1000.mgf"
    reference_mgf = "test_data/high_nine/high_nine_database.mgf"
    index_file = f"{reference_mgf}.efficient_index.pkl"
    output_dir = Path("high_nine_results_sequential_progressive")

    # 检查索引文件
    if not Path(index_file).exists():
        print(f"Index file not found: {index_file}")
        print("Building index first...")
        builder = EfficientIndexBuilder()
        index = builder.build_index(reference_mgf)
        builder.save_index(index, index_file)

    # 渐进beam策略: 5→25→125→100
    beam_schedule = {0: 5, 1: 25, 2: 125, 3: 100}

    # 创建处理器并开始处理
    processor = ProgressiveSequentialProcessor(
        test_mgf=test_mgf,
        reference_mgf=reference_mgf,
        index_file=index_file,
        output_dir=output_dir,
        beam_schedule=beam_schedule
    )

    processor.process_all()

if __name__ == "__main__":
    main()
