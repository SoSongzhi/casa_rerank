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
import logging
import einops
import collections

from casanovo.denovo.model import Spec2Pep
from casanovo.denovo.dataloaders import DeNovoDataModule
from casanovo.config import Config

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
        
        # 复用已加载的Casanovo模型（来自重排序器）用于解码
        try:
            self.casa_model: Spec2Pep = self.reranker.model  # type: ignore
        except Exception:
            self.casa_model = None

    def is_unmodified(self, seq: str) -> bool:
        """判断肽段是否不带修饰（不包含括号/方括号/加号/数字）"""
        if seq is None:
            return False
        return re.search(r"[\[\]\(\)\+\d]", seq) is None

    def _extract_true_seq_from_params(self, params: dict) -> str:
        seq = params.get('seq', '')
        if not seq and 'title' in params:
            match = re.search(r'[Ss]eq[=:]([A-Z\[\]0-9\-\+\.]+)', params['title'])
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
        
        # 初始化新状态：预扫描可处理（无修饰）的谱图索引
        eligible_indices = []
        total_spectra = 0
        with mgf.MGF(self.test_mgf) as reader:
            for idx, spec in enumerate(reader):
                total_spectra += 1
                true_seq = self._extract_true_seq_from_params(spec.get('params', {}))
                if self.is_unmodified(true_seq):
                    eligible_indices.append(idx)

        state = {
            'total_spectra': total_spectra,
            'eligible_indices': eligible_indices,
            'total_eligible': len(eligible_indices),
            'processed_count': 0,              # 处理到eligible中的第几个
            'processed_indices': [],           # 实际已处理的原始谱图索引
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
                # 仅处理不带修饰的样本
                if not self.is_unmodified(true_seq):
                    return None

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
    
    def _prepare_config_with_beams(self, base_yaml: str, beams: int) -> Path:
        """基于beam50.yaml生成指定n_beams的临时配置，top_match固定50，batch_size=1"""
        temp_config = self.temp_mgf_dir / f"beam_{beams}_single.yaml"
        try:
            try:
                with open(base_yaml, "r", encoding="utf-8") as cf:
                    raw = cf.read()
            except Exception:
                raw = ""
            # 设置predict_batch_size
            if raw:
                raw = re.sub(r'(?m)^\s*predict_batch_size\s*:\s*\d+\s*$', 'predict_batch_size: 1', raw)
                if 'predict_batch_size' not in raw:
                    raw += "\npredict_batch_size: 1\n"
                # 设置n_beams 和 top_match
                raw = re.sub(r'(?m)^\s*n_beams\s*:\s*\d+\s*$', f'n_beams: {beams}', raw)
                if 'n_beams' not in raw:
                    raw += f"\nn_beams: {beams}\n"
                raw = re.sub(r'(?m)^\s*top_match\s*:\s*\d+\s*$', 'top_match: 50', raw)
                if 'top_match' not in raw:
                    raw += "\ntop_match: 50\n"
            else:
                raw = f"n_beams: {beams}\ntop_match: 50\npredict_batch_size: 1\n"
            with open(temp_config, "w", encoding="utf-8") as sf:
                sf.write(raw)
        except Exception as e:
            print(f"Warning: failed to prepare config for beams={beams}: {e}")
            temp_config = Path(base_yaml)
        return temp_config

    def _progressive_decode_single(self, spectrum_file: Path, branch_k: int = 20, beam_k: int = 50, top_match: int = 50):
        """使用Casanovo Python API进行逐步beam解码：每步每父beam扩展branch_k，合并后全局保留beam_k，输出top_match候选。"""
        if self.casa_model is None:
            # 回退：从重排序器的模型路径再载入
            logging.info("Loading Spec2Pep model for progressive decode from reranker path...")
            self.casa_model = Spec2Pep.load_from_checkpoint("casanovo_v5_0_0_v5_0_0.ckpt", map_location=self.reranker.device)  # type: ignore
            self.casa_model.eval()
            self.casa_model.to(self.reranker.device)

        # 用数据模块做与CLI一致的预处理
        cfg = Config()
        lance_dir = self.temp_mgf_dir / f"lance_{int(time.time())}"
        dm = DeNovoDataModule(
            lance_dir=str(lance_dir),
            test_paths=[str(spectrum_file)],
            eval_batch_size=1,
            min_peaks=cfg.min_peaks,
            max_peaks=cfg.max_peaks,
            min_mz=cfg.min_mz,
            max_mz=cfg.max_mz,
            min_intensity=cfg.min_intensity,
            remove_precursor_tol=cfg.remove_precursor_tol,
            max_charge=cfg.max_charge,
            n_workers=0
        )
        dm.setup(stage="test", annotated=False)
        loader = dm.predict_dataloader()

        # 为避免 depthcharge tokenizer 在GPU/CPU混用导致的设备不一致，强制在CPU上进行progressive解码
        original_device = self.reranker.device
        device = torch.device('cpu')
        model = self.casa_model
        model.to(device)
        model.top_match = top_match
        model.n_beams = beam_k  # 用作缓存大小

        candidates = []
        with torch.no_grad():
            for batch in loader:
                mzs = batch["mz_array"].to(device)
                intensities = batch["intensity_array"].to(device)
                precursor_mz = batch["precursor_mz"].to(device)
                precursor_charge = batch["precursor_charge"].to(device)
                precursors = torch.stack([mzs.new_tensor(0.0).expand_as(precursor_mz), precursor_charge.float(), precursor_mz], dim=1)

                # 基于beam_search_decode改写，局部branch_k，全球beam_k
                memories, mem_masks = model.encoder(mzs, intensities)
                B = mzs.shape[0]
                L = model.max_peptide_len + 1
                V = model.vocab_size
                S = beam_k

                scores = torch.full((B, L, V, S), float('nan'), device=device)
                tokens = torch.zeros(B, L, S, dtype=torch.int64, device=device)

                pred_cache = collections.OrderedDict((i, []) for i in range(B))

                pred = model.decoder(tokens=torch.zeros(B, 0, dtype=torch.int64, device=device), memory=memories, memory_key_padding_mask=mem_masks, precursors=precursors)

                # 第一步：每beam取branch_k，但起始只有一个父beam -> 直接取branch_k
                top_indices = torch.topk(pred[:, 0, :], branch_k, dim=1)[1]
                # 填充到S列，前branch_k为有效，其余复制最后一个，避免NaN
                tokens[:, 0, :branch_k] = top_indices
                tokens[:, 0, branch_k:] = top_indices[:, -1:].expand(B, S - branch_k)
                scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=S)

                model._batch_size = B
                model._beam_size = S
                model._cumulative_masses = torch.zeros(B * S, device=device)

                # 展开到(B*S)
                precursors_rep = einops.repeat(precursors, "B L -> (B S) L", S=S)
                mem_masks_rep = einops.repeat(mem_masks, "B L -> (B S) L", S=S)
                memories_rep = einops.repeat(memories, "B L V -> (B S) L V", S=S)
                tokens_rep = einops.rearrange(tokens, "B L S -> (B S) L")
                scores_rep = einops.rearrange(scores, "B L V S -> (B S) L V")

                # 主循环
                for step in range(0, model.max_peptide_len):
                    finished_beams, beam_fits_precursor, discarded_beams = model._finish_beams(tokens_rep, precursors_rep, step)

                    beams_to_cache = finished_beams & ~discarded_beams
                    if torch.any(beams_to_cache):
                        model._cache_finished_beams(tokens_rep, scores_rep, step, beams_to_cache, beam_fits_precursor, pred_cache)

                    finished_beams |= discarded_beams
                    if torch.all(finished_beams):
                        break

                    # 仅对活跃beam计算下一步
                    active = ~finished_beams
                    if torch.any(active):
                        active_tokens = tokens_rep[active, : step + 1]
                        active_precursors = precursors_rep[active]
                        active_memories = memories_rep[active]
                        active_mem_masks = mem_masks_rep[active]
                        active_scores = model.decoder(tokens=active_tokens, precursors=active_precursors, memory=active_memories, memory_key_padding_mask=active_mem_masks)
                        scores_rep[active, : step + 2, :] = active_scores

                    # 自定义选择：每个父beam取branch_k，再全局保留S
                    # 还原到(B,S)分组
                    tokens_bs = einops.rearrange(tokens_rep, "(B S) L -> B L S", S=S)
                    scores_bsv = einops.rearrange(scores_rep, "(B S) L V -> B L V S", S=S)

                    # 在当前步为每个父beam选择branch_k个token
                    logits = scores_bsv[:, step, :, :]  # B, V, S
                    logits = einops.rearrange(logits, "B V S -> B S V")
                    topk_vals, topk_idx = torch.topk(logits, k=branch_k, dim=2)

                    # 组合所有扩展，形成(B, S*branch_k)候选，然后按均值分数排序取S
                    # 计算到当前的均值分数（用nanmean避免NaN）
                    prev_tokens = tokens_bs[:, : step + 1, :]
                    prev_scores = torch.gather(scores_bsv[:, : step + 1, :, :], dim=2, index=einops.repeat(prev_tokens, "B L S -> B L 1 S").expand(-1, -1, 1, -1))
                    prev_scores = prev_scores[:, :, 0, :]  # B, L, S
                    prev_mean = torch.nanmean(prev_scores, dim=1)  # B, S

                    # 新token的得分取对应概率
                    new_vals = topk_vals  # B, S, K
                    # 合成候选得分：简单平均（等价于附加一步后新的均值）
                    new_mean = (prev_mean.unsqueeze(-1) * (step + 1) + new_vals) / (step + 2)  # B, S, K

                    # 选全局Top-S
                    new_mean_flat = einops.rearrange(new_mean, "B S K -> B (S K)")
                    best_vals, best_idx = torch.topk(new_mean_flat, k=S, dim=1)
                    # 反推对应的父beam和token
                    parent_idx = best_idx // branch_k  # B, S
                    token_sel = best_idx % branch_k    # B, S
                    v_idx = torch.gather(topk_idx, 2, token_sel.unsqueeze(-1)).squeeze(-1)  # B, S

                    # 重组tokens/scores到下一轮
                    gather_parent = parent_idx
                    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(gather_parent)
                    tokens_bs[:, : step + 1, :] = tokens_bs[b_idx, : step + 1, gather_parent]
                    tokens_bs[:, step + 1, :] = v_idx

                    scores_bsv[:, : step + 2, :, :] = scores_bsv[b_idx, : step + 2, :, gather_parent]
                    # 回写展平表示
                    tokens_rep = einops.rearrange(tokens_bs, "B L S -> (B S) L")
                    scores_rep = einops.rearrange(scores_bsv, "B L V S -> (B S) L V")

                # 收集结果（与模型一致）
                top_list = list(model._get_top_peptide(pred_cache))[0]
                for (pep_score, _aa_scores, seq) in top_list:
                    if self.is_unmodified(seq):
                        candidates.append({
                            'peptide': seq,
                            'score': float(pep_score)
                        })

        # 将模型移回原设备（供后续rerank继续使用GPU）
        try:
            model.to(original_device)
        except Exception:
            pass

        # 清理lance目录
        try:
            import shutil
            shutil.rmtree(lance_dir, ignore_errors=True)
        except Exception:
            pass

        # 排序并返回Top-50
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_match]

    def run_denovo_single(self, spectrum_file, spec_idx):
        """对单个谱图运行De Novo预测（自定义progressive beam：第0步20，第1步20*20->50，后续50*20->50）"""
        start_time = time.time()
        try:
            candidates = self._progressive_decode_single(Path(spectrum_file), branch_k=20, beam_k=50, top_match=50)
            return candidates, time.time() - start_time
        except Exception as e:
            print(f"Progressive decode failed for spectrum {spec_idx}: {e}")
            import traceback; traceback.print_exc()
            return None, time.time() - start_time
    
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
                            # 仅保留不带修饰的候选
                            if peptide and self.is_unmodified(peptide):
                                candidates.append({
                                    'peptide': peptide,
                                    'score': score
                                })
            
            # 按分数排序（降序）
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"  ├─ De Novo candidates found: {len(candidates)}")
            if candidates:
                print(f"  ├─ Top 3 candidates:")
                for i, candidate in enumerate(candidates[:3], 1):
                    print(f"     {i}. {candidate['peptide']:20} (score: {candidate['score']:.3f})")
            
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
        print(f"⏱️  Timing: De Novo {denovo_time:.2f}s | Rerank {rerank_time:.2f}s | Total {total_time:.2f}s")
        print(f"🎯 True Sequence: {true_seq}")
        
        # 显示Casanovo Top 5
        if candidates:
            print(f"\n🔬 Casanovo Top 5 Candidates:")
            for i, candidate in enumerate(candidates[:5], 1):
                peptide = candidate['peptide']
                score = candidate['score']
                is_correct = False
                if true_seq:
                    pred_seq = self.reranker.normalize_peptide(peptide)
                    true_seq_clean = self.reranker.normalize_peptide(true_seq)
                    is_correct = pred_seq == true_seq_clean and true_seq_clean != ''
                check = '✓' if is_correct else ' '
                print(f"   {i}. {peptide:20} (score: {score:.3f}) {check}")
        
        # 显示重排序结果
        if result and result.get('peptide'):
            pred_source = result.get('source', 'Unknown')
            similarity_score = result.get('similarity', -1.0)
            denovo_score = result.get('denovo_score', -1.0)
            
            # 来源图标
            source_icon = {
                'Database': '🗄️',
                'Prosit': '🧬', 
                'DeNovoFallback': '🔄',
                'DeNovo': '🔬',
                'NoResults': '❌',
                'Error': '⚠️'
            }.get(pred_source, '❓')
            
            print(f"\n🏆 Rerank Top 1:")
            print(f"   Peptide: {result['peptide']} {source_icon}({pred_source})")
            if similarity_score >= 0:
                print(f"   Similarity: {similarity_score:.4f}")
            if denovo_score >= 0:
                print(f"   De Novo Score: {denovo_score:.3f}")
            
            # 判断重排序结果是否正确
            is_rerank_correct = False
            if true_seq and result['peptide']:
                pred_seq = self.reranker.normalize_peptide(result['peptide'])
                true_seq_clean = self.reranker.normalize_peptide(true_seq)
                is_rerank_correct = pred_seq == true_seq_clean and true_seq_clean != ''
            
            # 对比Casanovo Top 1 vs Rerank Top 1
            if candidates:
                casanovo_top1 = candidates[0]
                casanovo_correct = False
                if true_seq and casanovo_top1['peptide']:
                    pred_seq = self.reranker.normalize_peptide(casanovo_top1['peptide'])
                    true_seq_clean = self.reranker.normalize_peptide(true_seq)
                    casanovo_correct = pred_seq == true_seq_clean and true_seq_clean != ''
                
                print(f"\n📊 Comparison:")
                print(f"   Casanovo Top 1: {casanovo_top1['peptide']:20} (rank: 1, score: {casanovo_top1['score']:.3f}) {'✓' if casanovo_correct else '✗'}")
                
                # 找到重排序结果在原始Casanovo列表中的位置
                rerank_peptide = result['peptide']
                original_rank = None
                for i, candidate in enumerate(candidates):
                    if candidate['peptide'] == rerank_peptide:
                        original_rank = i + 1
                        break
                
                if original_rank is not None:
                    print(f"   Rerank Top 1:   {result['peptide']:20} (original rank: {original_rank}, similarity: {similarity_score:.4f}) {'✓' if is_rerank_correct else '✗'}")
                else:
                    print(f"   Rerank Top 1:   {result['peptide']:20} (new candidate, similarity: {similarity_score:.4f}) {'✓' if is_rerank_correct else '✗'}")
                
                # 显示重排序效果
                if casanovo_top1['peptide'] == result['peptide']:
                    print(f"   📊 Reranking: No change (kept original top 1)")
                else:
                    if original_rank:
                        if original_rank > 1:
                            print(f"   🔄 Reranking: Promoted from rank {original_rank} to rank 1")
                        else:
                            print(f"   🔄 Reranking: Changed prediction")
                    else:
                        print(f"   🔄 Reranking: Selected different candidate")
                    
                    if casanovo_correct and not is_rerank_correct:
                        print(f"   ⚠️  Result: Reranking made it worse")
                    elif not casanovo_correct and is_rerank_correct:
                        print(f"   🎉 Result: Reranking fixed it! (improved accuracy)")
                    elif not casanovo_correct and not is_rerank_correct:
                        print(f"   📈 Result: Both wrong, but different approach")
                    else:
                        print(f"   ✅ Result: Both correct")
                        
                # 显示相似度如何影响排名
                if similarity_score >= 0:
                    print(f"   📈 Similarity Analysis:")
                    print(f"      Top similarity: {similarity_score:.4f}")
                    if original_rank and original_rank > 1:
                        print(f"      Original score: {casanovo_top1['score']:.3f}")
                        print(f"      Selected score: {candidates[original_rank-1]['score']:.3f}")
                        print(f"      ✅ Similarity overrode De Novo score")
                    elif original_rank == 1:
                        print(f"      ✅ De Novo top 1 confirmed by similarity")
                else:
                    print(f"   ❓ Similarity: No similarity score available")
        else:
            print(f"\n❌ Reranking failed - using Casanovo Top 1")
        
        print(f"\n⏳ ETA: {remaining:.1f}s | Avg: {avg_time:.2f}s/spectrum")
    
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
        print(f"Starting sequential processing of {total_eligible} unmodified spectra (out of {self.state['total_spectra']})...")
        print(f"Resume from eligible position {self.state['processed_count']}")
        print("="*70)

        correct_count = 0
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
