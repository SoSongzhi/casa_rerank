"""
渐进扩展Beam Search
策略: 5 → 25 → 125 → 100 → 100 ...

使用方法:
    python progressive_beam_search.py \
        --mgf_file input.mgf \
        --spectrum_index 0 \
        --output results.txt
"""

import torch
import numpy as np
import einops
import collections
import heapq
from typing import Dict, List, Tuple
from pathlib import Path
import tempfile
import shutil
import logging
from pyteomics import mass

from casanovo.denovo.model import Spec2Pep
from casanovo.config import Config
from casanovo.denovo.dataloaders import DeNovoDataModule

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ProgressiveBeamSpec2Pep(Spec2Pep):
    """
    渐进扩展Beam Search的模型
    
    Beam演化:
    - 第1步: 5个beam
    - 第2步: 25个beam
    - 第3步: 125个beam
    - 第4步: 100个beam (收缩)
    - 第5步+: 100个beam (维持)
    """
    
    def __init__(self, *args, beam_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 默认的beam演化策略
        if beam_schedule is None:
            self.beam_schedule = {
                0: 5,    # 第1步
                1: 25,   # 第2步
                2: 125,  # 第3步
                3: 100,  # 第4步开始
            }
        else:
            self.beam_schedule = beam_schedule
        
        # 最大beam size
        self.max_beam_size = max(self.beam_schedule.values())
        
        logger.info(f"渐进Beam Search策略: {self.beam_schedule}")
        logger.info(f"最大beam size: {self.max_beam_size}")
    
    def calculate_peptide_mass_with_mods(self, tokens, charge=1):
        """
        计算带修饰的肽段质量（从token序列）
        
        Parameters:
        -----------
        tokens : torch.Tensor
            Token序列 (1D tensor)
        charge : int
            电荷态
        
        Returns:
        --------
        precursor_mz : float
            前体离子的m/z值
        """
        # Unimod ID到质量的映射
        unimod_masses = {
            'UNIMOD:35': 15.994915,   # Oxidation (M)
            'UNIMOD:4': 57.021464,    # Carbamidomethyl (C)
            'UNIMOD:7': 0.984016,     # Deamidation (N/Q)
            'UNIMOD:1': 42.010565,    # Acetyl (N-term)
            'UNIMOD:5': 43.005814,    # Carbamyl (N-term)
            'UNIMOD:28': -17.026549,  # Gln->pyro-Glu (Q)
            'UNIMOD:27': -18.010565,  # Glu->pyro-Glu (E)
            'UNIMOD:385': -17.026549, # Ammonia-loss (N-term)
            'UNIMOD:21': 79.966331,   # Phospho (S/T/Y)
            'UNIMOD:34': 14.015650,   # Methyl
        }
        
        # 将tokens转换为肽段序列
        # 确保tokens是2D的 (batch_size=1, seq_len)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        peptide = self.tokenizer.detokenize(tokens)[0]  # 取第一个结果
        
        # 提取氨基酸序列和修饰
        total_mod_mass = 0.0
        clean_seq = ""
        
        i = 0
        while i < len(peptide):
            if peptide[i] in '[(':
                # 找到修饰的结束位置
                end_char = ']' if peptide[i] == '[' else ')'
                end = peptide.find(end_char, i)
                if end == -1:
                    break
                
                mod_str = peptide[i+1:end]
                
                # 解析修饰
                if mod_str.startswith('UNIMOD:'):
                    # UNIMOD格式
                    if mod_str in unimod_masses:
                        total_mod_mass += unimod_masses[mod_str]
                else:
                    # 数值格式: +15.994915 或 15.994915 或 +.98
                    try:
                        mod_mass = float(mod_str.replace('+', ''))
                        total_mod_mass += mod_mass
                    except ValueError:
                        pass
                
                i = end + 1
            else:
                # 普通氨基酸
                if peptide[i].isalpha():
                    clean_seq += peptide[i]
                i += 1
        
        # 计算基础肽段质量（不含修饰）
        try:
            base_mass = mass.calculate_mass(sequence=clean_seq, charge=0)
            # 加上修饰质量
            total_mass = base_mass + total_mod_mass
            # 计算m/z
            precursor_mz = (total_mass + charge * 1.007276) / charge
            return precursor_mz
        except Exception as e:
            # 降级到不含修饰的计算
            return mass.calculate_mass(sequence=clean_seq, charge=charge)
    
    def get_beam_size(self, step):
        """获取指定步骤的beam size"""
        if step in self.beam_schedule:
            return self.beam_schedule[step]
        # 步骤超过定义的最大步骤时，使用最后一个值
        max_defined_step = max(self.beam_schedule.keys())
        return self.beam_schedule[max_defined_step]
    
    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重写_finish_beams方法以解决设备兼容问题
        """
        # logger.info(f"[DEBUG] 使用重写的_finish_beams (step={step})")
        device = self.device
        batch_size = tokens.shape[0]
        
        # 确保所有输入都在正确的设备上
        tokens = tokens.to(device)
        precursors = precursors.to(device)
        
        # 获取索引和质量数据
        nterm_idx = self.nterm_idx
        neg_mass_idx = self.neg_mass_idx
        token_masses = self.token_masses
        
        # 检查当前步的tokens
        current_tokens = tokens[:, step]
        
        # 初始化返回tensors
        beam_fits_precursor = torch.zeros(batch_size, dtype=torch.bool, device=device)
        finished_beams = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ends_stop_token = current_tokens == self.stop_token
        finished_beams[ends_stop_token] = True
        
        discarded_beams = torch.zeros(batch_size, dtype=torch.bool, device=device)
        discarded_beams[current_tokens == 0] = True
        
        # 检查无效的修饰组合
        if step > 1:
            dim0 = torch.arange(batch_size, device=device)
            final_pos = torch.full((batch_size,), step, device=device)
            final_pos[ends_stop_token] = step - 1
            
            # 检查多个N端修饰
            last_token_is_nterm = torch.isin(tokens[dim0, final_pos], nterm_idx)
            prev_token_is_nterm = torch.isin(tokens[dim0, final_pos - 1], nterm_idx)
            multiple_mods = last_token_is_nterm & prev_token_is_nterm
            
            # 检查内部N端修饰
            positions = torch.arange(tokens.shape[1], device=device)
            mask = (final_pos - 1).unsqueeze(1) >= positions
            token_mask = torch.where(mask, tokens, torch.zeros_like(tokens))
            internal_mods = torch.any(torch.isin(token_mask, nterm_idx), dim=1)
            
            discarded_beams = discarded_beams | multiple_mods | internal_mods
        
        # 获取precursor信息
        precursor_charges = precursors[:, 1]
        precursor_mzs = precursors[:, 2]
        
        # 计算肽段长度
        peptide_lens = torch.full((batch_size,), step + 1, device=device)
        if self.tokenizer.reverse:
            has_stop_at_start = tokens[:, 0] == self.stop_token
            peptide_lens[has_stop_at_start] -= 1
        else:
            has_stop_at_end = ends_stop_token
            peptide_lens[has_stop_at_end] -= 1
        
        # 丢弃太短的肽段
        too_short = finished_beams & (peptide_lens < self.min_peptide_len)
        discarded_beams[too_short] = True
        
        # 检查质量容差
        beams_to_check = ~discarded_beams
        
        if torch.any(beams_to_check):
            idx = torch.nonzero(beams_to_check).squeeze(-1)
            
            sequences_to_check = []
            charges_to_check = precursor_charges[idx]
            
            for i, beam_idx in enumerate(idx):
                seq = tokens[beam_idx, : step + 1]
                if seq[-1] == self.stop_token:
                    seq = seq[:-1]
                sequences_to_check.append(seq)
            
            if sequences_to_check:
                max_len = max(len(seq) for seq in sequences_to_check)
                if max_len > 0:
                    # 创建padded tensor (在device上)
                    padded_sequences = torch.zeros(
                        len(sequences_to_check),
                        max_len,
                        dtype=torch.int64,
                        device=device,
                    )
                    for i, seq in enumerate(sequences_to_check):
                        if len(seq) > 0:
                            padded_sequences[i, : len(seq)] = seq
                    
                    # ===== 关键修复：确保设备一致性 =====
                    # 移到CPU进行tokenizer计算
                    padded_cpu = padded_sequences.cpu()
                    charges_cpu = charges_to_check.cpu()
                    
                    # ===== 使用带修饰的质量计算 =====
                    recalc_mzs = []
                    for i, seq in enumerate(padded_sequences):
                        # 计算带修饰的质量
                        mz = self.calculate_peptide_mass_with_mods(
                            seq, charge=int(charges_to_check[i].item())
                        )
                        recalc_mzs.append(mz)
                    
                    # 转换为tensor并移到device
                    recalc_mzs = torch.tensor(recalc_mzs, dtype=torch.float64, device=device)
                    charges_device = charges_to_check.to(device).double()
                    
                    # 计算中性质量 (现在都在同一device上)
                    recalc_neutral_masses = (
                        recalc_mzs - 1.007276
                    ) * charges_device
                    
                    # 更新cumulative mass
                    self._cumulative_masses[idx] = recalc_neutral_masses.to(
                        self._cumulative_masses.dtype
                    )
                    
                    current_mzs = recalc_mzs
                else:
                    current_mzs = torch.zeros(len(idx), dtype=torch.float64, device=device)
            else:
                current_mzs = torch.zeros(len(idx), dtype=torch.float64, device=device)
            
            precursor_mzs_obs = precursor_mzs[idx].double()
            
            # 创建同位素误差范围
            isotope_range = torch.arange(
                self.isotope_error_range[0],
                self.isotope_error_range[1] + 1,
                device=device,
                dtype=torch.float64,
            )
            
            # 计算同位素校正
            isotope_corr = (
                isotope_range.unsqueeze(0)
                * 1.00335
                / charges_device.unsqueeze(1)
            )
            
            # 计算PPM差异
            delta_ppms = (
                (
                    current_mzs.unsqueeze(1)
                    - (precursor_mzs_obs.unsqueeze(1) - isotope_corr)
                )
                / precursor_mzs_obs.unsqueeze(1)
                * 1e6
            )
            
            # 检查是否在容差范围内
            matches_any = (torch.abs(delta_ppms) < self.precursor_mass_tol).any(dim=1)
            
            temp_matches = torch.zeros_like(beam_fits_precursor)
            temp_matches[idx] = matches_any
            
            # 决定是否强制终止不匹配的beams
            still_alive = ~finished_beams[idx]
            to_terminate = still_alive & ~matches_any
            
            # 处理负质量氨基酸可能修正质量的情况
            if torch.any(to_terminate) and self.neg_mass_idx.numel() > 0:
                exceeding_indices = torch.where(to_terminate)[0]
                neg_masses = self.token_masses[self.neg_mass_idx]
                
                exceeding_masses = recalc_neutral_masses[exceeding_indices]
                exceeding_charges = charges_device[exceeding_indices].double()
                exceeding_precursor_mzs = precursor_mzs_obs[exceeding_indices]
                
                # 计算加上负质量氨基酸后的潜在m/z
                potential_masses = exceeding_masses.unsqueeze(1) + neg_masses.double().unsqueeze(0)
                potential_mzs = (
                    potential_masses.unsqueeze(2)
                    / exceeding_charges.unsqueeze(1).unsqueeze(2)
                    + 1.007276
                )
                
                isotope_corr_expanded = (
                    isotope_range.unsqueeze(0).unsqueeze(0)
                    * 1.00335
                    / exceeding_charges.unsqueeze(1).unsqueeze(2)
                )
                observed_mzs_expanded = (
                    exceeding_precursor_mzs.unsqueeze(1).unsqueeze(2)
                    - isotope_corr_expanded
                )
                delta_ppms_neg = (
                    (potential_mzs - observed_mzs_expanded)
                    / exceeding_precursor_mzs.unsqueeze(1).unsqueeze(2)
                    * 1e6
                )
                
                any_neg_aa_works = torch.any(
                    torch.abs(delta_ppms_neg) < self.precursor_mass_tol,
                    dim=(1, 2),
                )
                any_not_strictly_exceeding = torch.any(
                    delta_ppms_neg <= self.precursor_mass_tol, dim=(1, 2)
                )
                
                # 更新可以被负质量氨基酸"拯救"的beams
                can_be_saved = any_neg_aa_works | any_not_strictly_exceeding
                temp_matches[idx[exceeding_indices]] |= can_be_saved
                to_terminate[exceeding_indices] = ~can_be_saved
            
            # 终止确认超出容差的beams
            to_terminate = idx[to_terminate]
            finished_beams[to_terminate] = True
            
            # 更新beam_fits_precursor（仅对已完成的beams）
            beam_fits_precursor |= temp_matches & finished_beams
        
        return finished_beams, beam_fits_precursor, discarded_beams
    
    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]],
    ):
        """
        重写_cache_finished_beams以支持动态beam size
        """
        cache_indices = torch.nonzero(beams_to_cache).squeeze(-1).cpu().tolist()
        device = self.device
        
        # 使用当前的beam size计算spec_idx
        current_beam_size = self._current_beam_size
        
        for i in cache_indices:
            # 计算spectrum索引（使用当前beam size）
            spec_idx = i // current_beam_size
            
            # 如果spec_idx超出范围，跳过
            if spec_idx >= self._batch_size:
                continue
            
            pred_tokens = tokens[i, : step + 1]
            
            # 移除stop token
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            
            # 检查重复
            pred_peptide_cpu = pred_peptide.cpu()
            duplicate = False
            for pred_cached in pred_cache[spec_idx]:
                if torch.equal(pred_cached[-1], pred_peptide_cpu):
                    duplicate = True
                    break
            
            if duplicate:
                continue
            
            # 计算scores
            smx = self.softmax(scores[i : i + 1, : step + 1, :])
            range_tensor = torch.arange(len(pred_tokens), device=device)
            aa_scores = smx[0, range_tensor, pred_tokens].cpu().numpy()
            
            if not has_stop_token:
                aa_scores = np.append(aa_scores, 0)
            
            # 计算peptide score
            from casanovo.denovo.model import _peptide_score
            peptide_score = _peptide_score(aa_scores, beam_fits_precursor[i].item())
            
            aa_scores = aa_scores[:-1]
            
            # 添加到cache
            if len(pred_cache[spec_idx]) < self.max_beam_size:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            
            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide_cpu),
                ),
            )
    
    def beam_search_decode(
        self,
        mzs: torch.Tensor,
        intensities: torch.Tensor,
        precursors: torch.Tensor,
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        渐进扩展的Beam search
        """
        memories, mem_masks = self.encoder(mzs, intensities)
        device = self.device
        
        # 确保所有输入都在正确的设备上
        memories = memories.to(device)
        if mem_masks is not None:
            mem_masks = mem_masks.to(device)
        precursors = precursors.to(device)
        
        batch = mzs.shape[0]
        length = self.max_peptide_len + 1
        vocab = self.vocab_size
        max_beam = self.max_beam_size
        
        # 使用max_beam初始化所有tensor（预分配最大空间）
        scores = torch.full(
            size=(batch, length, vocab, max_beam),
            fill_value=torch.nan,
            device=device,
        )
        
        tokens = torch.zeros(
            batch, length, max_beam, dtype=torch.int64, device=device
        )
        
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))
        
        # ===== 第1步: beam=5 =====
        step0_beam = self.get_beam_size(0)
        logger.info(f"\n第1步: beam size = {step0_beam}")
        
        pred = self.decoder(
            tokens=torch.zeros(batch, 0, dtype=torch.int64, device=device),
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )
        
        # 选top step0_beam个
        top_indices = torch.topk(pred[:, 0, :], step0_beam, dim=1)[1]
        tokens[:, 0, :step0_beam] = top_indices
        
        # 填充scores
        first_step_scores = einops.repeat(pred, "B L V -> B L V S", S=step0_beam)
        scores[:, :1, :, :step0_beam] = first_step_scores
        
        # 初始化cumulative masses
        token_masses = self.token_masses.to(device)
        cumulative_masses = torch.zeros(batch, max_beam, device=device)
        
        for b in range(batch):
            for s in range(step0_beam):
                token_idx = tokens[b, 0, s].item()
                if token_idx < len(token_masses):
                    cumulative_masses[b, s] = token_masses[token_idx]
        
        # ===== 主解码循环 =====
        current_beam = step0_beam
        
        # 存储临时属性
        self._batch_size = batch
        self._max_beam_size = max_beam
        self._current_beam_size = current_beam
        
        try:
            for step in range(0, self.max_peptide_len):
                # 获取当前步骤的beam size
                target_beam = self.get_beam_size(step + 1)
                
                if target_beam != current_beam:
                    logger.info(f"第{step+2}步: beam size {current_beam} → {target_beam}")
                
                # 准备当前步骤的tensors (只使用current_beam个，确保在device上)
                precursors_exp = einops.repeat(
                    precursors.to(device), "B L -> (B S) L", S=current_beam
                ).to(device)
                mem_masks_exp = einops.repeat(
                    mem_masks.to(device), "B L -> (B S) L", S=current_beam
                ).to(device)
                memories_exp = einops.repeat(
                    memories.to(device), "B L V -> (B S) L V", S=current_beam
                ).to(device)
                
                tokens_step = einops.rearrange(
                    tokens[:, :, :current_beam], "B L S -> (B S) L"
                )
                scores_step = einops.rearrange(
                    scores[:, :, :, :current_beam], "B L V S -> (B S) L V"
                )
                cumulative_masses_step = einops.rearrange(
                    cumulative_masses[:, :current_beam], "B S -> (B S)"
                )
                
                # 更新临时属性 (确保在正确的设备上)
                self._cumulative_masses = cumulative_masses_step.to(device)
                self._current_beam_size = current_beam
                
                # 确保所有tensors在正确的设备上
                tokens_step = tokens_step.to(device)
                precursors_exp = precursors_exp.to(device)
                
                # 检查完成的beams
                finished_beams, beam_fits_precursor, discarded_beams = self._finish_beams(
                    tokens_step, precursors_exp, step
                )
                
                # 缓存完成的beams
                beams_to_cache = finished_beams & ~discarded_beams
                if torch.any(beams_to_cache):
                    self._cache_finished_beams(
                        tokens_step, scores_step, step,
                        beams_to_cache, beam_fits_precursor,
                        pred_cache
                    )
                
                # 检查是否所有beam都完成
                finished_beams |= discarded_beams
                if torch.all(finished_beams):
                    break
                
                # 更新活跃beams的scores
                active_beams = ~finished_beams
                if torch.any(active_beams):
                    active_tokens = tokens_step[active_beams, : step + 1]
                    active_precursors = precursors_exp[active_beams]
                    active_memories = memories_exp[active_beams]
                    active_mem_masks = mem_masks_exp[active_beams]
                    
                    active_scores = self.decoder(
                        tokens=active_tokens,
                        precursors=active_precursors,
                        memory=active_memories,
                        memory_key_padding_mask=active_mem_masks,
                    )
                    
                    scores_step[active_beams, : step + 2, :] = active_scores
                
                # 选择top-k beams并更新到max_beam空间
                tokens_new, scores_new, cumulative_masses_new = self._get_progressive_topk_beams(
                    tokens_step, scores_step, cumulative_masses_step,
                    finished_beams, batch, current_beam, target_beam, step + 1
                )
                
                # 更新到主tensors (确保设备一致)
                tokens[:, :, :target_beam] = tokens_new.to(device)
                scores[:, :, :, :target_beam] = scores_new.to(device)
                cumulative_masses[:, :target_beam] = cumulative_masses_new.to(device)
                
                # 更新current_beam
                current_beam = target_beam
                
        finally:
            # 清理临时属性
            temp_attrs = ["_cumulative_masses", "_batch_size", "_max_beam_size", "_current_beam_size"]
            for attr in temp_attrs:
                if hasattr(self, attr):
                    delattr(self, attr)
        
        return list(self._get_top_peptide(pred_cache))
    
    def _get_progressive_topk_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        cumulative_masses: torch.Tensor,
        finished_beams: torch.Tensor,
        batch: int,
        current_beam: int,
        target_beam: int,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        渐进式选择top-k beams
        支持beam size的动态变化
        """
        vocab = self.vocab_size
        device = self.device
        token_masses = self.token_masses.to(device)
        
        # Reshape
        tokens_r = einops.rearrange(tokens, "(B S) L -> B L S", S=current_beam)
        scores_r = einops.rearrange(scores, "(B S) L V -> B L V S", S=current_beam)
        cumulative_masses_r = einops.rearrange(cumulative_masses, "(B S) -> B S", S=current_beam)
        
        # 获取previous tokens和scores
        prev_tokens = einops.repeat(
            tokens_r[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores_r[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )
        
        # 当前步的所有可能scores
        step_scores = torch.zeros(
            batch, step + 1, current_beam * vocab, device=device
        ).type_as(scores)
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores_r[:, step, :, :], "B V S -> B (V S)"
        )
        
        # Active mask
        active_mask = (
            ~finished_beams.reshape(batch, current_beam).repeat(1, vocab)
        ).float()
        active_mask[:, :current_beam] = 1e-8  # mask padding token
        
        # 计算mean scores并选择top-k
        mean_scores = torch.nanmean(step_scores, dim=1)
        _, top_idx = torch.topk(mean_scores * active_mask, target_beam, dim=1)
        
        # 转换索引
        indices = torch.unravel_index(top_idx.flatten(), (vocab, current_beam))
        v_idx = indices[0].reshape(top_idx.shape).to(device)
        s_idx = indices[1].reshape(top_idx.shape).to(device)
        
        # 创建新的tensors
        tokens_new = torch.zeros(
            batch, tokens_r.shape[1], target_beam,
            dtype=torch.int64, device=device
        )
        scores_new = torch.full(
            (batch, scores_r.shape[1], vocab, target_beam),
            fill_value=torch.nan, device=device
        )
        
        # 批量索引
        s_idx_flat = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(
            torch.arange(batch, device=device), "B -> (B S)", S=target_beam
        )
        
        # 填充tokens
        tokens_new[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx_flat], "(B S) L -> B L S", S=target_beam
        )
        tokens_new[:, step, :] = v_idx
        
        # 填充scores
        scores_new[:, : step + 1, :, :] = einops.rearrange(
            scores_r[b_idx, : step + 1, :, s_idx_flat],
            "(B S) L V -> B L V S",
            S=target_beam,
        )
        
        # 更新cumulative masses (确保在正确设备上)
        parent_masses = torch.gather(cumulative_masses_r, dim=1, index=s_idx).to(device)
        new_token_masses = token_masses[v_idx].to(device)
        cumulative_masses_new = (parent_masses + new_token_masses).to(device)
        
        # 确保返回的tensors都在正确设备上
        return tokens_new.to(device), scores_new.to(device), cumulative_masses_new


def extract_single_spectrum_to_file(mgf_file, spectrum_index, output_file):
    """提取单个谱图到新文件"""
    current_idx = 0
    spectrum_lines = []
    found = False
    
    with open(mgf_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('BEGIN IONS'):
                if current_idx == spectrum_index:
                    spectrum_lines = [line]
                    found = True
            elif line.strip().startswith('END IONS'):
                if current_idx == spectrum_index:
                    spectrum_lines.append(line)
                    break
                current_idx += 1
            elif found:
                spectrum_lines.append(line)
    
    if not found:
        raise ValueError(f"未找到谱图索引 {spectrum_index}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(spectrum_lines)


def predict_with_progressive_beam(
    mgf_file,
    spectrum_index=None,
    beam_schedule=None,
    model_path=None,
    config_path=None,
    output_file=None
):
    """
    使用渐进Beam Search进行预测
    
    如果指定了spectrum_index，会先提取该谱图到临时文件，
    确保使用MGF文件的原始索引，避免数据加载器过滤导致的索引偏移
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 如果指定了单个谱图索引，先提取该谱图
    temp_mgf = None
    if spectrum_index is not None:
        logger.info(f"提取MGF中的谱图 {spectrum_index}...")
        temp_mgf = f"temp_spectrum_{spectrum_index}_extracted.mgf"
        extract_single_spectrum_to_file(mgf_file, spectrum_index, temp_mgf)
        mgf_to_process = temp_mgf
        process_index = 0  # 提取后的文件只有1个谱图，索引为0
    else:
        mgf_to_process = mgf_file
        process_index = None
    
    # 加载模型
    if model_path is None:
        import appdirs
        cache_dir = Path(appdirs.user_cache_dir("casanovo", False, opinion=False))
        model_files = list(cache_dir.glob("*.ckpt"))
        if model_files:
            model_path = str(model_files[0])
    
    config = Config(config_path)
    logger.info(f"加载模型: {model_path}")
    
    # 加载基础模型
    base_model = Spec2Pep.load_from_checkpoint(model_path, map_location=device)
    
    # 创建渐进beam模型
    hparams = dict(base_model.hparams)
    
    # 设置输出数量为最大beam size
    if beam_schedule is None:
        beam_schedule = {0: 5, 1: 25, 2: 125, 3: 100}
    
    max_beam = max(beam_schedule.values())
    hparams['top_match'] = max_beam
    hparams['n_beams'] = beam_schedule[0]  # 初始beam size
    
    model = ProgressiveBeamSpec2Pep(
        **hparams,
        beam_schedule=beam_schedule
    )
    model.load_state_dict(base_model.state_dict(), strict=False)
    model.eval()
    model.to(device)
    
    # 加载数据（使用可能提取后的MGF文件）
    lance_dir = tempfile.mkdtemp(prefix="casanovo_lance_")
    
    try:
        data_module = DeNovoDataModule(
            lance_dir=lance_dir,
            test_paths=[mgf_to_process],
            eval_batch_size=1,
            min_peaks=config.min_peaks,
            max_peaks=config.max_peaks,
            min_mz=config.min_mz,
            max_mz=config.max_mz,
            min_intensity=config.min_intensity,
            remove_precursor_tol=config.remove_precursor_tol,
            max_charge=config.max_charge,
            n_workers=0
        )
        
        data_module.setup(stage="test", annotated=False)
        predict_loader = data_module.predict_dataloader()
        
        # 预测
        all_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(predict_loader):
                # 如果指定了单个谱图，只处理索引0（提取后的文件）
                if process_index is not None and batch_idx != process_index:
                    continue
                
                # 显示原始索引
                display_idx = spectrum_index if spectrum_index is not None else batch_idx
                logger.info(f"\n{'='*60}")
                logger.info(f"处理谱图 {display_idx}")
                logger.info(f"{'='*60}")
                
                mzs = batch["mz_array"].to(device)
                intensities = batch["intensity_array"].to(device)
                precursor_mz = batch["precursor_mz"].to(device)
                precursor_charge = batch["precursor_charge"].to(device)
                
                # 确保维度正确
                if len(precursor_mz.shape) == 2:
                    precursor_mz = precursor_mz.squeeze(-1)
                if len(precursor_charge.shape) == 2:
                    precursor_charge = precursor_charge.squeeze(-1)
                
                precursor_masses = (precursor_mz - 1.007276) * precursor_charge
                precursors = torch.stack([precursor_masses, precursor_charge, precursor_mz], dim=-1)
                
                # 运行渐进beam search
                results = model.beam_search_decode(mzs, intensities, precursors)
                
                # 处理结果
                for spec_results in results:
                    spectrum_results = []
                    for score, aa_scores, peptide in spec_results:
                        spectrum_results.append({
                            'spectrum_index': display_idx,  # 使用原始索引
                            'peptide': peptide,
                            'score': score,
                        })
                    all_results.extend(spectrum_results)
                    
                    logger.info(f"\n获得 {len(spectrum_results)} 个候选peptide")
                
                if process_index is not None:
                    break
        
        # 保存结果
        if output_file and all_results:
            with open(output_file, 'w') as f:
                f.write("Spectrum_Index\tRank\tPeptide\tScore\n")
                for idx, res in enumerate(all_results, 1):
                    f.write(f"{res['spectrum_index']}\t{idx}\t{res['peptide']}\t{res['score']:.6f}\n")
            logger.info(f"\n结果已保存到: {output_file}")
            
            # 显示前10个
            logger.info(f"\n{'='*60}")
            logger.info("Top 10 候选peptide:")
            logger.info(f"{'='*60}")
            for idx, res in enumerate(all_results[:10], 1):
                logger.info(f"{idx}. {res['peptide']:<20} score={res['score']:.6f}")
        
        return all_results
        
    finally:
        try:
            shutil.rmtree(lance_dir)
        except:
            pass
        
        # 清理临时MGF文件
        if temp_mgf and Path(temp_mgf).exists():
            try:
                Path(temp_mgf).unlink()
            except:
                pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='渐进Beam Search预测')
    parser.add_argument('--mgf_file', required=True, help='MGF文件')
    parser.add_argument('--spectrum_index', type=int, help='谱图索引（可选）')
    parser.add_argument('--output', default='progressive_beam_results.txt', help='输出文件')
    parser.add_argument('--beam_schedule', type=str, help='自定义beam演化（如"0:5,1:25,2:125,3:100"）')
    
    args = parser.parse_args()
    
    # 解析beam_schedule
    beam_schedule = None
    if args.beam_schedule:
        beam_schedule = {}
        for pair in args.beam_schedule.split(','):
            step, size = pair.split(':')
            beam_schedule[int(step)] = int(size)
    
    results = predict_with_progressive_beam(
        mgf_file=args.mgf_file,
        spectrum_index=args.spectrum_index,
        beam_schedule=beam_schedule,
        output_file=args.output
    )
    
    print(f"\n✓ 完成！共获得 {len(results)} 个候选peptide")

