#!/usr/bin/env python
"""
使用高效索引的快速重排序器

特性:
1. 精确匹配（完全相同的序列）
2. 取前10个匹配的谱图
3. 计算Top-3最高相似度的平均值
4. 零遍历快速查找
"""

import pickle
import numpy as np
import pandas as pd
import torch
import requests
import re
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from pyteomics import mgf, mass

from casanovo.denovo.model import Spec2Pep
from casanovo.config import Config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class EfficientReranker:
    """使用预计算索引的高效重排序器"""

    def __init__(self, model_path=None, config_path=None, koina_url="https://koina.wilhelmlab.org"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.config = Config(config_path)

        # 加载模型
        if model_path is None:
            import appdirs
            cache_dir = Path(appdirs.user_cache_dir("casanovo", False, opinion=False))
            model_files = list(cache_dir.glob("*.ckpt"))
            if model_files:
                model_path = str(model_files[0])

        logger.info(f"Loading model: {model_path}")
        self.model = Spec2Pep.load_from_checkpoint(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        self.koina_url = koina_url
        self.encoding_cache = {}  # 编码缓存
        self.precomputed_index = None  # 预计算索引

    def normalize_peptide(self, peptide):
        """标准化序列 - 去除修饰并将L转换为I"""
        clean_seq = re.sub(r'\[.*?\]', '', peptide)
        clean_seq = re.sub(r'^[\[\]A-Za-z0-9\-\+\.]+\-', '', clean_seq)
        # 将L转换为I（亮氨酸和异亮氨酸在质谱中无法区分）
        clean_seq = clean_seq.replace('L', 'I')
        return clean_seq

    def load_precomputed_index(self, index_file):
        """加载预计算索引"""
        logger.info(f"Loading precomputed index: {index_file}")
        with open(index_file, 'rb') as f:
            self.precomputed_index = pickle.load(f)
        logger.info(f"Loaded {len(self.precomputed_index)} unique sequences")
        return self.precomputed_index

    def encode_spectrum_from_arrays(self, mz_array, intensity_array, precursor_mz, precursor_charge):
        """从m/z和intensity数组直接编码"""
        # 过滤和排序峰
        valid_idx = intensity_array > 0
        mz_array = mz_array[valid_idx]
        intensity_array = intensity_array[valid_idx]

        # 按m/z排序
        sorted_idx = np.argsort(mz_array)
        mz_array = mz_array[sorted_idx]
        intensity_array = intensity_array[sorted_idx]

        # 归一化强度
        if len(intensity_array) > 0:
            intensity_array = intensity_array / np.max(intensity_array)

        # 限制峰数
        max_peaks = self.config.max_peaks
        if len(mz_array) > max_peaks:
            top_idx = np.argsort(intensity_array)[-max_peaks:]
            top_idx = np.sort(top_idx)  # 保持m/z顺序
            mz_array = mz_array[top_idx]
            intensity_array = intensity_array[top_idx]

        # Padding
        if len(mz_array) < max_peaks:
            pad_len = max_peaks - len(mz_array)
            mz_array = np.concatenate([mz_array, np.zeros(pad_len)])
            intensity_array = np.concatenate([intensity_array, np.zeros(pad_len)])

        # 转tensor
        mzs = torch.tensor([mz_array], dtype=torch.float32).to(self.device)
        intensities = torch.tensor([intensity_array], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            memories, _ = self.model.encoder(mzs, intensities)
            embedding = memories.mean(dim=1).cpu().numpy()[0]

        return embedding

    def get_spectrum_from_mgf(self, mgf_file, spec_index):
        """从MGF文件获取指定索引的谱图"""
        with mgf.MGF(mgf_file) as reader:
            for idx, spec in enumerate(reader):
                if idx == spec_index:
                    pepmass = spec['params'].get('pepmass', [0])
                    if isinstance(pepmass, (list, tuple)):
                        precursor_mz = pepmass[0] if len(pepmass) > 0 else 0
                    else:
                        precursor_mz = pepmass if pepmass else 0

                    charge = spec['params'].get('charge', [2])
                    if isinstance(charge, (list, tuple)):
                        charge = charge[0] if len(charge) > 0 else 2
                    else:
                        charge = charge if charge else 2

                    if isinstance(charge, str):
                        charge = int(charge.replace('+', '').replace('-', ''))
                    else:
                        charge = int(charge)

                    return {
                        'mz': spec['m/z array'],
                        'intensity': spec['intensity array'],
                        'precursor_mz': float(precursor_mz),
                        'charge': charge
                    }
        return None

    def generate_prosit_spectrum(self, peptide, charge, nce=25.0):
        """使用Prosit生成理论谱图"""
        clean_seq = self.normalize_peptide(peptide)

        url = f"{self.koina_url}/v2/models/Prosit_2020_intensity_HCD/infer"
        payload = {
            "id": "prosit_prediction",
            "inputs": [
                {"name": "peptide_sequences", "shape": [1, 1], "datatype": "BYTES", "data": [clean_seq]},
                {"name": "precursor_charges", "shape": [1, 1], "datatype": "INT32", "data": [int(charge)]},
                {"name": "collision_energies", "shape": [1, 1], "datatype": "FP32", "data": [float(nce)]}
            ]
        }

        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
            if response.status_code != 200:
                return None

            result = response.json()
            intensities_data = None
            for output in result.get('outputs', []):
                if output.get('name') == 'intensities':
                    intensities_data = output.get('data', [])
                    break

            if not intensities_data:
                return None

            # 解析为峰列表
            mz_list, intensity_list = self._prosit_to_peaks(clean_seq, charge, intensities_data)

            if len(mz_list) == 0:
                return None

            return {
                'mz': np.array(mz_list),
                'intensity': np.array(intensity_list),
                'precursor_mz': mass.calculate_mass(sequence=clean_seq, charge=charge),
                'charge': charge
            }
        except Exception as e:
            logger.warning(f"Prosit failed for {peptide}: {e}")
            return None

    def _prosit_to_peaks(self, sequence, charge, intensities):
        """将Prosit强度转为峰列表"""
        mz_list, intensity_list = [], []
        peptide_len = len(sequence)
        idx = 0

        # b ions
        for ion_charge in [1, 2, 3]:
            for pos in range(1, peptide_len):
                fragment = sequence[:pos]
                for mod in ['', '-H2O', '-NH3']:
                    if idx < len(intensities):
                        intensity = intensities[idx]
                        if intensity > 0:
                            b_mz = mass.fast_mass(fragment, ion_type='b', charge=ion_charge)
                            if mod == '-H2O':
                                b_mz -= 18.01056 / ion_charge
                            elif mod == '-NH3':
                                b_mz -= 17.02655 / ion_charge
                            mz_list.append(b_mz)
                            intensity_list.append(max(float(intensity), 0.00001))
                    idx += 1

        # y ions
        for ion_charge in [1, 2, 3]:
            for pos in range(1, peptide_len):
                fragment = sequence[-pos:]
                for mod in ['', '-H2O', '-NH3']:
                    if idx < len(intensities):
                        intensity = intensities[idx]
                        if intensity > 0:
                            y_mz = mass.fast_mass(fragment, ion_type='y', charge=ion_charge)
                            if mod == '-H2O':
                                y_mz -= 18.01056 / ion_charge
                            elif mod == '-NH3':
                                y_mz -= 17.02655 / ion_charge
                            mz_list.append(y_mz)
                            intensity_list.append(max(float(intensity), 0.00001))
                    idx += 1

        return mz_list, intensity_list

    def rerank_with_efficient_index(
        self,
        query_mgf,
        query_index,
        candidates,
        use_prosit=True,
        top_k=3
    ):
        """
        使用高效索引重排序

        Parameters:
        - query_mgf: 查询谱图MGF
        - query_index: 谱图索引
        - candidates: list of dict with 'peptide' and 'score'
        - use_prosit: 是否对未匹配的使用Prosit
        - top_k: 取前k个最高相似度的平均（默认3）
        """
        if self.precomputed_index is None:
            raise ValueError("Precomputed index not loaded. Call load_precomputed_index() first.")

        # 1. 编码查询谱图
        query_spec = self.get_spectrum_from_mgf(query_mgf, query_index)
        if query_spec is None:
            raise ValueError(f"Failed to load spectrum at index {query_index}")

        query_embedding = self.encode_spectrum_from_arrays(
            query_spec['mz'], query_spec['intensity'],
            query_spec['precursor_mz'], query_spec['charge']
        )

        # 2. 为每个候选计算相似度
        results = []

        for candidate in candidates:
            peptide = candidate['peptide']
            denovo_score = candidate['score']
            clean_peptide = self.normalize_peptide(peptide)

            # O(1) 精确查找
            if clean_peptide in self.precomputed_index:
                ref_spectra = self.precomputed_index[clean_peptide]["spectra"]

                # 限制为前10个
                ref_spectra = ref_spectra[:10]

                # 计算所有匹配谱图的相似度
                similarities = []
                for ref_spec in ref_spectra:
                    # 检查缓存
                    cache_key = f"ref_{ref_spec['index']}"
                    if cache_key in self.encoding_cache:
                        ref_embedding = self.encoding_cache[cache_key]
                    else:
                        ref_embedding = self.encode_spectrum_from_arrays(
                            ref_spec['mz'], ref_spec['intensity'],
                            ref_spec['precursor_mz'], ref_spec['charge']
                        )
                        self.encoding_cache[cache_key] = ref_embedding

                    sim = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        ref_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

                # 选择Top-K最高相似度的平均
                top_k_similarities = sorted(similarities, reverse=True)[:top_k]
                final_similarity = np.mean(top_k_similarities)

                results.append({
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': final_similarity,
                    'matched_count': len(ref_spectra),
                    'all_similarities': similarities,
                    'top_k_similarities': top_k_similarities,
                    'source': 'Database'
                })

            elif use_prosit:
                # 使用Prosit生成
                prosit_spec = self.generate_prosit_spectrum(peptide, query_spec['charge'])

                if prosit_spec:
                    prosit_embedding = self.encode_spectrum_from_arrays(
                        prosit_spec['mz'], prosit_spec['intensity'],
                        prosit_spec['precursor_mz'], prosit_spec['charge']
                    )

                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        prosit_embedding.reshape(1, -1)
                    )[0][0]

                    results.append({
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': similarity,
                        'matched_count': 0,
                        'source': 'Prosit'
                    })
                else:
                    results.append({
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': -1.0,
                        'matched_count': 0,
                        'source': 'Failed'
                    })
            else:
                # 不使用Prosit
                results.append({
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': -1.0,
                    'matched_count': 0,
                    'source': 'NotFound'
                })

        # 3. 按相似度排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity', ascending=False).reset_index(drop=True)
        results_df['rerank'] = range(1, len(results_df) + 1)

        return results_df
