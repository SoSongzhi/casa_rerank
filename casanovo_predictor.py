#!/usr/bin/env python
"""
Casanovo Python API 包装器
用于替代命令行调用，实现独立运行
"""

import logging
import tempfile
import shutil
from pathlib import Path
import torch
from casanovo.denovo.model import Spec2Pep
from casanovo.config import Config
from casanovo.denovo.dataloaders import DeNovoDataModule

logger = logging.getLogger(__name__)


class CasanovoPredictor:
    """
    Casanovo预测器 - 使用Python API直接预测
    """

    def __init__(self, model_path, config_path=None):
        """
        初始化预测器

        Parameters:
        -----------
        model_path : str
            模型checkpoint路径
        config_path : str, optional
            配置文件路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # 加载配置
        self.config = Config(config_path) if config_path else Config()

        # 加载模型
        logger.info(f"Loading model: {model_path}")
        self.model = Spec2Pep.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)
        logger.info("Model loaded successfully")

    def predict(self, mgf_file, output_file, n_beams=5, top_match=1):
        """
        预测MGF文件中的肽段序列

        Parameters:
        -----------
        mgf_file : str
            输入MGF文件路径
        output_file : str
            输出文件路径（.txt格式）
        n_beams : int
            Beam search宽度
        top_match : int
            每个谱图返回的候选数

        Returns:
        --------
        bool
            是否成功
        """
        try:
            # 更新模型参数
            self.model.n_beams = n_beams
            self.model.top_match = top_match

            # 创建临时Lance目录
            lance_dir = tempfile.mkdtemp(prefix="casanovo_lance_")

            try:
                # 创建数据模块
                logger.info(f"Processing {mgf_file}...")
                data_module = DeNovoDataModule(
                    lance_dir=lance_dir,
                    test_paths=[str(mgf_file)],
                    eval_batch_size=self.config.predict_batch_size,
                    min_peaks=self.config.min_peaks,
                    max_peaks=self.config.max_peaks,
                    min_mz=self.config.min_mz,
                    max_mz=self.config.max_mz,
                    min_intensity=self.config.min_intensity,
                    remove_precursor_tol=self.config.remove_precursor_tol,
                    max_charge=self.config.max_charge,
                    n_workers=0
                )

                # 设置测试阶段
                data_module.setup(stage="test", annotated=False)
                predict_loader = data_module.predict_dataloader()

                # 进行预测
                logger.info(f"Starting prediction with beam={n_beams}, top_match={top_match}...")
                results = []

                with torch.no_grad():
                    for batch_idx, batch in enumerate(predict_loader):
                        # 移动数据到设备
                        mzs = batch["mz_array"].to(self.device)
                        intensities = batch["intensity_array"].to(self.device)
                        precursor_mz = batch["precursor_mz"].to(self.device)
                        precursor_charge = batch["precursor_charge"].to(self.device)

                        # Beam search预测
 # Combine precursor info
 precursors = torch.stack([precursor_mz, precursor_charge.float()], dim=1)

 # Beam search预测
 predictions = self.model.beam_search_decode(mzs, intensities, precursors)
 # Combine precursor info
 precursors = torch.stack([precursor_mz, precursor_charge.float()], dim=1)

 # Beam search预测
 predictions = self.model.beam_search_decode(mzs, intensities, precursors)
 # Combine precursor info
 precursors = torch.stack([precursor_mz, precursor_charge.float()], dim=1)

 # Beam search预测
 predictions = self.model.beam_search_decode(mzs, intensities, precursors)
 # Combine precursor info
 precursors = torch.stack([precursor_mz, precursor_charge.float()], dim=1)

 # Beam search预测
 predictions = self.model.beam_search_decode(mzs, intensities, precursors)
 # Combine precursor info
 precursors = torch.stack([precursor_mz, precursor_charge.float()], dim=1)

 # Beam search预测
 predictions = self.model.beam_search_decode(mzs, intensities, precursors)
                        for pred in predictions:
                            for rank, (peptide, score) in enumerate(pred, start=1):
                                results.append({
                                    'spectrum_index': batch_idx,
                                    'rank': rank,
                                    'peptide': peptide,
                                    'score': score
                                })

                        if (batch_idx + 1) % 100 == 0:
                            logger.info(f"  Processed {batch_idx + 1} spectra...")

                # 保存结果到文件
                logger.info(f"Saving results to {output_file}...")
                with open(output_file, 'w') as f:
                    f.write("spectrum_index\trank\tpeptide\tscore\n")
                    for r in results:
                        f.write(f"{r['spectrum_index']}\t{r['rank']}\t{r['peptide']}\t{r['score']:.6f}\n")

                logger.info(f"Prediction completed! Total predictions: {len(results)}")
                return True

            finally:
                # 清理临时目录
                try:
                    shutil.rmtree(lance_dir)
                except:
                    pass

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python casanovo_predictor.py <model.ckpt> <input.mgf> <output.txt> [n_beams] [top_match]")
        sys.exit(1)

    model_path = sys.argv[1]
    mgf_file = sys.argv[2]
    output_file = sys.argv[3]
    n_beams = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    top_match = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    predictor = CasanovoPredictor(model_path)
    success = predictor.predict(mgf_file, output_file, n_beams, top_match)

    sys.exit(0 if success else 1)
