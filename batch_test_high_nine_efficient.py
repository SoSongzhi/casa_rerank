#!/usr/bin/env python
"""
High-Nine 数据集批量测试 - 使用高效索引重排序

改进点:
1. 使用预计算索引（零遍历）
2. Top-3 相似度平均策略
3. 精确匹配（完全相同的序列）
"""

import pandas as pd
import subprocess
import time
from pathlib import Path
import re
from pyteomics import mgf
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from efficient_reranker import EfficientReranker
from build_efficient_index import EfficientIndexBuilder

# 配置路径
test_mgf = "test_data/high_nine/high_nine_validation_1000.mgf"
reference_mgf = "test_data/high_nine/high_nine_database.mgf"
index_file = f"{reference_mgf}.efficient_index.pkl"
output_dir = Path("high_nine_results_efficient")
output_dir.mkdir(exist_ok=True)

# 自动统计谱图总数
num_spectra = 0
try:
    with mgf.MGF(test_mgf) as _reader:
        for _ in _reader:
            num_spectra += 1
except Exception as _e:
    print(f"Failed to count spectra from {test_mgf}: {_e}")
    raise

print("="*70)
print(f"Batch Test: High-Nine Dataset (Efficient Reranker)")
print(f"Processing ALL {num_spectra} spectra with Beam=50")
print("="*70)

# Step 1: 确保索引存在
if not Path(index_file).exists():
    print(f"\nBuilding index (first time only)...")
    builder = EfficientIndexBuilder()
    index = builder.build_index(reference_mgf)
    builder.save_index(index, index_file)
else:
    print(f"\nUsing existing index: {index_file}")

# Step 2: 提取全部谱图
print(f"\nStep 1: Extracting all {num_spectra} spectra...")
test_all_mgf = output_dir / f"test_{num_spectra}_spectra.mgf"
ground_truth = []

try:
    with mgf.MGF(test_mgf) as reader:
        with open(test_all_mgf, 'w') as writer:
            for idx, spec in enumerate(reader):
                true_seq = spec['params'].get('seq', '')
                if not true_seq and 'title' in spec['params']:
                    match = re.search(r'[Ss]eq[=:]([A-Z\[\]0-9\-\+\.]+)', spec['params']['title'])
                    if match:
                        true_seq = match.group(1)

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

                ground_truth.append({
                    'spectrum_index': idx,
                    'true_sequence': true_seq,
                    'precursor_mz': precursor_mz,
                    'charge': charge
                })

                mgf.write([spec], writer)

    print(f"Extracted {len(ground_truth)} spectra to: {test_all_mgf}")
except Exception as e:
    print(f"Error reading MGF: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 保存 ground truth
gt_df = pd.DataFrame(ground_truth)
gt_df.to_csv(output_dir / "ground_truth.csv", index=False)

print(f"\nGround truth sample (first 10):")
for i, item in enumerate(ground_truth[:10]):
    print(f"  Spectrum {item['spectrum_index']}: {item['true_sequence']}")

# Step 3: 使用 beam50 配置运行 Casanovo denovo
print(f"\nStep 2: Running Casanovo denovo (beam=50)...")
config_file = "beam50.yaml"
denovo_output = output_dir / f"denovo_predictions_{int(time.time())}"

# 生成安全配置
safe_config = output_dir / "beam50_oomsafe.yaml"
try:
    raw = ""
    try:
        with open(config_file, "r", encoding="utf-8") as cf:
            raw = cf.read()
    except Exception:
        raw = ""
    import re as _re
    if raw:
        raw = _re.sub(r'(?m)^\s*predict_batch_size\s*:\s*\d+\s*$', 'predict_batch_size: 1', raw)
        if 'predict_batch_size' not in raw:
            raw += "\npredict_batch_size: 1\n"
        raw = _re.sub(r'(?m)^\s*n_beams\s*:\s*\d+\s*$', 'n_beams: 50', raw)
        if 'n_beams' not in raw:
            raw += "\nn_beams: 50\n"
        raw = _re.sub(r'(?m)^\s*top_match\s*:\s*\d+\s*$', 'top_match: 50', raw)
        if 'top_match' not in raw:
            raw += "\ntop_match: 50\n"
    else:
        raw = "n_beams: 50\ntop_match: 50\npredict_batch_size: 1\n"
    with open(safe_config, "w", encoding="utf-8") as sf:
        sf.write(raw)
    use_config = str(safe_config)
except Exception as e:
    print(f"Failed to prepare safe config, fallback to {config_file}: {e}")
    use_config = str(config_file)

start_time = time.time()
result = subprocess.run(
    [
        "casanovo", "sequence",
        str(test_all_mgf),
        "--config", use_config,
        "--output_root", str(denovo_output)
    ],
    capture_output=True,
    text=True
)
denovo_time = time.time() - start_time

if result.returncode != 0:
    print(f"Casanovo failed!")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")

    # 回退方案：降低 beam 避免 OOM 或 topk 越界
    err_msg = (result.stderr or "") + (result.stdout or "")
    need_retry = ("out of memory" in err_msg.lower()) or ("selected index k out of range" in err_msg.lower())
    if need_retry:
        print("\nRetrying with safer settings: n_beams=20, top_match=50, predict_batch_size=1 ...")
        fallback_cfg = output_dir / "beam20_safe.yaml"
        try:
            raw2 = ""
            try:
                with open(config_file, "r", encoding="utf-8") as cf:
                    raw2 = cf.read()
            except Exception:
                raw2 = ""
            import re as _re2
            if raw2:
                raw2 = _re2.sub(r'(?m)^\s*n_beams\s*:\s*\d+\s*$', 'n_beams: 20', raw2)
                if 'n_beams' not in raw2:
                    raw2 += "\nn_beams: 20\n"
                raw2 = _re2.sub(r'(?m)^\s*top_match\s*:\s*\d+\s*$', 'top_match: 50', raw2)
                if 'top_match' not in raw2:
                    raw2 += "\ntop_match: 50\n"
                raw2 = _re2.sub(r'(?m)^\s*predict_batch_size\s*:\s*\d+\s*$', 'predict_batch_size: 1', raw2)
                if 'predict_batch_size' not in raw2:
                    raw2 += "\npredict_batch_size: 1\n"
            else:
                raw2 = "n_beams: 20\ntop_match: 50\npredict_batch_size: 1\n"
            with open(fallback_cfg, "w", encoding="utf-8") as sf:
                sf.write(raw2)

            denovo_output_retry = output_dir / f"denovo_predictions_retry_{int(time.time())}"
            start_time = time.time()
            result2 = subprocess.run(
                [
                    "casanovo", "sequence",
                    str(test_all_mgf),
                    "--config", str(fallback_cfg),
                    "--output_root", str(denovo_output_retry)
                ],
                capture_output=True,
                text=True
            )
            denovo_time = time.time() - start_time

            if result2.returncode != 0:
                print("Retry also failed.")
                print(f"stdout: {result2.stdout}")
                print(f"stderr: {result2.stderr}")
                exit(1)
            else:
                denovo_output = denovo_output_retry
                print(f"Retry succeeded in {denovo_time:.1f}s")
        except Exception as e:
            print(f"Retry preparation failed: {e}")
            exit(1)
    else:
        exit(1)
else:
    print(f"Casanovo completed in {denovo_time:.1f}s ({denovo_time/num_spectra:.2f}s per spectrum)")

# Step 4: 解析 mztab
mztab_file = f"{denovo_output}.mztab"
print(f"\nStep 3: Parsing denovo results from {mztab_file}...")

denovo_results = []
with open(mztab_file, 'r', encoding="utf-8") as f:
    header = None
    for line in f:
        if line.startswith('PSH'):
            header = line.strip().split('\t')[1:]
        elif line.startswith('PSM'):
            values = line.strip().split('\t')[1:]
            if header:
                row = dict(zip(header, values))
                denovo_results.append({
                    'spectrum_id': row.get('spectra_ref', ''),
                    'peptide': row.get('sequence', ''),
                    'score': float(row.get('search_engine_score[1]', 0))
                })

denovo_df = pd.DataFrame(denovo_results)
print(f"Total predictions: {len(denovo_df)}")
print(f"Unique spectra: {denovo_df['spectrum_id'].nunique()}")

# 提取每个谱图候选（最多50个）
all_candidates = []
for spec_idx in range(num_spectra):
    spec_preds = denovo_df[denovo_df['spectrum_id'].str.contains(str(spec_idx), na=False)].copy()
    spec_preds = spec_preds.head(50)

    if len(spec_preds) > 0:
        spec_preds['spectrum_index'] = spec_idx
        all_candidates.append(spec_preds[['spectrum_index', 'peptide', 'score']])

if len(all_candidates) == 0:
    print("No candidates found!")
    exit(1)

candidates_df = pd.concat(all_candidates, ignore_index=True)
candidates_csv = output_dir / "all_candidates.csv"
candidates_df.to_csv(candidates_csv, index=False)

print(f"\nSaved {len(candidates_df)} candidates to {candidates_csv}")
print(f"Average {len(candidates_df)/max(1, num_spectra):.1f} candidates per spectrum")

# Step 5: 批量重排序（使用高效索引）
print("\n" + "="*70)
print("Step 4: Batch reranking with efficient index (Top-3 average)...")
print("="*70)

# 初始化高效重排序器
reranker = EfficientReranker(model_path="casanovo_v5_0_0_v5_0_0.ckpt", config_path="beam50.yaml")
reranker.load_precomputed_index(index_file)

all_rerank_results = []
rerank_start = time.time()

import sys

for spec_idx in range(num_spectra):
    # 更频繁的进度显示：每10个光谱
    if (spec_idx + 1) % 10 == 0:
        elapsed = time.time() - rerank_start
        avg_time = elapsed / (spec_idx + 1)
        remaining = (num_spectra - spec_idx - 1) * avg_time
        progress = (spec_idx + 1) / num_spectra * 100
        print(f"[{spec_idx+1}/{num_spectra}] Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | Speed: {avg_time:.2f}s/spectrum", flush=True)

    spec_candidates = candidates_df[candidates_df['spectrum_index'] == spec_idx].copy()
    candidates_list = spec_candidates[['peptide', 'score']].to_dict('records')

    if len(candidates_list) == 0:
        continue

    try:
        result = reranker.rerank_with_efficient_index(
            str(test_all_mgf),
            spec_idx,
            candidates_list,
            use_prosit=True,
            top_k=3  # 使用 Top-3 平均
        )

        result['spectrum_index'] = spec_idx
        result['true_sequence'] = ground_truth[spec_idx]['true_sequence']
        all_rerank_results.append(result)
    except Exception as e:
        print(f"  Error on spectrum {spec_idx}: {e}")
        continue

rerank_time = time.time() - rerank_start

# 合并结果
if len(all_rerank_results) == 0:
    print("\nNo results!")
    exit(1)

final_results = pd.concat(all_rerank_results, ignore_index=True)
final_csv = output_dir / "reranked_results_efficient.csv"
final_results.to_csv(final_csv, index=False)

print(f"\n{'='*70}")
print(f"Saved all results to: {final_csv}")
print(f"{'='*70}")

# Step 6: 统计准确率
print("\n" + "="*70)
print("ACCURACY ANALYSIS")
print("="*70)

correct_count = 0
correct_in_candidates = 0
has_ground_truth = 0
db_match_count = 0
prosit_count = 0

for spec_idx in range(num_spectra):
    true_seq = ground_truth[spec_idx]['true_sequence']
    if not true_seq or true_seq == '':
        continue

    has_ground_truth += 1

    spec_results = final_results[final_results['spectrum_index'] == spec_idx].copy()
    if len(spec_results) == 0:
        continue

    top1 = spec_results.iloc[0]
    is_correct = reranker.normalize_peptide(top1['peptide']) == reranker.normalize_peptide(true_seq)
    if is_correct:
        correct_count += 1

    # 统计来源
    if top1['source'] == 'Database':
        db_match_count += 1
    elif top1['source'] == 'Prosit':
        prosit_count += 1

if has_ground_truth == 0:
    print("No ground truth sequences found!")
    exit(1)

accuracy = correct_count / has_ground_truth * 100

print(f"Spectra with ground truth: {has_ground_truth}/{num_spectra}")
print(f"Top-1 Accuracy: {correct_count}/{has_ground_truth} = {accuracy:.2f}%")

print(f"\nTop-1 Source Distribution:")
print(f"  Database matches: {db_match_count} ({db_match_count/has_ground_truth*100:.1f}%)")
print(f"  Prosit predictions: {prosit_count} ({prosit_count/has_ground_truth*100:.1f}%)")

# 统计数据库匹配的详细信息
db_results = final_results[final_results['source'] == 'Database']
if len(db_results) > 0:
    print(f"\nDatabase Match Statistics:")
    print(f"  Total database matches: {len(db_results)}")
    print(f"  Average matched spectra count: {db_results['matched_count'].mean():.1f}")
    print(f"  Max matched spectra count: {db_results['matched_count'].max():.0f}")

print("\n" + "="*70)
print("Timing:")
print(f"  Denovo: {denovo_time:.1f}s ({denovo_time/max(1, num_spectra):.2f}s/spectrum)")
print(f"  Rerank: {rerank_time:.1f}s ({rerank_time/max(1, num_spectra):.2f}s/spectrum)")
print(f"  Total:  {denovo_time + rerank_time:.1f}s")
print("="*70)

print("\n✅ Test completed with efficient reranker!")
print(f"Results saved to: {output_dir}")
