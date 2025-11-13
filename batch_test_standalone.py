#\!/usr/bin/env python
# Standalone batch test
import pandas as pd
import time
from pathlib import Path
import re
from pyteomics import mgf
import sys

sys.path.insert(0, str(Path(__file__).parent))

from efficient_reranker import EfficientReranker
from build_efficient_index import EfficientIndexBuilder
from casanovo_predictor import CasanovoPredictor

test_mgf = 'test_data/high_nine/high_nine_validation_1000.mgf'
reference_mgf = 'test_data/high_nine/high_nine_database.mgf'
index_file = f'{reference_mgf}.efficient_index.pkl'
model_path = 'casanovo_v4_2_0.ckpt'
config_file = 'beam50.yaml'
output_dir = Path('high_nine_results_efficient')
output_dir.mkdir(exist_ok=True)

print('='*70)
print('High-Nine Batch Test - Standalone (Python API)')
print('='*70)
