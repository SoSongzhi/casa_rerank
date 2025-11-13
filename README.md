# High-Nine Batch Test - Standalone Package

This is a standalone package for running batch tests on the High-Nine dataset using efficient reranking.

## 📁 Directory Structure

```
high_nine_standalone/
├── batch_test_high_nine_efficient.py  # Main test script
├── efficient_reranker.py              # Efficient reranker module
├── build_efficient_index.py           # Index builder
├── beam50.yaml                        # Beam search configuration
├── casanovo_v4_2_0.ckpt              # Model weights
├── casanovo/                          # Casanovo package modules
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── version.py
│   ├── denovo/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── transformers.py
│   │   └── evaluate.py
│   └── data/
│       ├── __init__.py
│       ├── ms_io.py
│       └── psm.py
├── test_data/high_nine/               # Symlinks to data files
│   ├── high_nine_validation_1000.mgf -> (original location)
│   ├── high_nine_database.mgf -> (original location)
│   └── high_nine_database.mgf.efficient_index.pkl -> (original location)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Activate Environment

```bash
conda activate casa
```

### 2. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

### 3. Run Batch Test

```bash
cd high_nine_standalone
python batch_test_high_nine_efficient.py
```

## 📊 What This Does

1. **De novo sequencing**: Uses Casanovo with Beam=50 to generate peptide candidates
2. **Efficient indexing**: Uses pre-computed index for fast reference matching
3. **Reranking**: Reranks candidates using Top-3 average similarity strategy
4. **Evaluation**: Compares with ground truth and reports accuracy

## 📈 Test Parameters

- **Test spectra**: 1,000 spectra (high_nine_validation_1000.mgf)
- **Reference database**: ~50,000 spectra (high_nine_database.mgf)
- **Beam width**: 50
- **Reranking strategy**: Top-3 average cosine similarity
- **Expected runtime**: 30-60 minutes

## 📁 Output

Results are saved in `high_nine_results_efficient/`:

- `batch_summary.txt` - Overall accuracy statistics
- `casanovo_predictions.txt` - De novo predictions
- `ground_truth.csv` - True sequences
- `spectrum_*.csv` - Detailed reranking results per spectrum

## 🔧 Configuration

### Modify Beam Width

Edit `beam50.yaml` and change:

```yaml
n_beams: 50  # Change to desired beam width
```

### Modify Data Paths

Edit `batch_test_high_nine_efficient.py` lines 26-27:

```python
test_mgf = "test_data/high_nine/high_nine_validation_1000.mgf"
reference_mgf = "test_data/high_nine/high_nine_database.mgf"
```

### Use Different Model

Edit `efficient_reranker.py` to specify model path:

```python
reranker = EfficientReranker(model_path="your_model.ckpt")
```

## 📝 Notes

- **Data files**: The data files in `test_data/high_nine/` are symbolic links pointing to the original location. If you move this folder, update the symlinks or copy the actual data files.

- **Model weights**: `casanovo_v4_2_0.ckpt` is included in this package.

- **Memory usage**: The script automatically creates a memory-safe configuration if OOM occurs.

## 🐛 Troubleshooting

### Issue: "File not found" error for data files

**Solution**: The symlinks may be broken. Copy the actual data files:

```bash
cp /c/Users/research/Desktop/casanovo/test_data/high_nine/*.mgf test_data/high_nine/
cp /c/Users/research/Desktop/casanovo/test_data/high_nine/*.pkl test_data/high_nine/
```

### Issue: "Module not found" error

**Solution**: Make sure you're in the `high_nine_standalone` directory and have activated the `casa` environment.

### Issue: OOM (Out of Memory)

**Solution**: The script will automatically create `beam50_oomsafe.yaml` with reduced batch size.

## 📞 Support

For issues, check the main Casanovo repository: https://github.com/Noble-Lab/casanovo

---

**Created**: 2024-11-13
**Version**: 1.0
