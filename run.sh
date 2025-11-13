#!/bin/bash
# Quick start script for High-Nine batch test

echo "======================================"
echo "High-Nine Batch Test - Quick Start"
echo "======================================"
echo ""

# Check if in conda environment
if [[ "$CONDA_DEFAULT_ENV" != "casa" ]]; then
    echo "Warning: Not in 'casa' conda environment"
    echo "Please run: conda activate casa"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Check if data files exist
if [ ! -f "test_data/high_nine/high_nine_validation_1000.mgf" ]; then
    echo "ERROR: Data files not found!"
    echo "Please check symlinks in test_data/high_nine/"
    exit 1
fi

echo "Starting batch test..."
echo "This will take approximately 30-60 minutes"
echo ""

python batch_test_high_nine_efficient.py

echo ""
echo "======================================"
echo "Batch test completed!"
echo "Results saved in: high_nine_results_efficient/"
echo "======================================"
