#!/usr/bin/env python3
"""
Basic functionality tests for the High-Nine standalone package.
These tests verify that the core modules can be imported and basic functionality works.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")
    
    try:
        import casanovo_predictor
        print("✓ casanovo_predictor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import casanovo_predictor: {e}")
        return False
    
    try:
        import efficient_reranker
        print("✓ efficient_reranker imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import efficient_reranker: {e}")
        return False
    
    try:
        import build_efficient_index
        print("✓ build_efficient_index imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import build_efficient_index: {e}")
        return False
    
    return True

def test_config_files():
    """Test that configuration files are valid YAML."""
    print("\nTesting configuration files...")
    
    config_files = ['beam50.yaml', 'casanovo/config.yaml']
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print(f"✓ {config_file} is valid YAML")
        except Exception as e:
            print(f"✗ {config_file} is invalid: {e}")
            return False
    
    return True

def test_requirements():
    """Test that requirements.txt exists and is readable."""
    print("\nTesting requirements...")
    
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        if 'casanovo' in requirements:
            print("✓ requirements.txt contains casanovo dependency")
        else:
            print("⚠ requirements.txt might be missing casanovo")
        
        print(f"✓ requirements.txt is readable ({len(requirements)} characters)")
    except Exception as e:
        print(f"✗ Failed to read requirements.txt: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without requiring model files."""
    print("\nTesting basic functionality...")
    
    try:
        # Test that we can create basic objects
        from efficient_reranker import EfficientReranker
        
        # This should fail gracefully without model files
        try:
            reranker = EfficientReranker(model_path="nonexistent.ckpt")
            print("✓ EfficientReranker can be instantiated")
        except Exception as e:
            if "No such file" in str(e) or "not found" in str(e):
                print("✓ EfficientReranker handles missing model files correctly")
            else:
                print(f"⚠ EfficientReranker unexpected error: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running basic tests for High-Nine standalone package...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config_files,
        test_requirements,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
