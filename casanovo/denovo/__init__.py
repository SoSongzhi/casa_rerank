# Minimal denovo package init for standalone usage
# Only import what is actually needed

try:
 from .model_runner import ModelRunner
except ImportError:
 # ModelRunner not needed for inference-only usage
 pass
