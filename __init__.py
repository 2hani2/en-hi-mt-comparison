"""EN->HI MT comparison package."""
import os

# Must be set before PyTorch touches the GPU / before transformers is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less memory fragmentation
os.environ.setdefault("USE_TF", "0")                                           # Colab ships TensorFlow; we never need it
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
