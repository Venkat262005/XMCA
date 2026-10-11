"""XMCA-Net-Lite: Research-grade Retrieval-Augmented VLM Diagnostic System."""

import os

# Prevent OpenMP runtime clashes across FAISS, PyTorch, and LightGBM
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import lightgbm
except Exception:
    pass

__version__ = "1.0.0"
