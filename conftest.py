import os
import sys
from pathlib import Path

# Prevent OpenMP runtime clashes across FAISS, PyTorch, and LightGBM
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import lightgbm  # Pre-load LightGBM before FAISS to initialize OpenMP safely
except Exception:
    pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
