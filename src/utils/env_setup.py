"""Environment and reproducible setup utilities.

Import this module at the very top of scripts that use TensorFlow so CUDA-related
environment variables are set before importing TensorFlow.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def configure_cuda(use_cpu: bool = False, visible_gpus: Optional[str] = None) -> None:
    """Configure CUDA visibility.

    Args:
        use_cpu: If True, forces CPU usage by setting CUDA_VISIBLE_DEVICES="-1".
        visible_gpus: If provided, sets CUDA_VISIBLE_DEVICES to this string (e.g. "0" or "0,1").
    """
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus


def set_seed(seed: int = 42) -> None:
    """Set seeds for python, numpy and tensorflow (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        import tensorflow as tf  # imported here to avoid forcing TF import at module import time
        tf.random.set_seed(seed)
    except Exception:
        pass


BASE_DIR = Path.cwd().resolve()
LOG_DIR = BASE_DIR / "logs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
for d in (LOG_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
}
