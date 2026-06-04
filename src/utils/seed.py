"""Reproducibility utilities."""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for bit-reproducible training (paper §4.3 CFR-hard check).

    CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS handle, otherwise
    use_deterministic_algorithms raises on the first MatMul.
    warn_only=True keeps training alive for ops without a deterministic CUDA kernel.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # older torch without warn_only
