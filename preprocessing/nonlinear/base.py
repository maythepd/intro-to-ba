"""
Base utilities and checkpoint management for non-linear and neural representation learning.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# Windows sandbox CPU patch to avoid Loky subprocess probe hang
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib
    from joblib.externals.loky.backend import context
    context._count_physical_cores = lambda: 4
except Exception:
    pass

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves PyTorch model state_dict along with optional metadata to disk.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, str(path))
    return str(path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads PyTorch model state_dict from a checkpoint file.
    
    Returns:
        (model, metadata_dict)
    """
    dev = device or get_device()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(str(path), map_location=dev)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint.get("metadata", {})
    else:
        model.load_state_dict(checkpoint)
        metadata = {}

    model.to(dev)
    model.eval()
    return model, metadata


def checkpoint_exists(checkpoint_path: Union[str, Path]) -> bool:
    """Checks whether the checkpoint file exists and is non-empty."""
    p = Path(checkpoint_path)
    return p.is_file() and p.stat().st_size > 0


def is_checkpoint_compatible(
    checkpoint_path: Union[str, Path],
    in_features: Optional[int] = None,
) -> bool:
    """
    Checks whether the checkpoint file exists and matches the required input feature dimension.
    Prevents PyTorch shape mismatch RuntimeErrors when feature sets differ across runs.
    """
    if not checkpoint_exists(checkpoint_path):
        return False
    if in_features is None:
        return True
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "metadata" in checkpoint:
            saved_in = checkpoint["metadata"].get("in_features")
            if saved_in is not None and saved_in != in_features:
                return False
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            first_weight = checkpoint["state_dict"].get("encoder.0.weight")
            if first_weight is not None and first_weight.shape[1] != in_features:
                return False
        return True
    except Exception:
        return False

