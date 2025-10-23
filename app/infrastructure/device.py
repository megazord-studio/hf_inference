"""Device detection and management utilities.

Provides consistent device selection for PyTorch and Transformers pipelines.
"""

import torch


def device_str() -> str:
    """
    Return the device string for PyTorch ('cuda:0' or 'cpu').

    Returns:
        Device identifier string suitable for PyTorch operations.

    Example:
        >>> dev = device_str()
        >>> tensor = torch.tensor([1, 2, 3]).to(dev)
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def device_arg(dev: str) -> str:
    """
    Return device argument for transformers pipelines.

    Transformers >=4.41 accepts device str, torch.device, or int.
    This function ensures compatibility by returning the string directly.

    Args:
        dev: Device string (e.g., 'cuda:0' or 'cpu')

    Returns:
        Device argument suitable for transformers pipeline() calls.

    Example:
        >>> from transformers import pipeline
        >>> dev = device_str()
        >>> pl = pipeline("text-generation", device=device_arg(dev))
    """
    return dev


__all__ = ["device_str", "device_arg"]
