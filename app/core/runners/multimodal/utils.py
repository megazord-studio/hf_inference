"""Shared utilities for multimodal runners.

Small, focused helper functions for device, dtype, and tensor handling.
No side effects; purely functional where possible.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Optional

import torch

log = logging.getLogger("app.runners.multimodal")


def select_dtype(device: Any) -> torch.dtype:
    """Select appropriate dtype based on device capabilities."""
    if device and getattr(device, "type", None) == "cuda":
        cc = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
        return torch.bfloat16 if cc >= 8 else torch.float16
    if device and getattr(device, "type", None) == "mps":
        return torch.float16
    return torch.float32


def is_cuda(device: Any) -> bool:
    """Check if device is CUDA."""
    return bool(device and getattr(device, "type", None) == "cuda")


def to_device(model: Any, device: Any) -> Any:
    """Move model to device, handling None and errors gracefully."""
    if model is None:
        return None
    try:
        return model.to(device) if device else model
    except Exception:
        return model


def move_to_device(enc: Any, device: Any) -> Dict[str, Any]:
    """Move encoding dict to device, normalizing to plain dict."""
    if not isinstance(enc, dict):
        try:
            enc = dict(enc)
        except Exception:
            enc = {k: getattr(enc, k) for k in dir(enc) if not k.startswith("_")}
    if not device:
        return enc
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}


def unify_model_dtype(model: Any, dtype: torch.dtype) -> None:
    """Unify all model parameters and buffers to a single dtype."""
    if model is None:
        return
    for p in model.parameters():
        if p.dtype != dtype:
            p.data = p.data.to(dtype)
    for name, buf in model.named_buffers():
        if buf.dtype != dtype:
            try:
                setattr(model, name, buf.to(dtype))
            except Exception:
                pass


def count_params(model: Any) -> int:
    """Count model parameters, handling None gracefully."""
    if model is None or not hasattr(model, "parameters"):
        return 0
    return sum(p.numel() for p in model.parameters())


def cap_max_new_tokens(requested: int, device: Any) -> int:
    """Cap max_new_tokens on CPU/MPS to prevent slow/hanging generations."""
    if is_cuda(device):
        return max(1, requested)
    # On CPU/MPS, keep very small caps for responsiveness
    safe_cap = 16
    return max(1, min(requested, safe_cap))


def resolve_max_new_tokens(
    options: Optional[Dict[str, Any]], device: Any, default: int = 32
) -> tuple[int, bool]:
    """Resolve max_new_tokens and indicate if it originated from the user.

    Returns a tuple of (value, is_user_override). User-provided values are never
    capped here so downstream callers can decide whether to apply device caps.
    """

    opts = options or {}
    for key in ("max_new_tokens", "max_length"):
        if key not in opts:
            continue
        value = opts.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            log.debug("resolve_max_new_tokens ignoring invalid %s=%s", key, value)
            continue
        return max(1, parsed), True

    try:
        fallback = int(default)
    except (TypeError, ValueError):
        fallback = 32
    return max(1, fallback), False


def safe_call(fn, error_msg: Optional[str] = None):
    """Execute a function safely, returning None on error."""
    try:
        return fn()
    except Exception as e:
        log.error("safe_call failed: %s", error_msg or e)
        return None


def require(val: Any, err: str) -> Any:
    """Require a value to be truthy, raising RuntimeError if not."""
    if not val:
        raise RuntimeError(f"multimodal: {err}")
    return val
