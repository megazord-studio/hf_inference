"""Response formatting and serialization utilities.

Provides consistent JSON serialization and output formatting for inference results.
"""

import json
from typing import Any
from typing import Optional

import numpy as np
import torch


def safe_json(obj: Any) -> Any:
    """
    Convert Python objects to JSON-serializable format.

    Handles numpy arrays, torch tensors, and other non-serializable types
    commonly returned by ML models.

    Args:
        obj: Object to convert (dict, list, tensor, etc.)

    Returns:
        JSON-serializable version of the object

    Example:
        >>> import numpy as np
        >>> result = {"scores": np.array([0.9, 0.1]), "text": "hello"}
        >>> serializable = safe_json(result)
        >>> json.dumps(serializable)  # Works!
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (str, int, bool, type(None), float)):
        return obj
    if isinstance(obj, (np.generic,)):
        return float(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.item())
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return str(obj)


def safe_print_output(obj: Any) -> None:
    """
    Print object as formatted JSON, converting non-serializable types.

    Useful for debugging and CLI output.

    Args:
        obj: Object to print

    Example:
        >>> result = {"prediction": torch.tensor([1, 2, 3])}
        >>> safe_print_output(result)
        Output type: <class 'dict'>
        {
          "prediction": [1, 2, 3]
        }
    """
    clean = safe_json(obj)
    print(f"Output type: {type(clean)}")
    print(json.dumps(clean, indent=2, ensure_ascii=False))


def soft_skip(reason: str, hint: Optional[str] = None) -> None:
    """
    Output a skip message in structured JSON format.

    Used by runners to indicate graceful skips (e.g., unsupported models).

    Args:
        reason: Why the operation was skipped
        hint: Optional suggestion for user

    Example:
        >>> soft_skip("CUDA out of memory", "Try a smaller model or CPU")
    """
    out = {"skipped": True, "reason": reason}
    if hint:
        out["hint"] = hint
    safe_print_output(out)


def soft_hint_error(
    title: str, reason: str, hint: Optional[str] = None
) -> None:
    """
    Output an error message in structured JSON format.

    Used by runners to provide actionable error information.

    Args:
        title: Error title/category
        reason: Detailed error reason
        hint: Optional suggestion for user

    Example:
        >>> soft_hint_error(
        ...     "Model loading failed",
        ...     "Model weights not found",
        ...     "Ensure model_id is correct"
        ... )
    """
    out = {"error": title, "reason": reason}
    if hint:
        out["hint"] = hint
    safe_print_output(out)


__all__ = ["safe_json", "safe_print_output", "soft_skip", "soft_hint_error"]
