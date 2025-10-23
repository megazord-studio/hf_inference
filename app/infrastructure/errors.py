"""Error detection and handling utilities.

Provides consistent error detection for common failure modes in model loading
and inference operations.
"""


def is_cuda_oom(e: Exception) -> bool:
    """
    Check if exception is a CUDA out-of-memory error.

    Args:
        e: Exception to inspect

    Returns:
        True if the error is a CUDA OOM error

    Example:
        >>> try:
        ...     # Large model loading
        ...     model = AutoModel.from_pretrained("huge-model")
        ... except Exception as e:
        ...     if is_cuda_oom(e):
        ...         return {"error": "GPU out of memory", "hint": "Try CPU or smaller model"}
    """
    msg = repr(e).lower()
    return "cuda out of memory" in msg or "cuda oom" in msg


def is_missing_model_error(e: Exception) -> bool:
    """
    Check if exception indicates a model not found on Hugging Face Hub.

    Args:
        e: Exception to inspect

    Returns:
        True if the model identifier is invalid or not found

    Example:
        >>> try:
        ...     model = AutoModel.from_pretrained("nonexistent/model")
        ... except Exception as e:
        ...     if is_missing_model_error(e):
        ...         return {"error": "Model not found"}
    """
    msg = repr(e)
    return (
        "is not a local folder and is not a valid model identifier listed on"
        in msg
    )


def is_no_weight_files_error(e: Exception) -> bool:
    """
    Check if exception indicates missing model weight files.

    Args:
        e: Exception to inspect

    Returns:
        True if the model repository lacks pytorch_model.bin or safetensors

    Example:
        >>> try:
        ...     model = AutoModel.from_pretrained("incomplete/model")
        ... except Exception as e:
        ...     if is_no_weight_files_error(e):
        ...         return {"error": "Model weights not found"}
    """
    msg = repr(e)
    return (
        "does not appear to have a file named pytorch_model.bin" in msg
        or "model.safetensors" in msg
    )


def is_gated_repo_error(e: Exception) -> bool:
    """
    Check if exception indicates a gated repository access error.

    Gated repositories require authentication or manual access approval.

    Args:
        e: Exception to inspect

    Returns:
        True if the error is related to gated repository access

    Example:
        >>> try:
        ...     model = AutoModel.from_pretrained("meta-llama/Llama-2-70b")
        ... except Exception as e:
        ...     if is_gated_repo_error(e):
        ...         return {"error": "Gated model", "hint": "Request access on HF Hub"}
    """
    msg = repr(e).lower()
    return (
        ("gated repo" in msg)
        or ("401 client error" in msg)
        or ("access to model" in msg and "restricted" in msg)
    )


__all__ = [
    "is_cuda_oom",
    "is_missing_model_error",
    "is_no_weight_files_error",
    "is_gated_repo_error",
]
