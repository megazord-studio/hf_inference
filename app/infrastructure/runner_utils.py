"""Functional utilities for runners.

Provides higher-order functions and common patterns for runner implementations.
"""

from typing import Any
from typing import Callable
from typing import Dict

from app.infrastructure.errors import is_gated_repo_error
from app.infrastructure.errors import is_missing_model_error
from app.infrastructure.errors import is_no_weight_files_error
from app.types import RunnerSpec


# Type alias for runner functions
RunnerFunc = Callable[[RunnerSpec, str], Dict[str, Any]]


def with_standard_error_handling(
    runner_func: RunnerFunc,
) -> RunnerFunc:
    """
    Higher-order function: wrap runner with standard error handling.
    
    This HOF adds consistent error handling for common failure modes:
    - Gated repository errors
    - Missing model errors
    - Missing weight file errors
    
    Args:
        runner_func: Pure runner function to wrap
    
    Returns:
        Wrapped runner with error handling
    
    Example:
        >>> def my_runner(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
        ...     # Implementation
        ...     return {"result": "ok"}
        >>> 
        >>> safe_runner = with_standard_error_handling(my_runner)
        >>> result = safe_runner(spec, "cpu")
    """
    def wrapped_runner(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
        try:
            return runner_func(spec, dev)
        except Exception as e:
            # Check for known error patterns
            if is_gated_repo_error(e):
                return {
                    "skipped": True,
                    "reason": "gated model (no access/auth)",
                    "hint": "Request access via HF and login.",
                }
            if is_missing_model_error(e):
                return {
                    "skipped": True,
                    "reason": "model not found on Hugging Face",
                    "hint": "Check model ID spelling.",
                }
            if is_no_weight_files_error(e):
                return {
                    "skipped": True,
                    "reason": "model weights not found",
                    "hint": "Model may not be properly uploaded to HF.",
                }
            # Unknown error - return generic response
            task_name = spec.get("task", "unknown")
            return {
                "error": f"{task_name} failed",
                "reason": repr(e),
            }
    
    return wrapped_runner


def validate_spec_fields(
    required_payload_fields: list[str],
) -> Callable[[RunnerFunc], RunnerFunc]:
    """
    Higher-order function: validate spec has required payload fields.
    
    Args:
        required_payload_fields: List of required field names in payload
    
    Returns:
        Decorator function
    
    Example:
        >>> @validate_spec_fields(["prompt"])
        ... def text_runner(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
        ...     prompt = spec["payload"]["prompt"]
        ...     return {"text": generate(prompt)}
    """
    def decorator(runner_func: RunnerFunc) -> RunnerFunc:
        def validated_runner(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
            payload = spec.get("payload", {})
            
            # Check for missing required fields
            missing = [f for f in required_payload_fields if f not in payload]
            if missing:
                task_name = spec.get("task", "unknown")
                return {
                    "error": f"{task_name} validation failed",
                    "reason": f"Missing required fields: {', '.join(missing)}",
                }
            
            # All fields present, run the function
            return runner_func(spec, dev)
        
        return validated_runner
    
    return decorator


def with_file_handling(
    file_key: str,
    fallback_path_key: str,
) -> Callable[[RunnerFunc], RunnerFunc]:
    """
    Higher-order function: handle file upload or path in spec.
    
    Provides consistent file handling across runners that accept
    file uploads or file paths.
    
    Args:
        file_key: Key in spec["files"] (e.g., "image", "audio")
        fallback_path_key: Key in spec["payload"] for path fallback
    
    Returns:
        Decorator function
    
    Example:
        >>> @with_file_handling("image", "image_path")
        ... def image_runner(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
        ...     # spec["_file"] will contain the processed file
        ...     img = spec["_file"]
        ...     return process(img)
    """
    from app.infrastructure.file_io import ensure_image
    from app.infrastructure.file_io import get_upload_file_image
    
    def decorator(runner_func: RunnerFunc) -> RunnerFunc:
        def file_handled_runner(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
            # Handle file upload or path
            file_obj = None
            upload = spec.get("files", {}).get(file_key)
            
            if file_key == "image":
                file_obj = get_upload_file_image(upload)
                if file_obj is None:
                    path = spec.get("payload", {}).get(fallback_path_key, f"{file_key}.jpg")
                    file_obj = ensure_image(path)
            # Add more file types as needed
            
            # Add processed file to spec for runner to use
            enhanced_spec = {**spec, "_file": file_obj}
            return runner_func(enhanced_spec, dev)  # type: ignore
        
        return file_handled_runner
    
    return decorator


__all__ = [
    "RunnerFunc",
    "with_standard_error_handling",
    "validate_spec_fields",
    "with_file_handling",
]
