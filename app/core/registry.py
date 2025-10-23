"""Functional task runner registry with immutable operations.

Provides pure functions for managing task-to-runner mappings without mutable state.
All operations return new registry instances rather than mutating existing ones.
"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple

from app.types import RunnerSpec


# Type alias for runner functions
RunnerFunc = Callable[[RunnerSpec, str], Dict[str, Any]]

# Type alias for immutable registry (frozen mapping)
Registry = Mapping[str, RunnerFunc]


# Pure registry operations (no mutation, all return new values)

def create_registry(mappings: Optional[Dict[str, RunnerFunc]] = None) -> Registry:
    """
    Create a new immutable registry from mappings.
    
    This is a pure function that creates an immutable registry.
    
    Args:
        mappings: Optional initial task -> runner mappings
    
    Returns:
        Immutable registry (Mapping)
    
    Example:
        >>> registry = create_registry({
        ...     "text-generation": run_text_gen,
        ...     "image-classification": run_img_class,
        ... })
    """
    return dict(mappings or {})


def register_runner(
    registry: Registry, task: str, runner: RunnerFunc
) -> Tuple[Registry, Optional[str]]:
    """
    Register a runner, returning new registry and optional error.
    
    Pure function: returns new registry without mutating the original.
    
    Args:
        registry: Existing registry
        task: Task identifier
        runner: Runner function
    
    Returns:
        Tuple of (new_registry, error_message)
        If error_message is not None, registration failed.
    
    Example:
        >>> new_reg, error = register_runner(registry, "new-task", my_runner)
        >>> if error is None:
        ...     registry = new_reg
    """
    if task in registry:
        return registry, f"Task '{task}' is already registered"
    
    # Create new registry with added runner (immutable)
    new_registry = {**registry, task: runner}
    return new_registry, None


def get_runner(registry: Registry, task: str) -> Optional[RunnerFunc]:
    """
    Get runner function for a task (pure function).
    
    Args:
        registry: Registry to query
        task: Task identifier
    
    Returns:
        Runner function if registered, None otherwise
    
    Example:
        >>> runner = get_runner(registry, "text-generation")
        >>> if runner:
        ...     result = runner(spec, "cuda:0")
    """
    return registry.get(task)


def is_task_supported(registry: Registry, task: str) -> bool:
    """
    Check if a task is supported (pure function).
    
    Args:
        registry: Registry to query
        task: Task identifier
    
    Returns:
        True if task is registered
    
    Example:
        >>> if is_task_supported(registry, "image-classification"):
        ...     # Handle request
        ...     pass
    """
    return task in registry


def get_supported_tasks(registry: Registry) -> List[str]:
    """
    Get list of all supported task identifiers (pure function).
    
    Args:
        registry: Registry to query
    
    Returns:
        Sorted list of task names
    
    Example:
        >>> tasks = get_supported_tasks(registry)
        >>> "text-generation" in tasks
        True
    """
    return sorted(registry.keys())


def bulk_register(
    registry: Registry, mappings: Dict[str, RunnerFunc]
) -> Tuple[Registry, List[str]]:
    """
    Register multiple runners at once (pure function).
    
    Args:
        registry: Existing registry
        mappings: Dictionary of task -> runner mappings
    
    Returns:
        Tuple of (new_registry, list_of_errors)
        If list_of_errors is empty, all registrations succeeded.
    
    Example:
        >>> new_reg, errors = bulk_register(registry, {
        ...     "text-generation": run_text_gen,
        ...     "image-classification": run_img_class,
        ... })
        >>> if not errors:
        ...     registry = new_reg
    """
    current = registry
    errors: List[str] = []
    
    for task, runner in mappings.items():
        current, error = register_runner(current, task, runner)
        if error:
            errors.append(f"{task}: {error}")
    
    return current, errors


__all__ = [
    "RunnerFunc",
    "Registry",
    "create_registry",
    "register_runner",
    "get_runner",
    "is_task_supported",
    "get_supported_tasks",
    "bulk_register",
]
