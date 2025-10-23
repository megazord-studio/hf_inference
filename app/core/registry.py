"""Task runner registry with validation and type safety.

Provides a central registry for mapping task names to their runner functions,
with validation and error handling.
"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from app.types import RunnerSpec


# Type alias for runner functions
RunnerFunc = Callable[[RunnerSpec, str], Dict[str, Any]]


class RunnerRegistry:
    """
    Registry for task runner functions.

    Provides registration, lookup, and validation of task runners with
    proper error handling for unknown tasks.

    Example:
        >>> registry = RunnerRegistry()
        >>> registry.register("text-generation", my_text_gen_runner)
        >>> runner = registry.get("text-generation")
        >>> runner is not None
        True
        >>> registry.is_supported("unknown-task")
        False
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._runners: Dict[str, RunnerFunc] = {}

    def register(self, task: str, runner: RunnerFunc) -> None:
        """
        Register a runner function for a task.

        Args:
            task: Task identifier (e.g., "text-generation")
            runner: Callable that accepts (RunnerSpec, device_str) and returns Dict

        Raises:
            ValueError: If task is already registered (prevents accidental overwrites)

        Example:
            >>> registry = RunnerRegistry()
            >>> registry.register("text-generation", run_text_gen)
        """
        if task in self._runners:
            raise ValueError(
                f"Task '{task}' is already registered. "
                f"Use update() to modify existing registrations."
            )
        self._runners[task] = runner

    def update(self, task: str, runner: RunnerFunc) -> None:
        """
        Update an existing runner registration.

        Args:
            task: Task identifier
            runner: New runner function

        Example:
            >>> registry.update("text-generation", improved_runner)
        """
        self._runners[task] = runner

    def get(self, task: str) -> Optional[RunnerFunc]:
        """
        Get runner function for a task.

        Args:
            task: Task identifier

        Returns:
            Runner function if registered, None otherwise

        Example:
            >>> runner = registry.get("text-generation")
            >>> if runner:
            ...     result = runner(spec, "cuda:0")
        """
        return self._runners.get(task)

    def is_supported(self, task: str) -> bool:
        """
        Check if a task is supported.

        Args:
            task: Task identifier

        Returns:
            True if task is registered

        Example:
            >>> if registry.is_supported("image-classification"):
            ...     # Handle request
            ...     pass
        """
        return task in self._runners

    def supported_tasks(self) -> List[str]:
        """
        Get list of all supported task identifiers.

        Returns:
            Sorted list of task names

        Example:
            >>> tasks = registry.supported_tasks()
            >>> "text-generation" in tasks
            True
        """
        return sorted(self._runners.keys())

    def bulk_register(self, mappings: Dict[str, RunnerFunc]) -> None:
        """
        Register multiple runners at once.

        Useful for initialization from a static mapping.

        Args:
            mappings: Dictionary of task -> runner mappings

        Raises:
            ValueError: If any task is already registered

        Example:
            >>> registry.bulk_register({
            ...     "text-generation": run_text_gen,
            ...     "image-classification": run_img_class,
            ... })
        """
        for task, runner in mappings.items():
            self.register(task, runner)


__all__ = ["RunnerRegistry", "RunnerFunc"]
