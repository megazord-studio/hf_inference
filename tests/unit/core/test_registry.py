"""Unit tests for the functional core registry module.

Scenario: Functional task runner registry provides immutable operations for
managing task-to-function mappings with validation.

Test coverage:
- Registry creation
- Registration of new tasks (returns new registry)
- Duplicate registration detection
- Task lookup (supported and unsupported)
- Bulk registration
- Registry listing
- Immutability verification
"""

import pytest

from app.core.registry import bulk_register
from app.core.registry import create_registry
from app.core.registry import get_runner
from app.core.registry import get_supported_tasks
from app.core.registry import is_task_supported
from app.core.registry import register_runner
from app.types import RunnerSpec


def dummy_runner(spec: RunnerSpec, dev: str) -> dict:
    """Dummy runner for testing (pure function)."""
    return {"result": "ok", "task": spec.get("task"), "device": dev}


def another_runner(spec: RunnerSpec, dev: str) -> dict:
    """Another dummy runner for testing (pure function)."""
    return {"result": "ok", "task": spec.get("task"), "different": True}


class TestFunctionalRegistry:
    """Test scenarios for functional registry (immutable operations)."""

    def test_given_empty_mappings_when_creating_registry_then_empty_registry_created(
        self,
    ) -> None:
        """Given: No initial mappings
        When: Creating a new registry
        Then: Empty registry is created
        """
        registry = create_registry()
        assert len(registry) == 0
        assert get_supported_tasks(registry) == []

    def test_given_empty_registry_when_registering_task_then_new_registry_returned(
        self,
    ) -> None:
        """Given: Empty registry
        When: Registering a new task
        Then: New registry is returned with task added, original unchanged
        """
        original = create_registry()
        new_registry, error = register_runner(original, "text-generation", dummy_runner)
        
        assert error is None
        assert len(original) == 0  # Original unchanged (immutable)
        assert len(new_registry) == 1
        assert get_runner(new_registry, "text-generation") == dummy_runner

    def test_given_registry_when_registering_duplicate_task_then_returns_error(
        self,
    ) -> None:
        """Given: Registry with existing task
        When: Attempting to register same task again
        Then: Returns error message and original registry
        """
        registry = create_registry({"text-generation": dummy_runner})
        new_registry, error = register_runner(registry, "text-generation", another_runner)
        
        assert error is not None
        assert "already registered" in error
        assert new_registry == registry  # Registry unchanged

    def test_given_registry_when_checking_supported_task_then_returns_true(
        self,
    ) -> None:
        """Given: Registry with registered task
        When: Checking if task is supported
        Then: Returns True (pure function, no mutation)
        """
        registry = create_registry({"text-generation": dummy_runner})
        assert is_task_supported(registry, "text-generation") is True

    def test_given_registry_when_checking_unsupported_task_then_returns_false(
        self,
    ) -> None:
        """Given: Registry without a specific task
        When: Checking if unknown task is supported
        Then: Returns False (pure function)
        """
        registry = create_registry({"text-generation": dummy_runner})
        assert is_task_supported(registry, "image-classification") is False

    def test_given_registry_when_getting_unknown_task_then_returns_none(
        self,
    ) -> None:
        """Given: Registry without a specific task
        When: Getting runner for unknown task
        Then: Returns None (pure function)
        """
        registry = create_registry({"text-generation": dummy_runner})
        runner = get_runner(registry, "unknown-task")
        assert runner is None

    def test_given_registry_when_listing_tasks_then_returns_sorted_list(
        self,
    ) -> None:
        """Given: Registry with multiple tasks
        When: Requesting supported tasks list
        Then: Returns alphabetically sorted task names (pure function)
        """
        registry = create_registry({
            "text-generation": dummy_runner,
            "image-classification": another_runner,
            "audio-classification": dummy_runner,
        })
        
        tasks = get_supported_tasks(registry)
        assert tasks == [
            "audio-classification",
            "image-classification",
            "text-generation",
        ]

    def test_given_empty_registry_when_bulk_registering_then_all_tasks_added(
        self,
    ) -> None:
        """Given: Empty registry
        When: Bulk registering multiple tasks
        Then: New registry with all tasks is returned
        """
        original = create_registry()
        mappings = {
            "text-generation": dummy_runner,
            "image-classification": another_runner,
        }
        new_registry, errors = bulk_register(original, mappings)
        
        assert errors == []
        assert len(original) == 0  # Original unchanged
        assert len(new_registry) == 2
        assert is_task_supported(new_registry, "text-generation")
        assert is_task_supported(new_registry, "image-classification")
        assert get_runner(new_registry, "text-generation") == dummy_runner
        assert get_runner(new_registry, "image-classification") == another_runner

    def test_given_registry_with_task_when_bulk_registering_duplicate_then_returns_errors(
        self,
    ) -> None:
        """Given: Registry with existing task
        When: Bulk registering with duplicate task
        Then: Returns list of errors
        """
        registry = create_registry({"text-generation": dummy_runner})
        mappings = {
            "text-generation": another_runner,
            "image-classification": another_runner,
        }
        
        new_registry, errors = bulk_register(registry, mappings)
        
        assert len(errors) > 0
        assert any("text-generation" in err for err in errors)

    def test_given_registered_runner_when_calling_it_then_returns_expected_result(
        self,
    ) -> None:
        """Given: Registry with a pure runner function
        When: Retrieving and calling the runner
        Then: Runner executes correctly (pure function behavior)
        """
        registry = create_registry({"text-generation": dummy_runner})
        runner = get_runner(registry, "text-generation")
        assert runner is not None
        
        spec: RunnerSpec = {
            "model_id": "gpt2",
            "task": "text-generation",
            "payload": {}
        }
        result = runner(spec, "cpu")
        
        assert result["result"] == "ok"
        assert result["task"] == "text-generation"
        assert result["device"] == "cpu"
    
    def test_given_registry_when_multiple_operations_then_original_never_mutated(
        self,
    ) -> None:
        """Given: A registry
        When: Performing multiple operations
        Then: Original registry is never mutated (immutability test)
        """
        original = create_registry({"text-generation": dummy_runner})
        
        # Perform various operations
        reg1, _ = register_runner(original, "new-task", another_runner)
        reg2, _ = bulk_register(original, {"another-task": dummy_runner})
        _ = get_runner(original, "text-generation")
        _ = is_task_supported(original, "new-task")
        _ = get_supported_tasks(original)
        
        # Original should be unchanged
        assert len(original) == 1
        assert "text-generation" in original
        assert "new-task" not in original
        assert "another-task" not in original
        
        # New registries have their own state
        assert "new-task" in reg1
        assert "another-task" in reg2
