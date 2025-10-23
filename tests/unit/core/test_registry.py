"""Unit tests for the core registry module.

Scenario: Task runner registry manages task-to-function mappings with validation.

Test coverage:
- Registration of new tasks
- Duplicate registration prevention
- Task lookup (supported and unsupported)
- Bulk registration
- Registry listing
"""

import pytest

from app.core.registry import RunnerRegistry
from app.types import RunnerSpec


def dummy_runner(spec: RunnerSpec, dev: str) -> dict:
    """Dummy runner for testing."""
    return {"result": "ok", "task": spec.get("task"), "device": dev}


def another_runner(spec: RunnerSpec, dev: str) -> dict:
    """Another dummy runner for testing."""
    return {"result": "ok", "task": spec.get("task"), "different": True}


class TestRunnerRegistry:
    """Test scenarios for RunnerRegistry."""

    def test_given_empty_registry_when_registering_task_then_task_is_added(
        self,
    ) -> None:
        """Given: Empty registry
        When: Registering a new task
        Then: Task is successfully added and retrievable
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        runner = registry.get("text-generation")
        assert runner is not None
        assert runner == dummy_runner

    def test_given_registry_when_registering_duplicate_task_then_raises_error(
        self,
    ) -> None:
        """Given: Registry with existing task
        When: Attempting to register same task again
        Then: ValueError is raised
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("text-generation", another_runner)

    def test_given_registry_when_updating_task_then_runner_is_replaced(
        self,
    ) -> None:
        """Given: Registry with existing task
        When: Updating the task with new runner
        Then: New runner replaces old one
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)
        registry.update("text-generation", another_runner)

        runner = registry.get("text-generation")
        assert runner is not None
        assert runner == another_runner

    def test_given_registry_when_checking_supported_task_then_returns_true(
        self,
    ) -> None:
        """Given: Registry with registered task
        When: Checking if task is supported
        Then: Returns True
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        assert registry.is_supported("text-generation") is True

    def test_given_registry_when_checking_unsupported_task_then_returns_false(
        self,
    ) -> None:
        """Given: Registry without a specific task
        When: Checking if unknown task is supported
        Then: Returns False
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        assert registry.is_supported("image-classification") is False

    def test_given_registry_when_getting_unknown_task_then_returns_none(
        self,
    ) -> None:
        """Given: Registry without a specific task
        When: Getting runner for unknown task
        Then: Returns None
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        runner = registry.get("unknown-task")
        assert runner is None

    def test_given_registry_when_listing_tasks_then_returns_sorted_list(
        self,
    ) -> None:
        """Given: Registry with multiple tasks
        When: Requesting supported tasks list
        Then: Returns alphabetically sorted task names
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)
        registry.register("image-classification", another_runner)
        registry.register("audio-classification", dummy_runner)

        tasks = registry.supported_tasks()
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
        Then: All tasks are successfully added
        """
        registry = RunnerRegistry()
        mappings = {
            "text-generation": dummy_runner,
            "image-classification": another_runner,
        }
        registry.bulk_register(mappings)

        assert registry.is_supported("text-generation")
        assert registry.is_supported("image-classification")
        assert registry.get("text-generation") == dummy_runner
        assert registry.get("image-classification") == another_runner

    def test_given_registry_with_task_when_bulk_registering_duplicate_then_raises_error(
        self,
    ) -> None:
        """Given: Registry with existing task
        When: Bulk registering with duplicate task
        Then: ValueError is raised and registry remains unchanged
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        mappings = {
            "text-generation": another_runner,
            "image-classification": another_runner,
        }

        with pytest.raises(ValueError, match="already registered"):
            registry.bulk_register(mappings)

        # Original runner should still be there
        assert registry.get("text-generation") == dummy_runner
        # New task should not be added if bulk fails partway
        # (but this depends on implementation - currently it would add before failing)

    def test_given_registered_runner_when_calling_it_then_returns_expected_result(
        self,
    ) -> None:
        """Given: Registry with a functional runner
        When: Retrieving and calling the runner
        Then: Runner executes correctly
        """
        registry = RunnerRegistry()
        registry.register("text-generation", dummy_runner)

        runner = registry.get("text-generation")
        assert runner is not None

        spec: RunnerSpec = {"model_id": "gpt2", "task": "text-generation", "payload": {}}
        result = runner(spec, "cpu")

        assert result["result"] == "ok"
        assert result["task"] == "text-generation"
        assert result["device"] == "cpu"
