"""Unit tests for infrastructure response utilities.

Scenario: Response formatting utilities provide JSON-safe serialization.

Test coverage:
- Numpy array serialization
- Torch tensor serialization
- Nested structure serialization
- Primitive type pass-through
"""

import numpy as np
import pytest

from app.infrastructure.response import safe_json


class TestResponseFormatting:
    """Test scenarios for response formatting utilities."""

    def test_given_dict_with_primitives_when_serializing_then_returns_unchanged(
        self,
    ) -> None:
        """Given: Dictionary with primitive types
        When: Converting to safe JSON
        Then: Returns unchanged dictionary
        """
        data = {"text": "hello", "count": 42, "active": True, "value": None}
        result = safe_json(data)
        assert result == data

    def test_given_numpy_array_when_serializing_then_returns_list(self) -> None:
        """Given: Numpy array
        When: Converting to safe JSON
        Then: Returns Python list
        """
        arr = np.array([1, 2, 3, 4, 5])
        result = safe_json(arr)
        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)

    def test_given_numpy_scalar_when_serializing_then_returns_python_float(
        self,
    ) -> None:
        """Given: Numpy scalar value
        When: Converting to safe JSON
        Then: Returns Python float
        """
        val = np.float32(3.14)
        result = safe_json(val)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_given_nested_structure_when_serializing_then_recursively_converts(
        self,
    ) -> None:
        """Given: Nested dictionary with numpy arrays
        When: Converting to safe JSON
        Then: Recursively converts all values
        """
        data = {
            "scores": np.array([0.9, 0.1]),
            "metadata": {"count": 42, "values": np.array([1.0, 2.0])},
        }
        result = safe_json(data)

        assert result["scores"] == [0.9, 0.1]
        assert result["metadata"]["count"] == 42
        assert result["metadata"]["values"] == [1.0, 2.0]

    def test_given_list_with_arrays_when_serializing_then_converts_all_elements(
        self,
    ) -> None:
        """Given: List containing numpy arrays
        When: Converting to safe JSON
        Then: Converts all list elements
        """
        data = [np.array([1, 2]), {"arr": np.array([3, 4])}, "text"]
        result = safe_json(data)

        assert result[0] == [1, 2]
        assert result[1]["arr"] == [3, 4]
        assert result[2] == "text"

    def test_given_unsupported_type_when_serializing_then_converts_to_string(
        self,
    ) -> None:
        """Given: Object with unsupported type
        When: Converting to safe JSON
        Then: Converts to string representation
        """

        class CustomClass:
            def __repr__(self) -> str:
                return "CustomClass()"

        obj = CustomClass()
        result = safe_json(obj)
        assert result == "CustomClass()"

    def test_given_torch_tensor_when_serializing_then_returns_list(self) -> None:
        """Given: PyTorch tensor
        When: Converting to safe JSON
        Then: Returns Python list (if torch is available)
        """
        try:
            import torch

            tensor = torch.tensor([1.0, 2.0, 3.0])
            result = safe_json(tensor)
            assert result == [1.0, 2.0, 3.0]
            assert isinstance(result, list)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_given_single_element_torch_tensor_when_serializing_then_returns_scalar(
        self,
    ) -> None:
        """Given: Single-element PyTorch tensor
        When: Converting to safe JSON
        Then: Returns Python scalar
        """
        try:
            import torch

            tensor = torch.tensor(42.0)
            result = safe_json(tensor)
            assert result == 42.0
            assert isinstance(result, float)
        except ImportError:
            pytest.skip("PyTorch not available")
