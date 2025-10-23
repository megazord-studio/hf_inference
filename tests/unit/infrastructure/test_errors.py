"""Unit tests for infrastructure error detection utilities.

Scenario: Error detection utilities identify common failure modes.

Test coverage:
- CUDA OOM detection
- Missing model detection
- Missing weight files detection
- Gated repository access detection
"""

import pytest

from app.infrastructure.errors import is_cuda_oom
from app.infrastructure.errors import is_gated_repo_error
from app.infrastructure.errors import is_missing_model_error
from app.infrastructure.errors import is_no_weight_files_error


class TestErrorDetection:
    """Test scenarios for error detection utilities."""

    def test_given_cuda_oom_exception_when_checking_then_returns_true(
        self,
    ) -> None:
        """Given: Exception with CUDA out of memory message
        When: Checking if it's a CUDA OOM error
        Then: Returns True
        """
        error = RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        assert is_cuda_oom(error) is True

        error2 = RuntimeError("torch.cuda.OutOfMemoryError: CUDA OOM")
        assert is_cuda_oom(error2) is True

    def test_given_non_oom_exception_when_checking_then_returns_false(
        self,
    ) -> None:
        """Given: Exception without CUDA OOM message
        When: Checking if it's a CUDA OOM error
        Then: Returns False
        """
        error = RuntimeError("Some other CUDA error")
        assert is_cuda_oom(error) is False

        error2 = ValueError("Invalid input")
        assert is_cuda_oom(error2) is False

    def test_given_missing_model_exception_when_checking_then_returns_true(
        self,
    ) -> None:
        """Given: Exception indicating model not found
        When: Checking if it's a missing model error
        Then: Returns True
        """
        error = ValueError(
            "nonexistent/model is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'"
        )
        assert is_missing_model_error(error) is True

    def test_given_other_exception_when_checking_missing_model_then_returns_false(
        self,
    ) -> None:
        """Given: Exception without missing model message
        When: Checking if it's a missing model error
        Then: Returns False
        """
        error = ValueError("Some other error")
        assert is_missing_model_error(error) is False

    def test_given_no_weights_exception_when_checking_then_returns_true(
        self,
    ) -> None:
        """Given: Exception indicating missing weight files
        When: Checking if it's a no weights error
        Then: Returns True
        """
        error = OSError(
            "mymodel does not appear to have a file named pytorch_model.bin"
        )
        assert is_no_weight_files_error(error) is True

        error2 = OSError("Missing model.safetensors file")
        assert is_no_weight_files_error(error2) is True

    def test_given_other_exception_when_checking_no_weights_then_returns_false(
        self,
    ) -> None:
        """Given: Exception without missing weights message
        When: Checking if it's a no weights error
        Then: Returns False
        """
        error = OSError("File not found")
        assert is_no_weight_files_error(error) is False

    def test_given_gated_repo_exception_when_checking_then_returns_true(
        self,
    ) -> None:
        """Given: Exception indicating gated repository access
        When: Checking if it's a gated repo error
        Then: Returns True
        """
        error = ValueError("This is a gated repo")
        assert is_gated_repo_error(error) is True

        error2 = RuntimeError(
            "401 Client Error: Unauthorized for url: https://huggingface.co/..."
        )
        assert is_gated_repo_error(error2) is True

        error3 = RuntimeError("Access to model meta-llama/Llama-2-70b is restricted")
        assert is_gated_repo_error(error3) is True

    def test_given_other_exception_when_checking_gated_repo_then_returns_false(
        self,
    ) -> None:
        """Given: Exception without gated repo indicators
        When: Checking if it's a gated repo error
        Then: Returns False
        """
        error = ValueError("Invalid configuration")
        assert is_gated_repo_error(error) is False

        error2 = RuntimeError("Connection timeout")
        assert is_gated_repo_error(error2) is False
