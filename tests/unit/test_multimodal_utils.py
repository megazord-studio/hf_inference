"""Unit tests for multimodal runner utilities."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class FakeTokenizer:
    """Fake tokenizer for testing."""
    def __init__(self, image_token: str = "<image>"):
        self.image_token = image_token
        self.special_tokens_map = {}
        self.additional_special_tokens = []


class TestEnsureImageTokens:
    """Tests for _ensure_image_tokens method."""

    def _make_runner(self, image_token: str = "<image>"):
        """Create a minimal runner instance for testing."""
        from app.core.runners.multimodal import ImageTextToTextRunner
        runner = ImageTextToTextRunner(model_id="test", device=None)
        runner.processor = MagicMock()
        runner.processor.tokenizer = FakeTokenizer(image_token)
        return runner

    def test_no_change_when_correct_count(self):
        """Token already present with correct count - no modification."""
        runner = self._make_runner("<image>")
        question = "<image> What color is this?"
        result = runner._ensure_image_tokens(question, num_images=1)
        assert result == question

    def test_adds_single_token(self):
        """Single image token added when missing."""
        runner = self._make_runner("<image>")
        question = "What color is this?"
        result = runner._ensure_image_tokens(question, num_images=1)
        assert "<image>" in result
        assert result.count("<image>") == 1

    def test_adds_multiple_tokens(self):
        """Multiple image tokens added for multiple images."""
        runner = self._make_runner("<image>")
        question = "What is shown?"
        result = runner._ensure_image_tokens(question, num_images=3)
        assert result.count("<image>") == 3

    def test_replaces_wrong_count(self):
        """Replaces existing tokens to match correct count."""
        runner = self._make_runner("<image>")
        question = "<image> <image> What is shown?"
        result = runner._ensure_image_tokens(question, num_images=1)
        assert result.count("<image>") == 1

    def test_uses_custom_token(self):
        """Uses custom image token from tokenizer."""
        runner = self._make_runner("<img>")
        question = "What color is this?"
        result = runner._ensure_image_tokens(question, num_images=1)
        assert "<img>" in result
        assert result.count("<img>") == 1


class TestGetImageToken:
    """Tests for _get_image_token method."""

    def _make_runner(self):
        """Create a minimal runner instance for testing."""
        from app.core.runners.multimodal import ImageTextToTextRunner
        runner = ImageTextToTextRunner(model_id="test", device=None)
        return runner

    def test_returns_tokenizer_image_token(self):
        """Returns image_token attribute from tokenizer."""
        runner = self._make_runner()
        runner.processor = MagicMock()
        runner.processor.tokenizer = FakeTokenizer("<custom_img>")
        result = runner._get_image_token()
        assert result == "<custom_img>"

    def test_fallback_to_special_tokens_map(self):
        """Falls back to special_tokens_map if no image_token."""
        runner = self._make_runner()
        runner.processor = MagicMock()
        tok = FakeTokenizer("")
        tok.image_token = ""
        tok.special_tokens_map = {"image_token": "<image_special>"}
        runner.processor.tokenizer = tok
        result = runner._get_image_token()
        assert result == "<image_special>"

    def test_fallback_to_default(self):
        """Falls back to <image> if nothing found."""
        runner = self._make_runner()
        runner.processor = MagicMock()
        tok = FakeTokenizer("")
        tok.image_token = ""
        tok.special_tokens_map = {}
        tok.additional_special_tokens = []
        runner.processor.tokenizer = tok
        result = runner._get_image_token()
        assert result == "<image>"


class TestMoveToDevice:
    """Tests for _move_to_device method."""

    def _make_runner(self, device):
        """Create a minimal runner instance for testing."""
        from app.core.runners.multimodal import ImageTextToTextRunner
        runner = ImageTextToTextRunner(model_id="test", device=device)
        return runner

    def test_returns_dict_unchanged_when_no_device(self):
        """Returns encoding unchanged when device is None."""
        runner = self._make_runner(device=None)
        enc = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        result = runner._move_to_device(enc)
        assert result == enc

    def test_calls_to_on_tensors(self):
        """Calls .to(device) on tensors with 'to' method."""
        import torch
        runner = self._make_runner(device=torch.device("cpu"))
        tensor = torch.tensor([1, 2, 3])
        enc = {"input_ids": tensor, "value": 42}
        result = runner._move_to_device(enc)
        assert isinstance(result["input_ids"], torch.Tensor)
        assert result["value"] == 42


class TestStripProcessorOnlyKwargs:
    """Tests for _strip_processor_only_kwargs method."""

    def _make_runner(self, model_id: str):
        """Create a minimal runner instance for testing."""
        from app.core.runners.multimodal import ImageTextToTextRunner
        runner = ImageTextToTextRunner(model_id=model_id, device=None)
        return runner

    def test_strips_num_crops_for_gemma(self):
        """Strips num_crops from encoding for Gemma models."""
        runner = self._make_runner("google/gemma-3-1b-it")
        enc = {"input_ids": [1, 2], "num_crops": 4}
        runner._strip_processor_only_kwargs(enc)
        assert "num_crops" not in enc
        assert "input_ids" in enc

    def test_no_strip_for_non_gemma(self):
        """Does not strip num_crops for non-Gemma models."""
        runner = self._make_runner("other/model")
        enc = {"input_ids": [1, 2], "num_crops": 4}
        runner._strip_processor_only_kwargs(enc)
        assert "num_crops" in enc


class TestCapMaxNewTokens:
    """Tests for _cap_max_new_tokens method."""

    def _make_runner(self, is_cuda: bool):
        """Create a minimal runner instance for testing."""
        import torch

        from app.core.runners.multimodal import ImageTextToTextRunner
        device = torch.device("cuda") if is_cuda else torch.device("cpu")
        runner = ImageTextToTextRunner(model_id="test", device=device)
        return runner

    def test_no_cap_on_cuda(self):
        """No capping on CUDA devices."""
        runner = self._make_runner(is_cuda=True)
        # Skip test if CUDA not available
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        result = runner._cap_max_new_tokens(100)
        assert result == 100

    def test_caps_on_cpu(self):
        """Caps to 16 on CPU devices."""
        runner = self._make_runner(is_cuda=False)
        result = runner._cap_max_new_tokens(100)
        assert result == 16

    def test_minimum_is_one(self):
        """Returns minimum of 1 even for zero input."""
        runner = self._make_runner(is_cuda=False)
        result = runner._cap_max_new_tokens(0)
        assert result == 1
