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
    """Tests for ensure_image_tokens function."""

    def test_no_change_when_correct_count(self):
        """Token already present with correct count - no modification."""
        from app.core.runners.multimodal.tokenizer import ensure_image_tokens

        tok = FakeTokenizer("<image>")
        question = "<image> What color is this?"
        result = ensure_image_tokens(question, num_images=1, tokenizer=tok)
        assert result == question

    def test_adds_single_token(self):
        """Single image token added when missing."""
        from app.core.runners.multimodal.tokenizer import ensure_image_tokens

        tok = FakeTokenizer("<image>")
        question = "What color is this?"
        result = ensure_image_tokens(question, num_images=1, tokenizer=tok)
        assert "<image>" in result
        assert result.count("<image>") == 1

    def test_adds_multiple_tokens(self):
        """Multiple image tokens added for multiple images."""
        from app.core.runners.multimodal.tokenizer import ensure_image_tokens

        tok = FakeTokenizer("<image>")
        question = "What is shown?"
        result = ensure_image_tokens(question, num_images=3, tokenizer=tok)
        assert result.count("<image>") == 3

    def test_replaces_wrong_count(self):
        """Replaces existing tokens to match correct count."""
        from app.core.runners.multimodal.tokenizer import ensure_image_tokens

        tok = FakeTokenizer("<image>")
        question = "<image> <image> What is shown?"
        result = ensure_image_tokens(question, num_images=1, tokenizer=tok)
        assert result.count("<image>") == 1

    def test_uses_custom_token(self):
        """Uses custom image token from tokenizer."""
        from app.core.runners.multimodal.tokenizer import ensure_image_tokens

        tok = FakeTokenizer("<img>")
        question = "What color is this?"
        result = ensure_image_tokens(question, num_images=1, tokenizer=tok)
        assert "<img>" in result
        assert result.count("<img>") == 1


class TestGetImageToken:
    """Tests for get_image_token function."""

    def test_returns_tokenizer_image_token(self):
        """Returns image_token attribute from tokenizer."""
        from app.core.runners.multimodal.tokenizer import get_image_token

        tok = FakeTokenizer("<custom_img>")
        result = get_image_token(tok)
        assert result == "<custom_img>"

    def test_fallback_to_special_tokens_map(self):
        """Falls back to special_tokens_map if no image_token."""
        from app.core.runners.multimodal.tokenizer import get_image_token

        tok = FakeTokenizer("")
        tok.image_token = ""
        tok.special_tokens_map = {"image_token": "<image_special>"}
        result = get_image_token(tok)
        assert result == "<image_special>"

    def test_fallback_to_default(self):
        """Falls back to <image> if nothing found."""
        from app.core.runners.multimodal.tokenizer import get_image_token

        tok = FakeTokenizer("")
        tok.image_token = ""
        tok.special_tokens_map = {}
        tok.additional_special_tokens = []
        result = get_image_token(tok)
        assert result == "<image>"


class TestMoveToDevice:
    """Tests for move_to_device function."""

    def test_returns_dict_unchanged_when_no_device(self):
        """Returns encoding unchanged when device is None."""
        from app.core.runners.multimodal.utils import move_to_device

        enc = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        result = move_to_device(enc, device=None)
        assert result == enc

    def test_calls_to_on_tensors(self):
        """Calls .to(device) on tensors with 'to' method."""
        import torch

        from app.core.runners.multimodal.utils import move_to_device

        device = torch.device("cpu")
        tensor = torch.tensor([1, 2, 3])
        enc = {"input_ids": tensor, "value": 42}
        result = move_to_device(enc, device=device)
        assert isinstance(result["input_ids"], torch.Tensor)
        assert result["value"] == 42


class TestStripProcessorOnlyKwargs:
    """Tests for strip_processor_only_kwargs function."""

    def test_strips_num_crops_for_gemma(self):
        """Strips num_crops from encoding for Gemma models."""
        from app.core.runners.multimodal.tokenizer import strip_processor_only_kwargs

        enc = {"input_ids": [1, 2], "num_crops": 4}
        strip_processor_only_kwargs(enc, model_id="google/gemma-3-1b-it")
        assert "num_crops" not in enc
        assert "input_ids" in enc

    def test_no_strip_for_non_gemma(self):
        """Does not strip num_crops for non-Gemma models."""
        from app.core.runners.multimodal.tokenizer import strip_processor_only_kwargs

        enc = {"input_ids": [1, 2], "num_crops": 4}
        strip_processor_only_kwargs(enc, model_id="other/model")
        assert "num_crops" in enc


class TestCapMaxNewTokens:
    """Tests for cap_max_new_tokens function."""

    def test_no_cap_on_cuda(self):
        """No capping on CUDA devices."""
        import torch

        from app.core.runners.multimodal.utils import cap_max_new_tokens

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        result = cap_max_new_tokens(100, device)
        assert result == 100

    def test_caps_on_cpu(self):
        """Caps to 16 on CPU devices."""
        import torch

        from app.core.runners.multimodal.utils import cap_max_new_tokens

        device = torch.device("cpu")
        result = cap_max_new_tokens(100, device)
        assert result == 16

    def test_minimum_is_one(self):
        """Returns minimum of 1 even for zero input."""
        import torch

        from app.core.runners.multimodal.utils import cap_max_new_tokens

        device = torch.device("cpu")
        result = cap_max_new_tokens(0, device)
        assert result == 1
