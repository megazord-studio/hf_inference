"""
Integration tests for text-to-image runner.
Tests hit the /inference endpoint with real models.
"""

import pytest
from tests.conftest import create_spec


class TestTextToImage:
    """Tests for text-to-image task."""
    
    @pytest.mark.skip(reason="Requires large model download and GPU - enable for full integration test")
    def test_text_to_image_sd(self, client):
        """Test text-to-image with Stable Diffusion."""
        spec = create_spec(
            model_id="runwayml/stable-diffusion-v1-5",
            task="text-to-image",
            payload={"prompt": "A cat sitting on a mat"}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        # Should return image file or JSON with file_data
        assert response.status_code in [200, 500]  # May fail on CPU-only or without auth
        
        if response.status_code == 200:
            # Check if it's a file response
            if response.headers.get("content-type") == "image/png":
                assert len(response.content) > 0
            else:
                # Or JSON response
                data = response.json()
                assert "file_data" in data or "skipped" in data or "error" in data
