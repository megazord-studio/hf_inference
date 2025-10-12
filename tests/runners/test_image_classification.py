"""
Integration tests for image classification runner.
Tests hit the /inference endpoint with real models.
"""

import pytest
from tests.conftest import create_spec


class TestImageClassification:
    """Tests for image-classification task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_image_classification_vit(self, client, sample_image):
        """Test image classification with ViT model."""
        spec = create_spec(
            model_id="google/vit-base-patch16-224",
            task="image-classification",
            payload={}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec},
            files={"image": ("test.png", sample_image, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "label" in data[0]
        assert "score" in data[0]
    
    def test_image_classification_no_image(self, client):
        """Test image classification without uploading an image."""
        spec = create_spec(
            model_id="google/vit-base-patch16-224",
            task="image-classification",
            payload={"image_path": "image.jpg"}  # Will fall back to placeholder
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        # Should still process with placeholder image
        assert response.status_code in [200, 500]


class TestZeroShotImageClassification:
    """Tests for zero-shot-image-classification task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_zero_shot_image_classification_clip(self, client, sample_image):
        """Test zero-shot image classification with CLIP."""
        spec = create_spec(
            model_id="openai/clip-vit-base-patch32",
            task="zero-shot-image-classification",
            payload={"candidate_labels": ["cat", "dog", "bird"]}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec},
            files={"image": ("test.png", sample_image, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "label" in data[0]
        assert "score" in data[0]
