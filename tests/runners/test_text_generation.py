"""
Integration tests for text-generation runner.
Tests hit the /inference endpoint with real models.
"""

import pytest
from tests.conftest import create_spec


class TestTextGeneration:
    """Tests for text-generation task."""
    
    def test_healthz(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "device" in data
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_text_generation_gpt2(self, client):
        """Test text generation with GPT-2 model."""
        spec = create_spec(
            model_id="gpt2",
            task="text-generation",
            payload={"prompt": "Hello, I am"}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or isinstance(data, dict)
        if isinstance(data, list):
            assert len(data) > 0
            assert "generated_text" in data[0]
    
    def test_text_generation_unsupported_task(self, client):
        """Test with unsupported task."""
        spec = create_spec(
            model_id="gpt2",
            task="nonexistent-task",
            payload={"prompt": "test"}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_text_generation_invalid_json(self, client):
        """Test with invalid JSON spec."""
        response = client.post(
            "/inference",
            data={"spec": "invalid json {"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid JSON" in data["detail"]
