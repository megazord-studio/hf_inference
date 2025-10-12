"""
Integration tests for summarization and translation runners.
Tests hit the /inference endpoint with real models.
"""

import pytest
from tests.conftest import create_spec


class TestSummarization:
    """Tests for summarization task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_summarization_bart(self, client):
        """Test summarization with BART."""
        spec = create_spec(
            model_id="facebook/bart-large-cnn",
            task="summarization",
            payload={"prompt": "Switzerland trains are punctual and efficient. The rail network is extensive."}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, list):
            assert len(data) > 0
            assert "summary_text" in data[0]


class TestTranslation:
    """Tests for translation task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_translation_en_de(self, client):
        """Test English to German translation."""
        spec = create_spec(
            model_id="Helsinki-NLP/opus-mt-en-de",
            task="translation",
            payload={"prompt": "Good morning, how are you?"}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, list):
            assert len(data) > 0
            assert "translation_text" in data[0]


class TestSentimentAnalysis:
    """Tests for sentiment-analysis task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_sentiment_analysis(self, client):
        """Test sentiment analysis."""
        spec = create_spec(
            model_id="distilbert-base-uncased-finetuned-sst-2-english",
            task="sentiment-analysis",
            payload={"prompt": "I love this product!"}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, list):
            assert len(data) > 0
            assert "label" in data[0]
            assert "score" in data[0]
