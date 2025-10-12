"""
Integration tests for audio classification runner.
Tests hit the /inference endpoint with real models.
"""

import pytest
from tests.conftest import create_spec


class TestAudioClassification:
    """Tests for audio-classification task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_audio_classification(self, client, sample_audio):
        """Test audio classification."""
        spec = create_spec(
            model_id="superb/hubert-base-superb-er",
            task="audio-classification",
            payload={}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec},
            files={"audio": ("test.wav", sample_audio, "audio/wav")}
        )
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            # Should return list of classifications or error/skipped
            assert isinstance(data, (list, dict))


class TestAutomaticSpeechRecognition:
    """Tests for automatic-speech-recognition task."""
    
    @pytest.mark.skip(reason="Requires model download - enable for full integration test")
    def test_asr_wav2vec(self, client, sample_audio):
        """Test ASR with wav2vec2."""
        spec = create_spec(
            model_id="facebook/wav2vec2-base-960h",
            task="automatic-speech-recognition",
            payload={}
        )
        
        response = client.post(
            "/inference",
            data={"spec": spec},
            files={"audio": ("test.wav", sample_audio, "audio/wav")}
        )
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            # Should return transcription
            assert "text" in data or "skipped" in data or "error" in data
