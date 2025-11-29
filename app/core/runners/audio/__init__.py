"""Audio runners package - ASR, TTS, audio classification, etc.

Exports runner classes and task set for audio-related inference.
"""

from typing import Set
from typing import Type

from app.core.runners.base import BaseRunner

from .asr import AutomaticSpeechRecognitionRunner
from .classification import AudioClassificationRunner
from .extended import AudioTextToTextRunner
from .extended import AudioToAudioRunner
from .extended import TextToAudioRunner
from .extended import VoiceActivityDetectionRunner
from .tts import TextToSpeechRunner

AUDIO_TASKS: Set[str] = {
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-classification",
    "audio-to-audio",
    "text-to-audio",
    "audio-text-to-text",
    "voice-activity-detection",
}

_TASK_TO_RUNNER = {
    "automatic-speech-recognition": AutomaticSpeechRecognitionRunner,
    "text-to-speech": TextToSpeechRunner,
    "audio-classification": AudioClassificationRunner,
    "audio-to-audio": AudioToAudioRunner,
    "text-to-audio": TextToAudioRunner,
    "audio-text-to-text": AudioTextToTextRunner,
    "voice-activity-detection": VoiceActivityDetectionRunner,
}


def audio_runner_for_task(task: str) -> Type[BaseRunner]:
    """Return the runner class for an audio task."""
    return _TASK_TO_RUNNER[task]


__all__ = [
    "AUDIO_TASKS",
    "audio_runner_for_task",
    "AutomaticSpeechRecognitionRunner",
    "TextToSpeechRunner",
    "AudioClassificationRunner",
    "AudioToAudioRunner",
    "TextToAudioRunner",
    "AudioTextToTextRunner",
    "VoiceActivityDetectionRunner",
]
