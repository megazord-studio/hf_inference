from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any
from typing import Callable
from typing import Dict


def _lazy_runner(module_path: str, func_name: str) -> Callable[[Any, str], Any]:
    """Import the heavy runner module only when the task executes."""

    @wraps(_lazy_runner)
    def _runner(spec: Any, dev: str) -> Any:
        module = import_module(module_path, package="app.runners")
        func = getattr(module, func_name)
        return func(spec, dev)

    return _runner


RUNNERS: Dict[str, Any] = {
    "text-generation": _lazy_runner(".text_generation", "run_text_generation"),
    "text2text-generation": _lazy_runner(
        ".text2text_generation", "run_text2text"
    ),
    "zero-shot-classification": _lazy_runner(
        ".zero_shot_classification", "run_zero_shot_classification"
    ),
    "summarization": _lazy_runner(".summarization", "run_summarization"),
    "translation": _lazy_runner(".translation", "run_translation"),
    "question-answering": _lazy_runner(
        ".question_answering", "run_qa"
    ),
    "fill-mask": _lazy_runner(".fill_mask", "run_fill_mask"),
    "sentiment-analysis": _lazy_runner(".sentiment", "run_sentiment"),
    "token-classification": _lazy_runner(
        ".token_classification", "run_ner"
    ),
    "feature-extraction": _lazy_runner(
        ".feature_extraction", "run_feature_extraction"
    ),
    "table-question-answering": _lazy_runner(
        ".table_question_answering", "run_table_qa"
    ),
    "visual-question-answering": _lazy_runner(
        ".visual_question_answering", "run_vqa"
    ),
    "document-question-answering": _lazy_runner(
        ".document_question_answering", "run_doc_qa"
    ),
    "image-text-to-text": _lazy_runner(
        ".image_text_to_text", "run_vlm_image_text_to_text"
    ),
    "image-to-text": _lazy_runner(".image_to_text", "run_image_to_text"),
    "zero-shot-image-classification": _lazy_runner(
        ".zero_shot_image_classification", "run_zero_shot_image_classification"
    ),
    "image-classification": _lazy_runner(
        ".image_classification", "run_image_classification"
    ),
    "zero-shot-object-detection": _lazy_runner(
        ".zero_shot_object_detection", "run_zero_shot_object_detection"
    ),
    "object-detection": _lazy_runner(
        ".object_detection", "run_object_detection"
    ),
    "image-segmentation": _lazy_runner(
        ".image_segmentation", "run_image_segmentation"
    ),
    "depth-estimation": _lazy_runner(
        ".depth_estimation", "run_depth_estimation"
    ),
    "automatic-speech-recognition": _lazy_runner(
        ".automatic_speech_recognition", "run_asr"
    ),
    "audio-classification": _lazy_runner(
        ".audio_classification", "run_audio_classification"
    ),
    "zero-shot-audio-classification": _lazy_runner(
        ".zero_shot_audio_classification", "run_zero_shot_audio_classification"
    ),
    "text-to-speech": _lazy_runner(".text_to_speech", "run_tts"),
    "text-to-audio": _lazy_runner(".text_to_audio", "run_text_to_audio"),
    "text-to-image": _lazy_runner(".text_to_image", "run_text_to_image"),
    "image-feature-extraction": _lazy_runner(
        ".image_feature_extraction", "run_image_feature_extraction"
    ),
    "video-classification": _lazy_runner(
        ".video_classification", "run_video_classification"
    ),
    "mask-generation": _lazy_runner(
        ".mask_generation", "run_mask_generation"
    ),
    "image-to-image": _lazy_runner(".image_to_image", "run_image_to_image"),
}
