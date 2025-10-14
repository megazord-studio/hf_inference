from typing import Any
from typing import Dict

from .audio_classification import run_audio_classification
from .automatic_speech_recognition import run_asr
from .depth_estimation import run_depth_estimation
from .document_question_answering import run_doc_qa
from .feature_extraction import run_feature_extraction
from .fill_mask import run_fill_mask
from .image_classification import run_image_classification
from .image_feature_extraction import run_image_feature_extraction
from .image_segmentation import run_image_segmentation
from .image_text_to_text import run_vlm_image_text_to_text
from .image_to_image import run_image_to_image
from .image_to_text import run_image_to_text
from .mask_generation import run_mask_generation
from .object_detection import run_object_detection
from .question_answering import run_qa
from .sentiment import run_sentiment
from .summarization import run_summarization
from .table_question_answering import run_table_qa
from .text2text_generation import run_text2text
from .text_generation import run_text_generation
from .text_to_audio import run_text_to_audio
from .text_to_image import run_text_to_image
from .text_to_speech import run_tts
from .token_classification import run_ner
from .translation import run_translation
from .video_classification import run_video_classification
from .visual_question_answering import run_vqa
from .zero_shot_audio_classification import run_zero_shot_audio_classification
from .zero_shot_classification import run_zero_shot_classification
from .zero_shot_image_classification import run_zero_shot_image_classification
from .zero_shot_object_detection import run_zero_shot_object_detection

RUNNERS: Dict[str, Any] = {
    "text-generation": run_text_generation,
    "text2text-generation": run_text2text,
    "zero-shot-classification": run_zero_shot_classification,
    "summarization": run_summarization,
    "translation": run_translation,
    "question-answering": run_qa,
    "fill-mask": run_fill_mask,
    "sentiment-analysis": run_sentiment,
    "token-classification": run_ner,
    "feature-extraction": run_feature_extraction,
    "table-question-answering": run_table_qa,
    "visual-question-answering": run_vqa,
    "document-question-answering": run_doc_qa,
    "image-text-to-text": run_vlm_image_text_to_text,
    "image-to-text": run_image_to_text,
    "zero-shot-image-classification": run_zero_shot_image_classification,
    "image-classification": run_image_classification,
    "zero-shot-object-detection": run_zero_shot_object_detection,
    "object-detection": run_object_detection,
    "image-segmentation": run_image_segmentation,
    "depth-estimation": run_depth_estimation,
    "automatic-speech-recognition": run_asr,
    "audio-classification": run_audio_classification,
    "zero-shot-audio-classification": run_zero_shot_audio_classification,
    "text-to-speech": run_tts,
    "text-to-audio": run_text_to_audio,
    "text-to-image": run_text_to_image,
    "image-feature-extraction": run_image_feature_extraction,
    "video-classification": run_video_classification,
    "mask-generation": run_mask_generation,
    "image-to-image": run_image_to_image,
}
