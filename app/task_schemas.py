"""
Task UI schemas for the inference modal.

This module defines the form fields and input structure for each supported task type.
Each schema specifies what inputs, files, and advanced options to show in the UI.
"""

from __future__ import annotations

from typing import Any, Dict


# Schema builder functions - create consistent field structures
def text_field(name: str, label: str, placeholder: str, rows: int = 4, required: bool = True) -> Dict[str, Any]:
    """Create a textarea input field."""
    return {
        "type": "textarea",
        "name": name,
        "label": label,
        "placeholder": placeholder,
        "rows": rows,
        "required": required,
    }


def file_field(name: str, label: str, accept: str, preview: str | None = None) -> Dict[str, Any]:
    """Create a file upload field."""
    field: Dict[str, Any] = {
        "name": name,
        "label": label,
        "accept": accept,
        "required": True,
    }
    if preview:
        field["preview"] = preview
    return field


def schema(category: str, label: str, description: str, inputs: list[Dict[str, Any]] | None = None, files: list[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Create a complete task schema."""
    return {
        "category": category,
        "label": label,
        "description": description,
        "inputs": inputs or [],
        "files": files or [],
        "advanced": [],
    }


# Task schemas - organized by category
TASK_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Text generation tasks
    "text-generation": schema(
        "text",
        "Text generation",
        "Generate text from a prompt",
        inputs=[text_field("prompt", "Prompt", "Write a story about...", rows=6)],
    ),
    "text2text-generation": schema(
        "text",
        "Text to text",
        "Transform text (translation, summarization, etc)",
        inputs=[text_field("prompt", "Input text", "Enter text to transform...", rows=6)],
    ),
    "summarization": schema(
        "text",
        "Summarization",
        "Summarize long text",
        inputs=[text_field("prompt", "Text to summarize", "Paste a long paragraph...", rows=8)],
    ),
    "translation": schema(
        "text",
        "Translation",
        "Translate text between languages",
        inputs=[
            text_field("prompt", "Text to translate", "Enter text...", rows=6),
            {"type": "text", "name": "src_lang", "label": "Source language", "placeholder": "en_XX", "rows": 1, "required": False},
            {"type": "text", "name": "tgt_lang", "label": "Target language", "placeholder": "de_DE", "rows": 1, "required": False},
        ],
    ),
    
    # Text classification tasks
    "sentiment-analysis": schema(
        "text",
        "Sentiment analysis",
        "Analyze sentiment of text",
        inputs=[text_field("prompt", "Text", "Enter text to analyze...", rows=4)],
    ),
    "token-classification": schema(
        "text",
        "Token classification",
        "Identify named entities in text",
        inputs=[text_field("prompt", "Text", "Enter text...", rows=4)],
    ),
    "feature-extraction": schema(
        "text",
        "Feature extraction",
        "Extract text embeddings",
        inputs=[text_field("prompt", "Text", "Enter text...", rows=4)],
    ),
    "zero-shot-classification": schema(
        "text",
        "Zero-shot classification",
        "Classify text with custom labels",
        inputs=[
            text_field("prompt", "Text", "The Swiss Alps are breathtaking.", rows=3),
            {"type": "chips", "name": "candidate_labels", "label": "Labels", "placeholder": "travel, finance, sports", "rows": 2, "required": True},
        ],
    ),
    
    # Question answering
    "question-answering": schema(
        "text",
        "Question answering",
        "Answer questions based on context",
        inputs=[
            text_field("qa_question", "Question", "What is the capital?", rows=2),
            text_field("qa_context", "Context", "Provide context here...", rows=6),
        ],
    ),
    "fill-mask": schema(
        "text",
        "Fill mask",
        "Predict masked tokens",
        inputs=[
            text_field("mask_sentence", "Sentence", "The capital is <mask>.", rows=2),
            text_field("mask_sentence_alt", "Alternative (optional)", "Another sentence with <mask>.", rows=2, required=False),
        ],
    ),
    
    # Multimodal tasks
    "visual-question-answering": schema(
        "multimodal",
        "Visual QA",
        "Answer questions about images",
        inputs=[text_field("question", "Question", "What do you see?", rows=2)],
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "image-text-to-text": schema(
        "multimodal",
        "Image to text",
        "Generate text from image and prompt",
        inputs=[text_field("prompt", "Prompt (optional)", "Describe this image...", rows=2, required=False)],
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "document-question-answering": schema(
        "multimodal",
        "Document QA",
        "Ask questions about documents",
        inputs=[text_field("question", "Question", "What is the total?", rows=2)],
        files=[file_field("image", "Document image", "image/*", "image")],
    ),
    "table-question-answering": schema(
        "multimodal",
        "Table QA",
        "Ask questions about tables",
        inputs=[
            text_field("table_query", "Question", "Which product had highest revenue?", rows=2),
            {"type": "json", "name": "table", "label": "Table (JSON)", "placeholder": '[["Product","Revenue"],["A",1000]]', "rows": 6, "required": True},
        ],
    ),
    
    # Vision tasks
    "image-to-text": schema(
        "vision",
        "Image captioning",
        "Generate captions for images",
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "image-classification": schema(
        "vision",
        "Image classification",
        "Classify image content",
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "image-feature-extraction": schema(
        "vision",
        "Image embeddings",
        "Extract image feature vectors",
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "image-segmentation": schema(
        "vision",
        "Image segmentation",
        "Segment image regions",
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "depth-estimation": schema(
        "vision",
        "Depth estimation",
        "Estimate depth from image",
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "image-to-image": schema(
        "vision",
        "Image to image",
        "Transform images with text prompts",
        inputs=[text_field("prompt", "Prompt", "Add cyberpunk lighting...", rows=3)],
        files=[file_field("image", "Source image", "image/*", "image")],
    ),
    "object-detection": schema(
        "vision",
        "Object detection",
        "Detect objects in images",
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "zero-shot-image-classification": schema(
        "vision",
        "Zero-shot image classification",
        "Classify images with custom labels",
        inputs=[{"type": "chips", "name": "candidate_labels", "label": "Labels", "placeholder": "cat, dog, airplane", "rows": 2, "required": True}],
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "zero-shot-object-detection": schema(
        "vision",
        "Zero-shot object detection",
        "Detect objects with custom labels",
        inputs=[{"type": "chips", "name": "candidate_labels", "label": "Labels", "placeholder": "person, dog, bicycle", "rows": 2, "required": True}],
        files=[file_field("image", "Image", "image/*", "image")],
    ),
    "text-to-image": schema(
        "vision",
        "Text to image",
        "Generate images from text",
        inputs=[text_field("prompt", "Prompt", "A futuristic cityscape...", rows=4)],
    ),
    "mask-generation": schema(
        "vision",
        "Mask generation",
        "Segment Anything - not yet supported",
    ),
    
    # Audio tasks
    "automatic-speech-recognition": schema(
        "audio",
        "Speech recognition",
        "Transcribe audio to text",
        files=[file_field("audio", "Audio file", "audio/*", "audio")],
    ),
    "audio-classification": schema(
        "audio",
        "Audio classification",
        "Classify audio content",
        files=[file_field("audio", "Audio file", "audio/*", "audio")],
    ),
    "zero-shot-audio-classification": schema(
        "audio",
        "Zero-shot audio classification",
        "Classify audio with custom labels",
        inputs=[{"type": "chips", "name": "candidate_labels", "label": "Labels", "placeholder": "speech, music, noise", "rows": 2, "required": True}],
        files=[file_field("audio", "Audio file", "audio/*", "audio")],
    ),
    "text-to-speech": schema(
        "audio",
        "Text to speech",
        "Synthesize speech from text",
        inputs=[text_field("tts_text", "Text", "Hello from the Swiss mountains!", rows=4)],
    ),
    "text-to-audio": schema(
        "audio",
        "Text to audio",
        "Generate audio from text",
        inputs=[text_field("tta_prompt", "Prompt", "A mellow lo-fi beat...", rows=4)],
    ),
    
    # Video tasks
    "video-classification": schema(
        "video",
        "Video classification",
        "Classify video content",
        files=[file_field("video", "Video file", "video/*", "video")],
    ),
}


def get_schema(task: str) -> Dict[str, Any]:
    """Get the UI schema for a task, with fallback for unknown tasks."""
    if task in TASK_SCHEMAS:
        return TASK_SCHEMAS[task]
    
    # Fallback for unsupported tasks
    return {
        "category": "text",
        "label": task.replace("-", " ").title(),
        "description": f"No specific form for '{task}' - submit JSON via API",
        "inputs": [],
        "files": [],
        "advanced": [],
    }
