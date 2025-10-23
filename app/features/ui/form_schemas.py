from __future__ import annotations

from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import TypedDict

# Strongly typed field kinds to avoid magic strings elsewhere
FieldKind = Literal["text", "textarea", "file", "json"]


class UIField(TypedDict, total=False):
    """Descriptor used by templates to render a single form control."""

    name: str
    label: str
    type: FieldKind  # 'text' | 'textarea' | 'file' | 'json'
    required: bool
    accept: str  # for file inputs (e.g. "image/*")
    placeholder: str
    help: str


def _label_for(name: str, label: Optional[str]) -> str:
    """Return the given label or a prettified version of the field name."""
    return label or name.replace("_", " ")


# ---- Field builders --------------------------------------------------------


def text_field(
    name: str,
    label: Optional[str] = None,
    *,
    placeholder: str = "",
    required: bool = True,
) -> UIField:
    return {
        "name": name,
        "label": _label_for(name, label),
        "type": "text",
        "required": required,
        "placeholder": placeholder,
    }


def textarea_field(
    name: str,
    label: Optional[str] = None,
    *,
    placeholder: str = "",
    required: bool = True,
) -> UIField:
    return {
        "name": name,
        "label": _label_for(name, label),
        "type": "textarea",
        "required": required,
        "placeholder": placeholder,
    }


def file_field(
    name: str,
    label: Optional[str] = None,
    *,
    accept: str,
    required: bool = True,
) -> UIField:
    return {
        "name": name,
        "label": _label_for(name, label),
        "type": "file",
        "required": required,
        "accept": accept,
    }


def json_field(
    name: str,
    label: Optional[str] = None,
    *,
    placeholder: str = "[]",
    required: bool = False,
    help: str = "",
) -> UIField:
    return {
        "name": name,
        "label": _label_for(name, label),
        "type": "json",
        "required": required,
        "placeholder": placeholder,
        "help": help,
    }


def candidate_labels_field(name: str = "candidate_labels") -> UIField:
    """Common JSON field used across zero-shot tasks."""
    return json_field(
        name,
        label="Candidate labels (JSON array)",
        placeholder='["label_a","label_b"]',
        help="JSON array of strings.",
    )


# ---- Task → fields mapping -------------------------------------------------

_DEFAULT_FIELDS: List[UIField] = [
    textarea_field("prompt", placeholder="Enter prompt ..."),
]

TASK_FORM_SPECS: Dict[str, List[UIField]] = {
    # Text-only
    "text-generation": [
        textarea_field("prompt", placeholder="Write a poem about ...")
    ],
    "text2text-generation": [
        textarea_field("prompt", placeholder="Translate to German: ...")
    ],
    "summarization": [
        textarea_field("prompt", placeholder="Summarize this text ...")
    ],
    "translation": [
        textarea_field("prompt", placeholder="Text to translate ..."),
        text_field(
            "src_lang",
            label="Source language (optional)",
            placeholder="en or eng_Latn",
            required=False,
        ),
        text_field(
            "tgt_lang",
            label="Target language (optional)",
            placeholder="de or deu_Latn",
            required=False,
        ),
    ],
    "question-answering": [
        text_field(
            "qa_question", label="Question", placeholder="Who wrote Faust?"
        ),
        textarea_field(
            "qa_context",
            label="Context",
            placeholder="Johann Wolfgang von Goethe ...",
        ),
    ],
    "fill-mask": [
        text_field(
            "mask_sentence",
            label="Sentence with mask",
            placeholder="The capital of Switzerland is <mask>.",
        ),
        text_field(
            "mask_sentence_alt",
            label="Alt sentence (optional)",
            placeholder="The capital of Switzerland is [MASK].",
            required=False,
        ),
    ],
    "sentiment-analysis": [
        textarea_field("prompt", placeholder="I absolutely loved this place!")
    ],
    "token-classification": [
        textarea_field(
            "prompt", placeholder="Barack Obama was born in Hawaii."
        )
    ],
    "feature-extraction": [
        textarea_field("prompt", placeholder="This is a short sentence.")
    ],
    "text-to-image": [
        textarea_field(
            "prompt", placeholder="A cozy wooden cabin in snowy mountains ..."
        )
    ],
    "text-to-speech": [
        textarea_field(
            "tts_text", label="Text", placeholder="Hello from Switzerland!"
        )
    ],
    "text-to-audio": [
        textarea_field(
            "tta_prompt",
            label="Prompt",
            placeholder="Lo-fi chillhop beat with warm drums ...",
        )
    ],
    # Image + text
    "image-text-to-text": [
        file_field("image", label="Image", accept="image/*"),
        textarea_field("prompt", placeholder="Describe the image ..."),
    ],
    "visual-question-answering": [
        file_field("image", label="Image", accept="image/*"),
        text_field("question", placeholder="What is on the table?"),
    ],
    "document-question-answering": [
        file_field("image", label="Document image", accept="image/*"),
        text_field("question", placeholder="What is the total amount?"),
    ],
    "zero-shot-image-classification": [
        file_field("image", label="Image", accept="image/*"),
        candidate_labels_field(),
    ],
    "zero-shot-object-detection": [
        file_field("image", label="Image", accept="image/*"),
        candidate_labels_field(),
    ],
    "image-to-image": [
        file_field("image", label="Init image", accept="image/*"),
        textarea_field("prompt", placeholder="Make the sky sunset orange."),
    ],
    # Image-only
    "image-to-text": [file_field("image", label="Image", accept="image/*")],
    "image-classification": [
        file_field("image", label="Image", accept="image/*")
    ],
    "object-detection": [file_field("image", label="Image", accept="image/*")],
    "image-segmentation": [
        file_field("image", label="Image", accept="image/*")
    ],
    "image-feature-extraction": [
        file_field("image", label="Image", accept="image/*")
    ],
    "depth-estimation": [file_field("image", label="Image", accept="image/*")],
    "mask-generation": [file_field("image", label="Image", accept="image/*")],
    # Audio
    "automatic-speech-recognition": [
        file_field("audio", label="Audio", accept="audio/*")
    ],
    "audio-classification": [
        file_field("audio", label="Audio", accept="audio/*")
    ],
    "zero-shot-audio-classification": [
        file_field("audio", label="Audio", accept="audio/*"),
        candidate_labels_field(),
    ],
    # Video
    "video-classification": [
        file_field("video", label="Video", accept="video/*")
    ],
    # Tables
    "table-question-answering": [
        json_field(
            "table",
            label="Table (JSON 2D array)",
            placeholder='[["city","country"],["Bern","Switzerland"]]',
            help="List of rows; include header row if needed.",
        ),
        text_field(
            "table_query",
            label="Query",
            placeholder="Which country is Bern in?",
        ),
    ],
    # Text classification (zero-shot)
    "zero-shot-classification": [
        textarea_field(
            "prompt",
            placeholder="This restaurant was surprisingly good for the price.",
        ),
        candidate_labels_field(),
    ],
}


def get_fields_for_task(task: Optional[str]) -> List[UIField]:
    """Return fields for the requested task or a sensible default."""
    if not task:
        return []
    return TASK_FORM_SPECS.get(task, _DEFAULT_FIELDS)


__all__ = ["UIField", "get_fields_for_task", "TASK_FORM_SPECS"]
