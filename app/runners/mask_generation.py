from typing import Any
from typing import Dict

from app.types import RunnerSpec


def run_mask_generation(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Mask generation is not supported via transformers pipeline.
    Returns an error message.
    """
    return {
        "error": "mask-generation unsupported",
        "reason": "Segment Anything models are not exposed via transformers.pipeline.",
        "hint": "Use facebook/sam... with the segment-anything library, or switch to an image-segmentation model.",
    }
