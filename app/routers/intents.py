from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["intents"])

class Intent(BaseModel):
    id: str
    label: str
    description: str
    input_types: List[str]
    hf_tasks: List[str]

# Static intent catalog (aligns with frontend taxonomy & common HF tasks)
INTENTS: List[Intent] = [
    Intent(id="summarize", label="Summarize", description="Condense long text", input_types=["text"], hf_tasks=["summarization"]),
    Intent(id="translate", label="Translate", description="Translate text between languages", input_types=["text"], hf_tasks=["translation", "text2text-generation"]),
    Intent(id="qa", label="Question Answering", description="Answer questions given context", input_types=["text"], hf_tasks=["question-answering", "table-question-answering"]),
    Intent(id="classify-text", label="Text Classification", description="Classify or tag text", input_types=["text"], hf_tasks=["zero-shot-classification", "sentiment-analysis", "token-classification"]),
    Intent(id="embed-text", label="Text Embeddings", description="Generate embeddings for semantic tasks", input_types=["text"], hf_tasks=["feature-extraction"]),
    Intent(id="describe-image", label="Image Captioning", description="Generate a description of an image", input_types=["image"], hf_tasks=["image-to-text"]),
    Intent(id="detect-objects", label="Object Detection", description="Detect and localize objects", input_types=["image"], hf_tasks=["object-detection"]),
    Intent(id="segment-image", label="Image Segmentation", description="Generate segmentation masks", input_types=["image"], hf_tasks=["image-segmentation", "mask-generation"]),
    Intent(id="depth-estimate", label="Depth Estimation", description="Infer depth map from image", input_types=["image"], hf_tasks=["depth-estimation"]),
    Intent(id="image-to-image", label="Image-to-Image", description="Transform an input image", input_types=["image"], hf_tasks=["image-to-image"]),
    Intent(id="text-to-image", label="Text-to-Image", description="Generate image from text prompt", input_types=["text"], hf_tasks=["text-to-image"]),
]

@router.get("/intents", response_model=List[Intent])
async def list_intents() -> List[Intent]:
    return INTENTS
