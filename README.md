# HF Inference FastAPI (uv Edition)

A FastAPI-based inference server for HuggingFace models supporting 31+ different tasks.

## Quick Start (with `uv`)

### 1) Prerequisites

- Python 3.12+
- ffmpeg (for video processing)
- tesseract-ocr, libtesseract-dev, libleptonica-dev (for OCR tasks)
- [`uv`](https://docs.astral.sh/uv/) installed (`pipx install uv` or via your package manager)

### 3) Install dependencies

```bash
# Base deps
uv sync

uv sync --extra dev
```

## Running the Server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 [--reload]
```

## Making Requests

See [API.md](./API.md) for detailed API documentation and examples.

Quick example:

```bash
# Health check
curl http://localhost:8000/healthz

# Text generation
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"gpt2","task":"text-generation","payload":{"prompt":"Hello world"}}'

# Image classification
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"google/vit-base-patch16-224","task":"image-classification","payload":{}}' \
  -F 'image=@/path/to/image.jpg'
```

## Supported Tasks

The API supports 31+ different inference tasks including:
- Text generation, summarization, translation
- Image classification, object detection, segmentation
- Audio classification and speech recognition
- Video classification
- Text-to-image, text-to-audio generation
- Question answering (text, visual, document)
- And many more...

See [API.md](./API.md) for the complete list and usage examples.

## Testing

```bash
# Run tests
uv run pytest [-ra]
```
