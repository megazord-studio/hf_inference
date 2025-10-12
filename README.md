# HF Inference FastAPI

A FastAPI-based inference server for HuggingFace models supporting 31+ different tasks.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or with dev dependencies for testing
pip install -e ".[dev]"
```

### Running the Server

```bash
# Simple way
python start_server.py

# Or with custom options
python start_server.py --port 8080 --reload

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Making Requests

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

## System Requirements

- Python 3.12+
- ffmpeg (for video processing)
- tesseract-ocr, libtesseract-dev, libleptonica-dev (for OCR tasks)

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
pytest tests/ -v

# Run specific test file
pytest tests/runners/test_text_generation.py -v
```

Note: Most integration tests are skipped by default as they require downloading models. Remove the `@pytest.mark.skip` decorator to enable them.

## Architecture

- **app/main.py**: FastAPI application with /healthz and /inference endpoints
- **app/runners/**: Individual runner modules for each task type (31 runners)
- **app/helpers.py**: Utility functions for file handling and data processing
- **tests/**: Pytest-based integration tests

All runners return values instead of printing, and accept file uploads via multipart form data.
