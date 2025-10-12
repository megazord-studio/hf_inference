# FastAPI Inference API

This repository has been refactored into a FastAPI application for HuggingFace model inference.

## Endpoints

### Health Check
```bash
curl http://localhost:8000/healthz
```

Returns:
```json
{
  "status": "ok",
  "device": "cuda:0"  # or "cpu"
}
```

### Inference Endpoint

The main inference endpoint accepts multipart form data with:
- `spec`: JSON string containing model_id, task, and payload
- `image`: Optional image file (for image-based tasks)
- `audio`: Optional audio file (for audio-based tasks)
- `video`: Optional video file (for video-based tasks)

## Examples

### Text Generation
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"gpt2","task":"text-generation","payload":{"prompt":"Hello world"}}' 
```

### Image Classification
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"google/vit-base-patch16-224","task":"image-classification","payload":{}}' \
  -F 'image=@/path/to/image.jpg'
```

### Video Classification
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"MCG-NJU/videomae-base","task":"video-classification","payload":{}}' \
  -F 'video=@/path/to/video.mp4'
```

### Audio Classification
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"superb/hubert-base-superb-er","task":"audio-classification","payload":{}}' \
  -F 'audio=@/path/to/audio.wav'
```

### Zero-Shot Image Classification
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"openai/clip-vit-base-patch32","task":"zero-shot-image-classification","payload":{"candidate_labels":["cat","dog","bird"]}}' \
  -F 'image=@/path/to/image.jpg'
```

### Text-to-Image (returns image file)
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"runwayml/stable-diffusion-v1-5","task":"text-to-image","payload":{"prompt":"A sunset over mountains"}}' \
  --output generated_image.png
```

### Sentiment Analysis
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"distilbert-base-uncased-finetuned-sst-2-english","task":"sentiment-analysis","payload":{"prompt":"I love this!"}}'
```

### Question Answering
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"deepset/roberta-base-squad2","task":"question-answering","payload":{"qa_question":"Who wrote Faust?","qa_context":"Johann Wolfgang von Goethe was a German writer."}}'
```

## Running the Server

```bash
# Using Python directly
python app/main.py

# Or using uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (note: most tests are skipped by default as they require model downloads)
pytest tests/

# Run specific test file
pytest tests/runners/test_text_generation.py

# Run only non-skipped tests
pytest tests/ -v

# Run with model downloads enabled (will take time and bandwidth)
pytest tests/ -v --run-integration
```

## Supported Tasks

The API supports all tasks from the original implementation:
- text-generation
- text2text-generation
- zero-shot-classification
- summarization
- translation
- question-answering
- fill-mask
- sentiment-analysis
- token-classification
- feature-extraction
- table-question-answering
- visual-question-answering
- document-question-answering
- image-text-to-text
- image-to-text
- zero-shot-image-classification
- image-classification
- zero-shot-object-detection
- object-detection
- image-segmentation
- depth-estimation
- automatic-speech-recognition
- audio-classification
- zero-shot-audio-classification
- text-to-speech
- text-to-audio
- text-to-image
- image-feature-extraction
- video-classification
- mask-generation
- image-to-image

## Response Types

Responses vary by task:

- **Text tasks**: Return JSON with text results
- **Classification tasks**: Return JSON with labels and scores
- **Generation tasks (images/audio)**: Return file data (can be saved with `--output`)
- **Segmentation tasks**: Return JSON with base64-encoded masks
- **Embeddings**: Return JSON with embedding shape/metadata

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid spec, unsupported task)
- `500`: Internal server error (model inference failed)

Error responses include:
```json
{
  "error": "task-name failed",
  "reason": "detailed error message",
  "hint": "optional suggestion"
}
```

Skipped tasks (e.g., gated models without access):
```json
{
  "skipped": true,
  "reason": "gated model (no access/auth)",
  "hint": "Request access via HF and login"
}
```
