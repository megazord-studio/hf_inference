# Migration Guide: From CLI to FastAPI

This guide explains the differences between the old CLI-based inference system and the new FastAPI-based API.

## Overview

The package has been refactored from a command-line tool that reads `demo.yaml` to a REST API server that accepts HTTP requests.

## Key Changes

### Before (CLI)
```bash
# Load models from demo.yaml and run inference
python app/main.py

# Set environment variables for filtering
ONLY_TASK=text-generation python app/main.py
ONLY_MODEL=gpt2 python app/main.py
```

### After (FastAPI)
```bash
# Start the API server
python start_server.py

# Make HTTP requests
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"gpt2","task":"text-generation","payload":{"prompt":"Hello"}}'
```

## File Handling

### Before (CLI)
- Files referenced by paths in `demo.yaml`
- Output saved to `./outputs/` directory
- Used `assets/` folder for input files

### After (FastAPI)
- Files uploaded via multipart form data
- Output returned in HTTP response (no disk writes)
- No dependency on local file system

## Response Format

### Before (CLI)
```python
# Printed to stdout
print(json.dumps(result, indent=2))
```

### After (FastAPI)
```json
// Returned as HTTP response
{
  "result": "..."
}

// Or for file generation tasks
// Returns binary file data with appropriate Content-Type
```

## Task Mapping

All tasks from `demo.yaml` are supported. Here's how to migrate:

### Text Generation
**Before:**
```yaml
- model_id: gpt2
  task: text-generation
  payload:
    prompt: "Hello, I am"
```

**After:**
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"gpt2","task":"text-generation","payload":{"prompt":"Hello, I am"}}'
```

### Image Classification
**Before:**
```yaml
- model_id: google/vit-base-patch16-224
  task: image-classification
  payload:
    image_path: "assets/image.jpg"
```

**After:**
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"google/vit-base-patch16-224","task":"image-classification","payload":{}}' \
  -F 'image=@/path/to/image.jpg'
```

### Video Classification
**Before:**
```yaml
- model_id: MCG-NJU/videomae-base
  task: video-classification
  payload:
    video_path: "assets/video.mp4"
```

**After:**
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"MCG-NJU/videomae-base","task":"video-classification","payload":{}}' \
  -F 'video=@/path/to/video.mp4'
```

### Audio Tasks
**Before:**
```yaml
- model_id: facebook/wav2vec2-base-960h
  task: automatic-speech-recognition
  payload:
    audio_path: "assets/audio.wav"
```

**After:**
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"facebook/wav2vec2-base-960h","task":"automatic-speech-recognition","payload":{}}' \
  -F 'audio=@/path/to/audio.wav'
```

### Text-to-Image
**Before:**
```yaml
- model_id: runwayml/stable-diffusion-v1-5
  task: text-to-image
  payload:
    prompt: "A sunset over mountains"
# Output saved to ./outputs/sd_*.png
```

**After:**
```bash
# Returns image file directly
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"runwayml/stable-diffusion-v1-5","task":"text-to-image","payload":{"prompt":"A sunset over mountains"}}' \
  --output generated.png
```

## Error Handling

### Before (CLI)
```python
# Errors printed to stdout
{
  "error": "task failed",
  "reason": "..."
}
```

### After (FastAPI)
```bash
# HTTP status codes indicate success/failure
200 OK - Success
400 Bad Request - Invalid input
500 Internal Server Error - Inference failed
```

## Testing

### Before (CLI)
```bash
# Manual testing by running the script
python app/main.py
```

### After (FastAPI)
```bash
# Automated testing with pytest
pytest tests/ -v

# Test specific endpoint
pytest tests/runners/test_text_generation.py -v
```

## Integration

### Before (CLI)
```python
# Run as subprocess
import subprocess
result = subprocess.run(['python', 'app/main.py'], ...)
```

### After (FastAPI)
```python
# Use HTTP client
import requests
response = requests.post(
    'http://localhost:8000/inference',
    data={'spec': json.dumps({...})},
    files={'image': open('image.jpg', 'rb')}
)
result = response.json()
```

## Benefits of FastAPI Approach

1. **Stateless**: Each request is independent
2. **Scalable**: Can run multiple instances behind load balancer
3. **Language Agnostic**: Any language can call the API
4. **Auto Documentation**: Swagger UI at /docs
5. **Type Safety**: Pydantic models validate input
6. **No File System**: No cleanup needed
7. **Better Testing**: Integration tests with pytest
8. **Production Ready**: Can deploy with Docker, K8s, etc.

## Backward Compatibility

The `demo.yaml` file is still included for reference, but is no longer used by the application. You can use it as a guide for creating API requests.

To recreate the old behavior:
```python
# Load demo.yaml and make API requests
import yaml
import requests

with open('demo.yaml') as f:
    demos = yaml.safe_load(f)['demos']

for demo in demos:
    response = requests.post(
        'http://localhost:8000/inference',
        data={'spec': json.dumps(demo)}
    )
    print(response.json())
```
