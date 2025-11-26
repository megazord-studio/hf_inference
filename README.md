# HF Inference

FastAPI server + small frontend to try Hugging Face models through a single, consistent API.

- One `/api/inference` endpoint for text, vision, audio, and multimodal tasks.
- Optional streaming endpoints for text, diffusion, and TTS.
- Uses real HF Hub models via `transformers`, `diffusers`, `onnxruntime`, etc.

> This project is still under heavy development. Expect breaking changes.

## Install & run

```bash
# From the repo root (dev workflow)
uv sync
uv run fastapi dev app.main:app --host 0.0.0.0 --port 8000
```

The API will be served at `http://localhost:8000`.

## Core API

### `POST /api/inference`

Single JSON endpoint for all non-streaming tasks.

Request body (`InferenceRequest`):

- `model_id` – HF repo id, e.g. `gpt2`, `runwayml/stable-diffusion-v1-5`.
- `intent_id` – optional opaque id from the UI.
- `input_type` – high-level modality, e.g. `text`, `image`, `audio`, `video`.
- `inputs` – task-specific payload; typical keys:
  - `text` – prompt for text or text-to-image.
  - `image_base64` – `data:image/...;base64,...`.
  - `audio_base64` – `data:audio/wav;base64,...`.
- `task` – pipeline tag / task id (see below).
- `options` – model/task options, e.g. decoding params.

Response (`InferenceResponse`):

```jsonc
{
  "result": {
    "task": "text-generation",
    "task_output": { /* task-specific dict */ },
    "backend": "torch" | "onnx" | ...,
    "runtime_ms_model": 123,
    "resolved_model_id": "actual/revision"
  },
  "runtime_ms": 0,
  "model_id": "...",
  "model_meta": { /* from HF Hub when available */ }
}
```

#### Text generation example

```bash
curl -s -X POST http://localhost:8000/api/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "gpt2",
    "intent_id": "cli-example",
    "input_type": "text",
    "inputs": {"text": "Hello from hf_inference"},
    "task": "text-generation",
    "options": {"max_new_tokens": 32, "temperature": 0.8}
  }'
```

#### Text-to-image (diffusion) example

```bash
curl -s -X POST http://localhost:8000/api/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "runwayml/stable-diffusion-v1-5",
    "intent_id": "cli-example",
    "input_type": "text",
    "inputs": {"text": "a tiny cat on a skateboard"},
    "task": "text-to-image",
    "options": {"num_inference_steps": 20, "guidance_scale": 7.5}
  }'
```

The response contains `result.task_output.image_base64` with a `data:image/...;base64,...` URI.

#### Image classification example

```bash
IMG_PATH=./assets/image.jpg

curl -s -X POST http://localhost:8000/api/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "google/vit-base-patch16-224",
    "intent_id": "cli-example",
    "input_type": "image",
    "inputs": {"image_base64": "'"$(node -e 'console.log("data:image/jpeg;base64," + require("fs").readFileSync(process.argv[1]).toString("base64"))' "$IMG_PATH")"'"},
    "task": "image-classification",
    "options": {"top_k": 3}
  }'
```

> In practice, the frontend handles base64 encoding; this example shows the wire format.

#### Speech recognition (ASR) example

```bash
AUDIO=./assets/audio.wav

# Convert file to base64 data URI for the payload
DATA_URI="data:audio/wav;base64,$(base64 -w0 "$AUDIO")"

curl -s -X POST http://localhost:8000/api/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "openai/whisper-tiny.en",
    "intent_id": "cli-example",
    "input_type": "audio",
    "inputs": {"audio_base64": "'"$DATA_URI"'"}
  }'
```

## Streaming endpoints

For tasks with optional streaming support (e.g. text generation, TTS), use the `GET /api/streaming/...` endpoints.

### Text generation streaming example

```bash
curl -N -H 'Accept: text/event-stream' -X POST http://localhost:8000/api/streaming/text-generation \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "gpt2",
    "intent_id": "cli-example",
    "input_type": "text",
    "inputs": {"text": "Once upon a time"},
    "task": "text-generation",
    "options": {"max_new_tokens": 32, "temperature": 0.8}
  }'
```

### TTS streaming example

```bash
curl -N -H 'Accept: audio/wav' -X POST http://localhost:8000/api/streaming/text-to-speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "tts_model",
    "intent_id": "cli-example",
    "input_type": "text",
    "inputs": {"text": "Hello, world!"},
    "task": "text-to-speech",
    "options": {"sample_rate": 22050}
  }' > output.wav
```

> Note the use of `-N` to disable buffering for streaming responses.

## Supported tasks (examples) 📋

- Text: text-generation, text2text-generation, fill-mask, summarization, translation, question-answering, sentiment-analysis, token-classification
- Vision: image-classification, object-detection, image-segmentation, image-to-text, image-to-image, mask-generation, zero-shot-image-classification, zero-shot-object-detection
- Audio: audio-classification, automatic-speech-recognition, zero-shot-audio-classification, text-to-speech, text-to-audio
- Multimodal: image-text-to-text, visual-question-answering, table-question-answering, document-question-answering, depth-estimation, video-classification

Tip: GET /models without a task returns the exact list your server supports.

## Testing 🧪 (read before running)

Running the full pytest suite will download and execute many real, heavy models. Expect hundreds of GB of disk usage and
around 32GB of VRAM for smooth runs. Your SSD will hear about it.

Tips:

- Run a subset: `uv run pytest -k ["text_generation" | "image_classification"]`
- Faster downloads: set HF_HUB_ENABLE_HF_TRANSFER=1
- Put cache on a big disk: set HF_HOME or HUGGINGFACE_HUB_CACHE to a large path
- Inspect/clean cache: huggingface-cli cache info and huggingface-cli delete --help
- GPU memory: prefer -tiny/-base model variants if you hit OOM

Note: some tests assume online access to Hugging Face Hub and may be slow on first run while weights download.

## Configuration

CLI flags (also available via env):

- --host (HF_INF_HOST) default 0.0.0.0
- --port (HF_INF_PORT) default 8000
- --reload dev auto-reload
- --log-level (HF_INF_LOG_LEVEL) default info

Dev server (repo):

- `uv run uvicorn app.main:app --reload`

## Security notes (important) ⚠️

This project currently defaults to trust_remote_code=True in several loaders/pipelines. We verified this in the codebase
across utilities and multiple runners. Treat model loading as code execution.
Recommended:

- Pin model revisions (commit hashes)
- Audit model repositories before use
- Run inside hardened containers/VMs with minimal privileges
- Isolate network and secrets from the runtime process
- Prefer official models from trusted orgs for production

We plan to add a global toggle to disable trust_remote_code by default and allow explicit opt-in per request.

## Performance notes ⚡

- GPU recommended. CPU works but can be slow depending on the model.
- VRAM matters; some models require 8–16GB+. Smaller “-tiny/-base” variants help.
- Mixed precision often helps; some internal runners already opt into float16 where it’s safe.

## Development

Using uv and poe tasks.

- Install deps: `uv sync`
- Dev extras: `uv sync --extra dev`

Poe tasks (run with uv run poe <task>):

- test: run the test suite
  - `uv run poe test`
- format: format+lint with ruff
  - `uv run poe format`
- types: mypy type-checking
  - `uv run poe types`
- dev: start the dev server with auto-reload
  - `uv run poe dev`
- security: run safety and bandit
  - `uv run poe security`
- complexity: check code complexity with radon
  - `uv run poe complexity`
- deadcode: find unused code with vulture
  - `uv run poe deadcode`

## Contributing

See CONTRIBUTING.md

## Changelog

See CHANGELOG.md

## License

GPL-3.0-only. See LICENSE.

______________________________________________________________________

If this project saves you from writing one more one-off preprocessing script for “just this model,” it’s already doing
its job. A little less glue code; a lot more model poking. 😉
