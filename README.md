# HF Inference

A small FastAPI server and CLI that make trying Hugging Face models consistent. One endpoint. Many tasks. Fewer surprises.

> 🚧 Status: HEAVY DEVELOPMENT – not stable yet
>
> ⚠️ Security disclaimer: by default this project loads models with transformers trust_remote_code=True in multiple
> runners/utilities. Remote model repos may execute arbitrary Python when loading. Do not run untrusted models in
> production. Prefer sandboxing (containers/VMs), pin models, audit code, and isolate credentials/network.

## Quick links

- Install
- Quick start
- Docker
- API overview
- Supported tasks
- Examples
- Testing (read this)
- Security notes
- Performance notes
- Development
- Contributing

## Why this exists 🧭

Trying different HF models is great until each one “speaks” a slightly different API dialect. This project gives you:

- One simple REST endpoint for 31+ tasks
- Consistent request/response shapes across models
- A quick model catalog to discover candidates

In short: less boilerplate, fewer notebook tabs, more experiments per minute.

## What you get 🧰

- Single POST /inference endpoint that handles text, image, audio, and video tasks
- 31+ tasks across text, vision, audio, and multimodal
- Minimal model catalog API and a lightweight HTML UI for discovery
- CLI: hf-inference to start the server quickly

## Installation

Prereqs (recommended):

- Python 3.12+
- `ffmpeg` (video)
- `tesseract-ocr`, `libtesseract-dev`, `libleptonica-dev` (OCR tasks)

Install from PyPI:

- `pip install hf-inference`

Note: Large dependencies (PyTorch, Transformers, Diffusers). Expect hefty downloads and build times.

## Quick start 🚀

Start the server (default 0.0.0.0:8000):

- `uv run poe dev`

```bash
# Health check:
curl http://localhost:8000/healthz

# First request (text generation):
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"gpt2","task":"text-generation","payload":{"prompt":"Hello HF"}}'
```

## Docker 🐳

```bash
# Pull image (GPU):
docker pull ghcr.io/megazord-studio/hf-inference:gpu-latest

# Run (GPU, NVIDIA runtime required):
docker run --rm -it --gpus all -p 8000:8000 \
  ghcr.io/megazord-studio/hf-inference:gpu-latest \
  hf-inference --host 0.0.0.0 --port 8000

# Persist model/cache data (recommended):
docker run --rm -it --gpus all -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  ghcr.io/megazord-studio/hf-inference:gpu-latest \
  hf-inference --host 0.0.0.0 --port 8000
```

Notes:

- First run may download large models inside the container; use a volume to avoid redownloading. 🗂️
- A CPU-only image may be published separately; GPU image expects NVIDIA Container Toolkit.

## API overview 🔌

### POST /inference

Multipart form accepting:

- spec: JSON string
  - model_id: str (e.g., "gpt2" or "google/vit-base-patch16-224")
  - task: str (pipeline tag; see Supported tasks)
  - payload: object (task-specific kwargs)
- image: optional file
- audio: optional file
- video: optional file

Responses:

- JSON for textual results
- Streaming file for binary outputs when applicable (with Content-Disposition)

Example specs:

- Text generation: `{"model_id":"gpt2","task":"text-generation","payload":{"prompt":"..."}`}\`
- Image classification: `{"model_id":"google/vit-base-patch16-224","task":"image-classification","payload":{}}`
- ASR (speech): `{"model_id":"openai/whisper-tiny","task":"automatic-speech-recognition","payload":{}}`

### GET /healthz

- Returns `{ status, device }`

### GET /models?task=...

- Returns minimal metadata for public models of a task: id, likes, trendingScore, downloads, gated
- If task is missing, returns available_tasks (the supported tasks on this server)

### GET /

- Lightweight HTML table for quick sorting/filtering of a task’s models, backed by /models

## Supported tasks (examples) 📋

- Text: text-generation, text2text-generation, fill-mask, summarization, translation, question-answering, sentiment-analysis, token-classification
- Vision: image-classification, object-detection, image-segmentation, image-to-text, image-to-image, mask-generation, zero-shot-image-classification, zero-shot-object-detection
- Audio: audio-classification, automatic-speech-recognition, zero-shot-audio-classification, text-to-speech, text-to-audio
- Multimodal: image-text-to-text, visual-question-answering, table-question-answering, document-question-answering, depth-estimation, video-classification

Tip: GET /models without a task returns the exact list your server supports.

## Examples

```bash
# Text generation:
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"gpt2","task":"text-generation","payload":{"prompt":"Hello world"}}'

# Image classification:
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"google/vit-base-patch16-224","task":"image-classification","payload":{}}' \
  -F 'image=@/path/to/image.jpg'

# Speech recognition:
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"openai/whisper-tiny","task":"automatic-speech-recognition","payload":{}}' \
  -F 'audio=@/path/to/audio.wav'
```

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
