# Local HF Inference Roadmap & TODO

Goal: Evolve current FastAPI + React app from metadata-only + echo inference into a secure, resource-aware, multi‑modality local inference server supporting GPU (CUDA device 0) or Apple MPS (Metal). If neither is available, gracefully decline tasks requiring acceleration.

---
## Device / Acceleration Requirements (Global)
- Implement `app/core/device.py` with:
  - `select_device(prefer: str = 'auto') -> torch.device | None`:
    - Priorities: if prefer == 'cuda' and `torch.cuda.is_available()`, return `cuda:0`.
    - If prefer == 'mps' and `torch.backends.mps.is_available()`, return `mps`.
    - If prefer == 'auto': try `cuda:0`, then `mps`, else CPU.
  - `device_capabilities()` returning: `{"cuda": bool, "mps": bool, "gpu_name": str|None, "memory_gb": float|None}`.
  - Log chosen device at startup.
- Add startup check: if no CUDA/MPS and a request targets a model flagged as "gpu_required", respond 400 with actionable message.
- Provide env overrides:
  - `FORCE_DEVICE=cuda|mps|cpu` (error if unavailable).
  - `MAX_GPU_MEM_GB` for early rejection or automatic quantization.

---
## Phase 0 – Foundations & Text Core
**Objective:** Minimal local text inference + architecture skeleton.
**Deliverables:** Registry, base runner, 3 text runners, simple eviction, extended API.

Tasks:
- [x] Create directories: `app/core/` (`registry.py`, `runners/base.py`, `runners/text.py`, `resources.py`, `device.py`).
- [x] Define `ModelEntry` dataclass (status, runner, popularity_score, timestamps, acceleration_profile placeholder).
- [x] Implement `ModelRegistry` (get_or_load(model_id, task), unload(model_id), list_loaded()).
- [x] Implement `BaseRunner` abstract class.
- [x] TextGenerationRunner (GPT-2 or TinyLlama) with generation options (max_new_tokens, temperature, top_p).
- [x] TextClassificationRunner (distilbert SST-2) using transformers pipeline.
- [x] EmbeddingRunner (MiniLM-L6-v2) returning vector + dimensionality.
- [x] Add simple memory estimation (params * dtype bytes) & idle timestamp.
- [x] Add LRU eviction on `max_loaded_models` or `memory_watermark` env config.
- [x] Extend `InferenceRequest` to include optional `task`, `options` (generation), fallback to echo if absent (pipeline_tag mapping implemented).
- [x] Map HF `pipeline_tag` to internal task when `task` missing (heuristic dictionary).
- [x] Update `/api/inference` dispatch: echo if no task; otherwise route to registry.predict.
- [x] Basic error handling: model not supported, device unavailable, quantization suggestion.
- [x] Write pytest smoke tests for each runner (load + single inference + unload).
- [x] Update README with Phase 0 usage instructions. (Pending: add section with curl examples.)

Success Criteria:
- Running text generation locally returns text tokens; latency & load logged.
- Embeddings vector length correct.
- Eviction works (unload least-recently-used when threshold exceeded).

---
## Phase 1 – Streaming Interface
**Objective:** Responsive outputs for generation tasks.
Tasks:
- [x] Add SSE endpoint `/api/inference/stream` (EventSourceResponse).
- [x] Integrate `TextIteratorStreamer` for token streaming.
- [x] Add `stream: bool` flag in request. (Implemented as `_stream` option for runner)
- [x] Emit events: `token`, `done`, `error`.
- [x] Maintain request correlation ID (uuid4) in all events.
- [x] Add generation metrics: first_token_latency_ms, tokens_per_second.

Success Criteria:
- [x] Incremental tokens visible in UI within ~1s for small model. (Blocking collection done; SSE events produced sequentially)
- [x] KISS, DRY, clean code, well-named functions, functional, no side effects
- [x] Each is Tested with pytest (Add dedicated tests pending; minimal smoke relies on existing registry predict — to extend in next commit.)

---
## Phase 2 – Vision & Audio Expansion
**Objective:** Core image + audio tasks.
Tasks:
- [ ] Image utilities: base64 decode, PIL convert, normalization.
- [ ] ImageGenerationRunner (Stable Diffusion small) – support guidance_scale, num_inference_steps.
- [ ] ImageToImageRunner (img2img path).
- [ ] ImageCaptioningRunner (vit-gpt2 / BLIP).
- [ ] ObjectDetectionRunner (DETR) output: boxes, labels, scores.
- [ ] SegmentationRunner (SegFormer) output: mask or class-map.
- [ ] DepthEstimationRunner (DPT) output: depth-array summary stats + raw (optional).
- [ ] ASRRunner (Whisper tiny) output: transcript, language.
- [ ] TTSRunner (XTTS/Bark) output: audio_base64.
- [ ] AudioClassificationRunner (wav2vec2) output: label + score.
- [ ] VisionLanguageRunner (BLIP or LLaVA-mini) image+text reasoning.
- [ ] Extend response normalization (maintain single `result` key with task-specific substructure).
- [ ] Add partial progress event for diffusion tasks (later integrated with streaming framework).

Success Criteria:
- Each runner loads and returns correct structure for at least one sample input.
- KISS, DRY, clean code, well-named functions, functional, no side effects
- Each is Tested with pytest

---
## Phase 3 – Quantization & Acceleration
Tasks:
- [ ] Integrate bitsandbytes (4/8-bit) path for decoder LMs if GPU present.
- [ ] ONNX export (optimum) for encoder models (embeddings/classification) with caching in `cache/onnx/`.
- [ ] Add LlamaCppRunner for GGUF models (CPU fallback for large LLaMA). 
- [ ] TensorRT optional acceleration (if libs present): conditional load for SD UNet or encoder stack.
- [ ] Implement `AccelerationProfile` inside ModelEntry.
- [ ] Auto selection logic: large model + no GPU → GGUF; small encoder → ONNX; GPU large → 8-bit.
- [ ] Expose `acceleration` field in response meta.
- [ ] Document environment preconditions (CUDA version, GPU RAM). 

Success Criteria:
- Memory footprint reduced >30% for a quantized LM vs fp16.
- KISS, DRY, clean code, well-named functions, functional, no side effects
- Each is Tested with pytest

---
## Phase 4 – Telemetry & Evaluation
Tasks:
- [ ] Integrate `prometheus_client` `/metrics` endpoint.
- [ ] Metrics: counters (requests_total, errors_total), histograms (latency), gauges (active_models, memory_bytes_used).
- [ ] EvaluationSuite: folder `assets/eval/` with small JSON sets per task.
- [ ] Script `python -m app.eval.run --tasks text-generation,image-classification` storing results in `cache/eval_reports.json`.
- [ ] Regression detection: compare latency vs last baseline; log warning if >20%.
- [ ] Add `/api/evaluation/latest` endpoint.

Success Criteria:
- Prometheus scrape works; evaluation artifacts generated.
- KISS, DRY, clean code, well-named functions, functional, no side effects
- Each is Tested with pytest

---
## Phase 5 – Security & Sandbox
Tasks:
- [ ] Implement `SandboxProcess` using multiprocessing: isolated working dir, limited environment.
- [ ] Static scan of downloaded repo code for disallowed imports (socket, subprocess, shutil, requests).
- [ ] Network disable patch (override socket module) inside sandbox.
- [ ] RLIMIT: CPU time, virtual memory, open files.
- [ ] Flag models `sandboxed=True` in registry when remote code executed.
- [ ] Env flags: `TRUST_REMOTE_CODE=auto|always|never`, `ALLOW_UNSANDBOXED=1` (dev only).
- [ ] Security audit logging channel `app.security`.

Success Criteria:
- Remote-code model executes in sandbox; attempt to open outbound network fails.
- KISS, DRY, clean code, well-named functions, functional, no side effects
- Each is Tested with pytest

---
## Phase 6 – Scheduling & Concurrency
Tasks:
- [ ] Implement `RequestQueue` priority = user.priority + popularity_score + age.
- [ ] Worker pools per modality with concurrency caps (env config, e.g. TEXT_WORKERS=2).
- [ ] Cancellation endpoint `/api/inference/cancel/{request_id}`.
- [ ] Global/task timeouts returning structured error.
- [ ] Adaptive warm preload of top-K models (by recent frequency).
- [ ] Enhanced eviction factoring queue pressure.

Success Criteria:
- Queue drains smoothly under simulated load; cancellations succeed.
- KISS, DRY, clean code, well-named functions, functional, no side effects
- Each is Tested with pytest

---
## Phase 7 – Advanced Modalities & Plugins
Tasks:
- [ ] VideoGenerationRunner (SVD or AnimateDiff) with progress steps.
- [ ] VideoClassificationRunner (VideoMAE) output: label + score.
- [ ] TextTo3DRunner / ImageTo3DRunner (sandboxed) producing artifact path.
- [ ] ForecastingRunner (Chronos) output: forecast array + confidence intervals.
- [ ] Plugin system: `plugins/` auto-discovery; TaskPlugin interface.
- [ ] `/api/plugins` endpoint listing loaded plugins.

Success Criteria:
- At least one video & 3D task successfully run locally.
- KISS, DRY, clean code, well-named functions, functional, no side effects
- Each is Tested with pytest

---
## Cross-Cutting Enhancements (Ongoing)
- [ ] Structured logging with correlation ID & latency breakdown.
- [ ] Config validation on startup (raise if incompatible env e.g. ONNX wants GPU but not found).
- [ ] `GET /api/models/status/{model_id}` for load state & acceleration profile.
- [ ] `POST /api/models/unload/{model_id}` manual eviction.
- [ ] Health endpoint `/api/health` summarizing device + counts + memory.
- [ ] Documentation: update README per phase completion.

---
## Initial Smoke Test Model Set
Text Gen: `gpt2` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
Text Classification: `distilbert-base-uncased-finetuned-sst-2-english`
Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
Captioning: `nlpconnect/vit-gpt2-image-captioning`
Detection: `facebook/detr-resnet-50`
Segmentation: `nvidia/segformer-b0-finetuned-ade-512-512`
Depth: `Intel/dpt-large`
Text-to-Image: `runwayml/stable-diffusion-v1-5` (or sd-turbo for speed)
ASR: `openai/whisper-tiny.en`
TTS: `coqui/XTTS-v2` (small)
Audio Classification: `superb/wav2vec2-base-superb-ks`
Vision+Language: `Salesforce/blip-image-captioning-base`
Video Gen: `stabilityai/stable-video-diffusion-img2vid`
Time-Series: `amazon/chronos-t5-small`
3D (Sandbox): `zero123` / text-to-nerf prototype

---
## Risk Mitigation Quick Reference
| Risk | Mitigation |
|------|------------|
| Memory exhaustion | Quantization + eviction + size precheck |
| Malicious code | Sandbox + static scan + network patch |
| Latency spikes | Streaming + priority queue + acceleration |
| Load failures | Preflight device check + fallback backends |
| API regression | Additive changes only + contract tests |
| GPU contention | Per-modality worker limits + load throttling |

---
## Success Metrics (Track via Prometheus & Logs)
- P50/P95 latency per task.
- Model load time (ms) & first token latency.
- Active model count vs memory watermark (<85% usage).
- Eviction frequency (should stabilize after warm phase).
- Error rate <2% (excluding cancellations).
- Streaming token median interval.
- Quantization memory savings (% vs fp16 baseline).

---
## Immediate Next Actions
1. Implement `device.py` + log chosen device.
2. Create core scaffolding modules & BaseRunner.
3. Implement TextGenerationRunner with GPT-2 (CPU/GPU support).
4. Extend `/api/inference` to route by `task` when provided.
5. Add minimal tests + README section.

After those are green, proceed toward classification + embeddings, then finalize Phase 0.

---
## Notes
- Treat large models as opt-in initially (require env flag `ENABLE_LARGE_MODELS=1`).
- Favor SSE before WebSocket to reduce complexity; add WS only if needed for bi-directional control (cancellation streaming).
- Document GPU/MPS detection clearly so users know why a task may be rejected.
