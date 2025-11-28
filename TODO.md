# TODO

## General Instructions

All code should be implemented DRY, KISS, clean code.
Take care it has a well design folder structure.
Pick good naming for functions, variables, classes, files, folders, etc.

Try to implement the code functional with no side effects when possible.
Functions should only do one thing and do it well.
File should not become too large. Around 200 lines is a good limit.

Write tests for all the code. The tests should run real models.
It's ok if they take quite a while. Do not skip or mark them as slow.
Use the inference endpoint for the tests using a fastapi test client.

Imports should not be conditional. Assume libraries are installed. No
try/except around imports. Add necessary libraries using uv.

Use the `./frontend` as reference for what needs to be working in general.

Do not use environment variables to enable or disable capabilities; implement
capabilities as either always-on (hardcoded).

Do not add any fallbacks if a model does not run. It should be obvious that
something is not working as expected. Fallback would hide that fact.

## Phases

### Phase 1 – Critical runtime and API fixes

- ✅ Remove environment-variable-based behavior that changes runtime semantics (e.g. `LOG_LEVEL`, `HF_META_RETRIES`, `HF_META_TIMEOUT`, `ENRICH_MODELS_MAX`, `HF_INFERENCE_MAX_LOADED_MODELS`, `HF_INFERENCE_MEMORY_LIMIT_MB`, `FORCE_DEVICE`, `MAX_GPU_MEM_GB`) and replace it with hardcoded configuration or a static config object.
- ✅ Remove HTTP fallback and silent recovery paths for model metadata enrichment in `app/routers/inference.py::_fetch_full_model_meta`, `app/routers/models.py::_enrich_single_model`, and related helpers so model/hub failures are loud.
- ✅ Ensure non-streaming `/api/inference` responses return success with `result.task_output` (no `result.error`) and failures with structured `error` objects and proper HTTP status codes instead of always returning 200.
- ✅ Tighten the SPA catch-all in `app/main.py` so API paths return real FastAPI 404/405 (or explicit `HTTPException`) rather than a manual `{ "detail": "Not found" }` payload.
- ✅ Guard Hugging Face Hub calls (`_fetch_models`, enrichment helpers) with explicit limits/timeouts and raise on failures rather than logging and returning partial data.
- ✅ Replace broad `except Exception` blocks in inference/streaming/hub paths with narrow handling that rethrows unexpected errors so regressions are visible.
- ✅ Validate and sanitize per-task `inputs` on the backend, rejecting requests missing required keys (`text`, `image_base64`, `audio_base64`, etc.) instead of accepting arbitrary dicts.
- ✅ Ensure `ensure_task_supported` failures surface as HTTP errors for all endpoints (including streaming) so unsupported-device tasks never sneak through as soft errors.
- ✅ Remove the ONNX text-generation fallback in `ModelRegistry` that silently swaps runners; fail loudly when files are missing.
- ✅ Provide deterministic handling for `ModelRegistry` global state (`_models`, `_loading_futures`, eviction heuristics) so tests can reset/disable eviction and avoid flakiness.

### Phase 2 – Frontend–backend contract alignment

- Define shared contracts for `InferenceResponse`, `ModelSummary`, `ModelMeta`, streaming events, and error payloads, enforcing them via Pydantic models and TypeScript interfaces to prevent drift.
- Make runner failures speak clearly: propagate descriptive messages (with task/model context) from `ModelRegistry.predict` into structured error payloads instead of generic `inference_failed` strings.
- Standardize error handling in `useInference` and `useTextGenerationStream` so both HTTP failures and backend-declared logical errors populate a typed `error` field rendered by the UI.
- Make SSE payloads (`token`, `progress`, `done`, `error`) self-describing (e.g. include an internal `type`) and versioned, and update frontend parsers to branch on that type.
- Keep streaming vs non-streaming text-generation outputs semantically equivalent (same token sequences/final text) and document any unavoidable differences.
- Harmonize task/modality taxonomies across `app/core/tasks.py`, `taskModalities.ts`, `tasksCatalog.ts`, and `goalTaxonomy.ts`, clearly flagging unsupported tasks.
- Enforce input invariants on both frontend (`useModelExplorer`) and backend so missing modalities immediately trigger high-signal errors instead of runner crashes.
- Exercise the full contract with FastAPI `TestClient` integration tests that hit real models for representative tasks (text, image, audio, TTS) and assert both success and failure responses (unsupported tasks, missing inputs, device unavailable).
- Ensure model metadata enrichment (`model_meta`, `/models/enrich` gated flags) is fully represented in frontend types/components and surfaces explicit UI messaging when enrichment fails.

### Phase 3 – Improvements and polish

- Align backend `InferenceRequest`/`InferenceResponse` with frontend types by making `intent_id` optional in TS and adding fields like `task`, `options`, `task_output`, `error`, `runtime_ms_model`, `resolved_model_id`, `backend`.
- Update `useTextGenerationStream` to parse backend `error` events into `{ tokens, metrics, error: { message, details } }` instead of returning `'stream_error'`.
- Normalize logging namespaces so everything logs through `app.*`, with third-party loggers configured once in `configure_logging`.
- Replace per-request environment variable reads in hub helpers and registry/device code with a static configuration module imported at startup.
- Pass explicit limits to `api.list_models` (when possible) inside `_fetch_models`, and assert/log when manual iteration hits the configured cap.
- Add explicit return type hints for router functions (`models_status`, `unload_model`, `stream_inference`, `stream_text_to_image`, `stream_tts`, `preloaded_meta`, `enrich_models`).
- Unify `ModelEnriched`/`ModelSummary.cardData` definitions to avoid shadowing and ensure a single authoritative field.
- Document the SSE event contract in router docstrings and mirror it in typed frontend helpers/`RunRecord.streaming*` fields.
- Add focused tests for `models_preloaded.json`/`enrich_cache.json` disk persistence, TTL, and corruption scenarios.
- Rename ambiguous frontend variables/state (e.g. `m`, generic `data`) to descriptive names like `explorerState`, `inferenceResult`, `streamingMetrics`.
- Block the “Run” action when `extraArgsJson` is invalid, showing an inline error instead of silently sending `{}`.
- Guarantee `buildInferenceRequest` never sends empty `inputs` by adding a final validation step matching the backend guard.

### Phase 4 – Structural refactors

- Split `app/routers/inference.py` into focused modules (HTTP, SSE, metadata) to keep files under ~200 lines and separate concerns.
- Extract Hugging Face Hub access/caching into `app/core/hub_client.py` with pure functions (`list_popular_models`, `get_model_meta`, `enrich_models`).
- Introduce a domain layer (e.g. `TaskDispatcher`/`InferenceService`) around `REGISTRY.predict`/`SUPPORTED_TASKS` to centralize task routing, validation, and output normalization.
- Make caching helpers pure/state-driven so disk I/O is isolated and unit tests can cover cache transitions without touching the network.
- Ensure sync vs async boundaries are respected by running heavy `REGISTRY.predict` calls inside thread pools when invoked from async endpoints.
- Consolidate frontend API interaction into a typed client layer that wraps `/api/inference`, `/api/models/*`, `/api/intents`, and SSE endpoints, exposing reusable hooks.
- Add a lightweight state management layer (or structured React Query usage) to normalize models, intents, gated flags, and runs across `ExplorerFilters`, `ModelsGrid`, `RunPanel`, and `RunsDrawer`.
- Expand automated tests for streaming behavior (token order, `done` metrics, TTS chunking) and concurrent cache access, using real models where feasible.
- Document the full architecture (registry, device handling, runners, routers, task taxonomy, frontend flows) to speed onboarding and keep future work aligned.
- Define shared `ErrorResponse`/`ModelMeta` schemas reused across backend Pydantic models and frontend TS types, including streaming errors and model listings.
- Provide a deterministic `ModelRegistry` test mode/fixture that disables eviction and background async loading so real-model tests stay reproducible.
