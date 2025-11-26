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

## Specific Tasks

### Phase A: Shared Utilities & Refactors
- [x] Extract common base64 encode/decode for images, audio, video, 3D (glb) into `app/core/utils/media.py`.
- [x] Standardize output schemas with Pydantic models per task group; reference in inference endpoint response assembly and accordingly update frontend.
- [x] Introduce task registry mapping task -> runner class; replace manual if/else in registry.
- [x] Add caching layer for diffusion model schedulers to reduce load time.

### Phase B: Performance & Resource Management
- [x] Add async loading queue to ModelRegistry for heavy models (diffusers, video) to avoid blocking inference requests.
- [x] Implement memory-based auto-precision (use fp16 if GPU + supports) for large models.
- [x] Track per-model load time and expose in `/api/models/status`.
- [x] Add eviction policy weights (LRU + size + last error).

### Phase C: Streaming Enhancements
- [x] Extend SSE streaming to multimodal and text-to-image (progress events: step, done) and accordingly update frontend.
- [x] Add chunked audio streaming for TTS when generating long outputs and update frontend player to handle streaming audio.
- [x] Tests: verify presence of `event: progress` events for diffusion tasks.

### Phase E: Documentation & Examples
- [x] Completely refactor README with new tasks table and curl examples.
      Make it very slim, but incl. the tasks supported.
- [x] Reimplement the curl examples in the frontend (in the runner for selected model).

### Phase F: Test Suite Expansion
- [ ] Parametrize existing tests to reduce duplication.
- [ ] Centralize all test fixtures in `tests/conftest.py`.
- [ ] Ensure each new runner and model arch has at least one integration test hitting `/api/inference`.
- [ ] Add regression test verifying unloading works for heavy diffusion models.
