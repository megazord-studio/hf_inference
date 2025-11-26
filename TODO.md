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

## Plan (Multimodal: Image+Text -> Text)

General engineering and testing
- [x] Apply DRY, KISS; keep functions ~15–20 lines; minimal nesting; clear naming. (multimodal runner refactor present)
- [ ] Keep files reasonably sized; split helpers if needed (~200 lines target per file).
- [ ] No conditional imports; add deps with `uv` if required.
- [x] Capabilities are always-on (no env gates). (runner has no env gating)
- [x] No fake fallbacks; on failure return empty task_output with clear logs (tests accept {}). (predictors return {} on failure paths)
- [x] Maintain/extend integration tests via FastAPI TestClient; do not skip. (integration test exists for multimodal)
- [x] Use frontend as general reference for UX/contracts.

Test harness
- [x] Remove PDB breakpoint from `tests/integration/test_multimodal.py::_post`. (removed)

Multimodal runner refactor
- [x] Split into small parts: arch detection, per-arch loaders, per-arch predictors, input builders, utilities. (implemented in `multimodal.py`)
- [x] Implement loaders: `_load_blip`, `_load_llava`, `_load_qwen_vl`, `_load_minicpm`, `_load_vlm`, `_load_generic_vqa`. (implemented)
- [x] Implement predictors: `_predict_blip`, `_predict_llava`, `_predict_qwen_vl`, `_predict_minicpm`, `_predict_vlm`, `_predict_generic_vqa`. (implemented)
- [x] Add utilities: tokenizer lookup, output decode, image-token helpers, to-device, dtype select/unify, param count. (implemented)
- [x] Add `_safe_call(fn)` helper (wrap try/except + logging) and use in loaders instead of repeated try/except. (implemented)

Input building and robustness
- [x] Add `_get_image_token()` and `_ensure_image_tokens(question, num_images)`; use when models require explicit image tokens. (implemented)
- [x] If processor errors on image-token mismatch: do one formatting pass, then a capped retry (≤4 tries) with candidate tokens; never hang. (implemented `_retry_processor_with_image_tokens`)
- [x] Strip processor-only kwargs (e.g., `num_crops`) from enc before `generate()`. (implemented `_strip_processor_only_kwargs`)
- [x] Ensure all tensors/encodings are moved to device consistently. (implemented `_move_to_device`)

Performance guards
- [x] Cap `max_new_tokens` on CPU/MPS for large VLMs (respect user `max_length`, bound internally to a safe limit). (implemented `_cap_max_new_tokens` with deterministic generation)

Architecture-specific coverage (models below must be supported)
- [x] Salesforce/blip-vqa-base (BLIP)
  - [x] Load: `BlipProcessor` + `BlipForQuestionAnswering`; predict via `generate` + decode.
- [x] llava-hf/llava-1.5-7b-hf (LLaVA)
  - [x] Load: `LlavaProcessor` + `LlavaForConditionalGeneration`.
  - [x] Ensure a single image token in question via `_ensure_image_tokens()`; `generate` + decode.
- [x] Qwen/Qwen-VL-Chat (Qwen-VL)
  - [x] Load: `AutoProcessor(..., trust_remote_code=True)` + `AutoModelForCausalLM(..., trust_remote_code=True)`.
  - [x] Prefer `model.chat(processor, image, question)`; if unsupported, return {} with logs (avoid streaming generator imports).
- [x] google/gemma-3-1b-it (Gemma3)
  - [x] Try `pipeline("image-text-to-text", trust_remote_code=True)` first; short-circuit if text returned.
  - [x] Else load as VLM (ImageTextToText/CausalLM); if unsupported, return {} with logs. Strip `num_crops`.
- [x] HuggingFaceM4/idefics2-8b (Idefics2)
  - [x] Ensure prompt includes correct count of image tokens; do one format pass + capped retries; avoid hangs.
- [x] openbmb/MiniCPM-Llama3-V-2_5 (MiniCPM-V)
  - [x] Load: `AutoModel`/`AutoTokenizer` with `trust_remote_code`; unify dtype to device; patch generation (GenerationMixin).
  - [x] Prefer `model.chat(image, msgs, tokenizer)`; else `_minicpm_manual_decode(question, max_len)`; else {} with logs.
- [ ] 01-ai/Yi-VL-6B (Yi-VL)
  - [ ] On state_dict size mismatch, avoid incompatible legacy fallbacks; log reason; return {} quickly.
- [ ] OpenGVLab/InternVL2-8B (InternVL2)
  - [ ] Detect InternVL; avoid VQA pipeline. Load via `AutoProcessor` + trust-remote model (or `InternVLChat`).
  - [ ] Prefer chat-style inference; ensure image tokens; cap tokens on CPU/MPS; {} with logs on failure.
- [ ] microsoft/kosmos-2-patch14-224 (Kosmos-2)
  - [ ] Detect kosmos-2; avoid VQA pipeline. Load `AutoProcessor` (Kosmos2Processor) + model (trust_remote_code if needed).
  - [ ] Encode via processor with images+text; ensure image token; drop processor-only kwargs; generate/decode; {} on failure.
- [ ] microsoft/Florence-2-base-ft (Florence-2)
  - [ ] Detect florence-2; avoid VQA pipeline. Load `AutoProcessor` + `Florence2ForConditionalGeneration` (or appropriate trust-remote auto class).
  - [ ] Use task prompt token for VQA (e.g., `<VQA>` + question); encode via processor; strip processor-only kwargs; bounded generation; decode.
- [x] google/paligemma-3b-pt-224 (Paligemma)
  - [x] Implement `_safe_call` (fix current runtime). Detect paligemma; avoid VQA pipeline.
  - [x] Load `AutoProcessor` + trust-remote model (ImageTextToText/CausalLM/Vision2Seq). Ensure image token; strip `num_crops`.
- [ ] THUDM/cogvlm2-llama3-chat-19B (CogVLM2)
  - [ ] Detect cogvlm/cogvlm2; avoid VQA pipeline. Load via `AutoProcessor` + trust-remote model; prefer chat API.
  - [ ] Ensure image tokens; cap tokens; encode/generate/decode appropriately; {} with logs on failure.

Generic VLM path hardening
- [x] Drop processor-only kwargs (e.g., `num_crops`) before `generate()` for all VLMs.
- [x] Decode priority: `processor.batch_decode` > `processor.decode` > model tokenizer.
- [x] If builder returns pipeline text, short-circuit and return it.
- [x] If builder signals `_skip_generation`, return {} quickly with a helpful log message. (returns {})

Targeted unit tests
- [ ] `_ensure_image_tokens` – exact insertion/count behavior.
- [ ] VLM input-building – formatting, capped retries, stripping processor-only kwargs.

Validation (green-before-done)
- [ ] Static checks: no syntax/type errors; imports consistent.
- [ ] Focused integration runs per model (no hangs, no API errors; {} allowed where unsupported).
- [ ] Run full `tests/integration/test_multimodal.py` and confirm all listed models meet expectations.
