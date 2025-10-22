# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased] - YYYY-MM-DD

## [0.2.0] - 2025-10-22

### Added

- Interactive run modal with schema-driven forms
- File upload support with image/audio/video previews
- Progress timeline for inference status
- Inline media preview for generated outputs (images, audio, video)

### Changed

- Refactored inference modal controller for better readability
- Improved modal layout with scrollable Shoelace dialog
- Updated neon run button styling and accessibility
- Enlarged run column in models table

### Removed

- Legacy placeholder run modal template

## [0.1.0] - 2025-10-14

### Added

- Initial public release on PyPI
- FastAPI server with unified /inference endpoint
- 31+ Hugging Face tasks supported (text, vision, audio, video)
- Entry point `hf-inference`
- Tests, linting, typing setup
