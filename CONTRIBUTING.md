# Contributing to HF Inference

Thanks for your interest in contributing!

- Development environment: Python 3.12+, uv installed (pipx install uv)
- Install deps: uv sync and uv sync --extra dev
- Lint/format: uv run ruff format && uv run ruff check --fix
- Type-check: uv run mypy
- Tests: uv run pytest -ra

Workflow

- Create a feature branch from main.
- Add tests for new features and bug fixes.
- Keep PRs small and focused.
- Update CHANGELOG.md under Unreleased.

Commit style

- Use clear, descriptive messages. Examples: fix: ..., feat: ..., docs: ...

Release process (maintainers)

- Ensure CI is green and CHANGELOG.md is up to date.
- Bump version in pyproject.toml.
- Build distributions: uv build (produces wheel and sdist in dist/).
- Publish to PyPI:
  - TestPyPI: uv publish --repository testpypi --token "$TEST_PYPI_API_TOKEN"
  - PyPI: uv publish --token "$PYPI_API_TOKEN"
- Verify project page rendering and install from PyPI.

Security

- Report vulnerabilities privately via GitHub Security Advisories.

License

- By contributing, you agree to license your contributions under GPL-3.0-only.
