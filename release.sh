#!/usr/bin/env bash
set -euo pipefail

PROJECT="hf-inference"
PYPROJECT="pyproject.toml"
CHANGELOG="CHANGELOG.md"

# --- Preconditions ---
[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || { echo "Error: not on 'main' branch."; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "Error: working tree not clean."; exit 1; }
[[ -f "$PYPROJECT" ]] || { echo "Error: $PYPROJECT not found."; exit 1; }

# --- Ask user for version and token ---
read -r -p "Version to release (e.g. 0.1.0): " VERSION
[[ -n "${VERSION}" ]] || { echo "Error: version is required."; exit 1; }

echo -n "Enter PyPI token (starts with 'pypi-'): "
read -r -s PYPI_TOKEN
echo
[[ -n "${PYPI_TOKEN}" ]] || { echo "Error: token is required."; exit 1; }
export UV_PUBLISH_TOKEN="${PYPI_TOKEN}"

# --- Validate version in pyproject.toml ---
FILE_VERSION="$(grep -E '^\s*version\s*=\s*"' "$PYPROJECT" | head -n1 | sed -E 's/.*version\s*=\s*"([^"]+)".*/\1/')"
if [[ "$FILE_VERSION" != "$VERSION" ]]; then
  echo "Error: pyproject version (${FILE_VERSION}) does not match entered version (${VERSION})."
  echo "Update 'version' in ${PYPROJECT} to ${VERSION} before releasing."
  exit 1
fi

# --- Validate version in CHANGELOG.md ---
if [[ -f "$CHANGELOG" ]]; then
  if ! grep -Eq "^(##|\#\#\#)\s*\[?${VERSION}\]?(\s*-|\s*\(|\s*$)" "$CHANGELOG"; then
    echo "Error: no CHANGELOG entry found for version ${VERSION} in ${CHANGELOG}."
    echo "Please add an entry (e.g., '## ${VERSION} - YYYY-MM-DD') and try again."
    exit 1
  fi
else
  echo "Warning: ${CHANGELOG} not found."
fi

# --- Ensure tag does not already exist ---
! git rev-parse "v${VERSION}" >/dev/null 2>&1 || { echo "Error: tag v${VERSION} already exists."; exit 1; }

echo "Releasing ${PROJECT} v${VERSION} to PyPI..."

# --- Create and push tag ---
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"

# --- Build from the exact tag in a temporary worktree ---
WORKTREE_DIR=".release-worktree-v${VERSION}"
rm -rf "$WORKTREE_DIR" || true
git worktree add --detach "$WORKTREE_DIR" "v${VERSION}"
trap 'git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true; rm -rf "$WORKTREE_DIR" || true' EXIT

# --- Sanity check inside worktree ---
WT_VERSION="$(grep -E '^\s*version\s*=\s*"' "${WORKTREE_DIR}/${PYPROJECT}" | head -n1 | sed -E 's/.*version\s*=\s*"([^"]+)".*/\1/')"
[[ "$WT_VERSION" == "$VERSION" ]] || { echo "Error: worktree version mismatch."; exit 1; }

echo "Building dist from tag v${VERSION}..."
( cd "$WORKTREE_DIR" && uv build )

echo "Publishing to PyPI..."
( cd "$WORKTREE_DIR" && uv publish )

echo "✅ Success! ${PROJECT} v${VERSION} published to PyPI."
echo
echo "To install:"
echo "  uv pip install ${PROJECT}==${VERSION}"
