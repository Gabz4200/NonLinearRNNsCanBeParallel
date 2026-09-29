"""Pre-commit hook: upload the repo as a Kaggle dataset when credentials exist.

Skips (exit 0) without KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json, so
commits never block on missing credentials. Git introspection degrades to
"nogit" instead of crashing outside a checkout. Run via `uv run --no-sync`
so the local env (with kagglehub) is used without a lockfile sync.
"""

from __future__ import annotations

from nonlinearrnnscanbeparallel.scripts.upload_dataset import (
    _resolve_username,
    git_sha,
    is_dirty,
    upload_dataset,
)

if not _resolve_username():
    raise SystemExit(0)

_sha = git_sha()
_notes = f"pre-commit auto-upload {_sha}" + (" (dirty tree)" if is_dirty() else "")
upload_dataset("nonlinear-rnns-can-be-parallel", version_notes=_notes)
