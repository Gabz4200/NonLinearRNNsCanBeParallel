"""Upload this repository to Kaggle as a dataset."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

import kagglehub

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirrors .gitignore; kagglehub already ignores .git/ and .cache/.
_UPLOAD_IGNORE_PATTERNS = [
    "__pycache__/",
    "*.py[oc]",
    "*.egg-info/",
    ".venv/",
    ".ipynb_checkpoints/",
    "outputs/",
    "multirun/",
    "results/*/",
    "checkpoints/",
    "*.ckpt",
    "wandb/",
    "*.log",
    "lightning_logs/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".opencode/",
    ".omp",
    ".aislop/",
    "build/",
    "dist/",
    "wheels/",
    "coverage.xml",
    "htmlcov/",
]


def _resolve_username() -> str:
    username = os.environ.get("KAGGLE_USERNAME")
    if username:
        return username
    has_credentials = (
        (Path.home() / ".kaggle" / "kaggle.json").is_file()
        or os.environ.get("KAGGLE_KEY") is not None
        or os.environ.get("KAGGLE_API_TOKEN") is not None
    )
    if has_credentials:
        return str(kagglehub.whoami()["username"])
    return ""


def git_sha(repo_root: Path = _REPO_ROOT) -> str:
    """Short HEAD sha, or ``"nogit"`` outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=repo_root
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def is_dirty(repo_root: Path = _REPO_ROOT) -> bool:
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=repo_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out.strip())


def upload_dataset(
    slug: str,
    version_notes: str = "",
    repo_root: Path = _REPO_ROOT,
) -> str:
    """Upload repo contents as ``<KAGGLE_USERNAME>/<slug>``; returns the handle."""
    if not slug or "/" in slug or "\\" in slug:
        raise ValueError(f"Dataset slug must be a single name, got {slug!r}")
    username = _resolve_username()
    if not username:
        raise ValueError("Set KAGGLE_USERNAME (and KAGGLE_KEY, or ~/.kaggle/kaggle.json) to upload")
    handle = f"{username}/{slug}"
    kagglehub.dataset_upload(
        handle,
        str(repo_root),
        version_notes=version_notes,
        ignore_patterns=_UPLOAD_IGNORE_PATTERNS,
    )
    return handle


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Kaggle dataset slug (single name)")
    parser.add_argument("--version-notes", default="", help="Notes for this dataset version")
    args = parser.parse_args()
    handle = upload_dataset(args.slug, version_notes=args.version_notes)
    logging.getLogger(__name__).info("Uploaded https://www.kaggle.com/datasets/%s", handle)


if __name__ == "__main__":
    main()
