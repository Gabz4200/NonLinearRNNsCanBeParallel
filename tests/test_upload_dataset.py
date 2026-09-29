"""Tests for the Kaggle dataset upload script (validation only, no network)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import kagglehub
import pytest

from nonlinearrnnscanbeparallel.scripts.upload_dataset import (
    _resolve_username,
    git_sha,
    is_dirty,
    upload_dataset,
)


def test_when_username_in_env_then_resolves_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "gabrielsamaral")
    assert _resolve_username() == "gabrielsamaral"


def test_when_token_and_whoami_then_resolves_username(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test-token")
    monkeypatch.setattr(kagglehub, "whoami", lambda: {"username": "gabrielsamaral"})
    assert _resolve_username() == "gabrielsamaral"


def test_when_no_credentials_then_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert _resolve_username() == ""


def test_when_slug_contains_separator_then_raises() -> None:
    with pytest.raises(ValueError, match="single name"):
        upload_dataset("foo/bar")


def test_when_empty_slug_then_raises() -> None:
    with pytest.raises(ValueError, match="single name"):
        upload_dataset("")


def test_when_outside_git_then_sha_is_nogit_and_clean(tmp_path: Path) -> None:
    assert git_sha(tmp_path) == "nogit"
    assert is_dirty(tmp_path) is False


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_when_untracked_file_then_dirty(tmp_path: Path) -> None:
    subprocess.check_output(["git", "init"], text=True, cwd=tmp_path)
    assert is_dirty(tmp_path) is False
    (tmp_path / "note.txt").write_text("hi")
    assert is_dirty(tmp_path) is True
