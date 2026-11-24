"""Regression test for run_utils.git_state dirty_files parsing.

    python -m pytest tests/test_run_utils_git.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import run_utils as ru  # noqa: E402


def test_dirty_files_keeps_first_character(monkeypatch):
    # _git() strips its output, so the first porcelain line loses its leading
    # space (" M scripts/x.py" -> "M scripts/x.py"); later lines keep theirs.
    porcelain = " M scripts/0_img_metadata.py\n?? .pipeline/\nR  old name.py -> new.py".strip()
    answers = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("status", "--porcelain"): porcelain,
        ("rev-parse", "--abbrev-ref", "HEAD"): "dev-x",
        ("diff", "--shortstat"): "1 file changed",
    }
    monkeypatch.setattr(ru, "_git", lambda *a: answers[a])
    state = ru.git_state()
    assert state["dirty"] is True
    assert state["dirty_files"] == ["scripts/0_img_metadata.py", ".pipeline/", "old name.py -> new.py"]


def test_clean_tree_has_no_dirty_files(monkeypatch):
    answers = {("rev-parse", "HEAD"): "b" * 40, ("status", "--porcelain"): "",
               ("rev-parse", "--abbrev-ref", "HEAD"): "main"}
    monkeypatch.setattr(ru, "_git", lambda *a: answers[a])
    state = ru.git_state()
    assert state["dirty"] is False and "dirty_files" not in state
