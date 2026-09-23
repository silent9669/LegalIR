"""
Tests for exact Git commit SHA validation.
Authoritative production runs must never execute on 'main' or an unpinned SHA.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from src.release.fingerprints import assert_exact_git_sha, ShaMismatchError


VALID_SHA = "718efb7ba4565fa5b863f05927122484f8e58c2f"
DIFFERENT_SHA = "0000000000000000000000000000000000000000"


def _strict(monkeypatch):
    monkeypatch.setenv("LEGALIR_STRICT_GATES", "1")


def test_sha_match_passes():
    """Matching 40-char SHA passes without error."""
    with patch("src.release.fingerprints.get_git_head_sha", return_value=VALID_SHA):
        actual = assert_exact_git_sha(VALID_SHA, is_production=True)
        assert actual == VALID_SHA


def test_policy_a_enforces_exact_sha_without_strict_gates(monkeypatch):
    """Policy A: exact SHA is required independently of LEGALIR_STRICT_GATES."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    with patch("src.release.fingerprints.get_git_head_sha", return_value=VALID_SHA):
        with pytest.raises(ShaMismatchError, match="Git SHA mismatch"):
            assert_exact_git_sha(DIFFERENT_SHA, is_production=True)
        with pytest.raises(ShaMismatchError, match="Expected Git SHA must be provided"):
            assert_exact_git_sha("", is_production=True)
        with pytest.raises(ShaMismatchError, match="Literal branch"):
            assert_exact_git_sha("main", is_production=True)
        # Non-production with empty expected SHA passes through HEAD.
        assert assert_exact_git_sha("", is_production=False) == VALID_SHA


def test_sha_mismatch_fails_closed(monkeypatch):
    """Mismatched SHA raises ShaMismatchError in strict mode."""
    _strict(monkeypatch)
    with patch("src.release.fingerprints.get_git_head_sha", return_value=VALID_SHA):
        with pytest.raises(ShaMismatchError, match="Git SHA mismatch"):
            assert_exact_git_sha(DIFFERENT_SHA, is_production=True)


def test_unset_sha_fails_in_production(monkeypatch):
    """Empty or None expected SHA raises error in strict production."""
    _strict(monkeypatch)
    with patch("src.release.fingerprints.get_git_head_sha", return_value=VALID_SHA):
        with pytest.raises(ShaMismatchError, match="Expected Git SHA must be provided"):
            assert_exact_git_sha("", is_production=True)
        with pytest.raises(ShaMismatchError, match="Expected Git SHA must be provided"):
            assert_exact_git_sha(None, is_production=True)


def test_main_branch_string_fails_in_production(monkeypatch):
    """Literal 'main' or 'master' string is strictly rejected in strict production."""
    _strict(monkeypatch)
    with patch("src.release.fingerprints.get_git_head_sha", return_value=VALID_SHA):
        with pytest.raises(ShaMismatchError, match="Literal branch name 'main' is forbidden"):
            assert_exact_git_sha("main", is_production=True)
        with pytest.raises(ShaMismatchError, match="Literal branch name 'master' is forbidden"):
            assert_exact_git_sha("master", is_production=True)
